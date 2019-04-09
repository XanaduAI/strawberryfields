# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Unit tests for Gate classes in ops.py"""
import pytest

import numpy as np

from strawberryfields import ops
from strawberryfields.backends.base import BaseGaussian, BaseFock


# make test deterministic
np.random.random(42)
a = 0.1234
b = -0.543
c = 0.312


class TestEngineReset:
    """Test engine reset functionality"""

    def test_init_vacuum(self, setup_eng, tol):
        """Test that the engine is initialized to the vacuum state"""
        eng, _ = setup_eng(2)
        assert np.all(eng.backend.is_vacuum(tol))

    def test_reset_vacuum(self, setup_eng, pure, cutoff, tol):
        """Test that resetting the engine returns to the vacuum state"""
        eng, q = setup_eng(2)

        with eng:
            ops.Dgate(0.5) | 0

        eng.run()
        assert not np.all(eng.backend.is_vacuum(tol))

        eng.reset(pure=pure)
        assert np.all(eng.backend.is_vacuum(tol))

    @pytest.mark.backends("fock")
    def test_eng_reset(self, setup_eng, cutoff):
        """Test the Engine.reset() features."""
        eng, _ = setup_eng(2)

        state = eng.run()

        # change the cutoff dimension
        old_cutoff = eng.backend.get_cutoff_dim()
        assert state._cutoff == old_cutoff
        assert cutoff == old_cutoff

        new_cutoff = old_cutoff + 1
        eng.reset(cutoff_dim=new_cutoff)

        state = eng.run()
        temp = eng.backend.get_cutoff_dim()

        assert temp == new_cutoff
        assert state._cutoff == new_cutoff


@pytest.mark.parametrize("gate", ops.gates)
class TestGateApplication:
    """Engine tests that involve gate application"""

    @pytest.fixture
    def G(self, gate):
        """Initialize each gate"""
        if gate in ops.zero_args_gates:
            return gate

        if gate in ops.one_args_gates:
            return gate(a)

        if gate in ops.two_args_gates:
            return gate(a, b)

    def test_gate_dagger_vacuum(self, G, setup_eng, tol):
        """Test applying gate inverses after the gate cancels out"""
        eng, q = setup_eng(2)

        if isinstance(G, (ops.Vgate, ops.Kgate, ops.CKgate)) and isinstance(
            eng.backend, BaseGaussian
        ):
            pytest.skip("Non-Gaussian gates cannot be applied to the Gaussian backend")

        with eng:
            if G.ns == 1:
                G | q[0]
                G.H | q[0]
            elif G.ns == 2:
                G | (q[0], q[1])
                G.H | (q[0], q[1])

        state = eng.run()
        # we must end up back in vacuum since G and G.H cancel each other
        assert np.all(eng.backend.is_vacuum(tol))


class TestProperExecution:
    """Test that various frontend circuits execute through
    the backend with no error"""

    # TODO: Some of these tests should probably check *something* after execution

    def test_regref_transform(self, setup_eng):
        """Test circuit with regref transforms doesn't raise any errors"""
        eng, q = setup_eng(2)

        with eng:
            ops.MeasureX | q[0]
            ops.Sgate(q[0]) | q[1]
            # symbolic hermitian conjugate together with register reference
            ops.Dgate(q[0]).H | q[1]
            ops.Sgate(ops.RR(q[0], lambda x: x ** 2)) | q[1]
            ops.Dgate(ops.RR(q[0], lambda x: -x)).H | q[1]

        eng.run()

    def test_homodyne_measurement_vacuum(self, setup_eng, tol):
        """MeasureX and MeasureP leave the mode in the vacuum state"""
        eng, q = setup_eng(2)

        with eng:
            ops.Coherent(a, c) | q[0]
            ops.Coherent(b, c) | q[1]
            ops.MeasureX | q[0]
            ops.MeasureP | q[1]

        eng.run()
        assert np.all(eng.backend.is_vacuum(tol))

    def test_homodyne_measurement_vacuum_phi(self, setup_eng, tol):
        """Homodyne measurements leave the mode in the vacuum state"""
        eng, q = setup_eng(2)

        with eng:
            ops.Coherent(a, b) | q[0]
            ops.MeasureHomodyne(c) | q[0]

        eng.run()
        assert np.all(eng.backend.is_vacuum(tol))

    def test_program_subroutine(self, setup_eng, tol):
        """Simple quantum program with a subroutine and references."""
        eng, q = setup_eng(2)

        # define some gates
        D = ops.Dgate(0.5)
        BS = ops.BSgate(0.7 * np.pi, np.pi / 2)
        R = ops.Rgate(np.pi / 3)
        # get register references
        alice, bob = q

        def subroutine(a, b):
            """Subroutine for the quantum program"""
            R | a
            BS | (a, b)
            R.H | a

        # main program
        with eng:
            ops.All(ops.Vacuum()) | (alice, bob)
            D | alice
            subroutine(alice, bob)
            BS | (alice, bob)
            subroutine(bob, alice)

        state = eng.run()

        # state norm must be invariant
        if isinstance(eng.backend, BaseFock):
            assert np.allclose(state.trace(), 1, atol=tol, rtol=0)

    def test_checkpoints(self, setup_eng, tol):
        """Test history checkpoints work when creating and deleting modes."""
        eng, q = setup_eng(2)
        alice, bob = q

        # define some gates
        D = ops.Dgate(0.5)
        BS = ops.BSgate(2 * np.pi, np.pi / 2)
        R = ops.Rgate(np.pi)

        with eng:
            D | alice
            BS | (alice, bob)
            ops.Del | alice
            R | bob
            charlie, = ops.New(1)
            BS | (bob, charlie)
            ops.MeasureX | bob
            ops.Del | bob
            D.H | charlie
            ops.MeasureX | charlie

        eng.optimize()
        state = eng.run()

        # state norm must be invariant
        if isinstance(eng.backend, BaseFock):
            assert np.allclose(state.trace(), 1, atol=tol, rtol=0)

        def check_reg(self, expected_n=None):
            """Compare Engine.register with the mode list returned by the backend.
            They should always be in agreement after Engine.run(), Engine.reset_queue()
            and Engine.reset().
            """
            rr = eng.register
            modes = eng.backend.get_modes()
            # number of elements
            assert len(rr) == len(modes)

            if expected_n is not None:
                assert len(rr) == expected_n

            # check indices match
            assert np.all([r.ind for r in rr] == modes)
            # activity
            assert np.all([r.active for r in rr])

        # check that reset() works
        check_reg(1)
        eng.reset()

        new_reg = eng.register
        # original number of modes
        assert len(new_reg) == len(q)

        # the regrefs are reset as well
        assert np.all([r.val is None for r in new_reg])
        check_reg(2)

    def test_apply_history(self, setup_eng, pure):
        """Tests the reapply history argument works correctly with a backend"""
        eng, q = setup_eng(2)

        a = 0.23
        r = 0.1

        def inspect():
            res = []
            print_fn = lambda x: res.append(x.__str__())
            eng.print_applied(print_fn)
            return res

        with eng:
            ops.Dgate(a) | q[0]
            ops.Sgate(r) | q[1]

        state1 = eng.run()
        expected = [
            "Run 0:",
            "Dgate({}, 0) | (q[0])".format(a),
            "Sgate({}, 0) | (q[1])".format(r),
        ]

        assert inspect() == expected

        # reset backend, but reapply history
        eng.backend.reset(pure=pure)
        state2 = eng.run(apply_history=True)
        assert inspect() == expected
        assert state1 == state2

        # append more commands to the same backend
        with eng:
            ops.Rgate(r) | q[0]

        state3 = eng.run()
        expected = [
            "Run 0:",
            "Dgate({}, 0) | (q[0])".format(a),
            "Sgate({}, 0) | (q[1])".format(r),
            "Run 1:",
            "Rgate({}) | (q[0])".format(r),
        ]

        assert inspect() == expected
        assert not state2 == state3

        # reset backend, but reapply history
        eng.backend.reset(pure=pure)
        state4 = eng.run(apply_history=True)
        expected = [
            "Run 0:",
            "Dgate({}, 0) | (q[0])".format(a),
            "Sgate({}, 0) | (q[1])".format(r),
            "Rgate({}) | (q[0])".format(r),
        ]

        assert inspect() == expected
        assert state3 == state4

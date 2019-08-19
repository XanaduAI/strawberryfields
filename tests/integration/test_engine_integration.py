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
r"""Integration tests for the frontend engine.py module with the backends"""
import numbers
import pytest

import numpy as np

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.backends import BaseGaussian, BaseFock
from strawberryfields.backends import GaussianBackend, FockBackend
from strawberryfields.backends.states import BaseState


try:
    from strawberryfields.backends.tfbackend import TFBackend
except (ImportError, ModuleNotFoundError, ValueError) as e:
    eng_backend_params = [("gaussian", GaussianBackend), ("fock", FockBackend)]
else:
    eng_backend_params = [
        ("tf", TFBackend),
        ("gaussian", GaussianBackend),
        ("fock", FockBackend),
    ]


# make test deterministic
np.random.seed(42)
a = 0.1234
b = -0.543
c = 0.312


@pytest.mark.parametrize("name,expected", eng_backend_params)
def test_load_backend(name, expected, cutoff):
    """Test backends can be correctly loaded via strings"""
    eng = sf.Engine(name)
    assert isinstance(eng.backend, expected)


class TestEngineReset:
    """Test engine reset functionality"""

    def test_init_vacuum(self, setup_eng, tol):
        """Test that the engine is initialized to the vacuum state"""
        eng, prog = setup_eng(2)
        eng.run(prog)  # run an empty program
        assert np.all(eng.backend.is_vacuum(tol))

    def test_reset_vacuum(self, setup_eng, tol):
        """Test that resetting the engine returns to the vacuum state"""
        eng, prog = setup_eng(2)

        with prog.context:
            ops.Dgate(0.5) | 0

        eng.run(prog)
        assert not np.all(eng.backend.is_vacuum(tol))

        eng.reset()
        assert np.all(eng.backend.is_vacuum(tol))

    @pytest.mark.backends("fock")
    def test_eng_reset(self, setup_eng, cutoff):
        """Test the Engine.reset() features."""
        eng, prog = setup_eng(2)

        state = eng.run(prog).state
        backend_cutoff = eng.backend.get_cutoff_dim()
        assert state._cutoff == backend_cutoff
        assert cutoff == backend_cutoff

        # change the cutoff dimension
        new_cutoff = cutoff + 1
        eng.reset({"cutoff_dim": new_cutoff})

        state = eng.run(prog).state
        backend_cutoff = eng.backend.get_cutoff_dim()
        assert state._cutoff == backend_cutoff
        assert new_cutoff == backend_cutoff


class TestProperExecution:
    """Test that various frontend circuits execute through
    the backend with no error"""

    def test_no_return_state(self, setup_eng):
        """Engine returns no state object when none is requested."""
        eng, prog = setup_eng(2)
        res = eng.run(prog, run_options={"modes": []})
        assert res.state is None

    def test_return_state(self, setup_eng):
        """Engine returns a valid state object."""
        eng, prog = setup_eng(2)
        res = eng.run(prog)
        assert isinstance(res.state, BaseState)

    def test_return_samples(self, setup_eng):
        """Engine returns measurement samples."""
        eng, prog = setup_eng(2)
        with prog.context as q:
            ops.MeasureX | q[0]

        res = eng.run(prog, run_options=None)
        # one entry for each mode
        assert len(res.samples) == 2
        # the same samples can also be found in the regrefs
        assert [r.val for r in prog.register] == res.samples
        # first mode was measured
        assert isinstance(res.samples[0], (numbers.Number, np.ndarray))
        # second mode was not measured
        assert res.samples[1] is None

    # TODO: Some of these tests should probably check *something* after execution

    def test_regref_transform(self, setup_eng):
        """Test circuit with regref transforms doesn't raise any errors"""
        eng, prog = setup_eng(2)

        with prog.context as q:
            ops.MeasureX | q[0]
            ops.Sgate(q[0]) | q[1]
            # symbolic hermitian conjugate together with register reference
            ops.Dgate(q[0]).H | q[1]
            ops.Sgate(ops.RR(q[0], lambda x: x ** 2)) | q[1]
            ops.Dgate(ops.RR(q[0], lambda x: -x)).H | q[1]

        eng.run(prog)

    def test_homodyne_measurement_vacuum(self, setup_eng, tol):
        """MeasureX and MeasureP leave the mode in the vacuum state"""
        eng, prog = setup_eng(2)
        with prog.context as q:
            ops.Coherent(a, c) | q[0]
            ops.Coherent(b, c) | q[1]
            ops.MeasureX | q[0]
            ops.MeasureP | q[1]

        eng.run(prog)
        assert np.all(eng.backend.is_vacuum(tol))

    def test_homodyne_measurement_vacuum_phi(self, setup_eng, tol):
        """Homodyne measurements leave the mode in the vacuum state"""
        eng, prog = setup_eng(2)
        with prog.context as q:
            ops.Coherent(a, b) | q[0]
            ops.MeasureHomodyne(c) | q[0]

        eng.run(prog)
        assert np.all(eng.backend.is_vacuum(tol))

    def test_program_subroutine(self, setup_eng, tol):
        """Simple quantum program with a subroutine and references."""
        eng, prog = setup_eng(2)

        # define some gates
        D = ops.Dgate(0.5)
        BS = ops.BSgate(0.7 * np.pi, np.pi / 2)
        R = ops.Rgate(np.pi / 3)

        def subroutine(a, b):
            """Subroutine for the quantum program"""
            R | a
            BS | (a, b)
            R.H | a

        # main program
        with prog.context as q:
            # get register references
            alice, bob = q
            ops.All(ops.Vacuum()) | (alice, bob)
            D | alice
            subroutine(alice, bob)
            BS | (alice, bob)
            subroutine(bob, alice)

        state = eng.run(prog).state

        # state norm must be invariant
        if isinstance(eng.backend, BaseFock):
            assert np.allclose(state.trace(), 1, atol=tol, rtol=0)

    def test_subsystems(self, setup_eng, tol):
        """Check that the backend keeps in sync with the program when creating and deleting modes."""
        null = sf.Program(2)  # empty program
        eng, prog = setup_eng(2)

        # define some gates
        D = ops.Dgate(0.5)
        BS = ops.BSgate(2 * np.pi, np.pi / 2)
        R = ops.Rgate(np.pi)

        with prog.context as q:
            alice, bob = q
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

        def check_reg(p, expected_n=None):
            """Compare Program.register with the mode list returned by the backend.
            They should always be in agreement after Engine.run() and Engine.reset().
            """
            rr = p.register
            modes = eng.backend.get_modes()
            # number of elements
            assert len(rr) == len(modes)

            if expected_n is not None:
                assert len(rr) == expected_n

            # check indices match
            assert np.all([r.ind for r in rr] == modes)
            # activity
            assert np.all([r.active for r in rr])

        state = eng.run(null)
        check_reg(null, 2)
        state = eng.run(prog).state
        check_reg(prog, 1)

        # state norm must be invariant
        if isinstance(eng.backend, BaseFock):
            assert np.allclose(state.trace(), 1, atol=tol, rtol=0)

        # check that reset() works
        eng.reset()
        # the regrefs are reset as well
        assert np.all([r.val is None for r in prog.register])

    def test_empty_program(self, setup_eng):
        """Empty programs do not change the state of the backend."""
        eng, p1 = setup_eng(2)
        a = 0.23
        r = 0.1
        with p1.context as q:
            ops.Dgate(a) | q[0]
            ops.Sgate(r) | q[1]
        state1 = eng.run(p1).state

        # empty program
        p2 = sf.Program(p1)
        state2 = eng.run(p2).state
        assert state1 == state2

        p3 = sf.Program(p2)
        with p3.context as q:
            ops.Rgate(r) | q[0]
        state3 = eng.run(p3).state
        assert not state1 == state3

        state4 = eng.run(p2).state
        assert state3 == state4

    # TODO: when ``shots`` is incorporated into other backends, unmark this test
    @pytest.mark.backends("gaussian")
    def test_measurefock_shots(self, setup_eng):
        """Tests that passing shots with a program containing MeasureFock
           returns a result whose entries have the right shapes and values"""
        shots = 5
        expected = np.zeros(dtype=int, shape=(shots,))

        # all modes
        eng, p1 = setup_eng(3)
        with p1.context as q:
            ops.MeasureFock() | q
        samples = eng.run(p1, run_options={"shots": shots}).samples.astype(int)
        assert samples.shape == (shots, 3)
        assert all(samples[:, 0] == expected)
        assert all(samples[:, 1] == expected)
        assert all(samples[:, 2] == expected)

        # some modes
        eng, p2 = setup_eng(3)
        with p2.context as q:
            ops.MeasureFock() | (q[0], q[2])
        samples = eng.run(p2, run_options={"shots": shots}).samples
        assert samples.shape == (shots, 3)
        assert all(samples[:, 0].astype(int) == expected)
        assert all(s is None for s in samples[:, 1])
        assert all(samples[:, 2].astype(int) == expected)

        # one mode
        eng, p3 = setup_eng(3)
        with p3.context as q:
            ops.MeasureFock() | q[0]
        samples = eng.run(p3, run_options={"shots": shots}).samples
        assert samples.shape == (shots, 3)
        assert all(samples[:, 0].astype(int) == expected)
        assert all(s is None for s in samples[:, 1])
        assert all(s is None for s in samples[:, 2])

    # TODO: when ``shots`` is incorporated into other backends, delete this test
    @pytest.mark.backends("tf", "fock")
    def test_measurefock_shots_exception(self, setup_eng):
        shots = 5
        eng, p1 = setup_eng(3)
        with p1.context as q:
            ops.MeasureFock() | q

        backend_name = eng.backend.__str__()
        with pytest.raises(NotImplementedError,
                           match=r"""(Measure|MeasureFock) has not been implemented in {} """
                                  """for the arguments {{'shots': {}}}""".format(backend_name, shots)):
            eng.run(p1, run_options={"shots": shots}).samples

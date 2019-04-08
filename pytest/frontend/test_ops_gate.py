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
r"""Integration tests to make sure hbar values set correctly in returned results"""
import pytest

pytestmark = pytest.mark.frontend

import numpy as np

import strawberryfields as sf

from strawberryfields import ops
from strawberryfields.engine import Engine, MergeFailure, RegRefError
from strawberryfields import utils
from strawberryfields.parameters import Parameter

# make test deterministic
np.random.random(42)
a = np.random.random()
b = np.random.random()
c = np.random.random()


@pytest.mark.parametrize("state", ops.state_preparations)
class TestStatePreparationBasics:
    """Basic properties of state preparation operations."""

    def test_merge(self, state):
        """Test that merging states simply returns the second state"""
        # state to test against
        V = ops.Vacuum()
        G = state()

        # merging with another state returns the last one
        assert np.all(G.merge(V) == V)
        assert np.all(V.merge(G) == G)

    def test_exceptions(self, state):
        """Test exceptions raised if state prep used on invalid modes"""
        G = state()
        eng, _ = sf.Engine(2)

        with eng:
            # all states act on a single mode
            with pytest.raises(ValueError):
                G.__or__([0, 1])

            # can't repeat the same index
            with pytest.raises(RegRefError):
                ops.All(G).__or__([0, 0])


@pytest.mark.parametrize("gate", ops.gates)
class TestGateBasics:
    """Test the basic properties of gates"""

    @pytest.fixture(autouse=True)
    def eng(self):
        """Dummy engine context for each test"""
        eng, _ = sf.Engine(2)
        Engine._current_context = eng
        yield eng
        Engine._current_context = None

    @pytest.fixture
    def Q(self):
        """The common test gate"""
        return ops.Xgate(0.5)

    @pytest.fixture
    def G(self, gate):
        """Initialize each gate"""
        if gate in ops.zero_args_gates:
            return gate

        if gate in ops.one_args_gates:
            return gate(a)

        if gate in ops.two_args_gates:
            return gate(a, b)

    @pytest.fixture
    def H(self, gate):
        """Second gate fixture of the same class, with same phase as G"""
        if gate in ops.zero_args_gates:
            return gate

        if gate in ops.one_args_gates:
            return gate(c)

        if gate in ops.two_args_gates:
            return gate(c, b)

    def test_merge_inverse(self, G):
        """gate merged with its inverse is the identity"""
        assert G.merge(G.H) is None

    def test_merge_different_gate(self, G, Q):
        """gates cannot be merged with a different type of gate"""
        if isinstance(G, Q.__class__):
            pytest.skip("Gates are the same type.")

        with pytest.raises(MergeFailure):
            Q.merge(G)

        with pytest.raises(MergeFailure):
            G.merge(Q)

    def test_wrong_number_subsystems(self, G):
        """wrong number of subsystems"""
        if G.ns == 1:
            with pytest.raises(ValueError):
                G.__or__([0, 1])
        else:
            with pytest.raises(ValueError):
                G.__or__(0)

    def test_repeated_index(self, G):
        """multimode gates: can't repeat the same index"""
        if G.ns == 2:
            with pytest.raises(RegRefError):
                G.__or__([0, 0])

    def test_non_trivial_merging(self, G, H):
        """test the merging of two gates (with default values
        for optional parameters)"""
        if G in ops.zero_args_gates:
            pytest.skip("Gates with no arguments are not merged")

        a, b = np.random.random([2])
        merged = G.merge(H)

        # should not be the identity
        assert merged is not None

        # first parameters should be added
        assert merged.p[0] == G.p[0] + H.p[0]

        # merge G with conjugate transpose of H
        merged = G.merge(H.H)

        # should not be the identity
        assert merged is not None

        # first parameters should be subtracted
        assert merged.p[0] == G.p[0] - H.p[0]


class TestChannelBasics:
    """Test the basic properties of channels"""

    def test_loss_merging(self, tol):
        """test the merging of two Loss channels (with default values
        for optional parameters)"""
        G = ops.LossChannel(a)
        merged = G.merge(ops.LossChannel(b))
        assert np.allclose(merged.p[0].x, a * b, atol=tol, rtol=0)

    def test_thermalloss_merging_same_nbar(self, tol):
        """test the merging of two Loss channels with same nbar"""
        G = ops.ThermalLossChannel(a, c)
        merged = G.merge(ops.ThermalLossChannel(b, c))
        assert np.allclose(merged.p[0].x, a * b, atol=tol, rtol=0)

    def test_thermalloss_merging_different_nbar(self, tol):
        """test the merging of two Loss channels with same nbar raises exception"""
        G = ops.ThermalLossChannel(a, 2 * c)
        with pytest.raises(MergeFailure):
            merged = G.merge(ops.ThermalLossChannel(b, c))

    def test_thermalloss_merging_different_nbar(self, tol):
        """test the merging of Loss and ThermalLoss raises exception"""
        G = ops.ThermalLossChannel(a, 2 * c)
        with pytest.raises(MergeFailure):
            merged = G.merge(ops.LossChannel(b))


class TestEngineGateInteraction:
    """Test gates correctly dispatch/modify the engine"""

    @pytest.fixture
    def eng(self):
        """Dummy engine context for each test"""
        eng, _ = sf.Engine(2)
        Engine._current_context = eng
        yield eng
        Engine._current_context = None

    @pytest.mark.parametrize("gate", ops.one_args_gates + ops.two_args_gates)
    def test_dispatch_one_mode_gates(self, gate):
        """test one mode gates automatically add to the queue"""
        eng, _ = sf.Engine(2)
        G = gate(a)

        if G.ns == 2:
            pytest.skip("test only for 1 mode gates.")

        with eng:
            G | 0
            ops.All(G) | (0, 1)

        assert len(eng.cmd_queue) == 3
        assert all(cmd.op == G for cmd in eng.cmd_queue)
        assert eng.cmd_queue[0].reg[0].ind == 0
        assert eng.cmd_queue[1].reg[0].ind == 0
        assert eng.cmd_queue[2].reg[0].ind == 1

    @pytest.mark.parametrize("gate", ops.one_args_gates + ops.two_args_gates)
    def test_dispatch_two_mode_gates(self, gate):
        """test two mode gates automatically add to the queue"""
        eng, _ = sf.Engine(3)
        G = gate(a)

        if G.ns == 1:
            pytest.skip("test only for 1 mode gates.")

        with eng:
            G | (0, 1)
            G | (0, 2)

        assert len(eng.cmd_queue) == 2
        assert all(cmd.op == G for cmd in eng.cmd_queue)
        assert eng.cmd_queue[0].reg[0].ind == 0
        assert eng.cmd_queue[0].reg[1].ind == 1
        assert eng.cmd_queue[1].reg[0].ind == 0
        assert eng.cmd_queue[1].reg[1].ind == 2

    def test_create_or_exception(self):
        """New must not be called via its __or__ method"""
        with pytest.raises(ValueError):
            ops.New.__or__(0)

    def test_create_non_positive_integer(self, eng):
        """number of new modes must be a positive integer"""
        with pytest.raises(ValueError):
            ops.New.__call__(-2)

        with pytest.raises(ValueError):
            ops.New.__call__(1.5)

    def test_delete_not_existing(self, eng):
        """deleting nonexistent modes not allowed"""
        with pytest.raises(RegRefError):
            ops.Del.__or__(100)

    def test_delete(self, eng):
        """test deleting a mode"""
        q = eng.register
        assert q[1].active
        ops.Del | q[1]
        assert not q[1].active

    def test_create(self, eng):
        """test creating a mode"""
        q = eng.register
        new_q, = ops.New(1)
        assert new_q.active

    def test_delete_already_deleted(self, eng):
        """deleting a mode that was already deleted"""
        q = eng.register
        ops.Del | q[1]
        with pytest.raises(RegRefError):
            ops.Del.__or__(1)

    def test_create_delete_multiple_modes(self):
        """test creating and deleting multiple modes"""
        eng, (alice, bob, charlie) = sf.Engine(3)

        with eng:
            edward, frank, grace = ops.New(3)
            ops.Del | (alice, grace)

        # register should only return the active subsystems
        q = eng.register
        assert len(q) == eng.num_subsystems
        assert len(q) == 4
        # Engine.reg_refs contains all the regrefs, active and inactive
        assert len(eng.reg_refs) == 6

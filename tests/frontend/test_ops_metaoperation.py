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
r"""Unit tests for MetaOperation classes in ops.py"""
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

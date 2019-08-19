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

import strawberryfields.program_utils as pu
from strawberryfields import ops
from strawberryfields.program import Program
from strawberryfields.program_utils import MergeFailure, RegRefError, CircuitError
from strawberryfields import utils
from strawberryfields.parameters import Parameter

# make test deterministic
np.random.seed(42)
a = np.random.random()
b = np.random.random()


class TestProgramGateInteraction:
    """Test gates correctly dispatch/modify the program"""

    @pytest.fixture
    def prog(self):
        """Dummy program context for each test"""
        prog = Program(2)
        pu.Program_current_context = prog
        yield prog
        pu.Program_current_context = None

    @pytest.mark.parametrize("gate", ops.one_args_gates + ops.two_args_gates)
    def test_dispatch_one_mode_gates(self, gate):
        """test one mode gates automatically add to the queue"""
        prog = Program(2)

        if gate in ops.two_args_gates:
            G = gate(a, b)
        else:
            G = gate(a)

        if G.ns == 2:
            pytest.skip("test only for 1 mode gates.")

        with prog.context:
            G | 0
            ops.All(G) | (0, 1)

        assert len(prog) == 3
        assert all(cmd.op == G for cmd in prog.circuit)
        assert prog.circuit[0].reg[0].ind == 0
        assert prog.circuit[1].reg[0].ind == 0
        assert prog.circuit[2].reg[0].ind == 1

    @pytest.mark.parametrize("gate", ops.one_args_gates + ops.two_args_gates)
    def test_dispatch_two_mode_gates(self, gate):
        """test two mode gates automatically add to the queue"""
        prog = Program(3)

        if gate in ops.two_args_gates:
            G = gate(a, b)
        else:
            G = gate(a)

        if G.ns == 1:
            pytest.skip("test only for 1 mode gates.")

        with prog.context:
            G | (0, 1)
            G | (0, 2)

        assert len(prog) == 2
        assert all(cmd.op == G for cmd in prog.circuit)
        assert prog.circuit[0].reg[0].ind == 0
        assert prog.circuit[0].reg[1].ind == 1
        assert prog.circuit[1].reg[0].ind == 0
        assert prog.circuit[1].reg[1].ind == 2

    def test_create_or_exception(self):
        """_New_modes must not be called via its __or__ method"""
        with pytest.raises(ValueError, match='Wrong number of subsystems'):
            ops._New_modes(1).__or__(0)

    def test_create_outside_program_context(self):
        """New() must be only called inside a Program context."""
        with pytest.raises(RuntimeError, match='can only be called inside a Program context'):
            ops.New()

    def test_create_non_positive_integer(self, prog):
        """number of new modes must be a positive integer"""
        with pytest.raises(ValueError, match='is not a positive integer'):
            ops.New(-2)
        with pytest.raises(ValueError, match='is not a positive integer'):
            ops.New(1.5)

    def test_create_locked(self, prog):
        """No new modes can be created in a locked Program."""
        prog.lock()
        with pytest.raises(CircuitError, match='The Program is locked, no new subsystems can be created'):
            ops.New(1)

    def test_delete_locked(self, prog):
        """No modes can be deleted in a locked Program."""
        prog.lock()
        with pytest.raises(CircuitError, match='The Program is locked, no more Commands can be appended to it'):
            ops.Del | 0

    def test_delete_not_existing(self, prog):
        """deleting nonexistent modes not allowed"""
        with pytest.raises(RegRefError, match='does not exist'):
            ops.Del.__or__(100)

    def test_delete(self, prog):
        """test deleting a mode"""
        q = prog.register
        assert q[1].active
        assert prog.num_subsystems == 2
        ops.Del | q[1]
        assert not q[1].active
        assert prog.num_subsystems == 1

    def test_create(self, prog):
        """test creating a mode"""
        q = prog.register
        assert prog.num_subsystems == 2
        new_q, = ops.New(1)
        assert new_q.active
        assert prog.num_subsystems == 3

    def test_delete_already_deleted(self, prog):
        """deleting a mode that was already deleted"""
        q = prog.register
        ops.Del | q[1]
        with pytest.raises(RegRefError, match='has already been deleted'):
            ops.Del.__or__(1)

    def test_create_delete_multiple_modes(self):
        """test creating and deleting multiple modes"""
        prog = Program(3)

        with prog.context as (alice, bob, charlie):
            edward, frank, grace = ops.New(3)
            ops.Del | (alice, grace)

        # register should only return the active subsystems
        q = prog.register
        assert len(q) == prog.num_subsystems
        assert len(q) == 4
        # Program.reg_refs contains all the regrefs, active and inactive
        assert len(prog.reg_refs) == 6


class TestOperationDeprecation:
    """Tests for operation deprecation"""

    def test_measure_deprecation(self):
        """Test that use of the Measure shorthand correctly
        raises a deprecation warning"""

        msg = r"The shorthand '{}' has been deprecated, please use '{}\(\)' instead"

        with pytest.warns(UserWarning, match=msg.format("Measure", "MeasureFock")):
            ops.Measure

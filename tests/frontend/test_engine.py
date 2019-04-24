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
import logging

logging.getLogger()

import pytest

pytestmark = pytest.mark.frontend

import numpy as np

import strawberryfields as sf
from strawberryfields import engine
from strawberryfields import program
from strawberryfields import ops
from strawberryfields.backends.base import BaseBackend




@pytest.fixture
def eng(backend):
    """Engine fixture."""
    return sf.Engine(backend)

@pytest.fixture
def prog():
    """Program fixture."""
    return sf.Program(2)


def test_load_backend(monkeypatch):
    """Backend can be correctly loaded via strings"""
    eng = sf.Engine('base')
    assert isinstance(eng.backend, BaseBackend)


class TestProgram:
    """Tests the Program class."""

    def test_with_block(self, prog):
        """Gate application using a with block."""
        # command queue is empty
        assert len(prog) == 0

        with prog.context as q:
            ops.Dgate(0.5) | q[0]
        # now there is one gate in the queue
        assert len(prog) == 1

        with prog.context as q:
            ops.BSgate(0.5, 0.3) | (q[1], q[0])
        assert len(prog) == 2

    def test_identity_command(self, prog):
        """Tests that the None command acts as the identity"""
        identity = program.Command(None, prog.register[0])
        prog.circuit.append(identity)
        assert len(prog) == 1
        prog = prog.compile('base')
        assert len(prog) == 0

    def test_parent_program(self):
        """Continuing one program with another."""
        D = ops.Dgate(0.5)
        prog = sf.Program(3)
        with prog.context as q:
            D       | q[1]
            ops.Del | q[0]
        cont = sf.Program(prog)
        with cont.context as q:
            D | q[0]
            r = ops.New(1)
            D | r
        assert cont.can_follow(prog)
        assert prog.reg_refs == cont.init_reg_refs
        assert prog.unused_indices == cont.init_unused_indices

    def test_print_commands(self, eng, prog):
        """Program.print and Engine.print_applied return correct strings."""
        prog = sf.Program(2)
        res = []
        print_fn = lambda x: res.append(x.__str__())

        # prog should now be empty
        prog.print(print_fn)
        assert res == []

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
            ops.Dgate(bob).H | charlie
            ops.Del | bob
            ops.MeasureX | charlie

        res = []
        prog.print(print_fn)

        expected = [
            "Dgate(0.5, 0) | (q[0])",
            "BSgate(6.283, 1.571) | (q[0], q[1])",
            "Del | (q[0])",
            "Rgate(3.142) | (q[1])",
            "New(1)",
            "BSgate(6.283, 1.571) | (q[1], q[2])",
            "MeasureX | (q[1])",
            "Dgate(RR(q[1]), 0).H | (q[2])",
            "Del | (q[1])",
            "MeasureX | (q[2])",
        ]

        assert res == expected

        state = eng.run(prog, compile=False)  # FIXME optimization can change gate order, however this is not a good way of avoiding it
        res = []
        eng.print_applied(print_fn)
        assert res == ["Run 0:"] + expected


class TestRegRefs:
    """Testing register references."""

    def test_corrupt_index(self, prog):
        """User messes up the RegRef indices."""
        with prog.context as q:
            q[0].ind = 1
            with pytest.raises(program.RegRefError, match="RegRef state has become inconsistent"):
                ops.Dgate(0.5) | q[0]

    def test_nonexistent_index(self, prog):
        """Acting on a non-existent mode raises an error."""
        with prog.context as q:
            with pytest.raises(IndexError):
                ops.Dgate(0.5) | q[3]
            with pytest.raises(program.RegRefError, match="does not exist"):
                ops.Dgate(0.5) | 3

    def test_deleted_index(self, prog):
        """Test that acting on a deleted mode raises an error"""
        with prog.context as q:
            ops.Del | q[0]
            with pytest.raises(program.RegRefError, match="been deleted"):
                ops.Dgate(0.5) | 0

            with pytest.raises(program.RegRefError, match="been deleted"):
                ops.Dgate(0.5) | q[0]

    def test_unknown_regref(self, prog):
        """Applying an operation on a RegRef not belonging to the program."""
        r = program.RegRef(3)
        with prog.context:
            with pytest.raises(program.RegRefError, match="Unknown RegRef."):
                ops.Dgate(0.5) | r

    def test_invalid_measurement(self, eng, prog):
        """Cannot use a measurement before it exists."""
        with prog.context as q:
            ops.Dgate(q[0]) | q[1]
        with pytest.raises(program.CircuitError, match="nonexistent measurement result"):
            eng.run(prog)

    def test_invalid_regref(self, prog):
        """Cannot refer to a register with a non-integral or RegRef object."""
        with prog.context:
            with pytest.raises(program.RegRefError, match="using integers and RegRefs"):
                ops.Dgate(0) | 1.2

    def test_regref_val(self, eng, prog):
        """RegRefs storing measured values."""
        with prog.context as q:
            ops.MeasureX | q[0]

        eng.run(prog)
        assert q[0].val is not None
        assert q[1].val is None
        eng.reset()
        # the regrefs are reset as well
        temp = [r.val is None for r in prog.register]
        assert np.all(temp)

        eng.run(prog)
        assert q[0].val is not None
        assert q[1].val is None
        prog._clear_regrefs()
        assert np.all([r.val is None for r in prog.register])


class TestEngine:
    """Test the Engine class and its interaction with Program instances."""

    def test_no_return_state(self, eng, prog):
        """Engine returns None when no state is requested"""
        with prog.context as q:
            ops.Dgate(0.34) | q[0]

        res = eng.run(prog, return_state=False)
        assert res is None

    def test_history(self, eng, prog):
        """Engine history."""
        with prog.context as q:
            ops.Dgate(0.5) | q[0]

        # no programs have been run
        assert not eng.run_progs
        eng.run(prog)
        # one program has been run
        assert len(eng.run_progs) == 1
        assert eng.run_progs[-1].source == prog

    def test_reset(self, eng):
        """Running independent programs with an engine reset in between."""
        D = ops.Dgate(0.5)
        p1 = sf.Program(2)
        with p1.context as q:
            D       | q[1]
        assert not eng.run_progs
        eng.run(p1)
        assert len(eng.run_progs) == 1

        eng.reset()
        assert not eng.run_progs

        p2 = sf.Program(3)
        with p2.context as q:
            D | q[2]
        eng.run(p2)
        assert len(eng.run_progs) == 1

    def test_regref_mismatch(self, eng):
        """Running incompatible programs sequentially gives an error."""
        p1 = sf.Program(3)
        p2 = sf.Program(p1)
        p1.locked = False
        with p1.context as q:
            ops.Del | q[0]

        with pytest.raises(RuntimeError, match="Register mismatch"):
            eng.run([p1, p2])

    def test_sequential_programs(self, eng):
        """Running several program segments sequentially."""
        D = ops.Dgate(0.2)
        p1 = sf.Program(3)
        with p1.context as q:
            D       | q[1]
            ops.Del | q[0]
        assert not eng.run_progs
        eng.run(p1)
        assert len(eng.run_progs) == 1

        p2 = sf.Program(p1)
        with p2.context as q:
            D | q[1]
        eng.run(p2)
        assert len(eng.run_progs) == 2

        # p2 does not alter the register so it can be repeated
        eng.run([p2] * 3)
        assert len(eng.run_progs) == 5

        eng.reset()
        assert not eng.run_progs

    def test_apply_history(self, eng):
        """Tests the reapply history argument"""
        a = 0.23
        r = 0.1

        def inspect():
            res = []
            print_fn = lambda x: res.append(x.__str__())
            eng.print_applied(print_fn)
            return res

        p1 = sf.Program(2)
        with p1.context as q:
            ops.Dgate(a) | q[1]
            ops.Sgate(r) | q[1]

        eng.run(p1)
        expected = [
            "Run 0:",
            "Dgate({}, 0) | (q[1])".format(a),
            "Sgate({}, 0) | (q[1])".format(r),
        ]
        assert inspect() == expected

        # run the program again
        eng.reset()
        eng.run(p1)
        assert inspect() == expected

        # apply more commands to the same backend
        p2 = sf.Program(2)
        with p2.context as q:
            ops.Rgate(r) | q[1]

        eng.run(p2)
        expected = [
            "Run 0:",
            "Dgate({}, 0) | (q[1])".format(a),
            "Sgate({}, 0) | (q[1])".format(r),
            "Run 1:",
            "Rgate({}) | (q[1])".format(r),
        ]

        assert inspect() == expected

        # reapply history
        eng.reset()
        eng.run([p1, p2])
        #expected = [
        #    "Run 0:",
        #    "Dgate({}, 0) | (q[1])".format(a),
        #    "Sgate({}, 0) | (q[1])".format(r),
        #    "Rgate({}) | (q[1])".format(r),
        #]

        assert inspect() == expected

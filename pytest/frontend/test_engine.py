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


def test_load_backend(monkeypatch):
    """Test backend can be correctly loaded via strings"""
    eng = sf.Engine('base')
    assert isinstance(eng.backend, BaseBackend)


def test_no_return_state(backend):
    """Tests that engine returns None when no state is requested"""
    prog = sf.Program(2)
    eng  = sf.Engine(backend)

    with prog.context as q:
        ops.Dgate(0.34) | q[0]

    res = eng.run(prog, return_state=False)
    assert res is None


class TestProgram:
    """Tests the Program class."""

    def test_with_block(self):
        """Test gate application using a with block."""
        prog = sf.Program(2)
        # command queue is empty
        assert len(prog) == 0

        with prog.context as q:
            ops.Dgate(0.5) | q[0]
        # now there is one gate in the queue
        assert len(prog) == 1

    def test_unlink(self):
        # TODO
        pass


class TestQueue:
    """Tests that test for gate application and the queue behaviour"""

    def test_identity_command(self, backend):
        """Tests that the None command acts as the identity"""
        prog = sf.Program(2)
        identity = program.Command(None, prog.reg[0])
        prog.circuit.append(identity)
        assert len(prog) == 1
        prog = prog.compile('base')
        assert len(prog) == 0

    def test_clear_queue(self):
        """Test clearing queue is successfull"""
        eng, _ = sf.Engine(2)

        with eng:
            ops.Dgate(0.5) | 0

        eng.print_queue(print_fn=logging.debug)
        assert len(eng.cmd_queue) == 1

        eng.reset_queue()
        assert not eng.cmd_queue

    def test_print_commands(self, backend):
        """Test print_queue and print_applied returns correct strings."""
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

        eng = sf.Engine(backend)
        state = eng.run(prog)
        res = []
        eng.print_applied(print_fn)
        assert res == ["Run 0:"] + expected


class TestRegRefs:
    """Testing register references."""

    def test_corrupt_index(self):
        """This should never happen!"""
        prog = sf.Program(2)
        with prog.context as q:
            q[0].ind = 1
            with pytest.raises(program.RegRefError, match="Should never happen!"):
                ops.Dgate(0.5) | q[0]

    def test_nonexistent_index(self):
        """Acting on a non-existent mode raises an error."""
        prog = sf.Program(2)
        with prog.context as q:
            with pytest.raises(IndexError):
                ops.Dgate(0.5) | q[3]
            with pytest.raises(program.RegRefError, match="does not exist"):
                ops.Dgate(0.5) | 3

    def test_deleted_index(self):
        """Test that acting on a deleted mode raises an error"""
        prog = sf.Program(2)
        with prog.context as q:
            ops.Del | q[0]
            with pytest.raises(program.RegRefError, match="been deleted"):
                ops.Dgate(0.5) | 0

            with pytest.raises(program.RegRefError, match="been deleted"):
                ops.Dgate(0.5) | q[0]

    def test_unknown_regref(self):
        """Applying an operation on a RegRef not belonging to the program."""
        prog = sf.Program(2)
        r = program.RegRef(3)
        with prog.context:
            with pytest.raises(program.RegRefError, match="Unknown RegRef."):
                ops.Dgate(0.5) | r

    def test_invalid_measurement(self, backend):
        """Cannot use a measurement before it exists."""
        prog = sf.Program(2)
        eng  = sf.Engine(backend)
        with prog.context as q:
            ops.Dgate(q[0]) | q[1]
        with pytest.raises(program.CircuitError, match="nonexistent measurement result"):
            eng.run(prog)

    def test_invalid_regref(self):
        """Cannot refer to a register with a non-integral or RegRef object."""
        prog = sf.Program(2)
        with prog.context:
            with pytest.raises(program.RegRefError, match="using integers and RegRefs"):
                ops.Dgate(0) | 1.2


class TestQueueHistory:
    """Testing for history checkpoints"""

    def test_with_block(self, backend):
        """Tests engine history."""
        prog = sf.Program(2)
        eng  = sf.Engine(backend)

        with prog.context as q:
            ops.Dgate(0.5) | q[0]
        prog = prog.compile('base')

        # no programs have been run
        assert not eng.run_progs
        eng.run(prog)
        # one program has been run
        assert len(eng.run_progs) == 1
        assert eng.run_progs[-1] == prog


    def test_checkpoints(self, backend):
        """Test history checkpoints work when creating and deleting modes."""

        eng = sf.Engine(backend)
        prog = sf.Program(2)

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

        eng.run(prog)

        assert not alice.active
        assert charlie.active

        # check that reset_queue() restores the latest RegRef checkpoint
        # (created by eng.run() above)
        eng.reset_queue()

        assert not alice.active
        assert charlie.active

        with eng:
            diana, = ops.New(1)
            ops.Del | charlie

        assert not charlie.active
        assert diana.active
        eng.reset_queue()
        assert charlie.active
        assert not diana.active

        eng.reset()

        new_reg = eng.register
        # original number of modes
        assert len(new_reg) == len(q)

        # the regrefs are reset as well
        assert np.all([r.val is None for r in new_reg])

    def test_create_delete_reset(self, backend):
        """Test various use cases creating and deleting modes,
        together with engine resets."""
        eng = sf.Engine(backend)

        # define some gates
        X = ops.Xgate(0.5)

        qumodes = 2

        # states of the subsystems: active, not_measured
        state = np.full((qumodes, 2), True)

        def prog1(q):
            """program 1"""
            X | q
            ops.MeasureX | q

        def prog2(q):
            """program 2"""
            X | q
            ops.MeasureX | q
            ops.Del | q
            state[q.ind, 0] = False  # no longer active

        def reset():
            """reset the engine"""
            eng.reset()
            state[:] = True

        def check():
            """Check the activity and measurement state of subsystems.

            Activity state is changed right away when a Del command is applied to Engine,
            but a measurement result only becomes available during Engine.run().
            """
            for r, (a, nm) in zip(reg, state):
                assert r.active == a
                assert (r.val is None) == nm

        def input(p, reg):
            """input a program to the engine"""
            with eng:
                p(reg)

            check()

        def input_and_run(p, reg):
            """input a program to the engine and run it"""
            with eng:
                p(reg)

            check()
            eng.run(backend)

            # now measured (assumes p will always measure reg!)
            state[reg.ind, 1] = False
            check()


        alice, bob = reg


        reset()
        ## (1) run several independent programs, resetting everything in between
        input_and_run(prog1, alice)
        reset()
        input_and_run(prog2, bob)
        reset()

        ## (2) interactive state evolution in multiple runs
        input_and_run(prog2, alice)
        input_and_run(prog1, bob)
        reset()

        ## (3) repeat a (possibly extended) program fragment (the fragment must not create/delete subsystems!)
        input_and_run(prog1, alice)
        eng.cmd_queue.extend(eng.cmd_applied[-1])
        check()
        input_and_run(prog1, bob)
        reset()

        ## (4) interactive use, "changed my mind"
        input_and_run(prog1, alice)
        input(prog2, bob)
        eng.reset_queue()  # scratch that, back to last checkpoint
        state[bob.ind, 0] = True  # bob should be active again
        check()
        input_and_run(prog1, bob)
        reset()

        ## (5) reset the state, run the same program again to get new measurement samples
        input_and_run(prog1, alice)
        input(prog2, bob)
        eng.reset(keep_history=True)

        state[
            alice.ind, 1
        ] = True  # measurement result was erased, activity did not change
        state[bob.ind, 1] = True
        check()
        eng.run(backend)
        state[alice.ind, 1] = False
        state[bob.ind, 1] = False
        check()

    def test_apply_history(self, backend):
        """Tests the reapply history argument"""
        eng, q = sf.Engine(2)

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

        eng.run(backend)
        expected = [
            "Run 0:",
            "Dgate({}, 0) | (q[0])".format(a),
            "Sgate({}, 0) | (q[1])".format(r),
        ]

        assert inspect() == expected

        # reapply history
        state2 = eng.run(apply_history=True)
        assert inspect() == expected

        # append more commands to the same backend
        with eng:
            ops.Rgate(r) | q[0]

        eng.run()
        expected = [
            "Run 0:",
            "Dgate({}, 0) | (q[0])".format(a),
            "Sgate({}, 0) | (q[1])".format(r),
            "Run 1:",
            "Rgate({}) | (q[0])".format(r),
        ]

        assert inspect() == expected

        # reapply history
        eng.run(apply_history=True)
        expected = [
            "Run 0:",
            "Dgate({}, 0) | (q[0])".format(a),
            "Sgate({}, 0) | (q[1])".format(r),
            "Rgate({}) | (q[0])".format(r),
        ]

        assert inspect() == expected

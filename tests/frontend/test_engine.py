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
r"""Unit tests for engine.py"""
import pytest

pytestmark = pytest.mark.frontend

import numpy as np

import strawberryfields as sf
from strawberryfields import engine
from strawberryfields import ops
from strawberryfields.backends.base import BaseBackend


@pytest.fixture
def eng(backend):
    """Engine fixture."""
    return sf.Engine(backend)


class TestEngine:
    """Test basic engine functionality"""

    def test_load_backend(self):
        """Backend can be correctly loaded via strings"""
        eng = sf.Engine("base")
        assert isinstance(eng.backend, BaseBackend)


class TestEngineProgramInteraction:
    """Test the Engine class and its interaction with Program instances."""

    def test_no_return_state(self, eng):
        """Engine returns None when no state is requested"""
        prog = sf.Program(2)

        with prog.context as q:
            ops.Dgate(0.34) | q[0]

        res = eng.run(prog, return_state=False)
        assert res is None

    def test_history(self, eng):
        """Engine history."""
        prog = sf.Program(2)

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
            D | q[1]
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
            D | q[1]
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

        assert inspect() == expected

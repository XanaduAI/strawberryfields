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
r"""Unit tests for the IO module"""
import pytest

import numpy as np
import blackbird

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields import io
from strawberryfields.program import Program, CircuitError


pytestmark = pytest.mark.frontend


# fmt: off
U = np.array([[0.219546940711-0.256534554457j, 0.611076853957+0.524178937791j, -0.102700187435+0.474478834685j,-0.027250232925+0.03729094623j],
              [0.451281863394+0.602582912475j, 0.456952590016+0.01230749109j, 0.131625867435-0.450417744715j, 0.035283194078-0.053244267184j],
              [0.038710094355+0.492715562066j,-0.019212744068-0.321842852355j, -0.240776471286+0.524432833034j,-0.458388143039+0.329633367819j],
              [-0.156619083736+0.224568570065j, 0.109992223305-0.163750223027j, -0.421179844245+0.183644837982j, 0.818769184612+0.068015658737j]
    ])
# fmt: on


@pytest.fixture
def eng(backend):
    """Engine fixture."""
    return sf.LocalEngine(backend)


@pytest.fixture(scope="module")
def prog():
    prog = Program(4, name="test_program")

    with prog.context as q:
        # state preparation
        ops.Vac | q[1]
        ops.Squeezed(0.12) | q[2]

        # one mode gates
        ops.Sgate(1) | q[0]
        ops.Dgate(a=0.54 + 0.5j) | q[1]

        # two mode gates
        ops.S2gate(0.543, -0.12) | (q[0], q[3])

        # decomposition
        ops.Interferometer(U) | q

        # measurement
        ops.MeasureX | q[0]
        ops.MeasureHomodyne(0.43, select=0.32) | q[2]
        ops.MeasureHomodyne(phi=0.43, select=0.32) | q[2]

    return prog


test_prog_not_compiled = """\
name test_program
version 1.0

complex array A0[4, 4] =
    0.219546940711-0.256534554457j, 0.611076853957+0.524178937791j, -0.102700187435+0.474478834685j, -0.027250232925+0.03729094623j
    0.451281863394+0.602582912475j, 0.456952590016+0.01230749109j, 0.131625867435-0.450417744715j, 0.035283194078-0.053244267184j
    0.038710094355+0.492715562066j, -0.019212744068-0.321842852355j, -0.240776471286+0.524432833034j, -0.458388143039+0.329633367819j
    -0.156619083736+0.224568570065j, 0.109992223305-0.163750223027j, -0.421179844245+0.183644837982j, 0.818769184612+0.068015658737j

Vacuum() | 1
Squeezed(0.12, 0.0) | 2
Sgate(1, 0.0) | 0
Dgate(0.54+0.5j, 0.0) | 1
S2gate(0.543, -0.12) | [0, 3]
Interferometer(A0) | [0, 1, 2, 3]
MeasureHomodyne(phi=0) | 0
MeasureHomodyne(select=0.32, phi=0.43) | 2
MeasureHomodyne(select=0.32, phi=0.43) | 2
"""


class TestSFToBlackbirdConversion:
    """Tests for the io.to_blackbird utility function"""

    def test_metadata(self):
        """Test metadata correctly converts"""
        # create a test program
        prog = Program(4, name="test_program")
        bb = io.to_blackbird(prog)

        assert bb.name == "test_program"
        assert bb.version == "1.0"
        assert bb.target["name"] is None

        bb = io.to_blackbird(prog.compile("gaussian"))

        assert bb.name == "test_program"
        assert bb.version == "1.0"
        assert bb.target["name"] == "gaussian"

    def test_gate_noarg(self):
        """Test gate with no argument converts"""
        # create a test program
        prog = Program(1)

        with prog.context as q:
            ops.Vac | q[0]

        bb = io.to_blackbird(prog)
        expected = {"op": "Vacuum", "modes": [0], "args": [], "kwargs": {}}

        assert bb.operations[0] == expected

    def test_gate_arg(self):
        """Test gate with argument converts"""
        # create a test program
        prog = Program(2)

        with prog.context as q:
            ops.Sgate(0.54, 0.324) | q[1]

        bb = io.to_blackbird(prog)
        expected = {"op": "Sgate", "modes": [1], "args": [0.54, 0.324], "kwargs": {}}

        assert bb.operations[0] == expected

    def test_gate_kwarg(self):
        """Test gate with keyword argument converts"""
        # create a test program
        prog = Program(2)

        with prog.context as q:
            ops.Dgate(a=0.54 + 0.324j) | q[1]

        bb = io.to_blackbird(prog)
        # Note: due to how SF stores quantum commands with the Parameter class,
        # all kwargs get converted to positional args internally.
        expected = {
            "op": "Dgate",
            "modes": [1],
            "args": [0.54 + 0.324j, 0],
            "kwargs": {},
        }

        assert bb.operations[0] == expected

    def test_two_mode_gate(self):
        """Test two mode gate converts"""
        prog = Program(4)

        with prog.context as q:
            ops.BSgate(0.54, -0.324) | (q[3], q[0])

        bb = io.to_blackbird(prog)
        expected = {
            "op": "BSgate",
            "modes": [3, 0],
            "args": [0.54, -0.324],
            "kwargs": {},
        }

        assert bb.operations[0] == expected

    def test_decomposition_operation_not_compiled(self):
        """Test decomposition operation"""
        # create a test program
        prog = Program(4)

        with prog.context as q:
            ops.Interferometer(U) | q

        bb = io.to_blackbird(prog)
        expected = {
            "op": "Interferometer",
            "modes": [0, 1, 2, 3],
            "args": [U],
            "kwargs": {},
        }

        assert bb.operations[0] == expected

    def test_decomposition_operation_compiled(self):
        """Test decomposition operation gets decomposed if compiled"""
        # create a test program
        prog = Program(1)

        with prog.context as q:
            ops.Pgate(0.43) | q[0]

        bb = io.to_blackbird(prog)
        expected = {"op": "Pgate", "modes": [0], "args": [0.43], "kwargs": {}}
        assert bb.operations[0] == expected

        bb = io.to_blackbird(prog.compile("gaussian"))
        assert bb.operations[0]["op"] == "Sgate"
        assert bb.operations[1]["op"] == "Rgate"

    def test_measure_noarg(self):
        """Test measurement with no argument converts"""
        # create a test program
        prog = Program(1)

        with prog.context as q:
            ops.MeasureFock() | q[0]

        bb = io.to_blackbird(prog)
        expected = {"op": "MeasureFock", "modes": [0], "args": [], "kwargs": {}}

        assert bb.operations[0] == expected

    def test_measure_postselect(self):
        """Test measurement with postselection"""
        # create a test program
        prog = Program(1)

        with prog.context as q:
            ops.MeasureFock(select=2) | q[0]

        bb = io.to_blackbird(prog)
        expected = {
            "op": "MeasureFock",
            "modes": [0],
            "args": [],
            "kwargs": {"select": [2]},
        }

        assert bb.operations[0] == expected

    def test_measure_arg(self):
        """Test measurement with argument converts"""
        # create a test program
        prog = Program(1)

        with prog.context as q:
            ops.MeasureHomodyne(0.43) | q[0]

        bb = io.to_blackbird(prog)
        expected = {
            "op": "MeasureHomodyne",
            "modes": [0],
            "args": [],
            "kwargs": {"phi": 0.43},
        }

        assert bb.operations[0] == expected

    def test_measure_arg_postselect(self):
        """Test measurement with argument and postselection converts"""
        # create a test program
        prog = Program(1)

        with prog.context as q:
            ops.MeasureHomodyne(0.43, select=0.543) | q[0]

        bb = io.to_blackbird(prog)
        expected = {
            "op": "MeasureHomodyne",
            "modes": [0],
            "args": [],
            "kwargs": {"phi": 0.43, "select": 0.543},
        }

        assert bb.operations[0] == expected

        # repeat with kwargs only
        prog = Program(1)

        with prog.context as q:
            ops.MeasureHomodyne(phi=0.43, select=0.543) | q[0]

        bb = io.to_blackbird(prog)
        assert bb.operations[0] == expected

    # TODO: determine best way to serialize regref transforms
    # def test_regref_func_str(self):
    #     """Test a regreftransform with a function string converts properly"""
    #     prog = Program(2)

    #     with prog.context as q:
    #         ops.Sgate(0.43) | q[0]
    #         ops.MeasureX | q[0]
    #         ops.Zgate(ops.RR(q[0], lambda x: 2 * x, func_str="2*q0")) | q[1]

    #     bb = io.to_blackbird(prog)
    #     expected = {"op": "Zgate", "modes": [1], "args": ["2*q0"], "kwargs": {}}

    #     assert bb.operations[-1] == expected

    def test_regref_no_func_str(self):
        """Test a regreftransform with no function string raises exception"""
        prog = Program(2)

        with prog.context as q:
            ops.Sgate(0.43) | q[0]
            ops.MeasureX | q[0]
            ops.Zgate(ops.RR(q[0], lambda x: 2 * x)) | q[1]

        with pytest.raises(ValueError, match="not supported by Blackbird"):
            io.to_blackbird(prog)


class TestBlackbirdToSFConversion:
    """Tests for the io.to_program utility function"""

    def test_empty_program(self):
        """Test empty program raises error"""

        bb_script = """\
        name test_program
        version 1.0
        """

        bb = blackbird.loads(bb_script)

        with pytest.raises(ValueError, match="contains no quantum operations"):
            io.to_program(bb)

    def test_gate_not_defined(self):
        """Test unknown gate raises error"""

        bb_script = """\
        name test_program
        version 1.0

        np | [0, 1]
        """

        bb = blackbird.loads(bb_script)

        with pytest.raises(NameError, match="operation np not defined"):
            io.to_program(bb)

    def test_metadata(self):
        """Test metadata converts"""

        bb_script = """\
        name test_program
        version 1.0
        Vac | 0
        """

        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert prog.name == bb.name
        assert prog.name == 'test_program'

    def test_gate_no_arg(self):
        """Test gate with no argument converts"""

        bb_script = """\
        name test_program
        version 1.0
        Vac | 0
        """

        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert len(prog) == 1
        assert prog.circuit[0].op.__class__.__name__ == "Vacuum"
        assert prog.circuit[0].reg[0].ind == 0

    def test_gate_arg(self):
        """Test gate with arguments converts"""

        bb_script = """\
        name test_program
        version 1.0
        Sgate(0.54, 0.12) | 0
        """

        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert len(prog) == 1
        assert prog.circuit[0].op.__class__.__name__ == "Sgate"
        assert prog.circuit[0].op.p[0].x == 0.54
        assert prog.circuit[0].op.p[1].x == 0.12
        assert prog.circuit[0].reg[0].ind == 0

    def test_gate_kwarg(self):
        """Test gate with keyword argument converts"""

        bb_script = """\
        name test_program
        version 1.0

        Dgate(a=0.54) | 0
        """

        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert len(prog) == 1
        assert prog.circuit[0].op.__class__.__name__ == "Dgate"
        assert prog.circuit[0].op.p[0].x == 0.54
        assert prog.circuit[0].reg[0].ind == 0

    def test_gate_multimode(self):
        """Test multimode gate converts"""

        bb_script = """\
        name test_program
        version 1.0

        BSgate(theta=0.54, phi=pi) | [0, 2]
        """

        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert len(prog) == 1
        assert prog.circuit[0].op.__class__.__name__ == "BSgate"
        assert prog.circuit[0].op.p[0].x == 0.54
        assert prog.circuit[0].op.p[1].x == np.pi
        assert prog.circuit[0].reg[0].ind == 0
        assert prog.circuit[0].reg[1].ind == 2

    def test_valid_compilation(self):
        """Test setting compilation target using the target keyword"""
        bb_script = """\
        name test_program
        version 1.0
        target gaussian
        Pgate(0.54) | 0
        """
        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert prog.target == 'gaussian'
        assert len(prog) == 2
        assert prog.circuit[0].op.__class__.__name__ == "Sgate"
        assert prog.circuit[0].reg[0].ind == 0
        assert prog.circuit[1].op.__class__.__name__ == "Rgate"
        assert prog.circuit[1].reg[0].ind == 0

    def test_invalid_compilation(self):
         """Test an invalid compilation target raises error on attempted compilation"""
         bb_script = """\
         name test_program
         version 1.0
         target gaussian
         Kgate(0.54) | 0
         """
         bb = blackbird.loads(bb_script)
         with pytest.raises(CircuitError, match="cannot be used with the target"):
             prog = io.to_program(bb)


class DummyResults:
    """Dummy results object"""


def dummy_run(self, program, compile_options=None, **eng_run_options):
    """A dummy run function, that when called returns a dummy
    results object, with run options available as an attribute.
    This allows run_options to be returned and inspected after calling eng.run
    """
    results = DummyResults()
    results.run_options = eng_run_options
    return results


class TestEngineIntegration:
    """Test that target options interface correctly with eng.run"""

    def test_shots(self, eng, monkeypatch):
        """Test that passing shots correctly propagates to an engine run"""
        bb_script = """\
        name test_program
        version 1.0
        target gaussian (shots=100)
        Pgate(0.54) | 0
        MeasureX | 0
        """
        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert prog.run_options == {"shots": 100}

        with monkeypatch.context() as m:
            m.setattr("strawberryfields.engine.BaseEngine._run", dummy_run)
            results = eng.run(prog)

        assert results.run_options == {"shots": 100}

    def test_shots_overwritten(self, eng, monkeypatch):
        """Test if run_options are passed to eng.run, they
        overwrite those stored in the compiled program"""
        bb_script = """\
        name test_program
        version 1.0
        target gaussian (shots=100)
        Pgate(0.54) | 0
        MeasureX | 0
        """
        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert prog.run_options == {"shots": 100}

        with monkeypatch.context() as m:
            m.setattr("strawberryfields.engine.BaseEngine._run", dummy_run)
            results = eng.run(prog, run_options={"shots": 1000})

        assert results.run_options == {"shots": 1000}

    def test_program_sequence(self, eng, monkeypatch):
        """Test that program run_options are successively
        updated if a sequence of programs is executed."""
        bb_script = """\
        name test_program
        version 1.0
        target gaussian (shots=100)
        Pgate(0.54) | 0
        MeasureX | 0
        """
        bb = blackbird.loads(bb_script)
        prog1 = io.to_program(bb)

        bb_script = """\
        name test_program
        version 1.0
        target gaussian (shots=1024)
        Pgate(0.54) | 0
        MeasureX | 0
        """
        bb = blackbird.loads(bb_script)
        prog2 = io.to_program(bb)

        assert prog1.run_options == {"shots": 100}
        assert prog2.run_options == {"shots": 1024}

        with monkeypatch.context() as m:
            m.setattr("strawberryfields.engine.BaseEngine._run", dummy_run)
            results = eng.run([prog1, prog2])

        assert results.run_options == prog2.run_options


class TestSave:
    """Test sf.save functionality"""

    def test_save_filename_path_object(self, prog, tmpdir):
        """Test saving a program to a file path using a path object"""
        filename = tmpdir.join("test.xbb")
        sf.save(filename, prog)

        with open(filename, "r") as f:
            res = f.read()

        assert res == test_prog_not_compiled

    def test_save_filename_string(self, prog, tmpdir):
        """Test saving a program to a file path using a string filename"""
        filename = str(tmpdir.join("test.xbb"))
        sf.save(filename, prog)

        with open(filename, "r") as f:
            res = f.read()

        assert res == test_prog_not_compiled

    def test_save_file_object(self, prog, tmpdir):
        """Test writing a program to a file object"""
        filename = tmpdir.join("test.xbb")

        with open(filename, "w") as f:
            sf.save(f, prog)

        with open(filename, "r") as f:
            res = f.read()

        assert res == test_prog_not_compiled

    def test_save_append_extension(self, prog, tmpdir):
        """Test appending the extension if not present"""
        filename = str(tmpdir.join("test.txt"))
        sf.save(filename, prog)

        with open(filename+'.xbb', "r") as f:
            res = f.read()

        assert res == test_prog_not_compiled


class TestLoad:
    """Test sf.load functionality"""

    def test_invalid_file(self, prog, tmpdir):
        """Test exception is raised if file is invalid"""
        with pytest.raises(ValueError, match="must be a string, path"):
            sf.load(1)

    def assert_programs_equal(self, prog1, prog2):
        """Asserts that two SF Program objects are equivalent"""
        for cmd1, cmd2 in zip(prog1.circuit, prog2.circuit):
            assert cmd1.op.__class__.__name__ == cmd2.op.__class__.__name__
            assert cmd1.reg[0].ind == cmd2.reg[0].ind
            if cmd1.op.p:
                assert np.all(cmd1.op.p[0].x == cmd2.op.p[0].x)

    def test_load_filename_path_object(self, prog, tmpdir):
        """Test loading a program using a path object"""
        filename = tmpdir.join("test.xbb")

        with open(filename, "w") as f:
            f.write(test_prog_not_compiled)

        res = sf.load(filename)

        # check loaded program is the same as expected
        self.assert_programs_equal(res, prog)

    def test_load_filename_string(self, prog, tmpdir):
        """Test loading a program using a string filename"""
        filename = str(tmpdir.join("test.xbb"))

        with open(filename, "w") as f:
            f.write(test_prog_not_compiled)

        res = sf.load(filename)

        # check loaded program is the same as expected
        self.assert_programs_equal(res, prog)

    def test_load_file_object(self, prog, tmpdir):
        """Test loading a program via a file object"""
        filename = tmpdir.join("test.xbb")

        with open(filename, "w") as f:
            sf.save(f, prog)

        with open(filename, "r") as f:
            res = sf.load(f)

        # check loaded program is the same as expected
        self.assert_programs_equal(res, prog)

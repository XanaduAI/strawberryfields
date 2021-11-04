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
import textwrap

import numpy as np
import blackbird

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields import io
from strawberryfields.program import Program
from strawberryfields.tdm.tdmprogram import TDMProgram
from strawberryfields.parameters import (
    MeasuredParameter,
    FreeParameter,
    par_is_symbolic,
    par_funcs as pf,
)


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
        ops.Dgate(np.abs(0.54 + 0.5j), np.angle(0.54 + 0.5j)) | q[1]

        # two mode gates
        ops.S2gate(0.543, -0.12) | (q[0], q[3])

        # decomposition
        ops.Interferometer(U) | q

        # measurement
        ops.MeasureX | q[0]
        ops.MeasureHomodyne(0.43, select=0.32) | q[2]
        ops.MeasureHomodyne(phi=0.43, select=0.32) | q[2]

    return prog


test_blackbird_prog_not_compiled = """\
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
Dgate(0.735934779718964, 0.7469555733762603) | 1
S2gate(0.543, -0.12) | [0, 3]
Interferometer(A0) | [0, 1, 2, 3]
MeasureHomodyne(0) | 0
MeasureHomodyne(0.43, select=0.32) | 2
MeasureHomodyne(0.43, select=0.32) | 2
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

        bb = io.to_blackbird(prog.compile(compiler="gaussian"))

        assert bb.name == "test_program"
        assert bb.version == "1.0"
        assert bb.target["name"] == "gaussian"

    def test_metadata_run_options(self):
        """Test run options correctly converts"""
        prog = Program(4, name="test_program")
        bb = io.to_blackbird(prog.compile(compiler="gaussian", shots=1024))

        assert bb.name == "test_program"
        assert bb.version == "1.0"
        assert bb.target["name"] == "gaussian"
        assert bb.target["options"] == {"shots": 1024}

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
            ops.Dgate(r=np.abs(0.54 + 0.324j), phi=np.angle(0.54 + 0.324j)) | q[1]

        bb = io.to_blackbird(prog)
        # Note: due to how SF stores quantum commands with the Parameter class,
        # all kwargs get converted to positional args internally.
        expected = {
            "op": "Dgate",
            "modes": [1],
            "args": [np.abs(0.54 + 0.324j), np.angle(0.54 + 0.324j)],
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

        bb = io.to_blackbird(prog.compile(compiler="gaussian"))
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

    def test_measure_darkcounts(self):
        """Test measurement with dark counts"""
        # create a test program
        prog = Program(1)

        with prog.context as q:
            ops.MeasureFock(dark_counts=2) | q[0]

        bb = io.to_blackbird(prog)
        expected = {
            "op": "MeasureFock",
            "modes": [0],
            "args": [],
            "kwargs": {"dark_counts": [2]},
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
            "args": [0.43],
            "kwargs": {},
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
            "args": [0.43],
            "kwargs": {"select": 0.543},
        }

        assert bb.operations[0] == expected

        # repeat with kwargs only
        prog = Program(1)

        with prog.context as q:
            ops.MeasureHomodyne(phi=0.43, select=0.543) | q[0]

        bb = io.to_blackbird(prog)
        assert bb.operations[0] == expected

    def test_measured_par_str(self):
        """Test a MeasuredParameter with some transformations converts properly"""
        prog = Program(2)
        with prog.context as q:
            ops.Sgate(0.43) | q[0]
            ops.MeasureX | q[0]
            ops.Zgate(2 * pf.sin(q[0].par)) | q[1]

        bb = io.to_blackbird(prog)
        assert bb.operations[-1]["op"] == "Zgate"
        assert bb.operations[-1]["modes"] == [1]

        assert isinstance(bb.operations[-1]["args"][0], blackbird.RegRefTransform)
        assert bb.operations[-1]["args"][0].func_str == "2*sin(q0)"
        assert bb.operations[-1]["args"][0].regrefs == [0]

        assert bb.operations[-1]["kwargs"] == {}

    def test_free_par_str(self):
        """Test a FreeParameter with some transformations converts properly"""
        prog = Program(2)
        r, alpha = prog.params("r", "alpha")
        with prog.context as q:
            ops.Sgate(r) | q[0]
            ops.Zgate(3 * pf.log(-alpha)) | q[1]

        bb = io.to_blackbird(prog)
        assert bb.operations[0] == {"op": "Sgate", "modes": [0], "args": ["{r}", 0.0], "kwargs": {}}
        assert bb.operations[1] == {
            "op": "Zgate",
            "modes": [1],
            "args": ["3*log(-{alpha})"],
            "kwargs": {},
        }

    def test_tdm_program(self):
        prog = TDMProgram(2)

        with prog.context([1, 2], [3, 4], [5, 6]) as (p, q):
            ops.Sgate(0.7, 0) | q[1]
            ops.BSgate(p[0]) | (q[0], q[1])
            ops.Rgate(p[1]) | q[1]
            ops.MeasureHomodyne(p[2]) | q[0]

        bb = io.to_blackbird(prog)

        assert bb.operations[0] == {"kwargs": {}, "args": [0.7, 0], "op": "Sgate", "modes": [1]}
        assert bb.operations[1] == {
            "kwargs": {},
            "args": ["p0", 0.0],
            "op": "BSgate",
            "modes": [0, 1],
        }
        assert bb.operations[2] == {"kwargs": {}, "args": ["p1"], "op": "Rgate", "modes": [1]}
        assert bb.operations[3] == {
            "kwargs": {},
            "args": ["p2"],
            "op": "MeasureHomodyne",
            "modes": [0],
        }

        assert bb.programtype == {"name": "tdm", "options": {"temporal_modes": 2}}
        assert list(bb._var.keys()) == ["p0", "p1", "p2"]
        assert np.all(bb._var["p0"] == np.array([[1, 2]]))
        assert np.all(bb._var["p1"] == np.array([[3, 4]]))
        assert np.all(bb._var["p2"] == np.array([[5, 6]]))
        assert bb.modes == {0, 1}


class TestBlackbirdToSFConversion:
    """Tests for the io.to_program utility function for Blackbird programs"""

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
        assert prog.name == "test_program"

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
        assert prog.circuit[0].op.p[0] == 0.54
        assert prog.circuit[0].op.p[1] == 0.12
        assert prog.circuit[0].reg[0].ind == 0

    def test_gate_kwarg(self):
        """Test gate with keyword argument converts"""

        bb_script = """\
        name test_program
        version 1.0

        Dgate(r=0.54, phi=0) | 0
        """

        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert len(prog) == 1
        assert prog.circuit[0].op.__class__.__name__ == "Dgate"
        assert prog.circuit[0].op.p[0] == 0.54
        assert prog.circuit[0].reg[0].ind == 0

    def test_gate_measured_par(self):
        """Test a gate with a MeasuredParameter argument."""

        bb_script = """\
        name test_program
        version 1.0

        MeasureX | 0
        Dgate(q0) | 1
        Rgate(2*q0) | 2
        """
        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert len(prog) == 3

        cmd = prog.circuit[1]
        assert cmd.op.__class__.__name__ == "Dgate"
        p = cmd.op.p[0]
        assert isinstance(p, MeasuredParameter)
        assert p.regref.ind == 0
        assert cmd.reg[0].ind == 1

        cmd = prog.circuit[2]
        assert cmd.op.__class__.__name__ == "Rgate"
        p = cmd.op.p[0]
        assert par_is_symbolic(p)  # symbolic expression
        assert cmd.reg[0].ind == 2

    def test_gate_free_par(self):
        """Test a FreeParameter with some transformations converts properly"""

        bb_script = """\
        name test_program
        version 1.0

        Dgate(1-{ALPHA}, 0) | 0     # keyword arg, compound expr
        Rgate(theta={foo_bar1}) | 0  # keyword arg, atomic
        Dgate({ALPHA}**2, 0) | 0        # positional arg, compound expr
        Rgate({foo_bar2}) | 0        # positional arg, atomic
        """
        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert prog.free_params.keys() == set(["foo_bar1", "foo_bar2", "ALPHA"])
        assert len(prog) == 4

        cmd = prog.circuit[0]
        assert cmd.op.__class__.__name__ == "Dgate"
        p = cmd.op.p[0]
        assert par_is_symbolic(p)
        assert cmd.reg[0].ind == 0

        cmd = prog.circuit[1]
        assert cmd.op.__class__.__name__ == "Rgate"
        p = cmd.op.p[0]
        assert isinstance(p, FreeParameter)
        assert p.name == "foo_bar1"
        assert cmd.reg[0].ind == 0

        cmd = prog.circuit[2]
        assert cmd.op.__class__.__name__ == "Dgate"
        p = cmd.op.p[0]
        assert par_is_symbolic(p)
        assert cmd.reg[0].ind == 0

        cmd = prog.circuit[3]
        assert cmd.op.__class__.__name__ == "Rgate"
        p = cmd.op.p[0]
        assert isinstance(p, FreeParameter)
        assert p.name == "foo_bar2"
        assert cmd.reg[0].ind == 0

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
        assert prog.circuit[0].op.p[0] == 0.54
        assert prog.circuit[0].op.p[1] == np.pi
        assert prog.circuit[0].reg[0].ind == 0
        assert prog.circuit[0].reg[1].ind == 2

    def test_tdm_program(self):
        """Test converting a tdm bb_script to a TDMProgram"""

        bb_script = textwrap.dedent(
            """\
        name None
        version 1.0
        type tdm (temporal_modes=3)

        int array p0 =
            1, 2
        int array p1 =
            3, 4
        int array p2 =
            5, 6

        Sgate(0.7, 0) | 1
        BSgate(p0, 0.0) | [0, 1]
        Rgate(p1) | 1
        MeasureHomodyne(phi=p2) | 0
        """
        )

        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert len(prog) == 4
        assert prog.circuit[0].op.__class__.__name__ == "Sgate"
        assert prog.circuit[0].op.p[0] == 0.7
        assert prog.circuit[0].op.p[1] == 0
        assert prog.circuit[0].reg[0].ind == 1

        assert prog.circuit[1].op.__class__.__name__ == "BSgate"
        assert prog.circuit[1].op.p[0] == FreeParameter("p0")
        assert prog.circuit[1].op.p[1] == 0.0
        assert prog.circuit[1].reg[0].ind == 0
        assert prog.circuit[1].reg[1].ind == 1

        assert prog.circuit[2].op.__class__.__name__ == "Rgate"
        assert prog.circuit[2].op.p[0] == FreeParameter("p1")
        assert prog.circuit[2].reg[0].ind == 1

        assert prog.circuit[3].op.__class__.__name__ == "MeasureHomodyne"
        assert prog.circuit[3].op.p[0] == FreeParameter("p2")
        assert prog.circuit[3].reg[0].ind == 0

        assert prog.concurr_modes == 2
        assert prog.timebins == 2
        assert prog.spatial_modes == 1
        assert prog.free_params == {
            "p0": FreeParameter("p0"),
            "p1": FreeParameter("p1"),
            "p2": FreeParameter("p2"),
        }
        assert all(prog.tdm_params[0] == np.array([1, 2]))
        assert all(prog.tdm_params[1] == np.array([3, 4]))
        assert all(prog.tdm_params[2] == np.array([5, 6]))

    def test_empty_tdm_program(self):
        """Test empty tdm program raises error"""

        bb_script = """\
        name test_program
        version 1.0
        type tdm (temporal_modes= 12)
        """

        bb = blackbird.loads(bb_script)

        with pytest.raises(ValueError, match="contains no quantum operations"):
            io.to_program(bb)

    def test_gate_not_defined_tdm(self):
        """Test unknown gate raises error for tdm program"""

        bb_script = textwrap.dedent(
            """\
        name test_program
        version 1.0
        type tdm (temporal_modes= 12)

        int array p42 =
            1, 3, 5

        np | [0, 1]
        """
        )

        bb = blackbird.loads(bb_script)

        with pytest.raises(NameError, match="operation np not defined"):
            io.to_program(bb)


prog_txt = textwrap.dedent(
    """\
    import strawberryfields as sf
    from strawberryfields import ops

    prog = sf.Program(3)
    eng = sf.Engine({engine_args})

    with prog.context as q:
        ops.Sgate(0.54, 0) | q[0]
        ops.BSgate(0.45, np.pi/2) | (q[0], q[2])
        ops.Sgate(3*np.pi/2, 0) | q[1]
        ops.BSgate(2*np.pi, 0.62) | (q[0], q[1])
        ops.MeasureFock() | q[0]

    results = eng.run(prog)
"""
)

prog_txt_no_engine = textwrap.dedent(
    """\
    import strawberryfields as sf
    from strawberryfields import ops

    prog = sf.Program(3)

    with prog.context as q:
        ops.Sgate(0.54, 0) | q[0]
        ops.BSgate(0.45, np.pi/2) | (q[0], q[2])
        ops.Sgate(3*np.pi/2, 0) | q[1]
        ops.BSgate(2*np.pi, 0.62) | (q[0], q[1])
        ops.MeasureFock() | q[0]
"""
)

prog_txt_tdm = textwrap.dedent(
    """\
    import strawberryfields as sf
    from strawberryfields import ops

    prog = sf.TDMProgram(N=[2, 3])
    eng = sf.Engine("gaussian")

    with prog.context([np.pi, 3*np.pi/2, 0], [1, 0.5, np.pi], [0, 0, 0]) as (p, q):
        ops.Sgate(0.123, np.pi/4) | q[2]
        ops.BSgate(p[0], 0.0) | (q[1], q[2])
        ops.Rgate(p[1]) | q[2]
        ops.MeasureHomodyne(p[0]) | q[0]
        ops.MeasureHomodyne(p[2]) | q[2]

    results = eng.run(prog)
"""
)

class TestSaveBlackbird:
    """Test sf.save functionality"""

    def test_save_filename_path_object(self, prog, tmpdir):
        """Test saving a program to a file path using a path object"""
        filename = tmpdir.join("test.xbb")
        sf.save(filename, prog)

        with open(filename, "r") as f:
            res = f.read()

        assert res == test_blackbird_prog_not_compiled

    def test_save_filename_string(self, prog, tmpdir):
        """Test saving a program to a file path using a string filename"""
        filename = str(tmpdir.join("test.xbb"))
        sf.save(filename, prog)

        with open(filename, "r") as f:
            res = f.read()

        assert res == test_blackbird_prog_not_compiled

    def test_save_file_object(self, prog, tmpdir):
        """Test writing a program to a file object"""
        filename = tmpdir.join("test.xbb")

        with open(filename, "w") as f:
            sf.save(f, prog)

        with open(filename, "r") as f:
            res = f.read()

        assert res == test_blackbird_prog_not_compiled

    def test_save_append_extension(self, prog, tmpdir):
        """Test appending the extension if not present"""
        filename = str(tmpdir.join("test.txt"))
        sf.save(filename, prog)

        with open(filename + ".xbb", "r") as f:
            res = f.read()

        assert res == test_blackbird_prog_not_compiled


class TestLoadBlackbird:
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
                assert np.all(cmd1.op.p[0] == cmd2.op.p[0])

    def test_load_filename_path_object(self, prog, tmpdir):
        """Test loading a program using a path object"""
        filename = tmpdir.join("test.xbb")

        with open(filename, "w") as f:
            f.write(test_blackbird_prog_not_compiled)

        res = sf.load(filename)

        # check loaded program is the same as expected
        self.assert_programs_equal(res, prog)

    def test_load_filename_string(self, prog, tmpdir):
        """Test loading a program using a string filename"""
        filename = str(tmpdir.join("test.xbb"))

        with open(filename, "w") as f:
            f.write(test_blackbird_prog_not_compiled)

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

    def test_loads(self, prog):
        """Test loading a program from a string"""
        self.assert_programs_equal(io.loads(test_blackbird_prog_not_compiled), prog)

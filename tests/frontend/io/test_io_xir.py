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
import inspect

import numpy as np
import xir
from xir.program import Declaration

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields import io
from strawberryfields.program import Program
from strawberryfields.tdm.tdmprogram import TDMProgram
from strawberryfields.parameters import (
    par_funcs as pf,
    FreeParameter,
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


test_xir_prog_not_compiled = """\
Vacuum | [1];
Squeezed(0.12, 0.0) | [2];
Sgate(1, 0.0) | [0];
Dgate(0.735934779718964, 0.7469555733762603) | [1];
S2gate(0.543, -0.12) | [0, 3];
Interferometer([[(0.219546940711-0.256534554457j), (0.611076853957+0.524178937791j), (-0.102700187435+0.474478834685j), (-0.027250232925+0.03729094623j)], [(0.451281863394+0.602582912475j), (0.456952590016+0.01230749109j), (0.131625867435-0.450417744715j), (0.035283194078-0.053244267184j)], [(0.038710094355+0.492715562066j), (-0.019212744068-0.321842852355j), (-0.240776471286+0.524432833034j), (-0.458388143039+0.329633367819j)], [(-0.156619083736+0.224568570065j), (0.109992223305-0.163750223027j), (-0.421179844245+0.183644837982j), (0.818769184612+0.068015658737j)]]) | [0, 1, 2, 3];
MeasureHomodyne(phi: 0) | [0];
MeasureHomodyne(phi: 0.43, select: 0.32) | [2];
MeasureHomodyne(phi: 0.43, select: 0.32) | [2];\
"""


class TestSFtoXIRConversion:
    """Tests for the io.to_xir utility function"""

    def test_empty_program(self):
        """Test that an empty program is correctly converted"""
        # create a test program
        sf_prog = Program(4, name="test_program")
        xir_prog = io.to_xir(sf_prog)

        assert xir_prog.serialize() == ""
        assert xir_prog.version == "0.1.0"

    def test_gate_noarg(self):
        """Test gate with no argument converts"""
        # create a test program
        sf_prog = Program(1)

        with sf_prog.context as q:
            ops.Vac | q[0]

        xir_prog = io.to_xir(sf_prog)

        expected = [("Vacuum", [], (0,))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

    def test_gate_arg(self):
        """Test gate with argument converts"""
        # create a test program
        sf_prog = Program(2)

        with sf_prog.context as q:
            ops.Sgate(0.54, 0.324) | q[1]

        xir_prog = io.to_xir(sf_prog)

        expected = [("Sgate", [0.54, 0.324], (1,))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

    def test_gate_kwarg(self):
        """Test gate with keyword argument converts"""
        # create a test program
        sf_prog = Program(2)

        with sf_prog.context as q:
            ops.Dgate(r=np.abs(0.54 + 0.324j), phi=np.angle(0.54 + 0.324j)) | q[1]

        xir_prog = io.to_xir(sf_prog)

        # Note: due to how SF stores quantum commands with the Parameter class,
        # all kwargs get converted to positional args internally.
        expected = [("Dgate", [np.abs(0.54 + 0.324j), np.angle(0.54 + 0.324j)], (1,))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

    def test_two_mode_gate(self):
        """Test two mode gate converts"""
        sf_prog = Program(4)

        with sf_prog.context as q:
            ops.BSgate(0.54, -0.324) | (q[3], q[0])

        xir_prog = io.to_xir(sf_prog)

        expected = [("BSgate", [0.54, -0.324], (3, 0))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

    def test_decomposition_operation_not_compiled(self):
        """Test decomposition operation"""
        # create a test program
        sf_prog = Program(4)

        with sf_prog.context as q:
            ops.Interferometer(U) | q

        xir_prog = io.to_xir(sf_prog)

        assert xir_prog.statements[0].name == "Interferometer"
        param = np.array(xir_prog.statements[0].params)
        assert np.allclose(param[0], U)
        assert xir_prog.statements[0].wires == (0, 1, 2, 3)

    def test_decomposition_operation_compiled(self):
        """Test decomposition operation gets decomposed if compiled"""
        # create a test program
        sf_prog = Program(1)

        with sf_prog.context as q:
            ops.Pgate(0.43) | q[0]

        xir_prog = io.to_xir(sf_prog)

        expected = [("Pgate", [0.43], (0,))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

        xir_prog = io.to_xir(sf_prog.compile(compiler="gaussian"))

        assert xir_prog.statements[0].name == "Sgate"
        assert xir_prog.statements[1].name == "Rgate"

    def test_measure_noarg(self):
        """Test measurement with no argument converts"""
        # create a test program
        sf_prog = Program(1)

        with sf_prog.context as q:
            ops.MeasureFock() | q[0]

        xir_prog = io.to_xir(sf_prog)

        # note that measurement parameters are always dicts
        expected = [("MeasureFock", {}, (0,))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

    def test_measure_postselect(self):
        """Test measurement with postselection"""
        # create a test program
        sf_prog = Program(1)

        with sf_prog.context as q:
            ops.MeasureFock(select=2) | q[0]

        xir_prog = io.to_xir(sf_prog)

        expected = [("MeasureFock", {"select": [2]}, (0,))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

    def test_measure_darkcounts(self):
        """Test measurement with dark counts"""
        # create a test program
        sf_prog = Program(1)

        with sf_prog.context as q:
            ops.MeasureFock(dark_counts=2) | q[0]

        xir_prog = io.to_xir(sf_prog)

        expected = [("MeasureFock", {"dark_counts": [2]}, (0,))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

    def test_measure_arg(self):
        """Test measurement with argument converts"""
        # create a test program
        sf_prog = Program(1)

        with sf_prog.context as q:
            ops.MeasureHomodyne(0.43) | q[0]

        xir_prog = io.to_xir(sf_prog)

        expected = [("MeasureHomodyne", {"phi": 0.43}, (0,))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

    def test_measure_arg_postselect(self):
        """Test measurement with argument and postselection converts"""
        # create a test program
        sf_prog = Program(1)

        with sf_prog.context as q:
            ops.MeasureHomodyne(0.43, select=0.543) | q[0]

        xir_prog = io.to_xir(sf_prog)

        expected = [("MeasureHomodyne", {"phi": 0.43, "select": 0.543}, (0,))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

        # repeat with kwargs only
        sf_prog = Program(1)

        with sf_prog.context as q:
            ops.MeasureHomodyne(phi=0.43, select=0.543) | q[0]

        xir_prog = io.to_xir(sf_prog)
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

    def test_free_par_str(self):
        """Test a FreeParameter with some transformations converts properly"""
        sf_prog = Program(2)
        r, alpha = sf_prog.params("r", "alpha")
        with sf_prog.context as q:
            ops.Sgate(r) | q[0]
            ops.Zgate(3 * pf.log(-alpha)) | q[1]

        xir_prog = io.to_xir(sf_prog)

        expected = [("Sgate", ["r", 0.0], (0,)), ("Zgate", ["3*log(-alpha)"], (1,))]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected

    def test_tdm_program(self):
        sf_prog = TDMProgram(2)

        with sf_prog.context([1, 2], [3, 4], [5, 6]) as (p, q):
            ops.Sgate(0.7, 0) | q[1]
            ops.BSgate(p[0]) | (q[0], q[1])
            ops.Rgate(p[1]) | q[1]
            ops.MeasureHomodyne(p[2]) | q[0]

        xir_prog = io.to_xir(sf_prog)

        expected = [
            ("Sgate", [0.7, 0], (1,)),
            ("BSgate", ["p0", 0.0], (0, 1)),
            ("Rgate", ["p1"], (1,)),
            ("MeasureHomodyne", {"phi": "p2"}, (0,)),
        ]
        assert [(stmt.name, stmt.params, stmt.wires) for stmt in xir_prog.statements] == expected


class TestXIRtoSFConversion:
    """Tests for the io.to_program utility function for XIR programs"""

    def test_empty_program(self):
        """Test empty program raises error"""
        xir_prog = xir.Program()

        with pytest.raises(
            ValueError, match="is empty and cannot be transformed into a Strawberry Fields program"
        ):
            io.to_program(xir_prog)

    def test_gate_not_defined(self):
        """Test unknown gate raises error"""
        xir_prog = xir.Program()
        xir_prog.add_statement(xir.Statement("np", [1, 2, 3], (0,)))

        with pytest.raises(NameError, match="operation 'np' not defined"):
            io.to_program(xir_prog)

    # def test_metadata(self):
    #     """Test metadata converts"""

    #     bb_script = """\
    #     name test_program
    #     version 1.0
    #     Vac | 0
    #     """

    #     xir_prog = xir.Program()

    #     sf_prog = io.to_program(bb)

    #     assert sf_prog.version == xir_prog.name
    #     assert sf_prog.name == "test_program"

    def test_gate_no_arg(self):
        """Test gate with no argument converts"""
        xir_prog = xir.Program()
        xir_prog.add_statement(xir.Statement("Vacuum", [], (0,)))

        sf_prog = io.to_program(xir_prog)

        assert len(sf_prog) == 1
        assert sf_prog.circuit
        assert sf_prog.circuit[0].op.__class__.__name__ == "Vacuum"
        assert sf_prog.circuit[0].reg[0].ind == 0

    def test_gate_arg(self):
        """Test gate with arguments converts"""
        xir_prog = xir.Program()
        xir_prog.add_statement(xir.Statement("Sgate", [0.54, 0.12], (0,)))

        sf_prog = io.to_program(xir_prog)

        assert len(sf_prog) == 1
        assert sf_prog.circuit
        assert sf_prog.circuit[0].op.__class__.__name__ == "Sgate"
        assert sf_prog.circuit[0].op.p[0] == 0.54
        assert sf_prog.circuit[0].op.p[1] == 0.12
        assert sf_prog.circuit[0].reg[0].ind == 0

    def test_gate_kwarg(self):
        """Test gate with keyword argument converts"""
        xir_prog = xir.Program()
        xir_prog.add_statement(xir.Statement("Dgate", {"r": 0.54, "phi": 0}, (0,)))

        sf_prog = io.to_program(xir_prog)

        assert len(sf_prog) == 1
        assert sf_prog.circuit
        assert sf_prog.circuit[0].op.__class__.__name__ == "Dgate"
        assert sf_prog.circuit[0].op.p[0] == 0.54
        assert sf_prog.circuit[0].reg[0].ind == 0

    def test_gate_multimode(self):
        """Test multimode gate converts"""
        xir_prog = xir.Program()
        xir_prog.add_statement(xir.Statement("BSgate", {"theta": 0.54, "phi": np.pi}, (0, 2)))

        sf_prog = io.to_program(xir_prog)

        assert len(sf_prog) == 1
        assert sf_prog.circuit
        assert sf_prog.circuit[0].op.__class__.__name__ == "BSgate"
        assert sf_prog.circuit[0].op.p[0] == 0.54
        assert sf_prog.circuit[0].op.p[1] == np.pi
        assert sf_prog.circuit[0].reg[0].ind == 0
        assert sf_prog.circuit[0].reg[1].ind == 2

    def test_tdm_program(self):
        """Test converting a TDM XIR program to a TDMProgram"""

        xir_prog = xir.Program()

        xir_prog.add_statement(xir.Statement("Sgate", [0.7, 0], (1,)))
        xir_prog.add_statement(xir.Statement("BSgate", ["p0", 0.0], (0, 1)))
        xir_prog.add_statement(xir.Statement("Rgate", ["p1"], (1,)))
        xir_prog.add_statement(xir.Statement("MeasureHomodyne", {"phi": "p2"}, (0,)))

        # TODO: invalid to pass non-string parameters to declaration; must do so
        # to store parameter values due to TDM workaround
        xir_prog.add_declaration(Declaration("p0", "func", [1, 2]))
        xir_prog.add_declaration(Declaration("p1", "func", [3, 4]))
        xir_prog.add_declaration(Declaration("p2", "func", [5, 6]))

        xir_prog.add_option("type", "tdm")
        xir_prog.add_option("N", [2])

        sf_prog = io.to_program(xir_prog)

        assert isinstance(sf_prog, TDMProgram)

        assert len(sf_prog) == 4
        assert sf_prog.circuit
        assert sf_prog.circuit[0].op.__class__.__name__ == "Sgate"
        assert sf_prog.circuit[0].op.p[0] == 0.7
        assert sf_prog.circuit[0].op.p[1] == 0
        assert sf_prog.circuit[0].reg[0].ind == 1

        assert sf_prog.circuit[1].op.__class__.__name__ == "BSgate"
        assert sf_prog.circuit[1].op.p[0] == FreeParameter("p0")
        assert sf_prog.circuit[1].op.p[1] == 0.0
        assert sf_prog.circuit[1].reg[0].ind == 0
        assert sf_prog.circuit[1].reg[1].ind == 1

        assert sf_prog.circuit[2].op.__class__.__name__ == "Rgate"
        assert sf_prog.circuit[2].op.p[0] == FreeParameter("p1")
        assert sf_prog.circuit[2].reg[0].ind == 1

        assert sf_prog.circuit[3].op.__class__.__name__ == "MeasureHomodyne"
        assert sf_prog.circuit[3].op.p[0] == FreeParameter("p2")
        assert sf_prog.circuit[3].reg[0].ind == 0

        assert sf_prog.concurr_modes == 2
        assert sf_prog.timebins == 2
        assert sf_prog.spatial_modes == 1
        assert sf_prog.free_params == {
            "p0": FreeParameter("p0"),
            "p1": FreeParameter("p1"),
            "p2": FreeParameter("p2"),
        }
        assert all(sf_prog.tdm_params[0] == np.array([1, 2]))
        assert all(sf_prog.tdm_params[1] == np.array([3, 4]))
        assert all(sf_prog.tdm_params[2] == np.array([5, 6]))

    def test_tdm_program(self):
        """Test converting a TDM XIR script to a TDMProgram"""
        xir_script = inspect.cleandoc(
            """
            options:
                type: tdm;
                N: [2, 3];
            end;

            func p0(pi, 3*pi/2, 0);
            func p1(1, 0.5, pi);
            func p2(0, 0, 0);

            Sgate(0.123, pi/4) | [2];
            BSgate(p0, 0.0) | [1, 2];
            Rgate(p1) | [2];
            MeasureHomodyne(phi: p0) | [0];
            MeasureHomodyne(phi: p2) | [2];
            """
        )

        xir_prog = xir.parse_script(xir_script, eval_pi=True)
        sf_prog = io.to_program(xir_prog)

        assert isinstance(sf_prog, TDMProgram)

        assert len(sf_prog) == 5
        assert sf_prog.circuit
        assert sf_prog.circuit[0].op.__class__.__name__ == "Sgate"
        assert sf_prog.circuit[0].op.p[0] == 0.123
        assert sf_prog.circuit[0].op.p[1] == np.pi/4
        assert sf_prog.circuit[0].reg[0].ind == 2

        assert sf_prog.circuit[1].op.__class__.__name__ == "BSgate"
        assert sf_prog.circuit[1].op.p[0] == FreeParameter("p0")
        assert sf_prog.circuit[1].op.p[1] == 0.0
        assert sf_prog.circuit[1].reg[0].ind == 1
        assert sf_prog.circuit[1].reg[1].ind == 2

        assert sf_prog.circuit[2].op.__class__.__name__ == "Rgate"
        assert sf_prog.circuit[2].op.p[0] == FreeParameter("p1")
        assert sf_prog.circuit[2].reg[0].ind == 2

        assert sf_prog.circuit[3].op.__class__.__name__ == "MeasureHomodyne"
        assert sf_prog.circuit[3].op.p[0] == FreeParameter("p0")
        assert sf_prog.circuit[3].reg[0].ind == 0

        assert sf_prog.circuit[4].op.__class__.__name__ == "MeasureHomodyne"
        assert sf_prog.circuit[4].op.p[0] == FreeParameter("p2")
        assert sf_prog.circuit[4].reg[0].ind == 2

        assert sf_prog.concurr_modes == 5
        assert sf_prog.timebins == 3
        assert sf_prog.spatial_modes == 2
        assert sf_prog.free_params == {
            "p0": FreeParameter("p0"),
            "p1": FreeParameter("p1"),
            "p2": FreeParameter("p2"),
        }
        assert all(sf_prog.tdm_params[0] == np.array([np.pi, 3*np.pi/2, 0]))
        assert all(sf_prog.tdm_params[1] == np.array([1, 0.5, np.pi]))
        assert all(sf_prog.tdm_params[2] == np.array([0, 0, 0]))


prog_txt = inspect.cleandoc(
    """
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

prog_txt_no_engine = inspect.cleandoc(
    """
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

prog_txt_tdm = inspect.cleandoc(
    """
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


class TestSaveXIR:
    """Test sf.save functionality"""

    def test_save_filename_path_object(self, prog, tmpdir):
        """Test saving a program to a file path using a path object"""
        filename = tmpdir.join("test.xbb")
        sf.save(filename, prog, ir="xir")

        with open(filename, "r") as f:
            res = f.read()

        assert res == test_xir_prog_not_compiled

    def test_save_filename_string(self, prog, tmpdir):
        """Test saving a program to a file path using a string filename"""
        filename = str(tmpdir.join("test.xbb"))
        sf.save(filename, prog, ir="xir")

        with open(filename, "r") as f:
            res = f.read()

        assert res == test_xir_prog_not_compiled

    def test_save_file_object(self, prog, tmpdir):
        """Test writing a program to a file object"""
        filename = tmpdir.join("test.xbb")

        with open(filename, "w") as f:
            sf.save(f, prog, ir="xir")

        with open(filename, "r") as f:
            res = f.read()

        assert res == test_xir_prog_not_compiled

    def test_save_append_extension(self, prog, tmpdir):
        """Test appending the extension if not present"""
        filename = str(tmpdir.join("test.txt"))
        sf.save(filename, prog, ir="xir")

        with open(filename + ".xbb", "r") as f:
            res = f.read()

        assert res == test_xir_prog_not_compiled


class TestLoadXIR:
    """Test sf.load functionality"""

    def test_invalid_file(self, prog, tmpdir):
        """Test exception is raised if file is invalid"""
        with pytest.raises(ValueError, match="must be a string, path"):
            sf.load(1, ir="xir")

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
            f.write(test_xir_prog_not_compiled)

        res = sf.load(filename, ir="xir")

        # check loaded program is the same as expected
        self.assert_programs_equal(res, prog)

    def test_load_filename_string(self, prog, tmpdir):
        """Test loading a program using a string filename"""
        filename = str(tmpdir.join("test.xbb"))

        with open(filename, "w") as f:
            f.write(test_xir_prog_not_compiled)

        res = sf.load(filename, ir="xir")

        # check loaded program is the same as expected
        self.assert_programs_equal(res, prog)

    def test_load_file_object(self, prog, tmpdir):
        """Test loading a program via a file object"""
        filename = tmpdir.join("test.xbb")

        with open(filename, "w") as f:
            sf.save(f, prog, ir="xir", add_decl=False)

        with open(filename, "r") as f:
            res = sf.load(f, ir="xir")

        # check loaded program is the same as expected
        self.assert_programs_equal(res, prog)

    def test_loads(self, prog):
        """Test loading a program from a string"""
        self.assert_programs_equal(io.loads(test_xir_prog_not_compiled, ir="xir"), prog)

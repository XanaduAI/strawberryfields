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
r"""Unit tests for the circuit drawer"""
import sys
import os
import datetime
import difflib
from textwrap import dedent

import pytest

pytestmark = pytest.mark.frontend

import strawberryfields as sf
from strawberryfields import ops

from strawberryfields.circuitdrawer import Circuit as CircuitDrawer
from strawberryfields.circuitdrawer import (
    UnsupportedGateException,
    NotDrawableException,
    ModeMismatchException,
)


LINE_RETURN = "\n"
WIRE_TERMINATOR = r"\\" + "\n"
CIRCUIT_BODY_TERMINATOR = "}\n"
CIRCUIT_BODY_START = " {" + "\n"
BEGIN_DOCUMENT = r"\begin{document}"
EMPTY_PAGESTYLE = r"\pagestyle{empty}"
DOCUMENT_CLASS = r"\documentclass{article}"
QCIRCUIT_PACKAGE = r"\usepackage{qcircuit}"
CIRCUIT_START = r"\Qcircuit"
COLUMN_SPACING = "@C={0}"
ROW_SPACING = "@R={0}"
QUANTUM_WIRE = r"\qw"
INIT_DOCUMENT = (
    DOCUMENT_CLASS + "\n" + EMPTY_PAGESTYLE + "\n" + QCIRCUIT_PACKAGE + "\n" + BEGIN_DOCUMENT + "\n" + CIRCUIT_START
)


@pytest.fixture
def drawer():
    """Returns a 3 mode circuit drawer"""
    return CircuitDrawer(3)


def failure_message(result, expected):
    """Build failure message with explicit error latex diff.

    Args:
        result (str): the latex resulting from a test.
        expected (str): the expected latex output.

    Returns:
        str: failure message.
    """
    return "Discrepancies in circuit builder tex output: \
    {0}".format(
        LINE_RETURN + LINE_RETURN.join(difflib.ndiff([result], [expected]))
    )


class TestCircuitDrawerClass:
    """Direct tests of the CircuitDrawer class"""

    def test_end_wire(self, drawer):
        drawer._end_wire()
        assert drawer._document.endswith(WIRE_TERMINATOR)

    def test_end_circuit(self, drawer):
        drawer._end_circuit()
        assert drawer._document.endswith(CIRCUIT_BODY_TERMINATOR)

    def test_begin_circuit(self, drawer):
        drawer._begin_circuit()
        assert drawer._document.endswith(CIRCUIT_BODY_START)

    def test_init_document(self, drawer):
        drawer._init_document()
        assert drawer._document.endswith(INIT_DOCUMENT)

    def test_set_row_spacing(self, drawer):
        drawer._set_row_spacing("1em")
        assert drawer._row_spacing == "1em"

    def test_set_column_spacing(self, drawer):
        drawer._set_column_spacing("2em")
        assert drawer._column_spacing == "2em"

    def test_apply_spacing(self, drawer):
        drawer._set_row_spacing("1em")
        drawer._set_column_spacing("2em")
        drawer._apply_spacing()
        assert drawer._pad_with_spaces(COLUMN_SPACING.format("2em")) in drawer._document
        assert drawer._pad_with_spaces(ROW_SPACING.format("1em")) in drawer._document

    def test_op_applications(self, drawer):
        drawer._x(0)
        drawer._z(0)
        drawer._cx(0, 1)
        drawer._cz(0, 1)
        drawer._bs(0, 1)
        drawer._s2(0, 1)
        drawer._ck(0, 1)
        drawer._k(0)
        drawer._v(0)
        drawer._p(0)
        drawer._r(0)
        drawer._s(0)
        drawer._d(0)

        expected_circuit_matrix = [
            [
                "\\gate{X}",
                "\\gate{Z}",
                "\\ctrl{1}",
                "\\ctrl{1}",
                "\\multigate{1}{BS}",
                "\\multigate{1}{S}",
                "\\ctrl{1}",
                "\\gate{K}",
                "\\gate{V}",
                "\\gate{P}",
                "\\gate{R}",
                "\\gate{S}",
                "\\gate{D}",
            ],
            ["\\qw"] * 2
            + ["\\targ", "\\gate{Z}", "\\ghost{BS}", "\\ghost{S}", "\\gate{K}"]
            + ["\\qw"] * 6,
            ["\\qw"] * 13,
        ]

        assert drawer._circuit_matrix == expected_circuit_matrix

    def test_add_column(self, drawer):
        drawer._add_column()
        for wire in drawer._circuit_matrix:
            assert wire[-1] == QUANTUM_WIRE

    def test_on_empty_column(self, drawer):
        drawer._add_column()
        assert drawer._on_empty_column()

    def test_mode_mismatch(self, drawer):
        class Fakeop:
            r"""
            Fake operation used to test mode mismatch handling, a beam splitter with only one
            register reference.
            """

            def __init__(self):
                self.reg = [sf.program.RegRef(0)]
                Fakeop.__name__ = "Command"

            # Custom string representation necessary to trick the circuitdrawer
            # into executing the unsupported operation.
            def __repr__(self):
                return "BSgate | (q[0])"

        with pytest.raises(ModeMismatchException):
            drawer.parse_op(Fakeop())


class TestEngineIntegration:
    """Tests for calling the circuit drawer via the engine"""

    def test_one_mode_gates_from_operators(self, drawer):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Xgate(1) | (q[0])
            ops.Zgate(1) | (q[0])
            ops.Kgate(1) | (q[0])
            ops.Vgate(1) | (q[0])
            ops.Pgate(1) | (q[0])
            ops.Rgate(1) | (q[0])
            ops.Sgate(1) | (q[0])
            ops.Dgate(1) | (q[0])

        for op in prog.circuit:
            method, mode = drawer._gate_from_operator(op)
            assert callable(method) and hasattr(drawer, method.__name__)
            assert mode == 1

    def test_two_mode_gates_from_operators(self, drawer):
        prog = sf.Program(3)

        with prog.context as q:
            ops.CXgate(1) | (q[0], q[1])
            ops.CZgate(1) | (q[0], q[1])
            ops.BSgate(1) | (q[0], q[1])
            ops.S2gate(1) | (q[0], q[1])
            ops.CKgate(1) | (q[0], q[1])

        for op in prog.circuit:
            method, mode = drawer._gate_from_operator(op)
            assert callable(method) and hasattr(drawer, method.__name__)
            assert mode == 2

    def test_parse_op(self, drawer):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Xgate(1) | (q[0])
            ops.Zgate(1) | (q[0])
            ops.CXgate(1) | (q[0], q[1])
            ops.CZgate(1) | (q[0], q[1])
            ops.BSgate(0, 1) | (q[0], q[1])
            ops.S2gate(0, 1) | (q[0], q[1])
            ops.CKgate(1) | (q[0], q[1])
            ops.Kgate(1) | (q[0])
            ops.Vgate(1) | (q[0])
            ops.Pgate(1) | (q[0])
            ops.Rgate(1) | (q[0])
            ops.Sgate(1) | (q[0])
            ops.Dgate(1) | (q[0])

        for op in prog.circuit:
            drawer.parse_op(op)

        expected_circuit_matrix = [
            [
                "\\gate{X}",
                "\\gate{Z}",
                "\\ctrl{1}",
                "\\ctrl{1}",
                "\\multigate{1}{BS}",
                "\\multigate{1}{S}",
                "\\ctrl{1}",
                "\\gate{K}",
                "\\gate{V}",
                "\\gate{P}",
                "\\gate{R}",
                "\\gate{S}",
                "\\gate{D}",
            ],
            ["\\qw"] * 2
            + ["\\targ", "\\gate{Z}", "\\ghost{BS}", "\\ghost{S}", "\\gate{K}"]
            + ["\\qw"] * 6,
            ["\\qw"] * 13,
        ]

        assert drawer._circuit_matrix == expected_circuit_matrix

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_fourier(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Fourier | (q[0])

        fourier_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{F}  & \qw \\
             & \qw  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == fourier_output, failure_message(result, fourier_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_x_0(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Xgate(1) | (q[0])

        x_test_0_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{X}  & \qw \\
             & \qw  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == x_test_0_output, failure_message(result, x_test_0_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_x_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Xgate(1) | (q[1])

        x_test_1_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \qw  & \qw \\
             & \gate{X}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == x_test_1_output, failure_message(result, x_test_1_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_xx_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Xgate(1) | (q[1])
            ops.Xgate(1) | (q[1])

        xx_test_1_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \qw  & \qw  & \qw \\
             & \gate{X}  & \gate{X}  & \qw \\
             & \qw  & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == xx_test_1_output, failure_message(result, xx_test_1_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_x_z_0(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Xgate(1) | (q[0])
            ops.Zgate(1) | (q[0])

        x_z_test_0_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{X}  & \gate{Z}  & \qw \\
             & \qw  & \qw  & \qw \\
             & \qw  & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == x_z_test_0_output, failure_message(result, x_z_test_0_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_x_0_z_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Xgate(1) | (q[0])
            ops.Zgate(1) | (q[1])

        x_0_z_1_test_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{X}  & \qw \\
             & \gate{Z}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == x_0_z_1_test_output, failure_message(result, x_0_z_1_test_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_z_0(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Zgate(1) | (q[0])

        z_test_0_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{Z}  & \qw \\
             & \qw  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == z_test_0_output, failure_message(result, z_test_0_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_z_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Zgate(1) | (q[1])

        z_test_1_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \qw  & \qw \\
             & \gate{Z}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == z_test_1_output, failure_message(result, z_test_1_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_zz_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Zgate(1) | (q[1])
            ops.Zgate(1) | (q[1])

        zz_test_1_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \qw  & \qw  & \qw \\
             & \gate{Z}  & \gate{Z}  & \qw \\
             & \qw  & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == zz_test_1_output, failure_message(result, zz_test_1_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_cx(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.CXgate(1) | (q[0], q[1])

        cx_test_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \ctrl{1}  & \qw \\
             & \targ  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == cx_test_output, failure_message(result, cx_test_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_cz(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.CZgate(1) | (q[0], q[1])

        cz_test_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \ctrl{1}  & \qw \\
             & \gate{Z}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == cz_test_output, failure_message(result, cz_test_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_bs(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.BSgate(0, 1) | (q[0], q[1])

        bs_test_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \multigate{1}{BS}  & \qw \\
             & \ghost{BS}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == bs_test_output, failure_message(result, bs_test_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_s2(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.S2gate(1) | (q[0], q[1])

        s2_test_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \multigate{1}{S}  & \qw \\
             & \ghost{S}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == s2_test_output, failure_message(result, s2_test_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_ck(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.CKgate(1) | (q[0], q[1])

        ck_test_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \ctrl{1}  & \qw \\
             & \gate{K}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == ck_test_output, failure_message(result, ck_test_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_k_0(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Kgate(1) | (q[0])

        k_test_0_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{K}  & \qw \\
             & \qw  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == k_test_0_output, failure_message(result, k_test_0_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_k_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Kgate(1) | (q[1])

        k_test_1_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \qw  & \qw \\
             & \gate{K}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == k_test_1_output, failure_message(result, k_test_1_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_v_0(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Vgate(1) | (q[0])

        v_test_0_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{V}  & \qw \\
             & \qw  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == v_test_0_output, failure_message(result, v_test_0_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_v_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Vgate(1) | (q[1])

        v_test_1_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \qw  & \qw \\
             & \gate{V}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == v_test_1_output, failure_message(result, v_test_1_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_p_0(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Pgate(1) | (q[0])

        p_test_0_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{P}  & \qw \\
             & \qw  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == p_test_0_output, failure_message(result, p_test_0_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_p_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Pgate(1) | (q[1])

        p_test_1_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \qw  & \qw \\
             & \gate{P}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == p_test_1_output, failure_message(result, p_test_1_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_r_0(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Rgate(1) | (q[0])

        r_test_0_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{R}  & \qw \\
             & \qw  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == r_test_0_output, failure_message(result, r_test_0_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_r_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Rgate(1) | (q[1])

        r_test_1_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \qw  & \qw \\
             & \gate{R}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == r_test_1_output, failure_message(result, r_test_1_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_s_0(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Sgate(1) | (q[0])

        s_test_0_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{S}  & \qw \\
             & \qw  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == s_test_0_output, failure_message(result, s_test_0_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_s_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Sgate(1) | (q[1])

        s_test_1_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \qw  & \qw \\
             & \gate{S}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == s_test_1_output, failure_message(result, s_test_1_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_d_0(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Dgate(1) | (q[0])

        d_test_0_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \gate{D}  & \qw \\
             & \qw  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == d_test_0_output, failure_message(result, d_test_0_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_d_1(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Dgate(1) | (q[1])

        d_test_1_output = dedent(
            r"""            \documentclass{article}
            \pagestyle{empty}
            \usepackage{qcircuit}
            \begin{document}
            \Qcircuit {
             & \qw  & \qw \\
             & \gate{D}  & \qw \\
             & \qw  & \qw \\
            }
            \end{document}"""
        )

        result = prog.draw_circuit(tex_dir=tmpdir)[1]
        assert result == d_test_1_output, failure_message(result, d_test_1_output)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_not_drawable(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.BSgate(0, 2) | (q[0], q[2])

        with pytest.raises(NotDrawableException):
            prog.draw_circuit(tex_dir=tmpdir)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_unsupported_gate(self, tmpdir):
        prog = sf.Program(3)

        class Fakegate(ops.Gate):
            r"""Extension of the base Gate class to use to test
            unsupported operation handling."""

            def __init__(self):
                super().__init__([0])

            def _apply(self, reg, backend, **kwargs):
                pass

        with prog.context as q:
            Fakegate() | (q[0])

        with pytest.raises(UnsupportedGateException):
            prog.draw_circuit(tex_dir=tmpdir)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_compile(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Dgate(1) | (q[1])
            ops.Rgate(1) | (q[1])
            ops.S2gate(1) | (q[0], q[1])

        document = prog.draw_circuit(tex_dir=tmpdir)[0]

        file_name = "output_{0}.tex".format(
            datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p")
        )
        assert document.split("/")[-1] == file_name

        output_file = tmpdir.join(file_name)
        assert os.path.isfile(output_file)

    @pytest.mark.skipif(
        sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6"
    )
    def test_compile_custom_dir(self, tmpdir):
        prog = sf.Program(3)

        with prog.context as q:
            ops.Dgate(1) | (q[1])
            ops.Rgate(1) | (q[1])
            ops.S2gate(1) | (q[0], q[1])

        subdir = tmpdir.join("subdir")
        document = prog.draw_circuit(tex_dir=subdir)[0]

        file_name = "output_{0}.tex".format(
            datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p")
        )
        assert document.split("/")[-1] == file_name

        output_file = subdir.join(file_name)
        assert os.path.isfile(output_file)

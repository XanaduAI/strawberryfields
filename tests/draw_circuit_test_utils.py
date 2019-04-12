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
r"""
Draw circuit test utils
==============

String constants and utility function(s) used by tests of the :class:`~strawberryfields.circuitdrawer.Circuit' class.
"""
import difflib

LINE_RETURN = '\n'
WIRE_TERMINATOR = r'\\' + '\n'
CIRCUIT_BODY_TERMINATOR = '}\n'
CIRCUIT_BODY_START = ' {' + '\n'
BEGIN_DOCUMENT = r'\begin{document}'
DOCUMENT_CLASS = r'\documentclass{article}'
QCIRCUIT_PACKAGE = r'\usepackage{qcircuit}'
CIRCUIT_START = r'\Qcircuit'
COLUMN_SPACING = '@C={0}'
ROW_SPACING = '@R={0}'
QUANTUM_WIRE = r'\qw'
INIT_DOCUMENT = DOCUMENT_CLASS + '\n' + QCIRCUIT_PACKAGE + '\n' + BEGIN_DOCUMENT + '\n' + CIRCUIT_START

expected_circuit_matrix = [
    ['\\gate{X}',
     '\\gate{Z}',
     '\\ctrl{1}',
     '\\ctrl{1}',
     '\\multigate{1}{BS}',
     '\\multigate{1}{S}',
     '\\ctrl{1}',
     '\\gate{K}',
     '\\gate{V}',
     '\\gate{P}',
     '\\gate{R}',
     '\\gate{S}',
     '\\gate{D}'
     ],
    ['\\qw',
     '\\qw',
     '\\targ',
     '\\gate{Z}',
     '\\ghost{BS}',
     '\\ghost{S}',
     '\\gate{K}',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw'
     ],
    ['\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw',
     '\\qw'
     ]
]

x_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{X}  & \qw \\
 & \qw  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

x_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \qw  & \qw \\
 & \gate{X}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

xx_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \qw  & \qw  & \qw \\
 & \gate{X}  & \gate{X}  & \qw \\
 & \qw  & \qw  & \qw \\
}
\end{document}"""

x_z_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{X}  & \gate{Z}  & \qw \\
 & \qw  & \qw  & \qw \\
 & \qw  & \qw  & \qw \\
}
\end{document}"""

k_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{K}  & \qw \\
 & \qw  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

k_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \qw  & \qw \\
 & \gate{K}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

v_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{V}  & \qw \\
 & \qw  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

v_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \qw  & \qw \\
 & \gate{V}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

p_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{P}  & \qw \\
 & \qw  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

p_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \qw  & \qw \\
 & \gate{P}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

r_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{R}  & \qw \\
 & \qw  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

r_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \qw  & \qw \\
 & \gate{R}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

z_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{Z}  & \qw \\
 & \qw  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

z_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \qw  & \qw \\
 & \gate{Z}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

zz_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \qw  & \qw  & \qw \\
 & \gate{Z}  & \gate{Z}  & \qw \\
 & \qw  & \qw  & \qw \\
}
\end{document}"""

x_0_z_1_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{X}  & \qw \\
 & \gate{Z}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

cx_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \ctrl{1}  & \qw \\
 & \targ  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

cz_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \ctrl{1}  & \qw \\
 & \gate{Z}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

bs_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \multigate{1}{BS}  & \qw \\
 & \ghost{BS}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

s2_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \multigate{1}{S}  & \qw \\
 & \ghost{S}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

ck_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \ctrl{1}  & \qw \\
 & \gate{K}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

s_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{S}  & \qw \\
 & \qw  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

s_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \qw  & \qw \\
 & \gate{S}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

d_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{D}  & \qw \\
 & \qw  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

d_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \qw  & \qw \\
 & \gate{D}  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""

fourier_output = r"""\documentclass{article}
\usepackage{qcircuit}
\begin{document}
\Qcircuit {
 & \gate{F}  & \qw \\
 & \qw  & \qw \\
 & \qw  & \qw \\
}
\end{document}"""


def failure_message(result, expected):
    """Build failure message with explicit error latex diff.

    Args:
        result (str): the latex resulting from a test.
        expected (str): the expected latex output.

    Returns:
        str: failure message.
    """
    return 'Discrepancies in circuit builder tex output: \
    {0}'.format(LINE_RETURN + LINE_RETURN.join(difflib.ndiff([result], [expected])))

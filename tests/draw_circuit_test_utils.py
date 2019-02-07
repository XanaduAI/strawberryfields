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

x_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{X} \\
 & \qw[1] \\
 & \qw[1] \\

}
\end{document}"""

x_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \qw[1] \\
 & \gate{X} \\
 & \qw[1] \\

}
\end{document}"""

xx_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \qw[1]  & \qw[1] \\
 & \gate{X}  & \gate{X} \\
 & \qw[1]  & \qw[1] \\

}
\end{document}"""

x_z_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{X}  & \gate{Z} \\
 & \qw[1]  & \qw[1] \\
 & \qw[1]  & \qw[1] \\

}
\end{document}"""

k_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{K} \\
 & \qw[1] \\
 & \qw[1] \\

}
\end{document}"""

k_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \qw[1] \\
 & \gate{K} \\
 & \qw[1] \\

}
\end{document}"""

v_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{V} \\
 & \qw[1] \\
 & \qw[1] \\

}
\end{document}"""

v_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \qw[1] \\
 & \gate{V} \\
 & \qw[1] \\

}
\end{document}"""

p_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{P} \\
 & \qw[1] \\
 & \qw[1] \\

}
\end{document}"""

p_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \qw[1] \\
 & \gate{P} \\
 & \qw[1] \\

}
\end{document}"""

r_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{R} \\
 & \qw[1] \\
 & \qw[1] \\

}
\end{document}"""

r_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \qw[1] \\
 & \gate{R} \\
 & \qw[1] \\

}
\end{document}"""

z_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{Z} \\
 & \qw[1] \\
 & \qw[1] \\

}
\end{document}"""

z_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \qw[1] \\
 & \gate{Z} \\
 & \qw[1] \\

}
\end{document}"""

zz_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \qw[1]  & \qw[1] \\
 & \gate{Z}  & \gate{Z} \\
 & \qw[1]  & \qw[1] \\

}
\end{document}"""

x_0_z_1_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{X} \\
 & \gate{Z} \\
 & \qw[1] \\

}
\end{document}"""

cx_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \ctrl{1} \\
 & \targ \\
 & \qw[1] \\

}
\end{document}"""

cz_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \ctrl{1} \\
 & \gate{Z} \\
 & \qw[1] \\

}
\end{document}"""

bs_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{BS} \\
 & \gate{BS} \\
 & \qw[1] \\

}
\end{document}"""

s2_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{S} \\
 & \gate{S} \\
 & \qw[1] \\

}
\end{document}"""

ck_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \ctrl{1} \\
 & \gate{K} \\
 & \qw[1] \\

}
\end{document}"""

s_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{S} \\
 & \qw[1] \\
 & \qw[1] \\

}
\end{document}"""

s_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \qw[1] \\
 & \gate{S} \\
 & \qw[1] \\

}
\end{document}"""

d_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{D} \\
 & \qw[1] \\
 & \qw[1] \\

}
\end{document}"""

d_test_1_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \qw[1] \\
 & \gate{D} \\
 & \qw[1] \\

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

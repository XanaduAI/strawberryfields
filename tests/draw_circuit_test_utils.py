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
 & \ctrl{1} \\
 & \gate{BS} \\
 & \qw[1] \\

}
\end{document}"""

s2_test_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \ctrl{1} \\
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
    return 'Discrepancies in circuit builder tex output: \
    {0}'.format(LINE_RETURN + LINE_RETURN.join(difflib.ndiff([result], [expected])))

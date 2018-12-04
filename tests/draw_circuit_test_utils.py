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

def failure_message(result, expected):
    return f'Discrepancies in circuit builder tex output: \
    {LINE_RETURN + LINE_RETURN.join(difflib.ndiff([result], [expected]))}'
"""
Unit tests for :mod:`strawberryfields.ops`

The tests in this module do not require the quantum program to be run, and thus do not use a backend.
"""

import logging
logging.getLogger()

import unittest

from defaults import BaseTest, strawberryfields as sf
from strawberryfields.ops import *

x_test_0 = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{X} \\
 & \qw[{0}] \\
 & \qw[{0}] \\

}
\end{document}
"""

x_test_1 = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{X} \\
 & \qw[{0}] \\
 & \qw[{0}] \\

}
\end{document}
"""

x_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{X} \\
 & \qw[{0}] \\
 & \qw[{0}] \\

}
\end{document}
"""

x_z_test_0_output = r"""\documentclass{article}
\usepackage{qcircuit}
\Qcircuit {
 & \gate{X} \\
 & \qw[{0}] \\
 & \qw[{0}] \\

}
\end{document}
"""

class CircuitDrawerTests(BaseTest):

    num_subsystems = 3
    cutoff = 10

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)

    def test_x_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == x_test_0_output)


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', circuit drawer module.')
    # run the tests in this file
    suite = unittest.TestSuite()
    ttt = unittest.TestLoader().loadTestsFromTestCase(CircuitDrawerTests)
    suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

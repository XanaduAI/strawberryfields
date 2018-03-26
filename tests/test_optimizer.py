"""
Unit tests for the :mod:`strawberryfields` circuit optimizer.
"""

import unittest
# HACK to import the module in this source distribution, not the installed one!
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from numpy.random import randn
from math import pi

import strawberryfields as sf
from strawberryfields.engine import *
from strawberryfields.ops import *
from defaults import BaseTest


class BasicTests(BaseTest):
    """Implementation-independent tests."""
    num_subsystems = 3
    def setUp(self):
        super().setUp()
        # construct a compiler engine
        self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)

    def test_optimization1(self):
        "Cancelling and merging single-mode gates."
        D = Dgate(1)
        S = Sgate(1)
        R = Rgate(-pi)
        with self.eng:
            Dgate(0.6) | 0
            Sgate(0.2) | 0
            Sgate(-0.2) | 0
            S   | 0
            D   | 0
            S.H | 1
            D.H | 0
            R.H | 0
            R   | 0
            S.H | 0
            Dgate(-0.2) | 0
            S   | 1

        self.eng.optimize()
        # one displacement gate remains
        self.assertEqual(len(self.eng.cmd_queue), 1)

    def test_optimization2(self):
        "Optimization test 2."
        with self.eng:
            Xgate(0.6) | 0
            Zgate(0.2) | 0

        self.eng.optimize()
        # two displacements in different directions remain, cannot be combined (for now)
        self.assertEqual(len(self.eng.cmd_queue), 2)


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', circuit optimizer.')

    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BaseTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

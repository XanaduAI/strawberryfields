"""
Unit tests for the :mod:`strawberryfields` circuit optimizer.
"""

import unittest

from numpy.random import rand, randn, permutation
from numpy import round

from defaults import BaseTest, strawberryfields as sf
from strawberryfields.engine import *
from strawberryfields.ops import *


# all single-mode gates with at least one parameter
single_mode_gates = [x for x in one_args_gates+two_args_gates if x.ns == 1]


class BasicTests(BaseTest):
    """Implementation-independent tests."""
    num_subsystems = 3
    def setUp(self):
        super().setUp()
        # construct a compiler engine
        self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)

    def test_merge_dagger(self):
        "Optimizer merging single-mode gates with their daggered versions."
        self.logTestName()
        for G in single_mode_gates:
            G = G(*randn(1))
            with self.eng:
                G   | 0
                G.H | 0
            self.eng.optimize()
            self.assertEqual(len(self.eng.cmd_queue), 0)
            self.eng.reset_queue()

    def test_merge_negated(self):
        "Optimizer merging single-mode gates with their negated versions."
        self.logTestName()
        for G in single_mode_gates:
            temp = randn(1)[0]
            g1 = G(temp)
            g2 = G(-temp)
            with self.eng:
                g1 | 0
                g2 | 0
            self.eng.optimize()
            self.assertEqual(len(self.eng.cmd_queue), 0)
            self.eng.reset_queue()

    def test_merge_chain(self):
        "Optimizer merging chains of single-mode gates."
        self.logTestName()

        # construct one instance of each gate type with a random parameter
        params = randn(len(single_mode_gates))
        gates = [G(p) for G, p in zip(single_mode_gates, params)]
        # random permutation
        gates = permutation(gates)

        # palindromic cancelling
        with self.eng:
            for G in gates:
                G   | 0
            for G in reversed(gates):
                G.H | 0
        self.eng.optimize()
        self.assertEqual(len(self.eng.cmd_queue), 0)
        self.eng.reset_queue()

        # pairwise cancelling
        with self.eng:
            for G in gates:
                G   | 0
                G.H | 0
        self.eng.optimize()
        self.assertEqual(len(self.eng.cmd_queue), 0)
        self.eng.reset_queue()

        # interleaved chains on two subsystems, palindromic cancelling
        which = round(rand(len(gates))).astype(int)  # subsystem 0 or 1?
        with self.eng:
            for G, s in zip(gates, which):
                G | s
            for G, s in zip(reversed(gates), reversed(which)):
                G.H | s
        self.eng.optimize()
        self.assertEqual(len(self.eng.cmd_queue), 0)


    def test_merge_incompatible(self):
        "Merging incompatible gates."
        self.logTestName()
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
    for t in (BasicTests,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

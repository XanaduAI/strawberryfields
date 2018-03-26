"""
Unit tests for :mod:`strawberryfields.ops`

The tests in this module do not require the quantum program to be run, and thus do not use a backend.
"""

import unittest
from collections import Counter
from numpy.random import randn, random
from numpy import sqrt

# HACK to import the module in this source distribution, not the installed one!
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import strawberryfields as sf
from strawberryfields.engine import Engine
from strawberryfields.utils import *
from strawberryfields.ops import *
from defaults import BaseTest, FockBaseTest


class GateTests(BaseTest):
    def setUp(self):
        self.tol = 1e-6
        self.hbar = 2
        # engine without backend
        self.eng = Engine(num_subsystems=3, hbar=self.hbar)


    def test_init_basics(self):
        "Basic properties of initializers."

        # initializer to test against
        V = Vacuum()
        def test_init(G):
            # init merged with another returns the last one
            self.assertAllTrue(G.merge(V) == V)
            self.assertAllTrue(V.merge(G) == G)
            # all inits act on a single mode
            self.assertRaises(ValueError, G.__or__, (0,1))
            # can't repeat the same index
            self.assertRaises(IndexError, All(G).__or__, (0,0))

        with self.eng:
            for G in state_preparations:
                test_init(G())
            # ket initializer requires a parameter
            test_init(Ket(randnc(10,1)))


    def test_gate_basics(self):
        "Basic properties of gates."

        # test gate
        Q = Xgate(0.5)
        def test_gate(G):
            # gate merged with its inverse is identity
            self.assertAllTrue(G.merge(G.H) == None)
            # gates cannot be merged with a different type of gate
            if not isinstance(G, Q.__class__):
                self.assertRaises(TypeError, Q.merge, G)
                self.assertRaises(TypeError, G.merge, Q)
            # wrong number of subsystems
            if G.ns == 1:
                self.assertRaises(ValueError, G.__or__, (0,1))
            else:
                self.assertRaises(ValueError, G.__or__, 0)

        with self.eng:
            for G in two_args_gates:
                # construct a random gate
                G = G(*randn(2))
                test_gate(G)

            for G in one_args_gates:
                # construct a random gate
                G = G(*randn(1))
                test_gate(G)

            for G in zero_args_gates:
                # preconstructed singleton instances
                test_gate(G)

            for G in two_mode_gates:
                G = G(*randn(1))
                # can't repeat the same index
                self.assertRaises(IndexError, G.__or__, (0,0))
            self.assertRaises(IndexError, All(Q).__or__, (0,0))


    def test_gate_merging(self):
        "Nontrivial merging of gates."

        def merge_gates(f):
            # test the merging of two gates (with default values for optional parameters)
            a, b = randn(2)
            G = f(a)
            temp = G.merge(f(b))
            self.assertAlmostEqual(temp.p[0], a+b, delta=self.tol)
            temp = G.merge(f(b).H)
            self.assertAlmostEqual(temp.p[0], a-b, delta=self.tol)

        for G in one_args_gates+two_args_gates:
            merge_gates(G)


    def test_loss_merging(self):
        "Nontrivial merging of loss."

        # test the merging of two gates (with default values for optional parameters)
        a, b = random(2)
        G = LossChannel(a)
        temp = G.merge(LossChannel(b))
        self.assertAlmostEqual(temp.p[0], sqrt(a*b), delta=self.tol)


    def test_dispatch(self):
        "Dispatching of gates to the engine."

        with self.eng:
            for G in one_args_gates+two_args_gates:
                # construct a random gate
                G = G(*randn(1))
                if G.ns == 1:
                    G | 0
                    All(G) | (0,1)
                else:
                    G | (0,1)
                    G | (2,0)

            for I in state_preparations:
                I = I()
                I | 0
                All(I) | (0,1)


    def test_create_delete(self):
        """Creating and deleting new modes."""

        # New must not be called via its __or__ method
        self.assertRaises(ValueError, New.__or__, 0)

        # get register references
        alice, bob, charlie = self.eng.register
        with self.eng:
            # number of new modes must be a positive integer
            self.assertRaises(ValueError, New.__call__, -2)
            self.assertRaises(ValueError, New.__call__, 1.5)

            # deleting nonexistent modes
            self.assertRaises(KeyError, Del.__or__, 100)

            Del | alice
            diana = New(1)
            # deleting already deleted modes
            self.assertRaises(IndexError, Del.__or__, alice)

            edward, grace = New(2)
            Del | (charlie, grace)

        #self.eng.print_program()
        temp = self.eng.reg_refs
        c = Counter(temp.values())
        #print(c)
        # reference map has entries for both existing and deleted subsystems
        self.assertEqual(len(temp), self.eng.num_subsystems +c[None])


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', gates module.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (GateTests,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

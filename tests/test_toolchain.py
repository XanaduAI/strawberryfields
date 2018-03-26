"""
Unit tests for the :mod:`strawberryfields` full toolchain.
"""

import unittest
# HACK to import the module in this source distribution, not the installed one!
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from numpy.random import (randn, uniform, randint)
from numpy import array
from numpy import pi

import strawberryfields as sf
from strawberryfields.engine import *
from strawberryfields.ops import *
from defaults import BaseTest, FockBaseTest, GaussianBaseTest
from strawberryfields.backends import BaseFock



class BasicTests(BaseTest):
    """Implementation-independent tests."""
    num_subsystems = 2
    def setUp(self):
        super().setUp()
        # construct a compiler engine
        self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)


    def test_engine(self):
        "Basic engine tests."

        # backed is initialized to vacuum state
        self.assertAllTrue(self.backend.is_vacuum(self.tol))

        D = Dgate(0.5)
        D = Sgate(0.543)
        with self.eng:
            # trying to act on a nonexistent mode
            self.assertRaises(KeyError, D.__or__, self.eng.num_subsystems)

        # successful gate application
        with self.eng:
            D | 0
        self.assertEqual(len(self.eng.cmd_queue), 1)

        # print the queue contents
        self.eng.print_queue()

        # clear the command queue
        self.eng.reset_queue()
        self.assertEqual(len(self.eng.cmd_queue), 0)

        # run engine
        self.eng.run(backend=self.backend)

        # clear the command queue and reset the backend state to vacuum
        self.eng.reset()
        self.assertAllTrue(self.backend.is_vacuum(self.tol))


    def test_gate_dagger(self):
        "Dagger functionality of the gates."

        def test_gate(G):
            self.eng.reset()
            q = self.eng.register
            with self.eng:
                if G.ns == 1:
                    G   | q[0]
                    G.H | q[0]
                else:
                    G   | (q[0],q[1])
                    G.H | (q[0],q[1])
            state = self.eng.run(backend=self.backend)
            # state norm must be invariant
            if isinstance(self.eng.backend, BaseFock):
                self.assertAllAlmostEqual(state.trace(), 1, delta=self.tol,
                                       msg="Trace after G.H * G is not 1 for gate G={}".format(G))
            # we must end up back in vacuum since G and G.H cancel each other
            self.assertAllTrue(self.backend.is_vacuum(self.tol),
                                   msg="State is not vacuum after G.H * G for gate G={}".format(G))

        for G in two_args_gates:
            # construct a random gate
            G = G(uniform(high=0.1), uniform(high=2 * pi))
            test_gate(G)

        for G in one_args_gates:
            if G in (Vgate,) and not self.args.fock_support:
                continue  # the V gate cannot be used on Gaussian backends
            # construct a random gate
            G = G(uniform(high=0.25))
            test_gate(G)


    def test_regrefs(self):
        """Testing register references."""
        q = self.eng.register
        # using a measurement result before it exists
        with self.eng:
            Dgate(RR(q[0])) | q[1]
        self.assertRaises(SFProgramError, self.eng.run)

        self.eng.reset()
        # proper use
        with self.eng:
            MeasureX | q[0]
            Sgate(RR(q[0]))   | q[1]
            Dgate(RR(q[0])).H | q[1]  # symbolic hermitian conjugate together with register reference
            Sgate(RR(q[0], lambda x: x**2)) | q[1]
            Dgate(RR(q[0], lambda x: -x)).H | q[1]
        self.eng.run(backend=self.backend)


    def test_homodyne_measurement(self):
        """Homodyne measurements."""
        q = self.eng.register
        with self.eng:
            Coherent(*randn(2)) | q[0]
            Coherent(*randn(2)) | q[1]
            MeasureX | q[0]
            MeasureP | q[1]
        self.eng.run(backend=self.backend)
        # homodyne measurements leave the mode in vacuum state
        self.assertAllTrue(self.backend.is_vacuum(self.tol))
        with self.eng:
            Coherent(*randn(2)) | q[0]
            MeasureHomodyne(*randn(1)) | q[0]
        self.eng.run(backend=self.backend)
        self.assertAllTrue(self.backend.is_vacuum(self.tol))


    def test_program_subroutine(self):
        """Simple quantum program with a subroutine and references."""
        # define some gates
        D = Dgate(0.5)
        BS = BSgate(2*pi, pi/2)
        R = Rgate(pi)
        # get register references
        alice, bob = self.eng.register

        def subroutine(a, b):
            "Subroutine for the quantum program"
            R   | a
            BS  | (a,b)
            R.H | a

        # main program
        with self.eng:
            All(Vacuum()) | (alice, bob)
            D   | alice
            subroutine(alice, bob)
            BS  | (alice, bob)
            subroutine(bob, alice)

        state = self.eng.run(backend=self.backend)
        # state norm must be invariant
        if isinstance(self.eng.backend, BaseFock):
            self.assertAllAlmostEqual(state.trace(), 1, delta=self.tol)


    def test_create_delete(self):
        """Creating and deleting new modes."""
        # define some gates
        D = Dgate(0.5)
        BS = BSgate(2*pi, pi/2)
        R = Rgate(pi)
        # get register references
        alice, bob = self.eng.register
        with self.eng:
            D        | alice
            BS       | (alice, bob)
            Del      | alice
            R        | bob
            charlie, = New(1)
            BS       | (bob, charlie)
            MeasureX | bob
            Del      | bob
            D.H      | charlie
            MeasureX | charlie
        #self.eng.print_program()
        self.eng.optimize()
        state = self.eng.run(backend=self.backend)
        # print('measurement result: a: {}, b: {}, c: {}'.format(alice.val, bob.val, charlie.val))
        # state norm must be invariant
        if isinstance(self.eng.backend, BaseFock):
            self.assertAllAlmostEqual(state.trace(), 1, delta=self.tol)


class FockBasisTests(FockBaseTest):
    """Fock-basis dependent tests."""
    num_subsystems = 2
    def setUp(self):
        super().setUp()
        # construct a compiler engine
        self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)

    def test_fock_measurement(self):
        """Fock measurements."""
        q = self.eng.register
        s = randint(0, self.D, (2,))
        with self.eng:
            Fock(s[0]) | q[0]
            Fock(s[1]) | q[1]
            Measure | q
        self.eng.run(backend=self.backend)
        # measurement result must be equal to preparation
        for a,b in zip(q, s):
            self.assertAllEqual(a.val, b)
        # Fock measurements put the modes into vacuum state
        self.assertAllTrue(self.backend.is_vacuum(self.tol))


    def test_coinflip(self):
        "Coin flip program."
        D = Dgate(1)
        reg = self.eng.register
        with self.eng:
            All(Vacuum()) | reg
            D   | reg[0]
            Measure | reg[0]

        state = self.eng.run(backend=self.backend)
        #print('Coin flip result: {}'.format(reg[0].val))
        # state norm must be invariant
        self.assertAllAlmostEqual(state.trace(), 1, delta=self.tol)


    def test_deferred_measurement(self):
        """Deferred measurements using subsystem references."""
        # define some gates
        D = Dgate(0.5)
        BS = BSgate(2*pi, pi/2)
        q = self.eng.register
        with self.eng:
            D   | q[0]
            #BS  | q[::-1]
            Measure | q[0]
            Dgate(RR(q[0], lambda x: 1 +1.3 * x**2)) | q[1]
            Measure | q[1]

        self.eng.optimize()
        state = self.eng.run(backend=self.backend)
        # print('measurement result: a: {}, b: {}'.format(q[0].val, q[1].val))
        # state norm must be invariant
        self.assertAllAlmostEqual(state.trace(), 1, delta=self.tol)
        # see what the ket looks like
        temp = self.backend.state()
        #print(temp)



class GaussianTests(GaussianBaseTest):
    """Gaussian-only tests."""
    num_subsystems = 2
    def setUp(self):
        super().setUp()
        # construct a compiler engine
        self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)

    def test_gaussian_measurement(self):
        """Gaussian-only measurements."""
        q = self.eng.register
        with self.eng:
            Coherent(*randn(2)) | q[0]
            Coherent(*randn(2)) | q[1]
            MeasureHD | q[0]
            MeasureHD | q[1]
        self.eng.run(backend=self.backend)
        # heterodyne measurements leave the mode in vacuum state
        self.assertAllTrue(self.backend.is_vacuum(self.tol))


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', full toolchain.')

    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTests, FockBasisTests, GaussianTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

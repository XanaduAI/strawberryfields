"""
Unit tests for utilities in :class:`strawberryfields.utils`.
"""

import os
import sys
import signal

import unittest

import numpy as np
from numpy import pi

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import *
from strawberryfields.backends.shared_ops import sympmat
from defaults import BaseTest, FockBaseTest, GaussianBaseTest

a = 0.32+0.1j
r = 0.212
phi = 0.123
n = 2

cutoff = 10

class InitialStates(BaseTest):
    num_subsystems = 1

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)

    def test_vacuum_state(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Vac | q[0]

        state = self.eng.run(backend=self.backend_name, cutoff_dim=cutoff)

        if self.backend_name == 'gaussian':
            psi = state.reduced_gaussian(0)
            psi_exact = vacuum_state(basis='gaussian', hbar=self.hbar)
        else:
            psi = state.ket()
            psi_exact = vacuum_state(basis='fock', fock_dim=cutoff, hbar=self.hbar)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)

    def test_coherent_state(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Dgate(a) | q[0]

        state = self.eng.run(backend=self.backend_name, cutoff_dim=cutoff)

        if self.backend_name == 'gaussian':
            psi = state.reduced_gaussian(0)
            psi_exact = coherent_state(a, basis='gaussian', hbar=self.hbar)
        else:
            psi = state.ket()
            psi_exact = coherent_state(a, basis='fock', fock_dim=cutoff, hbar=self.hbar)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)

    def test_coherent_state(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Dgate(a) | q[0]

        state = self.eng.run(backend=self.backend_name, cutoff_dim=cutoff)

        if self.backend_name == 'gaussian':
            psi = state.reduced_gaussian(0)
            psi_exact = coherent_state(a, basis='gaussian', hbar=self.hbar)
        else:
            psi = state.ket()
            psi_exact = coherent_state(a, basis='fock', fock_dim=cutoff, hbar=self.hbar)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)

    def test_squeezed_state(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Sgate(r, phi) | q[0]

        state = self.eng.run(backend=self.backend_name, cutoff_dim=cutoff)

        if self.backend_name == 'gaussian':
            psi = state.reduced_gaussian(0)
            psi_exact = squeezed_state(r, phi, basis='gaussian', hbar=self.hbar)
        else:
            psi = state.ket()
            psi_exact = squeezed_state(r, phi, basis='fock', fock_dim=cutoff, hbar=self.hbar)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)

    def test_displaced_squeezed_state(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Sgate(r, phi) | q[0]
            Dgate(a) | q[0]

        state = self.eng.run(backend=self.backend_name, cutoff_dim=cutoff)

        if self.backend_name == 'gaussian':
            psi = state.reduced_gaussian(0)
            psi_exact = displaced_squeezed_state(a, r, phi, basis='gaussian', hbar=self.hbar)
        else:
            psi = state.ket()
            psi_exact = displaced_squeezed_state(a, r, phi, basis='fock', fock_dim=cutoff, hbar=self.hbar)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)


class FockInitialStates(FockBaseTest):
    num_subsystems = 1

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)

    def test_fock_state(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Fock(n) | q[0]

        state = self.eng.run(backend=self.backend_name, cutoff_dim=cutoff)

        psi = state.ket().real
        psi_exact = fock_state(n, fock_dim=cutoff)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)

    def test_cat_state(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Catstate(a, 0) | q[0]

        state = self.eng.run(backend=self.backend_name, cutoff_dim=cutoff)

        psi = state.ket()
        psi_exact = cat_state(a, 0, fock_dim=cutoff)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)


class ConvertFunctions(BaseTest):
    num_subsystems = 1

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)

    def test_neg(self):
        a = np.random.random()
        q = self.eng.register
        rrt = neg(q[0])

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(a), -a)

    def test_mag(self):
        x = np.random.random()
        y = np.random.random()
        q = self.eng.register
        rrt = mag(q[0])

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x+1j*y), np.abs(x+1j*y))

    def test_phase(self):
        x = np.random.random()
        y = np.random.random()
        q = self.eng.register
        rrt = phase(q[0])

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x+1j*y), np.angle(x+1j*y))

    def test_scale(self):
        a = np.random.random()
        x = np.random.random()
        q = self.eng.register
        rrt = scale(q[0], a)

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x), a*x)

    def test_shift(self):
        a = np.random.random()
        x = np.random.random()
        q = self.eng.register
        rrt = shift(q[0], a)

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x), a+x)

    def test_scale_shift(self):
        a = np.random.random()
        b = np.random.random()
        x = np.random.random()
        q = self.eng.register
        rrt = scale_shift(q[0], a, b)

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x), a*x+b)

    def test_power_positive_frac(self):
        a = np.random.random()
        x = np.random.random()
        q = self.eng.register
        rrt = power(q[0], a)

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x), x**a)

    def test_power_negative_int(self):
        a = -3
        x = np.random.random()
        q = self.eng.register
        rrt = power(q[0], a)

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x), x**a)


class RandomStates(BaseTest):
    num_subsystems = 1
    M = 3
    Om = sympmat(3)

    def test_random_covariance_pure(self):
        V = random_covariance(self.M, hbar=self.hbar, pure=True)
        det = np.linalg.det(V) - (self.hbar/2)**(2*self.M)
        eigs = np.linalg.eigvalsh(V)

        # check square and even number of rows/columns
        self.assertAllEqual(V.shape, np.array([2*self.M]*2))
        # check symmetric
        self.assertAllAlmostEqual(V, V.T, delta=self.tol)
        # check positive definite
        self.assertTrue(np.all(eigs > 0))
        # check pure
        self.assertAlmostEqual(det, 0, delta=self.tol)

    def test_random_covariance_mixed(self):
        V = random_covariance(self.M, hbar=self.hbar, pure=False)
        det = np.linalg.det(V) - (self.hbar/2)**(2*self.M)
        eigs = np.linalg.eigvalsh(V)

        # check square and even number of rows/columns
        self.assertAllEqual(V.shape, np.array([2*self.M]*2))
        # check symmetric
        self.assertAllAlmostEqual(V, V.T, delta=self.tol)
        # check positive definite
        self.assertTrue(np.all(eigs > 0))
        # check not pure
        self.assertNotAlmostEqual(det, 0, delta=self.tol)

    def test_random_symplectic_passive(self):
        S = random_symplectic(self.M, passive=True)
        # check square and even number of rows/columns
        self.assertAllEqual(S.shape, np.array([2*self.M]*2))
        # check symplectic
        self.assertAllAlmostEqual(S @ self.Om @ S.T, self.Om, delta=self.tol)
        # check orthogonal
        self.assertAllAlmostEqual(S @ S.T, np.identity(2*self.M), delta=self.tol)

    def test_random_symplectic_active(self):
        S = random_symplectic(self.M, passive=False)
        # check square and even number of rows/columns
        self.assertAllEqual(S.shape, np.array([2*self.M]*2))
        # check symplectic
        self.assertAllAlmostEqual(S @ self.Om @ S.T, self.Om, delta=self.tol)
        # check not orthogonal
        self.assertTrue(np.all(S @ S.T != np.identity(2*self.M)))

    def test_random_interferometer(self):
        U = random_interferometer(self.M)
        # check unitary
        self.assertAllAlmostEqual(U @ U.conj().T, np.identity(self.M), delta=self.tol)



if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', Utils.')

    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [InitialStates, FockInitialStates, ConvertFunctions, RandomStates]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

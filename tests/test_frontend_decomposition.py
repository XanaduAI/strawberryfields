"""
Unit tests for decompositions in :class:`strawberryfields.ops`.
"""

import os
import sys
import signal

import unittest

import numpy as np
from numpy import pi
from scipy.linalg import block_diag

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import *
from strawberryfields.decompositions import clements, bloch_messiah, williamson

from strawberryfields.backends.gaussianbackend import ops
from strawberryfields.backends.gaussianbackend import gaussiancircuit
from strawberryfields.backends.shared_ops import rotation_matrix as rot

from defaults import BaseTest, FockBaseTest, GaussianBaseTest


class GaussianDecompositions(GaussianBaseTest):
    num_subsystems = 3

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)

        U1 = ops.haar_measure(3)
        U2 = ops.haar_measure(3)

        state = gaussiancircuit.GaussianModes(3, hbar=self.hbar)
        ns = np.abs(np.random.random(3))
        for i in range(3):
            state.init_thermal(ns[i], i)
        state.apply_u(U1)
        for i in range(3):
            state.squeeze(np.log(i+1+2), 0, i)
        state.apply_u(U2)
        self.V_mixed = state.scovmatxp()*self.hbar/2
        th, self.S = williamson(self.V_mixed)
        self.o, r, p = bloch_messiah(self.S)
        self.u1 = (self.o[:3, :3] + 1j*self.o[3:, :3])

        state = gaussiancircuit.GaussianModes(3, hbar=self.hbar)
        state.apply_u(U1)
        for i in range(3):
            state.squeeze(np.log(i+1+2), 0, i)
        state.apply_u(U2)
        self.V_pure = state.scovmatxp()*self.hbar/2

    def test_covariance_random_state_mixed(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            CovarianceState(self.V_mixed) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(state.cov(), self.V_mixed, delta=self.tol)

    def test_covariance_random_state_pure(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            CovarianceState(self.V_pure) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(state.cov(), self.V_pure, delta=self.tol)

    def test_gaussian_transform(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            GaussianTransform(self.S) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(state.cov(), self.S@self.S.T*self.hbar/2, delta=self.tol)

    def test_interferometer(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            All(Squeezed(0.5)) | q
            init = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
            Interferometer(self.u1) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(state.cov(), self.o@init.cov()@self.o.T, delta=self.tol)

    def test_identity_interferometer(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Interferometer(np.identity(6)) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(self.eng.cmd_applied, [], delta=self.tol)


class GaussianCovarianceInitialStates(GaussianBaseTest):
    num_subsystems = 3

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)

    def test_covariance_vacuum(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            CovarianceState(np.identity(6)*self.hbar/2) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(self.eng.cmd_applied, [], delta=self.tol)

    def test_covariance_squeezed(self):
        self.eng.reset()
        q = self.eng.register
        cov = (self.hbar/2)*np.diag([np.exp(-0.1)]*3 + [np.exp(0.1)]*3)

        with self.eng:
            CovarianceState(cov) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)
        self.assertAllEqual(len(self.eng.cmd_applied), 3)

    def test_covariance_thermal(self):
        self.eng.reset()
        q = self.eng.register
        cov = np.diag(self.hbar*(np.array([0.3,0.4,0.2]*2)+0.5))

        with self.eng:
            CovarianceState(cov) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)
        self.assertAllEqual(len(self.eng.cmd_applied), 3)

    def test_covariance_rotated_squeezed(self):
        self.eng.reset()
        q = self.eng.register

        r = 0.1
        phi = 0.2312
        v1 = (self.hbar/2)*np.diag([np.exp(-r),np.exp(r)])
        A = ops.changebasis(3)
        cov = A.T @ block_diag(*[rot(phi) @ v1 @ rot(phi).T]*3) @ A

        with self.eng:
            CovarianceState(cov) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)
        self.assertAllEqual(len(self.eng.cmd_applied), 3)


class FockCovarianceInitialStates(FockBaseTest):
    """Fidelity tests."""
    num_subsystems = 3

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)

    def test_covariance_vacuum(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            CovarianceState(np.identity(6)*self.hbar/2) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(self.eng.cmd_applied, [], delta=self.tol)
        self.assertAllAlmostEqual(state.fidelity_vacuum(), 1, delta=self.tol)

    def test_covariance_squeezed(self):
        self.eng.reset()
        q = self.eng.register
        r = 0.05
        phi = 0
        cov = (self.hbar/2)*np.diag([np.exp(-2*r)]*3 + [np.exp(2*r)]*3)
        in_state = squeezed_state(r, phi, basis='fock', fock_dim=self.D)

        with self.eng:
            CovarianceState(cov) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllEqual(len(self.eng.cmd_applied), 3)

        for n in range(3):
            self.assertAllAlmostEqual(state.fidelity(in_state, n), 1, delta=self.tol)

    def test_covariance_rotated_squeezed(self):
        self.eng.reset()
        q = self.eng.register

        r = 0.1
        phi = 0.2312
        in_state = squeezed_state(r, phi, basis='fock', fock_dim=self.D)

        v1 = (self.hbar/2)*np.diag([np.exp(-2*r),np.exp(2*r)])
        A = ops.changebasis(3)
        cov = A.T @ block_diag(*[rot(phi) @ v1 @ rot(phi).T]*3) @ A

        with self.eng:
            CovarianceState(cov) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllEqual(len(self.eng.cmd_applied), 3)
        for n in range(3):
            self.assertAllAlmostEqual(state.fidelity(in_state, n), 1, delta=self.tol)


if __name__ == '__main__':
    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [
        GaussianDecompositions,
        GaussianCovarianceInitialStates,
        FockCovarianceInitialStates
        ]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

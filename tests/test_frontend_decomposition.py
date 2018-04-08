"""
Unit tests for decompositions in :class:`strawberryfields.ops`.
"""

import os
import sys
import signal

import unittest

import numpy as np
from numpy import pi

import strawberryfields as sf
from strawberryfields.ops import *
from defaults import BaseTest, FockBaseTest, GaussianBaseTest

from strawberryfields.backends.gaussianbackend import ops
from strawberryfields.backends.gaussianbackend import gaussiancircuit
from strawberryfields.decompositions import clements, bloch_messiah, williamson


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

    def test_covariance_state_mixed(self):
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            CovarianceState(self.V_mixed) | q

        state = self.eng.run(backend=self.backend_name, cutoff_dim=self.D)
        self.assertAllAlmostEqual(state.cov(), self.V_mixed, delta=self.tol)

    def test_covariance_state_pure(self):
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


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', Utils.')

    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [GaussianDecompositions]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

"""
Unit tests for the :mod:`strawberryfields` provided algorithms.
"""

import unittest

import numpy as np
from numpy import pi

from defaults import BaseTest, FockBaseTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import scale


float_regex = r'(-)?(\d+\.\d+)'


class Teleportation(BaseTest):
    num_subsystems = 3
    delta = 0.1
    cutoff = 10

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        # attach the backend (NOTE: self.backend is shared between the tests, make sure it is reset before use!)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)

    def test_teleportation_fidelity(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Coherent(0.5+0.2j) | q[0]
            BS = BSgate(pi/4, 0)

            Squeezed(-2) | q[1]
            Squeezed(2) | q[2]
            BS | (q[1], q[2])

            BS | (q[0], q[1])
            MeasureHomodyne(0, select=0) | q[0]
            MeasureHomodyne(pi/2, select=0) | q[1]

        state = self.eng.run()
        fidelity = state.fidelity_coherent([0,0,0.5+0.2j])
        self.assertAllAlmostEqual(fidelity, 1, delta=self.delta)


class GateTeleportation(GaussianBaseTest):
    num_subsystems = 4
    delta = 0.05

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        self.eng.backend = self.backend
        self.backend.reset()

    def test_gaussian_states_match(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Squeezed(0.1) | q[0]
            Squeezed(-2)  | q[1]
            Squeezed(-2)  | q[2]

            Pgate(0.5) | q[1]

            CZgate(1) | (q[0], q[1])
            CZgate(1) | (q[1], q[2])

            Fourier.H | q[0]
            MeasureHomodyne(0, select=0) | q[0]
            Fourier.H | q[1]
            MeasureHomodyne(0, select=0) | q[1]

            Squeezed(0.1) | q[3]
            Fourier       | q[3]
            Pgate(0.5)    | q[3]
            Fourier       | q[3]

        self.eng.run()
        cov1 = self.eng.backend.state([2]).cov()
        cov2 = self.eng.backend.state([3]).cov()
        self.assertAllAlmostEqual(cov1, cov2, delta=self.delta)


class GaussianBosonSampling(BaseTest):
    num_subsystems = 4
    cutoff = 6
    measure_states = [[0,0,0,0], [1,1,0,0], [0,1,0,1], [1,1,1,1], [2,0,0,0]]
    results = [
            0.176378447614135,
            0.0685595637122246,
            0.002056097258977398,
            0.00834294639986785,
            0.01031294525345511
        ]

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)

    def test_fock_probs(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            S = Sgate(1)
            S | q[0]
            S | q[1]
            S | q[2]
            S | q[3]

            # rotation gates
            Rgate(0.5719)  | q[0]
            Rgate(-1.9782) | q[1]
            Rgate(2.0603)  | q[2]
            Rgate(0.0644)  | q[3]

            # beamsplitter array
            BSgate(0.7804, 0.8578)  | (q[0], q[1])
            BSgate(0.06406, 0.5165) | (q[2], q[3])
            BSgate(0.473, 0.1176)   | (q[1], q[2])
            BSgate(0.563, 0.1517)   | (q[0], q[1])
            BSgate(0.1323, 0.9946)  | (q[2], q[3])
            BSgate(0.311, 0.3231)   | (q[1], q[2])
            BSgate(0.4348, 0.0798)  | (q[0], q[1])
            BSgate(0.4368, 0.6157)  | (q[2], q[3])

        state = self.eng.run()
        probs = [state.fock_prob(i) for i in self.measure_states]
        probs = np.array(probs).T.flatten()
        res = np.tile(self.results, self.bsize).flatten()
        self.assertAllAlmostEqual(probs, res, delta=self.tol)


class BosonSampling(FockBaseTest):
    num_subsystems = 4
    cutoff = 6
    measure_states = [[1,1,0,1], [2,0,0,1]]
    results = [0.174689160486, 0.106441927246]

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)

    def test_fock_probs(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Fock(1) | q[0]
            Fock(1) | q[1]
            Vac     | q[2]
            Fock(1) | q[3]

            # rotation gates
            Rgate(0.5719)  | q[0]
            Rgate(-1.9782) | q[1]
            Rgate(2.0603)  | q[2]
            Rgate(0.0644)  | q[3]

            # beamsplitter array
            BSgate(0.7804, 0.8578)  | (q[0], q[1])
            BSgate(0.06406, 0.5165) | (q[2], q[3])
            BSgate(0.473, 0.1176)   | (q[1], q[2])
            BSgate(0.563, 0.1517)   | (q[0], q[1])
            BSgate(0.1323, 0.9946)  | (q[2], q[3])
            BSgate(0.311, 0.3231)   | (q[1], q[2])
            BSgate(0.4348, 0.0798)  | (q[0], q[1])
            BSgate(0.4368, 0.6157)  | (q[2], q[3])

        state = self.eng.run()
        probs = [state.fock_prob(i) for i in self.measure_states]
        probs = np.array(probs).T.flatten()
        res = np.tile(self.results, self.bsize).flatten()
        self.assertAllAlmostEqual(probs, res, delta=self.tol)


class HamiltonianSimulation(FockBaseTest):
    num_subsystems = 2
    cutoff = 4
    measure_states = [[0,2],[1,1],[2,0]]
    results = [0.52240124572, 0.235652876857, 0.241945877423]
    # set the Hamiltonian parameters
    J = 1           # hopping transition
    U = 1.5         # on-site interaction
    k = 20          # Lie product decomposition terms
    t = 1.086       # timestep
    theta = -J*t/k
    r = -U*t/(2*k)

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)

    def test_fock_probs(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Fock(2) | q[0]

            # Two node tight-binding
            # Hamiltonian simulation

            for i in range(self.k):
                BSgate(self.theta, pi/2) | (q[0], q[1])
                Kgate(self.r)  | q[0]
                Rgate(-self.r) | q[0]
                Kgate(self.r)  | q[1]
                Rgate(-self.r) | q[1]

        state = self.eng.run()
        probs = [state.fock_prob(i) for i in self.measure_states]
        probs = np.array(probs).T.flatten()
        res = np.tile(self.results, self.bsize).flatten()
        self.assertAllAlmostEqual(probs, res, delta=self.tol)

class GaussianCloning(GaussianBaseTest):
    num_subsystems = 4
    shots = 500
    a = 0.7+1.2j

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        self.eng.backend = self.backend
        self.backend.reset()

    def test_average_fidelity(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Coherent(self.a) | q[0]
            BSgate() | (q[0], q[1])
            BSgate()  | (q[1], q[2])
            MeasureX | q[1]
            MeasureP | q[2]
            Xgate(scale(q[1], sqrt(2))) | q[0]
            Zgate(scale(q[2], sqrt(2))) | q[0]
            BSgate() | (q[0], q[3])

        # check outputs are identical clones
        state = self.eng.run(modes=[0,3])
        coh = np.array([state.is_coherent(i) for i in range(2)])
        disp = state.displacement()
        self.assertAllTrue(coh)
        self.assertAllAlmostEqual(*disp, delta=self.tol)

        f = np.empty([self.shots])
        a = np.empty([self.shots], dtype=np.complex128)

        for i in range(self.shots):
            self.eng.reset(keep_history=True)
            state = self.eng.run(modes=[0])
            f[i] = state.fidelity_coherent([0.7+1.2j])
            a[i] = state.displacement()

        self.assertAlmostEqual(np.mean(f), 2./3., delta=0.1)
        self.assertAlmostEqual(np.mean(a), self.a, delta=0.1)
        self.assertAllAlmostEqual(np.cov([a.real, a.imag]), 0.25*np.identity(2), delta=0.1)


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', examples.')

    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [
        Teleportation,
        GateTeleportation,
        GaussianBosonSampling,
        BosonSampling,
        HamiltonianSimulation,
        GaussianCloning
    ]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

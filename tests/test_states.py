"""
Unit tests for :class:`strawberryfields.backends.states`.
"""

import unittest

import numpy as np
from numpy import pi
from scipy.special import factorial
from scipy.stats import multivariate_normal

from defaults import BaseTest, FockBaseTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import *
from strawberryfields import backends
from strawberryfields.backends.shared_ops import rotation_matrix as R


a = 0.3+0.1j
r = 0.23
phi = 0.123


def wigner(grid, mu, cov):
    mvn = multivariate_normal(mu, cov, allow_singular=True)
    return mvn.pdf(grid)


class BackendStateCreation(BaseTest):
    num_subsystems = 3

    def test_full_state_creation(self):
        self.logTestName()
        state = self.circuit.state(modes=None)
        self.assertEqual(state.num_modes, 3)
        self.assertEqual(state.hbar, self.hbar)
        self.assertEqual(state.mode_names, {0: 'q[0]', 1: 'q[1]', 2: 'q[2]'})
        self.assertEqual(state.mode_indices, {'q[0]': 0, 'q[1]': 1, 'q[2]': 2})

        if isinstance(self.backend, backends.BaseFock):
            print(self.D)
            self.assertEqual(state.cutoff_dim, self.D)

    def test_reduced_state_creation(self):
        self.logTestName()
        state = self.circuit.state(modes=[0, 2])
        self.assertEqual(state.num_modes, 2)
        self.assertEqual(state.mode_names, {0: 'q[0]', 1: 'q[2]'})
        self.assertEqual(state.mode_indices, {'q[0]': 0, 'q[2]': 1})

    def test_reduced_state_fidelity(self):
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_coherent_state(a, 0)
        self.circuit.prepare_squeezed_state(r, phi, 1)

        state = self.circuit.state(modes=[0])
        f = state.fidelity_coherent([a])
        self.assertAllAlmostEqual(f, 1, delta=self.tol)

    def test_reduced_state_fock_probs(self):
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_coherent_state(a, 0)
        self.circuit.prepare_squeezed_state(r, phi, 1)

        state = self.circuit.state(modes=[0])
        probs = np.array([state.fock_prob([i]) for i in range(self.D)]).T

        ref_state = np.array([np.exp(-0.5*np.abs(a)**2)*a**n/np.sqrt(factorial(n)) for n in range(self.D)])
        ref_probs = np.tile(np.abs(ref_state) ** 2, self.bsize)

        self.assertAllAlmostEqual(probs.flatten(), ref_probs.flatten(), delta=self.tol)


class FrontendStateCreation(BaseTest):
    num_subsystems = 3

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend
        self.eng.reset()

    def test_full_state_creation(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Coherent(a) | q[0]
            Squeezed(r, phi) | q[1]

        state = self.eng.run()
        self.assertEqual(state.num_modes, 3)
        self.assertEqual(state.hbar, self.hbar)
        self.assertEqual(state.mode_names, {0: 'q[0]', 1: 'q[1]', 2: 'q[2]'})
        self.assertEqual(state.mode_indices, {'q[0]': 0, 'q[1]': 1, 'q[2]': 2})


class BaseStateMethods(BaseTest):
    num_subsystems = 2

    def test_mean_photon(self):
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_coherent_state(a, 0)
        state = self.circuit.state()
        mean_photon = state.mean_photon(0)
        self.assertAllAlmostEqual(mean_photon, np.abs(a)**2, delta=self.tol)

    def test_rdm(self):
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_coherent_state(a, 0)
        self.circuit.prepare_coherent_state(0.1, 1)

        state = self.circuit.state()
        rdm = state.reduced_dm(0, cutoff=self.D)

        ket = coherent_state(a, fock_dim=self.D)
        rdm_exact = np.outer(ket, ket.conj())

        if self.batched:
            np.tile(rdm_exact, [self.bsize, 1])

        self.assertAllAlmostEqual(rdm, rdm_exact, delta=self.tol)


class BaseFockStateMethods(FockBaseTest):
    num_subsystems = 2

    def test_ket(self):
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.displacement(a, 0)

        state = self.circuit.state()
        if self.args.mixed:
            self.assertEqual(state.is_pure, False)
            return
        else:
            self.assertEqual(state.is_pure, True)

        ket = state.ket()
        ket0exact = coherent_state(a, fock_dim=self.D)

        ket0 = np.sum(ket,axis=-1)
        if self.batched:
            np.tile(ket0, [self.bsize, 1])

        self.assertAllAlmostEqual(ket0, ket0exact, delta=self.tol)


class BaseGaussianStateMethods(GaussianBaseTest):
    num_subsystems = 2
    # can push values higher in Gaussian backend
    a = 1+0.5j
    r = 2
    phi = -0.5

    def test_coherent_methods(self):
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_coherent_state(self.a, 0)
        self.circuit.prepare_squeezed_state(self.r, self.phi, 1)

        state = self.circuit.state()

        coherent_check = []
        for i in range(2):
            coherent_check.append(state.is_coherent(i))

        alpha_list = state.displacement()

        self.assertAllEqual(coherent_check, [True, False])
        self.assertAllAlmostEqual(alpha_list, [self.a,  0.], delta=self.tol)

    def test_squeezing_methods(self):
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_coherent_state(self.a, 0)
        self.circuit.prepare_squeezed_state(self.r, self.phi, 1)

        state = self.circuit.state()

        squeezing_check = []
        for i in range(2):
            squeezing_check.append(state.is_squeezed(i))

        z_list = np.array(state.squeezing())

        self.assertAllEqual(squeezing_check, [False, True])
        self.assertAllAlmostEqual(z_list, [[0.,0.],  [self.r, self.phi]], delta=self.tol)


class QuadExpectation(BaseTest):
    num_subsystems = 1

    def test_vacuum(self):
        self.logTestName()
        state = self.circuit.state()
        res = np.array(state.quad_expectation(0, phi=pi/4)).T
        res_exact = np.tile(np.array([0, self.hbar/2.]), self.bsize)
        self.assertAllAlmostEqual(res.flatten(), res_exact.flatten(), delta=self.tol)

    def test_squeezed_coherent(self):
        self.logTestName()
        qphi = 0.78

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        res = np.array(state.quad_expectation(0, phi=qphi)).T

        xphi_mean = (a.real*np.cos(qphi) + a.imag*np.sin(qphi)) * np.sqrt(2*self.hbar)
        xphi_var = (np.cosh(2*r) - np.cos(phi-2*qphi)*np.sinh(2*r)) * self.hbar/2
        res_exact = np.tile(np.array([xphi_mean, xphi_var]), self.bsize)

        self.assertAllAlmostEqual(res.flatten(), res_exact.flatten(), delta=self.tol)


class WignerSingleMode(BaseTest):
    num_subsystems = 1
    batched = False
    bsize = 1

    # wigner parameters
    xmin = -5
    xmax = 5
    dx = 0.1
    xvec = np.arange(xmin, xmax, dx)

    X, P = np.meshgrid(xvec, xvec)
    grid = np.empty(X.shape + (2,))
    grid[:, :, 0] = X
    grid[:, :, 1] = P

    def test_vacuum(self):
        self.logTestName()
        if self.batched:
            return

        state = self.circuit.state()
        W = state.wigner(0, self.xvec, self.xvec)

        # exact wigner function
        mu = [0,0]
        cov = np.identity(2)*self.hbar/2.
        Wexact = wigner(self.grid, mu, cov)

        self.assertAllAlmostEqual(W, Wexact, delta=self.tol)

    def test_squeezed_coherent(self):
        self.logTestName()
        if self.batched:
            return

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_coherent_state(a, 0)
        self.circuit.squeeze(r*np.exp(1j*phi), 0)

        state = self.circuit.state()
        W = state.wigner(0, self.xvec, self.xvec)
        rot = R(phi/2)

        # exact wigner function
        alpha = a*np.cosh(r) - np.conjugate(a)*np.exp(1j*phi)*np.sinh(r)
        mu = np.array([alpha.real, alpha.imag])*np.sqrt(2*self.hbar)
        cov = np.diag([np.exp(-2*r), np.exp(2*r)])
        cov = np.dot(rot, np.dot(cov, rot.T))*self.hbar/2.
        Wexact = wigner(self.grid, mu, cov)

        self.assertAllAlmostEqual(W, Wexact, delta=self.tol)


class WignerTwoMode(BaseTest):
    num_subsystems = 2

    # wigner parameters
    xmin = -5
    xmax = 5
    dx = 0.1
    xvec = np.arange(xmin, xmax, dx)

    X, P = np.meshgrid(xvec, xvec)
    grid = np.empty(X.shape + (2,))
    grid[:, :, 0] = X
    grid[:, :, 1] = P

    def test_two_mode_squeezed(self):
        self.logTestName()
        if self.batched:
            return

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_squeezed_state(r, 0, 0)
        self.circuit.prepare_squeezed_state(-r, 0, 1)

        state = self.circuit.state()
        W0 = state.wigner(0, self.xvec, self.xvec)
        W1 = state.wigner(1, self.xvec, self.xvec)

        # exact wigner function
        mu = np.array([0, 0])*np.sqrt(2*self.hbar)
        cov = np.diag([np.exp(-2*r), np.exp(2*r)])*self.hbar/2
        W0exact = wigner(self.grid, mu, cov)

        cov = np.diag([np.exp(2*r), np.exp(-2*r)])*self.hbar/2
        W1exact = wigner(self.grid, mu, cov)

        self.assertAllAlmostEqual(W0, W0exact, delta=self.tol)
        self.assertAllAlmostEqual(W1, W1exact, delta=self.tol)


class InitialStateFidelityTests(BaseTest):
    """Fidelity tests."""
    num_subsystems = 2

    def test_vacuum(self):
        self.logTestName()
        state = self.circuit.state()
        self.assertAllAlmostEqual(state.fidelity_vacuum(), 1, delta=self.tol)

    def test_coherent_fidelity(self):
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_coherent_state(a, 0)
        self.circuit.displacement(a, 1)
        state = self.circuit.state()

        if isinstance(self.backend, backends.BaseFock):
            in_state = coherent_state(a, basis='fock', fock_dim=self.D)
        else:
            in_state = coherent_state(a, basis='gaussian')

        self.assertAllAlmostEqual(state.fidelity(in_state, 0), 1, delta=self.tol)
        self.assertAllAlmostEqual(state.fidelity(in_state, 1), 1, delta=self.tol)
        self.assertAllAlmostEqual(state.fidelity_coherent([a,a]), 1, delta=self.tol)

    def test_squeezed_fidelity(self):
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_squeezed_state(r, phi, 0)
        self.circuit.squeeze(r*np.exp(1j*phi), 1)
        state = self.circuit.state()

        if isinstance(self.backend, backends.BaseFock):
            in_state = squeezed_state(r, phi, basis='fock', fock_dim=self.D)
        else:
            in_state = squeezed_state(r, phi, basis='gaussian')

        self.assertAllAlmostEqual(state.fidelity(in_state, 0), 1, delta=self.tol)
        self.assertAllAlmostEqual(state.fidelity(in_state, 1), 1, delta=self.tol)

    def test_squeezed_coherent_fidelity(self):
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)
        self.circuit.squeeze(r*np.exp(1j*phi), 1)
        self.circuit.displacement(a, 1)
        state = self.circuit.state()

        if isinstance(self.backend, backends.BaseFock):
            in_state = displaced_squeezed_state(a, r, phi, basis='fock', fock_dim=self.D)
        else:
            in_state = displaced_squeezed_state(a, r, phi, basis='gaussian')

        self.assertAllAlmostEqual(state.fidelity(in_state, 0), 1, delta=self.tol)
        self.assertAllAlmostEqual(state.fidelity(in_state, 1), 1, delta=self.tol)


mag_alphas = np.linspace(0, .8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)


class FockProbabilities(BaseTest):
    num_subsystems = 1
    def test_prob_fock_gaussian(self):
        self.logTestName()
        """Tests that probabilities of particular Fock states |n> are correct for a gaussian state."""
        for mag_alpha in mag_alphas:
            for phase_alpha in phase_alphas:
                self.circuit.reset(pure=self.kwargs['pure'])

                alpha = mag_alpha * np.exp(1j * phase_alpha)
                ref_state = np.array([np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n)) for n in range(self.D)])
                ref_probs = np.abs(ref_state) ** 2

                self.circuit.prepare_coherent_state(alpha, 0)
                state = self.circuit.state()

                for n in range(self.D):
                  prob_n = state.fock_prob([n])
                  self.assertAllAlmostEqual(prob_n, ref_probs[n], delta=self.tol)


class AllFockProbsSingleMode(FockBaseTest):
    num_subsystems = 1

    def test_all_fock_probs_pure(self):
        self.logTestName()
        """Tests that the numeric probabilities in the full Fock basis are correct for a one-mode pure state."""
        for mag_alpha in mag_alphas:
            for phase_alpha in phase_alphas:
                self.circuit.reset(pure=True)

                alpha = mag_alpha * np.exp(1j * phase_alpha)
                ref_state = np.array([np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n)) for n in range(self.D)])
                ref_probs = np.abs(ref_state) ** 2

                self.circuit.prepare_coherent_state(alpha, 0)
                state = self.circuit.state()

                probs = state.all_fock_probs().flatten()
                ref_probs = np.tile(ref_probs, self.bsize)
                self.assertAllAlmostEqual(probs, ref_probs, delta=self.tol)


class AllFockProbsTwoMode(FockBaseTest):
    num_subsystems = 2

    def test_prob_fock_state_nongaussian(self):
        self.logTestName()
        """Tests that probabilities of particular Fock states |n> are correct for a nongaussian state."""
        for mag_alpha in mag_alphas:
            for phase_alpha in phase_alphas:
                self.circuit.reset(pure=self.kwargs['pure'])

                alpha = mag_alpha * np.exp(1j * phase_alpha)
                ref_state = np.array([np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n)) for n in range(self.D)])
                ref_probs = np.abs(ref_state) ** 2

                self.circuit.prepare_coherent_state(alpha, 0)
                self.circuit.prepare_fock_state(self.D // 2, 1)
                state = self.circuit.state()

                for n in range(self.D):
                    prob_n = state.fock_prob([n, self.D // 2])
                    self.assertAllAlmostEqual(prob_n, ref_probs[n], delta=self.tol)

    def test_all_fock_state_probs(self):
        self.logTestName()
        """Tests that the numeric probabilities in the full Fock basis are correct for a two-mode gaussian state."""
        for mag_alpha in mag_alphas:
            for phase_alpha in phase_alphas:
                self.circuit.reset(pure=self.kwargs['pure'])

                alpha = mag_alpha * np.exp(1j * phase_alpha)
                ref_state1 = np.array([np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n)) for n in range(self.D)])
                ref_state2 = np.array([np.exp(-0.5 * np.abs(-alpha) ** 2) * (-alpha) ** n / np.sqrt(factorial(n)) for n in range(self.D)])
                ref_state = np.outer(ref_state1, ref_state2)
                ref_probs = np.abs(np.reshape(ref_state ** 2, -1))
                ref_probs = np.tile(ref_probs, self.bsize)

                self.circuit.prepare_coherent_state(alpha, 0)
                self.circuit.prepare_coherent_state(-alpha, 1)
                state = self.circuit.state()

                for n in range(self.D):
                    for m in range(self.D):
                        probs = state.all_fock_probs().flatten()
                        self.assertAllAlmostEqual(probs, ref_probs, delta=self.tol)

                if mag_alpha == 0.:
                    pass


if __name__ == '__main__':
    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [
        BackendStateCreation,
        FrontendStateCreation,
        BaseStateMethods,
        BaseFockStateMethods,
        BaseGaussianStateMethods,
        QuadExpectation,
        WignerSingleMode,
        WignerTwoMode,
        InitialStateFidelityTests,
        # FockProbabilities,
        # AllFockProbsSingleMode,
        # AllFockProbsTwoMode
    ]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

"""
Unit tests for :class:`strawberryfields.backends.states`.
"""

import unittest
import logging

import numpy as np
from numpy import pi
from scipy.special import factorial
from scipy.stats import multivariate_normal
from scipy.integrate import simps
from scipy.linalg import block_diag

from defaults import BaseTest, FockBaseTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import *
from strawberryfields import backends
from strawberryfields.backends.shared_ops import rotation_matrix as R, changebasis


a = 0.3+0.1j
r = 0.23
phi = 0.123

mag_alphas = np.linspace(0, .8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)

def wigner(grid, mu, cov):
    mvn = multivariate_normal(mu, cov, allow_singular=True)
    return mvn.pdf(grid)


class BackendStateCreationTests(BaseTest):
    num_subsystems = 3

    def test_full_state_creation(self):
        self.logTestName()
        state = self.circuit.state(modes=None)
        self.assertEqual(state.num_modes, 3)
        self.assertEqual(state.hbar, self.hbar)
        self.assertEqual(state.mode_names, {0: 'q[0]', 1: 'q[1]', 2: 'q[2]'})
        self.assertEqual(state.mode_indices, {'q[0]': 0, 'q[1]': 1, 'q[2]': 2})

        if isinstance(self.backend, backends.BaseFock):
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


class FrontendStateCreationTests(BaseTest):
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


class BaseStateMethodsTests(BaseTest):
    num_subsystems = 2

    def test_mean_photon_coherent(self):
        """Test that E(n) = |a|^2 and var(n) = |a|^2 for a coherent state"""
        self.logTestName()
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.displacement(a, 0)
        state = self.circuit.state()
        mean_photon, var = state.mean_photon(0)
        self.assertAllAlmostEqual(mean_photon, np.abs(a)**2, delta=self.tol)
        self.assertAllAlmostEqual(var, np.abs(a)**2, delta=self.tol)

    def test_mean_photon_squeezed(self):
        """Test that E(n)=sinh^2(r) and var(n)=2(sinh^2(r)+sinh^4(r)) for a squeezed state"""
        self.logTestName()

        r = 0.1
        a = 0.3+0.1j

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.squeeze(r*np.exp(1j*phi), 0)
        state = self.circuit.state()
        mean_photon, var = state.mean_photon(0)
        self.assertAllAlmostEqual(mean_photon, np.sinh(r)**2, delta=self.tol)
        self.assertAllAlmostEqual(var, 2*(np.sinh(r)**2+np.sinh(r)**4), delta=self.tol)

    def test_mean_photon_displaced_squeezed(self):
        """Test that E(n) = sinh^2(r)+|a|^2 for a displaced squeezed state"""
        self.logTestName()

        nbar = 0.123
        a = 0.12-0.05j
        r = 0.195

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.squeeze(r*np.exp(1j*phi), 0)
        self.circuit.displacement(a, 0)
        state = self.circuit.state()
        mean_photon, var = state.mean_photon(0)

        mag_a = np.abs(a)
        phi_a = np.angle(a)

        mean_ex = np.abs(a)**2 + np.sinh(r)**2
        self.assertAllAlmostEqual(mean_photon, mean_ex, delta=self.tol)

    def test_mean_photon_displaced_thermal(self):
        """Test that E(n)=|a|^2+nbar and var(n)=var_th+|a|^2(1+2nbar)"""
        self.logTestName()

        nbar = 0.123

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_thermal_state(nbar, 0)
        self.circuit.displacement(a, 0)
        state = self.circuit.state()
        mean_photon, var = state.mean_photon(0)

        mean_ex = np.abs(a)**2 + nbar
        var_ex = nbar**2 + nbar + np.abs(a)**2*(1+2*nbar)
        self.assertAllAlmostEqual(mean_photon, mean_ex, delta=self.tol)
        self.assertAllAlmostEqual(var, var_ex, delta=self.tol)

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


class BaseFockStateMethodsTests(FockBaseTest):
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


class BaseGaussianStateMethodsTests(GaussianBaseTest):
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


class QuadExpectationTests(BaseTest):
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


class PolyQuadExpectationSingleModeTests(BaseTest):
    num_subsystems = 3
    hbar = 2

    # qudrature rotation
    qphi = 0.78

    # construct the expected vector of means and covariance matrix
    mu = R(qphi).T @ np.array([a.real, a.imag]) * np.sqrt(2*hbar)
    cov = R(qphi).T @ squeezed_cov(r, phi, hbar=hbar) @ R(qphi)

    def sample_one_mode_normal_expectations(self, mu, cov, func, correction=0):
        """Returns the expectation value E(f) and the variance var(f)
        for some normal distribution X~N(mu, cov).

        Args:
            mu (array): means vector
            cov (array): covariance matrix
            func (function): function acting on the random variables X, P, XP,
                returning a second order polynomial

        Returns:
            tuple: tuple of expectation value and variance.
        """
        X, P = np.mgrid[-7:7:.01, -7:7:.01]
        grid = np.dstack((X, P))
        XP = np.prod(grid, axis=2)

        poly = func(X, P, XP)
        PDF = multivariate_normal.pdf(grid, mu, cov)

        Ex = simps(simps(poly * PDF, P[0]), X.T[0])
        ExSq = simps(simps(poly**2 * PDF, P[0]), X.T[0])

        var = ExSq - Ex**2 + correction

        return Ex, var

    def setUp(self):
        """Set up."""
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend

        cutoff = 12
        if self.D < cutoff:
            logging.warning("Provided cutoff value is too small for "
                            "tests in `PolyQuadExpectationSingleModeTests`. "
                            "Cutoff has been increased internally.")
            self.D = cutoff
        self.backend.reset(cutoff_dim=self.D)

        if isinstance(self.backend, backends.TFBackend):
            raise unittest.SkipTest('The poly_quad_expectation method is not yet supported on the TF backend.')

    def test_no_expectation(self):
        """Test the case E(0), var(0)"""
        self.logTestName()
        state = self.circuit.state()

        A = np.zeros([6, 6])
        d = np.zeros([6])
        k = 0

        mean, var = state.poly_quad_expectation(A, d, k)
        self.assertAlmostEqual(mean, 0, delta=self.tol)
        self.assertAlmostEqual(var, 0, delta=self.tol)

    def test_constant(self):
        """Test the case E(k), var(k)"""
        self.logTestName()
        state = self.circuit.state()

        A = np.zeros([6, 6])
        d = np.zeros([6])
        k = 0.543

        mean, var = state.poly_quad_expectation(A, d, k)
        self.assertAlmostEqual(mean, k, delta=self.tol)
        self.assertAlmostEqual(var, 0, delta=self.tol)

    def test_linear_vacuum(self):
        """Test that the correct results are returned for the vacuum state."""
        self.logTestName()

        state = self.circuit.state()

        A = np.zeros([6, 6])
        # create an arbitrary linear combination
        d = np.array([1, 1, 0, 1, 0, 0])
        k = 0

        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)
        self.assertAlmostEqual(mean, 0, delta=self.tol)
        self.assertAlmostEqual(var, len(d)*self.hbar/4, delta=self.tol)

    def test_x_squeezed(self):
        """Test that the correct E(x) is returned for the squeezed coherent state."""
        self.logTestName()

        A = None
        d = np.array([1, 0, 0, 0, 0, 0])
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.squeeze(r, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=0)

        self.assertAlmostEqual(mean, 0, delta=self.tol)
        self.assertAlmostEqual(var, np.exp(-2*r), delta=self.tol)

    def test_x_displaced(self):
        """Test that the correct E(x) is returned for the squeezed coherent state."""
        self.logTestName()

        A = None
        d = np.array([1, 0, 0, 0, 0, 0])
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.displacement(a, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=0)

        self.assertAlmostEqual(mean, a.real*np.sqrt(2*self.hbar), delta=self.tol)
        self.assertAlmostEqual(var, self.hbar/2, delta=self.tol)

    def test_x_squeezed_coherent(self):
        """Test that the correct E(x) is returned for the squeezed coherent state."""
        self.logTestName()

        A = None
        d = np.array([1, 0, 0, 0, 0, 0])
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        self.assertAlmostEqual(mean, self.mu[0], delta=self.tol)
        self.assertAlmostEqual(var, self.cov[0, 0], delta=self.tol)

    def test_p_squeezed_coherent(self):
        """Test that the correct E(p) is returned for the squeezed coherent state."""
        self.logTestName()

        A = None
        d = np.array([0, 0, 0, 1, 0, 0])
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        self.assertAlmostEqual(mean, self.mu[1], delta=self.tol)
        self.assertAlmostEqual(var, self.cov[1, 1], delta=self.tol)

    def test_linear_combination(self):
        """Test that the correct result is returned for E(ax+bp)"""
        self.logTestName()

        A = None
        d = np.array([0.4234, 0, 0, 0.1543, 0, 0])
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        # E(ax+bp) = aE(x) + bE(p)
        mean_expected = d[0]*self.mu[0] + d[3]*self.mu[1]
        self.assertAlmostEqual(mean, mean_expected, delta=self.tol)

        # var(ax+bp) = a**2 var(x)+b**2 var(p)+2ab cov(x,p)
        var_expected = self.cov[0, 0]*d[0]**2 + self.cov[1, 1]*d[3]**2 + 2*d[0]*d[3]*self.cov[0, 1]
        self.assertAlmostEqual(var, var_expected, delta=self.tol)

    def test_n_thermal(self):
        """Test expectation and variance of the number operator on a thermal state"""
        self.logTestName()

        nbar = 0.423

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_thermal_state(nbar, 0)
        state = self.circuit.state()

        # n = a^\dagger a = (X^2 +P^2)/2\hbar - I/2
        A = np.zeros([6, 6])
        A[0, 0] = 1/(2*self.hbar)
        A[3, 3] = 1/(2*self.hbar)
        k = -0.5

        mean, var = state.poly_quad_expectation(A, d=None, k=k, phi=0)
        self.assertAlmostEqual(mean, nbar, delta=self.tol)
        self.assertAlmostEqual(var, nbar*(nbar+1), delta=self.tol)

    def test_n_squeeze(self):
        """Test expectation and variance of the number operator on a squeeze state"""
        self.logTestName()

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_squeezed_state(r, phi, 0)
        state = self.circuit.state()

        # n = a^\dagger a = (X^2 +P^2)/2\hbar - I/2
        A = np.zeros([6, 6])
        A[0, 0] = 1/(2*self.hbar)
        A[3, 3] = 1/(2*self.hbar)
        k = -0.5

        mean, var = state.poly_quad_expectation(A, None, k, phi=self.qphi)

        mean_ex = np.sinh(r)**2
        var_ex = 0.5*np.sinh(2*r)**2

        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)

    def test_x_squared(self):
        """Test that the correct result is returned for E(x^2)"""
        self.logTestName()

        A = np.zeros([6, 6])
        A[0, 0] = 1

        d = None
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        mean_ex, var_ex = self.sample_one_mode_normal_expectations(
            self.mu, self.cov, lambda X, P, XP: A[0, 0]*X**2)
        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)

    def test_p_squared(self):
        """Test that the correct result is returned for E(p^2)"""
        self.logTestName()

        A = np.zeros([6, 6])
        A[3, 3] = 1

        d = None
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        mean_ex, var_ex = self.sample_one_mode_normal_expectations(
            self.mu, self.cov, lambda X, P, XP: P**2)
        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)

    def test_xp_vacuum(self):
        """Test that the correct result is returned for E(xp)"""
        self.logTestName()

        # set quadratic coefficient
        A = np.zeros([6, 6])
        A[3, 0] = A[0, 3] = 0.5

        d = None
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=0)

        mean_ex, var_ex = self.sample_one_mode_normal_expectations(
            np.zeros([2]), np.identity(2), lambda X, P, XP: XP,
            correction=-np.linalg.det(2*A[:, [0, 3]][[0, 3]]))

        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)

    def test_xp(self):
        """Test that the correct result is returned for E(xp)"""
        self.logTestName()

        # set quadratic coefficient
        A = np.zeros([6, 6])
        A[3, 0] = A[0, 3] = 0.5

        d = None
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        mean_ex, var_ex = self.sample_one_mode_normal_expectations(
            self.mu, self.cov, lambda X, P, XP: XP,
            correction=-np.linalg.det(2*A[:, [0, 3]][[0, 3]]))
        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)

    def test_arbitrary_quadratic(self):
        """Test that the correct result is returned for E(c0 x^2 + c1 p^2 + c2 xp + c3 x + c4 p + k)"""
        self.logTestName()

        c0 = 1/np.sqrt(2)
        c1 = 3
        c2 = 0.53
        c3 = 1/3.
        c4 = -1

        # define the arbitrary quadratic
        A = np.zeros([6, 6])
        A[0, 0] = c0
        A[3, 3] = c1

        A[3, 0] = c2/2
        A[0, 3] = c2/2

        # define the linear combination and constant term
        d = np.array([c3, 0, 0, c4, 0, 0])
        k = 5

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        mean_ex, var_ex = self.sample_one_mode_normal_expectations(
            self.mu, self.cov, lambda X, P, XP: c0*X**2 + c1*P**2 + c2*XP + c3*X + c4*P + k,
            correction=-np.linalg.det(2*A[:, [0, 3]][[0, 3]]))

        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)


class PolyQuadExpectationFockTests(FockBaseTest):
    num_subsystems = 1

    def setUp(self):
        """Set up."""
        super().setUp()

        if isinstance(self.backend, backends.TFBackend):
            raise unittest.SkipTest('The poly_quad_expectation method is not yet supported on the TF backend.')

    def test_n_fock_state(self):
        """Test that the correct E(n) and var(n) returned for Fock state |4>"""
        self.logTestName()
        n = 4

        # n = a^\dagger a = (X^2 +P^2)/2\hbar - I/2
        A = np.identity(2)/(2*self.hbar)
        k = -0.5

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_fock_state(n, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, None, k, phi=0)

        self.assertAlmostEqual(mean, n, delta=self.tol)
        self.assertAlmostEqual(var, 0, delta=self.tol)


class PolyQuadExpectationMultiModeTests(BaseTest):
    num_subsystems = 3
    hbar = 2
    qphi = 0.78

    def setUp(self):
        """Set up."""
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend

        cutoff = 7
        if self.D < cutoff:
            logging.warning("Provided cutoff value is too small for "
                            "tests in `PolyQuadExpectationMultiModeTests`. "
                            "Cutoff has been increased internally.")
            self.D = cutoff
        self.backend.reset(cutoff_dim=self.D)

        if isinstance(self.backend, backends.TFBackend):
            raise unittest.SkipTest('The poly_quad_expectation method is not yet supported on the TF backend.')

    def test_three_mode_arbitrary(self):
        """Test that the correct result is returned for an arbitrary quadratic polynomial"""
        self.logTestName()

        A = np.array([[ 0.7086495 , -0.39299695,  0.30536448,  0.48822049,  0.64987373, 0.7020327 ],
                      [-0.39299695,  0.0284145 ,  0.53202656,  0.20232385,  0.26288656, 0.20772833],
                      [ 0.30536448,  0.53202656,  0.28126466,  0.64192545, -0.36583748, 0.51704656],
                      [ 0.48822049,  0.20232385,  0.64192545,  0.51033017,  0.29129713, 0.77103581],
                      [ 0.64987373,  0.26288656, -0.36583748,  0.29129713,  0.37646972, 0.2383589 ],
                      [ 0.7020327 ,  0.20772833,  0.51704656,  0.77103581,  0.2383589 ,-0.96494418]])

        d = np.array([ 0.71785224,  -0.80064627,  0.08799823,  0.76189805,  0.99665321, -0.60777437])
        k = 0.123

        self.circuit.reset(pure=self.kwargs['pure'])

        a_list = [0.044+0.023j, 0.0432+0.123j, -0.12+0.04j]
        r_list = [0.1065, 0.032, -0.123]
        phi_list = [0.897, 0.31, 0.432]

        mu = np.zeros([6])
        cov = np.zeros([6, 6])

        # squeeze and displace each mode
        for i, (a_, r_, phi_) in enumerate(zip(a_list, r_list, phi_list)):
            self.circuit.prepare_displaced_squeezed_state(a_, r_, phi_, i)
            mu[2*i:2*i+2] = R(self.qphi).T @ np.array([a_.real, a_.imag]) * np.sqrt(2*self.hbar)
            cov[2*i:2*i+2, 2*i:2*i+2] = R(self.qphi).T @ squeezed_cov(r_, phi_, hbar=self.hbar) @ R(self.qphi)

        # apply a beamsplitter to the modes
        self.circuit.beamsplitter(1/np.sqrt(2), 1/np.sqrt(2), 0, 1)
        self.circuit.beamsplitter(1/np.sqrt(2), 1/np.sqrt(2), 1, 2)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        # apply a beamsplitter to vector of means and covariance matrices
        t = 1/np.sqrt(2)
        BS = np.array([[ t,  0,  -t,  0],
                       [ 0,  t,  0,  -t],
                       [ t,  0,  t,  0],
                       [ 0,  t,  0,  t]])

        S1 = block_diag(BS, np.identity(2))
        S2 = block_diag(np.identity(2), BS)

        C = changebasis(3)
        mu = C.T @ S2 @ S1 @ mu
        cov = C.T @ S2 @ S1 @ cov @ S1.T @ S2.T @ C

        modes = list(np.arange(6).reshape(2,-1).T)

        mean_ex = np.trace(A @ cov) + mu @ A @ mu + mu @ d + k
        var_ex = 2*np.trace(A @ cov @ A @ cov) + 4*mu.T @ A.T @ cov @ A @ mu + d.T @ cov @ d \
            + 2*mu.T @ A.T @ cov @ d + 2*d.T @ cov @ A @ mu \
            - np.sum([np.linalg.det(self.hbar*A[:, m][n]) for m in modes for n in modes])

        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)


class WignerSingleModeTests(BaseTest):
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


class WignerTwoModeTests(BaseTest):
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


class FockProbabilitiesTests(BaseTest):
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


class AllFockProbsSingleModeTests(FockBaseTest):
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


class AllFockProbsTwoModeTests(FockBaseTest):
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
        BackendStateCreationTests,
        FrontendStateCreationTests,
        BaseStateMethodsTests,
        BaseFockStateMethodsTests,
        BaseGaussianStateMethodsTests,
        QuadExpectationTests,
        PolyQuadExpectationSingleModeTests,
        PolyQuadExpectationFockTests,
        PolyQuadExpectationMultiModeTests,
        WignerSingleModeTests,
        WignerTwoModeTests,
        InitialStateFidelityTests,
        FockProbabilitiesTests,
        AllFockProbsSingleModeTests,
        AllFockProbsTwoModeTests
    ]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

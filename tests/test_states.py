"""
Unit tests for :class:`strawberryfields.backends.states`.
"""

import unittest

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


class PolyQuadExpectationSingleMode(BaseTest):
    num_subsystems = 3
    cutoff = 12
    hbar = 2

    # qudrature rotation
    qphi = 0#.432

    # construct the expected vector of means and covariance matrix
    mu = R(qphi).T @ np.array([a.real, a.imag]) * np.sqrt(2*hbar)
    cov = R(qphi).T @ squeezed_cov(r, phi, hbar=hbar) @ R(qphi)

    def sample_one_mode_normal_expectations(self, mu, cov, func):
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

        var = ExSq - Ex**2

        return Ex, var

    def setUp(self):
        """Set up."""
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)

    def test_no_expectation(self):
        """Test the case E(0), var(0)"""
        self.logTestName()
        state = self.circuit.state()

        A = np.zeros([6, 6])
        d = np.zeros([6])
        k = 0

        mean, var = state.poly_quad_expectation(A, d, k)
        self.assertEqual(mean, 0)
        self.assertEqual(var, 0)

    def test_constant(self):
        """Test the case E(k), var(k)"""
        self.logTestName()
        state = self.circuit.state()

        A = np.zeros([6, 6])
        d = np.zeros([6])
        k = 0.543

        mean, var = state.poly_quad_expectation(A, d, k)
        self.assertEqual(mean, k)
        self.assertEqual(var, 0)

    def test_linear_vacuum(self):
        """Test that the correct results are returned for the vacuum state."""
        self.logTestName()

        state = self.circuit.state()

        A = np.zeros([6, 6])
        # create an arbitrary linear combination
        d = np.array([1, 1, 0, 1, 0, 0])
        k = 0

        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)
        self.assertEqual(mean, 0)
        self.assertEqual(var, len(d)*self.hbar/4)

    def test_x_squeezed_coherent(self):
        """Test that the correct E(x) is returned for the squeezed coherent state."""
        self.logTestName()

        A = np.zeros([6, 6])
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

        A = np.zeros([6, 6])
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
        """Test that the correct result is returned for E(2x+p)"""
        self.logTestName()

        A = np.zeros([6, 6])
        d = np.array([2, 0, 0, 1, 0, 0])
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        # E(2x+p) = 2E(x) + E(p)
        mean_expected = d[0]*self.mu[0] + d[3]*self.mu[1]
        self.assertAlmostEqual(mean, mean_expected, delta=self.tol)

        # var(2x+p) = 4var(x)+var(p)+4cov(x,p)
        var_expected = self.cov[0, 0]*d[0]**2 + self.cov[1, 1]*d[3]**2 + 2*d[0]*d[3]*self.cov[0, 1]
        self.assertAlmostEqual(var, var_expected, delta=self.tol)

    def test_x_squared(self):
        """Test that the correct result is returned for E(x^2)"""
        self.logTestName()

        A = np.zeros([6, 6])
        A[0, 0] = 2

        d = np.zeros([6])
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        mean_ex, var_ex = self.sample_one_mode_normal_expectations(
            self.mu, self.cov, lambda X, P, XP: A[0, 0]*X**2/2)
        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)

    def test_p_squared(self):
        """Test that the correct result is returned for E(p^2)"""
        self.logTestName()

        A = np.zeros([6, 6])
        A[3, 3] = 2

        d = np.zeros([6])
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        mean_ex, var_ex = self.sample_one_mode_normal_expectations(
            self.mu, self.cov, lambda X, P, XP: A[3, 3]*P**2/2)
        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)

    def test_xp(self):
        """Test that the correct result is returned for E(xp)"""
        self.logTestName()

        # set quadratic coefficient
        A = np.zeros([6, 6])
        A[3, 0] = A[0, 3] = 1

        d = np.zeros([6])
        k = 0

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        mean_ex, var_ex = self.sample_one_mode_normal_expectations(
            self.mu, self.cov, lambda X, P, XP: A[3, 0]*XP)
        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)

    def test_arbitrary_quadratic(self):
        """Test that the correct result is returned for E(c0 x^2 + c1 p^2 + c2 xp + c3 x + c4 p + k)"""
        self.logTestName()

        c0 = 1/np.sqrt(2)
        c1 = 3
        c2 = 0.53
        c3 = 2
        c4 = -1

        # define the arbitrary quadratic
        A = np.zeros([6, 6])
        A[0, 0] = 2*c0
        A[3, 3] = 2*c1

        A[3, 0] = c2
        A[0, 3] = c2

        # define the linear combination and constant term
        d = np.array([c3, 0, 0, c4, 0, 0])
        k = 5

        # prepare a squeezed displaced state
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        mean_ex, var_ex = self.sample_one_mode_normal_expectations(
            self.mu, self.cov, lambda X, P, XP: c0*X**2 + c1*P**2 + c2*XP + c3*X + c4*P + k)

        self.assertAlmostEqual(mean, mean_ex, delta=self.tol)
        self.assertAlmostEqual(var, var_ex, delta=self.tol)


class PolyQuadExpectationMultiMode(BaseTest):
    num_subsystems = 3
    cutoff = 12
    hbar = 2
    qphi = 0.

    # define circuit parameters
    a_list = [0.344, 0.432+0.123j]
    r_list = [0.1065, 0.32]
    phi_list = [0.897, 0.31]

    def setUp(self):
        """Set up."""
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)

        mu = np.zeros([4])
        cov = np.zeros([4, 4])

        # construct the vector of means and covariance matrix for each mode
        # note that we use (x0,p0,x1,p1,...) ordering here.
        for i, (x, y, z) in enumerate(zip(self.a_list, self.r_list, self.phi_list)):
            mu[2*i:2*i+2] = R(self.qphi).T @ np.array([x.real, x.imag]) * np.sqrt(2*self.hbar)
            cov[2*i:2*i+2, 2*i:2*i+2] = R(self.qphi).T @ squeezed_cov(y, z, hbar=self.hbar) @ R(self.qphi)

        # apply a beamsplitter to vector of means and covariance matrices
        t = 1/np.sqrt(2)
        BS = np.array([[ t,  0,  -t,  0],
                       [ 0,  t,  0,  -t],
                       [ t,  0,  t,  0],
                       [ 0,  t,  0,  t]])

        self.mu = BS @ mu
        self.cov = BS @ block_diag(cov) @ BS.T

    def test_multi_mode_linear_combination(self):
        """Test that the correct result is returned for E(2x0 + x1 - p0)"""
        self.logTestName()

        # define the linear combination
        A = np.zeros([6, 6])
        d = np.array([2, 1, 0, -1, 0, 0])
        k = 0

        self.circuit.reset(pure=self.kwargs['pure'])

        # squeeze and displace each mode
        for i, (x, y, z) in enumerate(zip(self.a_list, self.r_list, self.phi_list)):
            self.circuit.prepare_displaced_squeezed_state(x, y, z, i)

        # apply a beamsplitter to the modes
        self.circuit.beamsplitter(1/np.sqrt(2), 1/np.sqrt(2), 0, 1)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        # E(2x0 + x1 - p0) = 2E(x0) + E(x1) - E(p0)
        mean_expected = d[0]*self.mu[0] + d[1]*self.mu[2] + d[3]*self.mu[1]
        self.assertAlmostEqual(mean, mean_expected, delta=self.tol)

        # var(2x0 + x1 - p0) = 4var(x0)+var(x1)+var(p0)-4cov(x0,p0) + 4cov(x0, x1) - 2*cov(x1, p0)
        var_expected = 4*self.cov[0, 0] + self.cov[2, 2] + self.cov[1, 1] \
            - 4*self.cov[0, 1] + 4*self.cov[0, 2] - 2*self.cov[1, 2]
        self.assertAlmostEqual(var, var_expected, delta=self.tol)

    def test_multi_mode_quadratic_combination(self):
        """Test that the correct result is returned for E(x0x1+x0p1)"""
        self.logTestName()

        A = np.zeros([6, 6])
        A[0, 1] = 1
        A[1, 0] = 1
        A[0, 4] = 1
        A[4, 0] = 1

        d = np.zeros([6])
        k = 0

        self.circuit.reset(pure=self.kwargs['pure'])

        # squeeze and displace each mode
        for i, (x, y, z) in enumerate(zip(self.a_list, self.r_list, self.phi_list)):
            self.circuit.prepare_displaced_squeezed_state(x, y, z, i)

        # apply a beamsplitter to the modes
        self.circuit.beamsplitter(1/np.sqrt(2), 1/np.sqrt(2), 0, 1)

        state = self.circuit.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=self.qphi)

        # E(x0x1+x0p1) = cov(x0,x1)+E(x0)E(X1) + cov(x0,p1) +E(x0)E(p1)
        mean_expected = self.cov[0, 2] + self.mu[0]*self.mu[2] + self.cov[0, 3] + self.mu[0]*self.mu[3]
        self.assertAlmostEqual(mean, mean_expected, delta=self.tol)

        # var(2x0 + x1 - p0) = 4var(x0)+var(x1)+var(p0)-4cov(x,p)
        # var_expected = 4*cov[0, 0, 0] + cov[1, 0, 0] + cov[0, 1, 1] - 4*cov[0, 0, 1]
        # self.assertAlmostEqual(var, var_expected, delta=self.tol)


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
        PolyQuadExpectationSingleMode,
        PolyQuadExpectationMultiMode,
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

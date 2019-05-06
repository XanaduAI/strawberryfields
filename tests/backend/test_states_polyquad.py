# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Unit tests for the poly_quad_expectations method in the states.py submodule"""
import pytest

import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import simps
from scipy.linalg import block_diag

from strawberryfields import backends
from strawberryfields import utils
from strawberryfields.backends.shared_ops import rotation_matrix as R, changebasis

# some tests require a higher cutoff for accuracy
CUTOFF = 12

a = 0.3 + 0.1j
r = 0.23
phi = 0.123
qphi = 0.78


@pytest.mark.backends("fock", "gaussian")
class TestSingleModePolyQuadratureExpectations:
    """Test single mode poly_quad_expectation methods"""

    @pytest.fixture
    def gaussian_state(self, hbar):
        """A test Gaussian state to use in testing"""
        # quadrature rotation

        # construct the expected vector of means and covariance matrix
        mu = R(qphi).T @ np.array([a.real, a.imag]) * np.sqrt(2 * hbar)
        cov = R(qphi).T @ utils.squeezed_cov(r, phi, hbar=hbar) @ R(qphi)

        return mu, cov

    @pytest.fixture
    def sample_normal_expectations(self, gaussian_state):
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

        def _sample(func, correction=0, mu=None, cov=None):
            """wrapped function"""
            if mu is None:
                mu = gaussian_state[0]

            if cov is None:
                cov = gaussian_state[1]

            X, P = np.mgrid[-7:7:0.01, -7:7:0.01]
            grid = np.dstack((X, P))
            XP = np.prod(grid, axis=2)

            poly = func(X, P, XP)
            PDF = multivariate_normal.pdf(grid, mu, cov)

            Ex = simps(simps(poly * PDF, P[0]), X.T[0])
            ExSq = simps(simps(poly ** 2 * PDF, P[0]), X.T[0])

            var = ExSq - Ex ** 2 + correction

            return Ex, var

        return _sample

    def test_no_expectation(self, setup_backend, tol):
        """Test the case E(0), var(0)"""
        backend = setup_backend(3)
        state = backend.state()

        A = np.zeros([6, 6])
        d = np.zeros([6])
        k = 0

        mean, var = state.poly_quad_expectation(A, d, k)
        assert np.allclose(mean, 0, atol=tol, rtol=0)
        assert np.allclose(var, 0, atol=tol, rtol=0)

    def test_constant(self, setup_backend, tol):
        """Test the case E(k), var(k)"""
        backend = setup_backend(3)
        state = backend.state()

        A = np.zeros([6, 6])
        d = np.zeros([6])
        k = 0.543

        mean, var = state.poly_quad_expectation(A, d, k)
        assert np.allclose(mean, k, atol=tol, rtol=0)
        assert np.allclose(var, 0, atol=tol, rtol=0)

    def test_linear_vacuum(self, setup_backend, tol, hbar):
        """Test that the correct results are returned for the vacuum state."""
        backend = setup_backend(3)
        state = backend.state()

        A = np.zeros([6, 6])
        # create an arbitrary linear combination
        d = np.array([1, 1, 0, 1, 0, 0])
        k = 0

        mean, var = state.poly_quad_expectation(A, d, k, phi=qphi)
        assert np.allclose(mean, 0, atol=tol, rtol=0)
        assert np.allclose(var, len(d) * hbar / 4, atol=tol, rtol=0)

    def test_x_squeezed(self, setup_backend, tol, pure, hbar):
        """Test that the correct E(x) is returned for the squeezed state."""
        backend = setup_backend(3)
        backend.reset(cutoff_dim=CUTOFF, pure=pure)

        A = None
        d = np.array([1, 0, 0, 0, 0, 0])
        k = 0

        # prepare a squeezed state
        backend.squeeze(r, 0)

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=0)

        assert np.allclose(mean, 0, atol=tol, rtol=0)
        assert np.allclose(var, np.exp(-2 * r)*hbar/2, atol=tol, rtol=0)

    def test_x_displaced(self, setup_backend, tol, hbar):
        """Test that the correct E(x) is returned for a displaced state."""
        backend = setup_backend(3)

        A = None
        d = np.array([1, 0, 0, 0, 0, 0])
        k = 0

        # prepare a displaced state
        backend.displacement(a, 0)

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=0)

        assert np.allclose(mean, a.real * np.sqrt(2 * hbar), atol=tol, rtol=0)
        assert np.allclose(var, hbar / 2, atol=tol, rtol=0)

    def test_x_displaced_squeezed(self, setup_backend, tol, gaussian_state):
        """Test that the correct E(x) is returned for the displaced squeezed state."""
        backend = setup_backend(3)
        mu, cov = gaussian_state

        A = None
        d = np.array([1, 0, 0, 0, 0, 0])
        k = 0

        # prepare a displaced squeezed state
        backend.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=qphi)

        assert np.allclose(mean, mu[0], atol=tol, rtol=0)
        assert np.allclose(var, cov[0, 0], atol=tol, rtol=0)

    def test_p_displaced_squeezed(self, setup_backend, tol, gaussian_state):
        """Test that the correct E(p) is returned for the displaced squeezed state."""
        backend = setup_backend(3)
        mu, cov = gaussian_state

        A = None
        d = np.array([0, 0, 0, 1, 0, 0])
        k = 0

        # prepare a displaced squeezed state
        backend.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=qphi)

        assert np.allclose(mean, mu[1], atol=tol, rtol=0)
        assert np.allclose(var, cov[1, 1], atol=tol, rtol=0)

    def test_linear_combination(self, setup_backend, tol, gaussian_state):
        """Test that the correct result is returned for E(ax+bp)"""
        backend = setup_backend(3)
        mu, cov = gaussian_state

        A = None
        d = np.array([0.4234, 0, 0, 0.1543, 0, 0])
        k = 0

        # prepare a displaced squeezed state
        backend.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=qphi)

        # E(ax+bp) = aE(x) + bE(p)
        mean_expected = d[0] * mu[0] + d[3] * mu[1]
        assert np.allclose(mean, mean_expected, atol=tol, rtol=0)

        # var(ax+bp) = a**2 var(x)+b**2 var(p)+2ab cov(x,p)
        var_expected = (
            cov[0, 0] * d[0] ** 2 + cov[1, 1] * d[3] ** 2 + 2 * d[0] * d[3] * cov[0, 1]
        )
        assert np.allclose(var, var_expected, atol=tol, rtol=0)

    def test_n_thermal(self, setup_backend, tol, hbar, pure):
        """Test expectation and variance of the number operator on a thermal state"""
        backend = setup_backend(3)
        backend.reset(cutoff_dim=CUTOFF, pure=pure)

        nbar = 0.423

        backend.prepare_thermal_state(nbar, 0)
        state = backend.state()

        # n = a^\dagger a = (X^2 +P^2)/2\hbar - I/2
        A = np.zeros([6, 6])
        A[0, 0] = 1 / (2 * hbar)
        A[3, 3] = 1 / (2 * hbar)
        k = -0.5

        mean, var = state.poly_quad_expectation(A, d=None, k=k, phi=0)
        assert np.allclose(mean, nbar, atol=tol, rtol=0)
        assert np.allclose(var, nbar * (nbar + 1), atol=tol, rtol=0)

    def test_n_squeeze(self, setup_backend, tol, hbar, pure):
        """Test expectation and variance of the number operator on a squeezed state"""
        backend = setup_backend(3)
        backend.reset(cutoff_dim=CUTOFF, pure=pure)

        backend.prepare_squeezed_state(r, phi, 0)
        state = backend.state()

        # n = a^\dagger a = (X^2 +P^2)/2\hbar - I/2
        A = np.zeros([6, 6])
        A[0, 0] = 1 / (2 * hbar)
        A[3, 3] = 1 / (2 * hbar)
        k = -0.5

        mean, var = state.poly_quad_expectation(A, None, k, phi=qphi)

        mean_ex = np.sinh(r) ** 2
        var_ex = 0.5 * np.sinh(2 * r) ** 2

        assert np.allclose(mean, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)

    def test_x_squared(self, setup_backend, tol, pure, sample_normal_expectations):
        """Test that the correct result is returned for E(x^2)"""
        backend = setup_backend(3)
        backend.reset(cutoff_dim=CUTOFF, pure=pure)

        A = np.zeros([6, 6])
        A[0, 0] = 1

        d = None
        k = 0

        # prepare a displaced squeezed state
        backend.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=qphi)

        mean_ex, var_ex = sample_normal_expectations(lambda X, P, XP: A[0, 0] * X ** 2)
        assert np.allclose(mean, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)

    def test_p_squared(self, setup_backend, tol, pure, sample_normal_expectations):
        """Test that the correct result is returned for E(p^2)"""
        backend = setup_backend(3)
        backend.reset(cutoff_dim=CUTOFF, pure=pure)

        A = np.zeros([6, 6])
        A[3, 3] = 1

        d = None
        k = 0

        # prepare a displaced squeezed state
        backend.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=qphi)

        mean_ex, var_ex = sample_normal_expectations(lambda X, P, XP: P ** 2)
        assert np.allclose(mean, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)

    def test_xp_vacuum(self, setup_backend, tol, sample_normal_expectations, hbar):
        """Test that the correct result is returned for E(xp) on the vacuum state"""
        backend = setup_backend(3)

        # set quadratic coefficient
        A = np.zeros([6, 6])
        A[3, 0] = A[0, 3] = 0.5

        d = None
        k = 0

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=0)

        mean_ex, var_ex = sample_normal_expectations(
            lambda X, P, XP: XP,
            correction=-np.linalg.det(hbar * A[:, [0, 3]][[0, 3]]),
            mu=np.zeros([2]),
            cov=np.identity(2)*hbar/2,
        )

        assert np.allclose(mean, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)

    def test_xp_displaced_squeezed(
        self, setup_backend, tol, pure, sample_normal_expectations, hbar
    ):
        """Test that the correct result is returned for E(xp) on a displaced squeezed state"""
        backend = setup_backend(3)
        backend.reset(cutoff_dim=CUTOFF, pure=pure)

        # set quadratic coefficient
        A = np.zeros([6, 6])
        A[3, 0] = A[0, 3] = 0.5

        d = None
        k = 0

        # prepare a displaced squeezed state
        backend.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=qphi)

        mean_ex, var_ex = sample_normal_expectations(
            lambda X, P, XP: XP, correction=-np.linalg.det(hbar * A[:, [0, 3]][[0, 3]])
        )
        assert np.allclose(mean, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)

    def test_arbitrary_quadratic(
        self, setup_backend, tol, pure, sample_normal_expectations, hbar
    ):
        """Test that the correct result is returned for E(c0 x^2 + c1 p^2 + c2 xp + c3 x + c4 p + k) on a displaced squeezed state"""
        backend = setup_backend(3)
        backend.reset(cutoff_dim=CUTOFF, pure=pure)

        c0 = 1 / np.sqrt(2)
        c1 = 3
        c2 = 0.53
        c3 = 1 / 3.0
        c4 = -1

        # define the arbitrary quadratic
        A = np.zeros([6, 6])
        A[0, 0] = c0
        A[3, 3] = c1

        A[3, 0] = c2 / 2
        A[0, 3] = c2 / 2

        # define the linear combination and constant term
        d = np.array([c3, 0, 0, c4, 0, 0])
        k = 5

        # prepare a displaced squeezed state
        backend.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=qphi)

        mean_ex, var_ex = sample_normal_expectations(
            lambda X, P, XP: c0 * X ** 2 + c1 * P ** 2 + c2 * XP + c3 * X + c4 * P + k,
            correction=-np.linalg.det(hbar * A[:, [0, 3]][[0, 3]]),
        )

        assert np.allclose(mean, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)


@pytest.mark.backends("fock", "gaussian")
class TestMultiModePolyQuadratureExpectations:
    """Test multi mode poly_quad_expectation methods"""

    def test_three_mode_arbitrary(self, setup_backend, pure, hbar, tol):
        """Test that the correct result is returned for an arbitrary quadratic polynomial"""
        backend = setup_backend(3)
        # increase the cutoff to 7 for accuracy
        backend.reset(cutoff_dim=7, pure=pure)

        # fmt:off
        A = np.array([[ 0.7086495 , -0.39299695,  0.30536448,  0.48822049,  0.64987373, 0.7020327 ],
                      [-0.39299695,  0.0284145 ,  0.53202656,  0.20232385,  0.26288656, 0.20772833],
                      [ 0.30536448,  0.53202656,  0.28126466,  0.64192545, -0.36583748, 0.51704656],
                      [ 0.48822049,  0.20232385,  0.64192545,  0.51033017,  0.29129713, 0.77103581],
                      [ 0.64987373,  0.26288656, -0.36583748,  0.29129713,  0.37646972, 0.2383589 ],
                      [ 0.7020327 ,  0.20772833,  0.51704656,  0.77103581,  0.2383589 ,-0.96494418]])

        # fmt:on
        d = np.array(
            [0.71785224, -0.80064627, 0.08799823, 0.76189805, 0.99665321, -0.60777437]
        )
        k = 0.123

        a_list = [0.044 + 0.023j, 0.0432 + 0.123j, -0.12 + 0.04j]
        r_list = [0.1065, 0.032, -0.123]
        phi_list = [0.897, 0.31, 0.432]

        mu = np.zeros([6])
        cov = np.zeros([6, 6])

        # squeeze and displace each mode
        for i, (a_, r_, phi_) in enumerate(zip(a_list, r_list, phi_list)):
            backend.prepare_displaced_squeezed_state(a_, r_, phi_, i)
            mu[2 * i : 2 * i + 2] = (
                R(qphi).T @ np.array([a_.real, a_.imag]) * np.sqrt(2 * hbar)
            )
            cov[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = (
                R(qphi).T @ utils.squeezed_cov(r_, phi_, hbar=hbar) @ R(qphi)
            )

        # apply a beamsplitter to the modes
        backend.beamsplitter(1 / np.sqrt(2), 1 / np.sqrt(2), 0, 1)
        backend.beamsplitter(1 / np.sqrt(2), 1 / np.sqrt(2), 1, 2)

        state = backend.state()
        mean, var = state.poly_quad_expectation(A, d, k, phi=qphi)

        # apply a beamsplitter to vector of means and covariance matrices
        t = 1 / np.sqrt(2)
        BS = np.array([[t, 0, -t, 0], [0, t, 0, -t], [t, 0, t, 0], [0, t, 0, t]])

        S1 = block_diag(BS, np.identity(2))
        S2 = block_diag(np.identity(2), BS)

        C = changebasis(3)
        mu = C.T @ S2 @ S1 @ mu
        cov = C.T @ S2 @ S1 @ cov @ S1.T @ S2.T @ C

        modes = list(np.arange(6).reshape(2, -1).T)

        mean_ex = np.trace(A @ cov) + mu @ A @ mu + mu @ d + k
        var_ex = (
            2 * np.trace(A @ cov @ A @ cov)
            + 4 * mu.T @ A.T @ cov @ A @ mu
            + d.T @ cov @ d
            + 2 * mu.T @ A.T @ cov @ d
            + 2 * d.T @ cov @ A @ mu
            - np.sum([np.linalg.det(hbar * A[:, m][n]) for m in modes for n in modes])
        )

        assert np.allclose(mean, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)

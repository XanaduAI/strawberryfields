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
r"""Unit tests for the states.py Wigner method"""
import pytest

import numpy as np
from scipy.stats import multivariate_normal

from thewalrus.symplectic import rotation as rotm

from strawberryfields import backends
from strawberryfields import utils


A = 0.3 + 0.1j
R = 0.23
PHI = 0.123

# wigner parameters
XMIN = -5
XMAX = 5
DX = 0.1
XVEC = np.arange(XMIN, XMAX, DX)

X, P = np.meshgrid(XVEC, XVEC)
GRID = np.empty(X.shape + (2,))
GRID[:, :, 0] = X
GRID[:, :, 1] = P


@pytest.fixture(autouse=True)
def skip_if_batched(batch_size):
    """Skip this test if in batch mode"""
    if batch_size is not None:
        pytest.skip("Wigner tests skipped in batch mode")


def wigner(GRID, mu, cov):
    """Generate the PDF distribution of a normal distribution"""
    mvn = multivariate_normal(mu, cov, allow_singular=True)
    return mvn.pdf(GRID)


def test_vacuum(setup_backend, hbar, tol):
    """Test Wigner function for a Vacuum state is a standard
    normal Gaussian"""
    backend = setup_backend(1)
    state = backend.state()
    W = state.wigner(0, XVEC, XVEC)

    # exact wigner function
    mu = [0, 0]
    cov = np.identity(2) * hbar / 2.0
    Wexact = wigner(GRID, mu, cov)

    assert np.allclose(W, Wexact, atol=tol, rtol=0)


def test_vacuum_one_point(setup_backend, hbar, tol):
    """Test Wigner function for a Vacuum state is a standard
    normal Gaussian at a single point"""
    backend = setup_backend(1)
    state = backend.state()
    vec = np.array([0])
    W = state.wigner(0, vec, vec)

    X, P = np.meshgrid(vec, vec)
    GRID = np.empty(X.shape + (2,))
    GRID[:, :, 0] = X
    GRID[:, :, 1] = P

    # exact wigner function
    mu = [0, 0]
    cov = np.identity(2) * hbar / 2.0
    Wexact = wigner(GRID, mu, cov)

    assert np.allclose(W, Wexact, atol=tol, rtol=0)


def test_squeezed_coherent(setup_backend, hbar, tol):
    """Test Wigner function for a squeezed coherent state
    matches the analytic result"""
    backend = setup_backend(1)
    backend.prepare_coherent_state(np.abs(A), np.angle(A), 0)
    backend.squeeze(R, PHI, 0)

    state = backend.state()
    W = state.wigner(0, XVEC, XVEC)
    rot = rotm(PHI / 2)

    # exact wigner function
    alpha = A * np.cosh(R) - np.conjugate(A) * np.exp(1j * PHI) * np.sinh(R)
    mu = np.array([alpha.real, alpha.imag]) * np.sqrt(2 * hbar)
    cov = np.diag([np.exp(-2 * R), np.exp(2 * R)])
    cov = np.dot(rot, np.dot(cov, rot.T)) * hbar / 2.0
    Wexact = wigner(GRID, mu, cov)

    assert np.allclose(W, Wexact, atol=0.01, rtol=0)


def test_two_mode_squeezed(setup_backend, hbar, tol):
    """Test Wigner function for a two mode squeezed state
    matches the analytic result"""
    backend = setup_backend(2)
    backend.prepare_squeezed_state(R, 0, 0)
    backend.prepare_squeezed_state(-R, 0, 1)

    state = backend.state()
    W0 = state.wigner(0, XVEC, XVEC)
    W1 = state.wigner(1, XVEC, XVEC)

    # exact wigner function
    mu = np.array([0, 0]) * np.sqrt(2 * hbar)
    cov = np.diag([np.exp(-2 * R), np.exp(2 * R)]) * hbar / 2
    W0exact = wigner(GRID, mu, cov)

    cov = np.diag([np.exp(2 * R), np.exp(-2 * R)]) * hbar / 2
    W1exact = wigner(GRID, mu, cov)

    assert np.allclose(W0, W0exact, atol=0.01, rtol=0)
    assert np.allclose(W1, W1exact, atol=0.01, rtol=0)


def fock_1_state_quad(setup_backend, hbar, tol):
    """Test the quadrature probability distribution
    functions for the |1> Fock state"""
    backend = setup_backend(1)
    backend.prepare_fock_state(1, 0)

    state = backend.state()

    x_vals = state.x_quad_values(0, XVEC, XVEC)
    p_vals = state.p_quad_values(0, XVEC, XVEC)

    # Exact probability distribution
    def exact(a):
        return (
            0.5 * np.sqrt(1 / (np.pi * hbar)) * np.exp(-1 * (a ** 2) / hbar) * (4 / hbar) * (a ** 2)
        )

    exact_x = np.array([exact(x) for x in XVEC])
    exact_p = np.array([exact(p) for p in XVEC])

    assert np.allclose(x_vals, exact_x, atol=tol, rtol=0)
    assert np.allclose(p_vals, exact_p, atol=tol, rtol=0)


def vacuum_state_quad(setup_backend, hbar, tol):
    """Test the quadrature probability distribution
    functions for the vacuum state"""
    backend = setup_backend(1)
    backend.prepare_vacuum_state(0)

    state = backend.state()

    x_vals = state.x_quad_values(0, XVEC, XVEC)
    p_vals = state.p_quad_values(0, XVEC, XVEC)

    # Exact probability distribution
    def exact(a):
        return np.sqrt(1 / (np.pi * hbar)) * np.exp(-1 * (a ** 2) / hbar)

    exact_x = np.array([exact(x) for x in XVEC])
    exact_p = np.array([exact(p) for p in XVEC])

    assert np.allclose(x_vals, exact_x, atol=tol, rtol=0)
    assert np.allclose(p_vals, exact_p, atol=tol, rtol=0)


def coherent_state_quad(setup_backend, hbar, tol):
    """Test the quadrature probability distribution
    functions for the coherent state with alpha = 1"""
    backend = setup_backend(1)
    backend.prepare_coherent_state(1, 0)

    state = backend.state()

    x_vals = state.x_quad_values(0, XVEC, XVEC)
    p_vals = state.p_quad_values(0, XVEC, XVEC)

    # Exact probability distribution
    def x_exact(a):
        return np.sqrt(1 / (np.pi * hbar)) * np.exp(
            -1 * ((a - 0.5 * np.sqrt(2 * hbar)) ** 2) / hbar
        )

    def p_exact(a):
        return np.sqrt(1 / (np.pi * hbar)) * np.exp(-1 * (a ** 2) / hbar)

    exact_x = np.array([x_exact(x) for x in XVEC])
    exact_p = np.array([p_exact(p) for p in XVEC])

    assert np.allclose(x_vals, exact_x, atol=tol, rtol=0)
    assert np.allclose(p_vals, exact_p, atol=tol, rtol=0)

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

from strawberryfields import backends
from strawberryfields import utils
from strawberryfields.backends.shared_ops import rotation_matrix as R


a = 0.3 + 0.1j
r = 0.23
phi = 0.123

# wigner parameters
xmin = -5
xmax = 5
dx = 0.1
xvec = np.arange(xmin, xmax, dx)

X, P = np.meshgrid(xvec, xvec)
grid = np.empty(X.shape + (2,))
grid[:, :, 0] = X
grid[:, :, 1] = P


@pytest.fixture(autouse=True)
def skip_if_batched(batch_size):
    """Skip this test if in batch mode"""
    if batch_size is not None:
        pytest.skip("Wigner tests skipped in batch mode")


def wigner(grid, mu, cov):
    """Generate the PDF distribution of a normal distribution"""
    mvn = multivariate_normal(mu, cov, allow_singular=True)
    return mvn.pdf(grid)


def test_vacuum(setup_backend, hbar, tol):
    """Test Wigner function for a Vacuum state is a standard
    normal Gaussian"""
    backend = setup_backend(1)
    state = backend.state()
    W = state.wigner(0, xvec, xvec)

    # exact wigner function
    mu = [0, 0]
    cov = np.identity(2) * hbar / 2.0
    Wexact = wigner(grid, mu, cov)

    assert np.allclose(W, Wexact, atol=tol, rtol=0)


def test_squeezed_coherent(setup_backend, hbar, tol):
    """Test Wigner function for a squeezed coherent state
    matches the analytic result"""
    backend = setup_backend(1)
    backend.prepare_coherent_state(a, 0)
    backend.squeeze(r * np.exp(1j * phi), 0)

    state = backend.state()
    W = state.wigner(0, xvec, xvec)
    rot = R(phi / 2)

    # exact wigner function
    alpha = a * np.cosh(r) - np.conjugate(a) * np.exp(1j * phi) * np.sinh(r)
    mu = np.array([alpha.real, alpha.imag]) * np.sqrt(2 * hbar)
    cov = np.diag([np.exp(-2 * r), np.exp(2 * r)])
    cov = np.dot(rot, np.dot(cov, rot.T)) * hbar / 2.0
    Wexact = wigner(grid, mu, cov)

    assert np.allclose(W, Wexact, atol=tol, rtol=0)


def test_two_mode_squeezed(setup_backend, hbar, tol):
    """Test Wigner function for a two mode squeezed state
    matches the analytic result"""
    backend = setup_backend(2)
    backend.prepare_squeezed_state(r, 0, 0)
    backend.prepare_squeezed_state(-r, 0, 1)

    state = backend.state()
    W0 = state.wigner(0, xvec, xvec)
    W1 = state.wigner(1, xvec, xvec)

    # exact wigner function
    mu = np.array([0, 0]) * np.sqrt(2 * hbar)
    cov = np.diag([np.exp(-2 * r), np.exp(2 * r)]) * hbar / 2
    W0exact = wigner(grid, mu, cov)

    cov = np.diag([np.exp(2 * r), np.exp(-2 * r)]) * hbar / 2
    W1exact = wigner(grid, mu, cov)

    assert np.allclose(W0, W0exact, atol=tol, rtol=0)
    assert np.allclose(W1, W1exact, atol=tol, rtol=0)

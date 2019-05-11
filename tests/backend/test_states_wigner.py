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
from strawberryfields.backends.shared_ops import rotation_matrix as rotm


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


def test_squeezed_coherent(setup_backend, hbar, tol):
    """Test Wigner function for a squeezed coherent state
    matches the analytic result"""
    backend = setup_backend(1)
    backend.prepare_coherent_state(A, 0)
    backend.squeeze(R * np.exp(1j * PHI), 0)

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

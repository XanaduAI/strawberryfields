# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Unit tests for strawberryfields.apps.train
"""
import pytest

import numpy as np

from strawberryfields.apps import train
from thewalrus.quantum import find_scaling_adjacency_matrix as rescale
from thewalrus.quantum import find_scaling_adjacency_matrix_torontonian as rescale_tor


# TODO: REMOVE
class Exp:

    def weights(self, params):
        return np.exp(-params)

    def __call__(self, params):
        return self.weights(params)

    def derivative(self, params):
        return -1*self.weights(params)


embedding = Exp()


@pytest.fixture
def params(dim):
    """Fixture for generating arbitrary parameters for the model"""
    return np.linspace(0, 1, dim)


def test_rescale_adjacency():
    """Test for the ``train.parametrization.rescale_adjacency`` function. We find the rescaled
    adjacency matrix when using threshold and PNR detection and check that: (i) the two matrices
    are not the same, (ii) they are rescaled by the correct factor."""
    adj = np.ones((10, 10))
    n_mean = 5

    r1 = train.parametrization.rescale_adjacency(adj, n_mean, True)
    r2 = train.parametrization.rescale_adjacency(adj, n_mean, False)
    s1 = rescale_tor(adj, n_mean)
    s2 = rescale(adj, n_mean)

    assert not np.allclose(r1, r2)
    assert np.allclose(r1, s1 * adj)
    assert np.allclose(r2, s2 * adj)


def test_A_to_cov():
    """Test if A_to_cov returns the correct covariance matrix for a simple fixed example"""
    adj = np.ones((2, 2))
    cov = train.parametrization.A_to_cov(adj)
    target = np.array([[-1, -4, -2, -2], [-4, -1, -2, -2], [-2, -2, -1, -4], [-2, -2, -4, -1]]) / 6
    assert np.allclose(cov, target)


@pytest.mark.usefixtures("adj", "params")
@pytest.mark.parametrize("dim", [7, 9])  # used in the adj fixture to determine number of modes
@pytest.mark.parametrize("n_mean", [4, 7])
class TestVGBS:
    """Tests for the class ``train.VGBS``"""

    def test_W(self, adj, n_mean, params):
        """Test that the _W method correctly gives the diagonal matrix of embedded parameters"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        W = gbs._W(params)
        assert np.allclose(np.diag(W), embedding(params))  # check that diagonal equals embedding
        assert np.allclose(np.diag(np.diag(W)), W)  # check that matrix is diagonal

    def test_A(self, adj, n_mean, params):
        """Test that the A and _A_scaled methods correctly return the WAW matrix of the A_init
        and A_init_scaled adjacency matrices, respectively, and that these outputs are different
        (due to the scaling)"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        A = gbs.A(params)
        A_scaled = gbs._A_scaled(params)
        W = gbs._W(params)

        assert np.allclose(A, W @ adj @ W)
        assert np.allclose(A_scaled, W @ gbs.A_init_scaled @ W)
        assert not np.allclose(A, A_scaled)

    # @pytest.mark.parametrize("threshold", [True, False])
    # def test_generate_samples(self, adj, n_mean, params):

    def test_mean_photons_per_mode(self, adj, n_mean, params, dim):
        gbs = train.VGBS(adj, n_mean, embedding, True)

        n_mean = gbs.mean_photons_per_mode(params)

        cov = train.parametrization.A_to_cov(gbs._A_scaled(params))
        cov_diag = np.diag(cov)

        nbar = [0.5 * (cov_diag[k] + cov_diag[k + dim] - 1) for k in range(dim)]
        targed_n_mean = np.array(nbar)

        print(np.sum(targed_n_mean))
        print(np.sum(n_mean))
        # assert np.allclose(n_mean, targed_n_mean)

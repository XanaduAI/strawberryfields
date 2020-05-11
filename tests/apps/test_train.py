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
        return -1 * self.weights(params)


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
    x = np.sqrt(0.5)
    A = np.array([[0, x], [x, 0]])
    cov = train.parametrization.A_to_cov(A)
    target = np.array(
        [
            [1.5, 0.0, 0.0, 1.41421356],
            [0.0, 1.5, 1.41421356, 0.0],
            [0.0, 1.41421356, 1.5, 0.0],
            [1.41421356, 0.0, 0.0, 1.5],
        ]
    )
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

    def test_mean_photons_per_mode(self, n_mean, dim):
        """Test that mean_photons_per_mode is correct when given a simple fully connected
        adjacency matrix. We expect each mode to have the same mean photon number and for that to
        add up to n_mean."""
        adj = np.ones((dim, dim))
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, False)
        n_mean_vec = np.round(gbs.mean_photons_per_mode(params), 5)

        assert np.allclose(np.sum(n_mean_vec), n_mean)  # check that the vector sums to n_mean
        assert len(np.unique(n_mean_vec)) == 1  # check that the vector is constant
        assert np.allclose(n_mean_vec[0], n_mean / dim)  # check that elements have correct values

    def test_mean_clicks_per_mode(self, n_mean, dim):
        """Test that mean_clicks_per_mode is correct when given a simple fully connected
        adjacency matrix. We expect each mode to have the same mean click number and for that to
        add up to n_mean."""
        adj = np.ones((dim, dim))
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, True)
        n_mean_vec = np.round(gbs.mean_clicks_per_mode(params), 5)

        assert np.allclose(np.sum(n_mean_vec), n_mean)  # check that the vector sums to n_mean
        assert len(np.unique(n_mean_vec)) == 1  # check that the vector is constant
        assert np.allclose(n_mean_vec[0], n_mean / dim)  # check that elements have correct values

    def test_photons_clicks_comparison(self, n_mean, dim):
        """Test that compares mean_photons_per_mode and mean_clicks_per_mode in the setting of a
        high mean photon number. Using the fully connected adjacency matrix, we expect elements of
        n_mean_vec_photon to be above one (there are more photons than modes) and also elements of
        n_mean_vec_click to be below one (you can never have more clicks on average than one)."""
        adj = np.ones((dim, dim))
        params = np.zeros(dim)
        gbs = train.VGBS(adj, 3 * n_mean, embedding, False)
        n_mean_vec_photon = np.round(gbs.mean_photons_per_mode(params), 5)
        n_mean_vec_click = np.round(gbs.mean_clicks_per_mode(params), 5)

        assert n_mean_vec_click[0] < 1
        assert n_mean_vec_photon[0] > 1


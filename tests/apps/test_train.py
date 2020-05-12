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
import thewalrus

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
@pytest.mark.parametrize("n_mean", [4, 6])
class TestVGBS:
    """Tests for the class ``train.VGBS``"""

    def test_W(self, adj, n_mean, params):
        """Test that the W method correctly gives the diagonal matrix of embedded parameters"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        W = gbs.W(params)
        assert np.allclose(np.diag(W), embedding(params))  # check that diagonal equals embedding
        assert np.allclose(np.diag(np.diag(W)), W)  # check that matrix is diagonal

    def test_A(self, adj, n_mean, params):
        """Test that the A and _A_scaled methods correctly return the WAW matrix of the A_init
        and A_init_scaled adjacency matrices, respectively, and that these outputs are different
        (due to the scaling)"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        A = gbs.A(params)
        A_scaled = gbs._A_scaled(params)
        W = gbs.W(params)

        assert np.allclose(A, W @ adj @ W)
        assert np.allclose(A_scaled, W @ gbs.A_init_scaled @ W)
        assert not np.allclose(A, A_scaled)

    def test_generate_samples(self, adj, n_mean, monkeypatch):
        """Test that _generate_samples correctly dispatches between torontonian and hafnian
        sampling based upon whether threshold=True or threshold=False"""
        gbs = train.VGBS(adj, n_mean, embedding, True)

        with monkeypatch.context() as m:
            m.setattr(thewalrus.samples, "torontonian_sample_state", lambda x, y: 0)
            s_threshold = gbs._generate_samples(gbs.A_init_scaled, 10)

        gbs.threshold = False

        with monkeypatch.context() as m:
            m.setattr(thewalrus.samples, "hafnian_sample_state", lambda x, y: 1)
            s_pnr = gbs._generate_samples(gbs.A_init_scaled, 10)

        assert s_threshold == 0
        assert s_pnr == 1

    def test_add_A_init_samples_bad_shape(self, adj, n_mean, dim):
        """Test that _add_A_init_samples raises a ValueError when input samples of incorrect
        shape, i.e. of dim + 1 modes"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        s = np.ones((2, dim + 1))
        with pytest.raises(ValueError, match="Must input samples of shape"):
            gbs._add_A_init_samples(s)

    def test_add_A_init_samples_none_there(self, adj, n_mean, dim):
        """Test that _add_A_init_samples correctly adds samples"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        s = np.ones((2, dim))
        gbs._add_A_init_samples(s)
        assert np.allclose(gbs._A_init_samples, s)

    def test_add_A_init_samples_already_there(self, adj, n_mean, dim):
        """Test that _add_A_init_samples correctly adds more samples when some are already there"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        s = np.ones((2, dim))
        gbs._A_init_samples = s
        gbs._add_A_init_samples(s)
        assert gbs._A_init_samples.shape == (4, dim)

    def test_mean_photons_per_mode(self, n_mean, dim):
        """Test that mean_photons_by_mode is correct when given a simple fully connected
        adjacency matrix and an identity W. We expect each mode to have the same mean photon number
        and for that to add up to n_mean."""
        adj = np.ones((dim, dim))
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, False)
        n_mean_vec = np.round(gbs.mean_photons_by_mode(params), 5)

        assert np.allclose(np.sum(n_mean_vec), n_mean)  # check that the vector sums to n_mean
        assert len(np.unique(n_mean_vec)) == 1  # check that the vector is constant
        assert np.allclose(n_mean_vec[0], n_mean / dim)  # check that elements have correct values

    def test_mean_clicks_per_mode(self, n_mean, dim):
        """Test that mean_clicks_by_mode is correct when given a simple fully connected
        adjacency matrix and an identity W. We expect each mode to have the same mean click number
        and for that to add up to n_mean."""
        adj = np.ones((dim, dim))
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, True)
        n_mean_vec = np.round(gbs.mean_clicks_by_mode(params), 5)

        assert np.allclose(np.sum(n_mean_vec), n_mean)  # check that the vector sums to n_mean
        assert len(np.unique(n_mean_vec)) == 1  # check that the vector is constant
        assert np.allclose(n_mean_vec[0], n_mean / dim)  # check that elements have correct values

    def test_photons_clicks_comparison(self, n_mean, dim):
        """Test that compares mean_photons_by_mode and mean_clicks_by_mode in the setting of a
        high mean photon number. Using the fully connected adjacency matrix and an identity W,
        we expect elements of n_mean_vec_photon to be above one (there are more photons than
        modes) and also elements of n_mean_vec_click to be below one (you can never have more
        clicks on average than one)."""
        adj = np.ones((dim, dim))
        params = np.zeros(dim)
        gbs = train.VGBS(adj, 3 * n_mean, embedding, False)
        n_mean_vec_photon = np.round(gbs.mean_photons_by_mode(params), 5)
        n_mean_vec_click = np.round(gbs.mean_clicks_by_mode(params), 5)

        assert n_mean_vec_click[0] < 1
        assert n_mean_vec_photon[0] > 1

    @pytest.mark.parametrize("threshold", [True, False])
    def test_n_mean(self, adj, n_mean, dim, threshold):
        """Test that n_mean returns the expected number of photons or clicks when using an
        identity W, so that we expect the mean number of photons to be equal to the input value
        of n_mean"""
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, threshold)
        assert np.allclose(gbs.n_mean(params), n_mean)


@pytest.mark.usefixtures("adj", "params")
@pytest.mark.parametrize("dim", [7, 9])  # used in the adj fixture to determine number of modes
@pytest.mark.parametrize("n_mean", [2, 3])
@pytest.mark.parametrize("threshold", [True, False])
def test_VGBS_integration(adj, params, n_mean, threshold, dim):
    """Integration test for the class ``train.VGBS``. We access the output adjacency matrix,
    mean photon number, and samples to check that they have the expected shape."""
    n_samples = 3
    gbs = train.VGBS(adj, n_mean, embedding, threshold)
    A_prime = gbs.A(params)
    n_mean_prime = gbs.n_mean(params)
    samples = gbs.generate_samples(params, n_samples)
    assert A_prime.shape == (dim, dim)
    assert isinstance(n_mean_prime, float)
    assert samples.shape == (n_samples, dim)

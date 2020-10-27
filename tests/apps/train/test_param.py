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
Unit tests for strawberryfields.apps.train.param
"""
# pylint: disable=no-self-use,protected-access,redefined-outer-name
import itertools

import numpy as np
import pytest
import thewalrus
from thewalrus.quantum import adj_scaling as rescale
from thewalrus.quantum import adj_scaling_torontonian as rescale_tor

import strawberryfields as sf
from strawberryfields.apps import train
from strawberryfields.apps.train.embed import Exp

pytestmark = pytest.mark.apps


@pytest.fixture
def embedding(dim):
    """Fixture for embedding"""
    return Exp(dim)


@pytest.fixture
def params(dim):
    """Fixture for generating arbitrary parameters for the model"""
    return np.linspace(0, 1, dim)


def test_rescale_adjacency():
    """Test for the ``train.param.rescale_adjacency`` function. We find the rescaled
    adjacency matrix when using threshold and PNR detection and check that: (i) the two matrices
    are not the same, (ii) they are rescaled by the correct factor."""
    adj = np.ones((10, 10))
    n_mean = 5

    r1 = train.param.rescale_adjacency(adj, n_mean, True)
    r2 = train.param.rescale_adjacency(adj, n_mean, False)
    s1 = rescale_tor(adj, n_mean)
    s2 = rescale(adj, n_mean)

    assert not np.allclose(r1, r2)
    assert np.allclose(r1, s1 * adj)
    assert np.allclose(r2, s2 * adj)


def test_prob_click():
    """Test for the ``train.param.prob_click`` function. For a simple 4 mode system,
    we generate the full probability distribution and check that it sums to one as well as that
    individual probabilities are between 0 and 1. We then focus on probabilities for samples that
    have two clicks and assert that they are all equal, as expected since our input adjacency
    matrix is fully connected.
    """
    dim = 4
    A = np.ones((dim, dim))
    A_scale = train.param.rescale_adjacency(A, 3, True)
    s = list(itertools.product([0, 1], repeat=dim))
    p = np.array([train.param.prob_click(A_scale, s_) for s_ in s])
    p_two_clicks = np.array([p[i] for i, s_ in enumerate(s) if sum(s_) == 2])
    assert np.allclose(np.sum(p), 1)  # check that distribution sums to one
    assert (p <= 1).all() and (p >= 0).all()
    assert np.allclose(p_two_clicks - p_two_clicks[0], np.zeros(len(p_two_clicks)))


def test_prob_photon_sample():
    """Test for the ``train.param.prob_photon_sample`` function. For a simple 4 mode
    system, we generate the probability distribution up to two photons and check that it sums to
    less than one as well as that individual probabilities are between 0 and 1. We then focus on
    probabilities for samples in the orbit [1, 1] and assert that they are all equal, as expected
    since our input adjacency matrix is fully connected.
    """
    dim = 4
    A = np.ones((dim, dim))
    A_scale = train.param.rescale_adjacency(A, 3, False)
    s = list(itertools.product(range(3), repeat=dim))
    p = np.array([train.param.prob_photon_sample(A_scale, s_) for s_ in s])
    p_two_clicks = np.array([p[i] for i, s_ in enumerate(s) if sum(s_) == 2 and max(s_) == 1])
    assert np.sum(p) <= 1
    assert (p <= 1).all() and (p >= 0).all()
    assert np.allclose(p_two_clicks - p_two_clicks[0], np.zeros(len(p_two_clicks)))


def test_A_to_cov(monkeypatch):
    """Test if A_to_cov returns the correct covariance matrix for a simple fixed example"""
    monkeypatch.setattr(sf, "hbar", 2.0)
    x = np.sqrt(0.5)
    A = np.array([[0, x], [x, 0]])
    cov = train.param.A_to_cov(A)
    target = np.array(
        [
            [3.0, 0.0, 0.0, 2.82842712],
            [0.0, 3.0, 2.82842712, 0.0],
            [0.0, 2.82842712, 3.0, 0.0],
            [2.82842712, 0.0, 0.0, 3.0],
        ]
    )
    assert np.allclose(cov, target)


@pytest.mark.usefixtures("adj", "params", "embedding")
@pytest.mark.parametrize("dim", [7, 9])  # used in the adj fixture to determine number of modes
@pytest.mark.parametrize("n_mean", [4, 6])
class TestVGBS:
    """Tests for the class ``train.VGBS``"""

    def test_W(self, adj, n_mean, params, embedding):
        """Test that the W method correctly gives the diagonal matrix of square root embedded
        parameters"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        W = gbs.W(params)
        assert np.allclose(np.diag(W) ** 2, embedding(params))  # check that diagonal squared
        # equals embedding

    def test_generate_samples(self, adj, n_mean, monkeypatch, embedding):
        """Test that generate_samples correctly dispatches between torontonian and hafnian
        sampling based upon whether threshold=True or threshold=False. This is done by
        monkeypatching torontonian_sample_state and hafnian_sample_state so that they simply
        return 0 and 1, respectively, instead of calculating samples. We then check that the
        returned samples are 0 in threshold mode and 1 when not in threshold mode."""
        gbs = train.VGBS(adj, n_mean, embedding, True)

        with monkeypatch.context() as m:
            m.setattr(thewalrus.samples, "torontonian_sample_state", lambda *args, **kwargs: 0)
            s_threshold = gbs.generate_samples(gbs.A_init, 10)

        gbs.threshold = False

        with monkeypatch.context() as m:
            m.setattr(thewalrus.samples, "hafnian_sample_state", lambda *args, **kwargs: 1)
            s_pnr = gbs.generate_samples(gbs.A_init, 10)

        assert s_threshold == 0
        assert s_pnr == 1

    def test_add_A_init_samples_bad_shape(self, adj, n_mean, dim, embedding):
        """Test that add_A_init_samples raises a ValueError when input samples of incorrect
        shape, i.e. of dim + 1 modes"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        s = np.ones((2, dim + 1))
        with pytest.raises(ValueError, match="Must input samples of shape"):
            gbs.add_A_init_samples(s)

    def test_add_A_init_samples_none_there(self, adj, n_mean, dim, embedding):
        """Test that add_A_init_samples correctly adds samples"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        s = np.ones((2, dim))
        gbs.add_A_init_samples(s)
        assert np.allclose(gbs.A_init_samples, s)

    def test_add_A_init_samples_already_there(self, adj, n_mean, dim, embedding):
        """Test that add_A_init_samples correctly adds more samples when some are already there"""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        gbs.A_init_samples = np.ones((2, dim))
        gbs.add_A_init_samples(np.zeros((2, dim)))
        assert gbs.A_init_samples.shape == (4, dim)
        assert np.allclose(gbs.A_init_samples[:2], np.ones((2, dim)))
        assert np.allclose(gbs.A_init_samples[2:3], np.zeros((2, dim)))

    def test_get_A_init_samples_none_there(self, adj, n_mean, monkeypatch, dim, embedding):
        """Test if get_A_init_samples generates the required samples when none are present in
        A_init_samples. To speed up sampling, we monkeypatch torontonian_sample_state to always
        return a numpy array of ones."""
        gbs = train.VGBS(adj, n_mean, embedding, True)
        with monkeypatch.context() as m:
            m.setattr(
                thewalrus.samples,
                "torontonian_sample_state",
                lambda *args, **kwargs: np.ones((args[1], dim)),
            )
            samples = gbs.get_A_init_samples(1000)
        assert np.allclose(samples, np.ones((1000, dim)))

    def test_get_A_init_samples_already_there(self, adj, n_mean, monkeypatch, dim, embedding):
        """Test if get_A_init_samples generates the required samples when some are already
        present in A_init_samples. We pre-load 200 samples of all zeros into VGBS and then
        request 1000 samples back. The sampling is monkeypatched to always return samples of
        ones, so that we hence expect 200 zero samples then 800 ones samples."""
        gbs = train.VGBS(adj, n_mean, embedding, True, np.zeros((200, dim)))
        with monkeypatch.context() as m:
            m.setattr(
                thewalrus.samples,
                "torontonian_sample_state",
                lambda *args, **kwargs: np.ones((args[1], dim)),
            )
            samples = gbs.get_A_init_samples(1000)
        assert np.allclose(samples[:200], np.zeros((200, dim)))
        assert np.allclose(samples[200:], np.ones((800, dim)))

    def test_get_A_init_samples_lots_there(self, adj, n_mean, dim, embedding):
        """Test if get_A_init_samples returns a portion of the pre-generated samples if
        ``n_samples`` is less than the number of samples stored."""
        gbs = train.VGBS(adj, n_mean, embedding, True, np.zeros((1000, dim)))
        samples = gbs.get_A_init_samples(200)
        assert samples.shape == (200, dim)

    @pytest.mark.parametrize("threshold", [True, False])
    def test_prob_sample_vacuum(self, n_mean, dim, threshold, embedding):
        """Test if prob_sample returns the correct probability of the vacuum, which can be
        calculated directly as the prefactor in the GBS distribution."""
        adj = np.ones((dim, dim))
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, threshold)
        sample = np.zeros(dim)
        p = gbs.prob_sample(params, sample)

        O = train.param._Omat(gbs.A(params))
        scale = np.sqrt(np.linalg.det(np.identity(2 * dim) - O))

        assert np.allclose(scale, p)

    def test_prob_sample_different(self, n_mean, dim, embedding):
        """Test if prob_sample returns different probabilities for the same sample when using
        threshold and pnr modes."""
        adj = np.ones((dim, dim))
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, True)
        sample = np.array([1, 1] + [0] * (dim - 2))
        p1 = gbs.prob_sample(params, sample)

        gbs.threshold = False
        p2 = gbs.prob_sample(params, sample)

        assert not np.allclose(p1, p2)

    def test_mean_photons_by_mode(self, n_mean, dim, embedding):
        """Test that mean_photons_by_mode is correct when given a simple fully connected
        adjacency matrix and an identity W. We expect each mode to have the same mean photon number
        and for that to add up to n_mean."""
        adj = np.ones((dim, dim))
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, False)
        n_mean_vec = gbs.mean_photons_by_mode(params)

        assert np.allclose(np.sum(n_mean_vec), n_mean)  # check that the vector sums to n_mean
        assert np.allclose(n_mean_vec - n_mean_vec[0], np.zeros(dim))  # check that the vector is
        # constant
        assert np.allclose(n_mean_vec[0], n_mean / dim)  # check that elements have correct values

    def test_mean_clicks_by_mode(self, n_mean, dim, embedding):
        """Test that mean_clicks_by_mode is correct when given a simple fully connected
        adjacency matrix and an identity W. We expect each mode to have the same mean click number
        and for that to add up to n_mean."""
        adj = np.ones((dim, dim))
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, True)
        n_mean_vec = gbs.mean_clicks_by_mode(params)

        assert np.allclose(np.sum(n_mean_vec), n_mean)  # check that the vector sums to n_mean
        assert np.allclose(n_mean_vec - n_mean_vec[0], np.zeros(dim))  # check that the vector is
        # constant
        assert np.allclose(n_mean_vec[0], n_mean / dim)  # check that elements have correct values

    def test_photons_clicks_comparison(self, n_mean, dim, adj, embedding):
        """Test that compares mean_photons_by_mode and mean_clicks_by_mode. We expect elements of
        n_mean_vec_photon to always be larger than n_mean_vec_click and also for elements of
        n_mean_vec_click to not exceed one."""
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, False)
        n_mean_vec_photon = gbs.mean_photons_by_mode(params)
        n_mean_vec_click = gbs.mean_clicks_by_mode(params)

        assert (n_mean_vec_click <= 1).all()
        assert (n_mean_vec_photon >= n_mean_vec_click).all()

    @pytest.mark.parametrize("threshold", [True, False])
    def test_n_mean(self, adj, n_mean, dim, threshold, embedding):
        """Test that n_mean returns the expected number of photons or clicks when using an
        identity W, so that we expect the mean number of photons to be equal to the input value
        of n_mean"""
        params = np.zeros(dim)
        gbs = train.VGBS(adj, n_mean, embedding, threshold)
        assert np.allclose(gbs.n_mean(params), n_mean)


@pytest.mark.usefixtures("adj", "params", "embedding")
@pytest.mark.parametrize("dim", [7, 9])  # used in the adj fixture to determine number of modes
@pytest.mark.parametrize("n_mean", [2, 3])
@pytest.mark.parametrize("threshold", [True, False])
def test_VGBS_integration(adj, params, n_mean, threshold, dim, embedding):
    """Integration test for the class ``train.VGBS``. We access the output adjacency matrix,
    mean photon number, and samples to check that they have the expected shape."""
    n_samples = 3
    gbs = train.VGBS(adj, n_mean, embedding, threshold)
    A_prime = gbs.A(params)
    n_mean_prime = gbs.n_mean(params)
    samples = gbs.generate_samples(A_prime, n_samples)
    assert A_prime.shape == (dim, dim)
    assert isinstance(n_mean_prime, float)
    assert samples.shape == (n_samples, dim)

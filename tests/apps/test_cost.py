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
Unit tests for strawberryfields.apps.train.cost
"""
import itertools

import numpy as np
import pytest

from strawberryfields.apps import train


def h(sample):
    """TODO"""
    return -sum([s * (i + 1) * (-1) ** (i + 1) for i, s in enumerate(sample)])


@pytest.fixture
def embedding(dim):
    """Fixture feature-based embedding"""
    feats = np.arange(0, dim * (dim - 1)).reshape((dim, dim - 1))
    return train.embed.ExpFeatures(feats)


@pytest.fixture
def A(dim):
    """Fully connected and non-normalized adjacency matrix"""
    return np.ones((dim, dim))


@pytest.fixture
def vgbs(A, n_mean, embedding, threshold):
    """VGBS for the fully connected adjacency matrix using the exponential embedding"""
    return train.VGBS(A, n_mean, embedding, threshold)


@pytest.fixture
def params(dim):
    """Trainable parameters initialized as a slight perturbation from all zeros"""
    return np.array([0.01 * i for i in range(dim - 1)])


@pytest.mark.parametrize("dim", [4])
@pytest.mark.parametrize("n_mean", [1])
@pytest.mark.usefixtures("vgbs")
class TestStochastic:
    """Tests for the class ``train.Stochastic``"""

    @pytest.mark.parametrize("threshold", [False])
    def test_h_reparametrized(self, vgbs, dim, params):
        """Test that _h_reparametrized behaves as expected by calculating the cost function over
        a fixed set of PNR samples using both the reparametrized and non-reparametrized methods"""
        cost_fn = train.Stochastic(h, vgbs)
        possible_samples = list(itertools.product([0, 1, 2], repeat=dim))

        probs = [vgbs.prob_sample(params, s) for s in possible_samples]
        probs_init = [vgbs.prob_sample(np.zeros(dim - 1), s) for s in possible_samples]

        cost = sum([probs[i] * h(s) for i, s in enumerate(possible_samples)])
        cost_reparam = sum([probs_init[i] * cost_fn._h_reparametrized(s, params) for i, \
                                                                                  s in enumerate(
            possible_samples)])

        assert np.allclose(cost, cost_reparam)

    @pytest.mark.parametrize("threshold", [False])
    def test_h_reparametized_example(self, dim, vgbs, params, embedding):
        """Test that _h_reparametrized returns the correct value when compared to working the
        result out by hand"""
        cost_fn = train.Stochastic(h, vgbs)
        sample = np.ones(dim)
        h_reparam = cost_fn._h_reparametrized(sample, params)
        h_reparam_expected = -1.088925964188385
        assert np.allclose(h_reparam, h_reparam_expected)

    @pytest.mark.parametrize("threshold", [False])
    def test_evaluate(self, vgbs, dim, params):
        """Test that evaluate returns the expected value when the VGBS class is preloaded with a
        dataset where half of the datapoints are zeros and half of the datapoints are ones. The
        expected result of the evaluate method is then simply the average of _h_reparametrized
        applied to a ones vector and a zeros vector."""
        n_samples = 10
        zeros = np.zeros((n_samples, dim))
        ones = np.ones((n_samples, dim))
        samples = np.vstack([zeros, ones])
        vgbs.add_A_init_samples(samples)

        cost_fn = train.Stochastic(h, vgbs)
        h0 = cost_fn._h_reparametrized(zeros[0], params)
        h1 = cost_fn._h_reparametrized(ones[0], params)
        eval_expected = (h0 + h1) * 0.5

        eval = cost_fn(params, 2 * n_samples)
        assert np.allclose(eval, eval_expected)

    @pytest.mark.parametrize("threshold", [True, False])
    def test_sample_difference_from_mean(self, dim, vgbs, n_mean):
        """Test that _sample_difference_from_mean is correct when considering a fully connected
        adjacency matrix. With n_mean = 1, we expect each mode to have a mean number of
        clicks/photons of 0.25. If we provide a sample of all ones, then we expect the difference
        to the mean vector to be a constant vector of 0.5."""
        cost_fn = train.Stochastic(h, vgbs)
        sample = np.ones(dim)
        params = np.zeros(dim - 1)

        diff = cost_fn._sample_difference_from_mean(sample, params)
        assert np.allclose(diff, 1 - n_mean * np.ones(dim) / dim)

    @pytest.mark.parametrize("threshold", [True, False])
    def test_gradient_one_sample(self, vgbs, dim, params, threshold):
        """Test that _gradient_one_sample returns the correct values when compared to
        calculations done by hand"""
        cost_fn = train.Stochastic(h, vgbs)
        sample = np.ones(dim)
        gradient = cost_fn._gradient_one_sample(sample, params)

        if threshold:
            grad_target = np.array([18.46861628, 22.53873502, 26.60885377])
        else:
            grad_target = np.array([17.17157601, 20.94382262, 24.71606922])

        assert np.allclose(gradient, grad_target)

    @pytest.mark.parametrize("threshold", [False])
    def test_gradient(self, vgbs, dim, params):
        """Test that gradient returns the expected value when the VGBS class is preloaded with a
        dataset where half of the datapoints are zeros and half of the datapoints are ones. The
        expected result of the gradient method is then simply the average of _gradient_one_sample
        applied to a ones vector and a zeros vector."""
        n_samples = 10
        zeros = np.zeros((n_samples, dim))
        ones = np.ones((n_samples, dim))
        samples = np.vstack([zeros, ones])
        vgbs.add_A_init_samples(samples)

        cost_fn = train.Stochastic(h, vgbs)
        g0 = cost_fn._gradient_one_sample(zeros[0], params)
        g1 = cost_fn._gradient_one_sample(ones[0], params)
        grad_expected = (g0 + g1) * 0.5

        grad = cost_fn.gradient(params, 2 * n_samples)
        assert np.allclose(grad_expected, grad)
        assert grad.shape == (dim - 1,)

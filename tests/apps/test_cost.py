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

pytestmark = pytest.mark.apps


def h(sample):
    """Cost function that applies to a single sample and gets lower if there are photons in odd
    numbered modes and higher with photons in even numbered modes. The higher-numbered modes have a
    greater effect due to the (i + 1) term."""
    return -sum([s * (i + 1) * (-1) ** (i + 1) for i, s in enumerate(sample)])


@pytest.fixture
def embedding(dim):
    """Fixture feature-based embedding"""
    feats = np.arange(0, dim * (dim - 1)).reshape((dim, dim - 1))
    return train.embed.ExpFeatures(feats)


@pytest.fixture
def simple_embedding(dim):
    """Fixture simple embedding"""
    return train.embed.Exp(dim)


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
        """Test that h_reparametrized behaves as expected by calculating the cost function over
        a fixed set of PNR samples using both the reparametrized and non-reparametrized methods"""
        cost_fn = train.Stochastic(h, vgbs)
        possible_samples = list(itertools.product([0, 1, 2], repeat=dim))

        probs = [vgbs.prob_sample(params, s) for s in possible_samples]
        probs_init = [vgbs.prob_sample(np.zeros(dim - 1), s) for s in possible_samples]

        cost = sum([probs[i] * h(s) for i, s in enumerate(possible_samples)])
        cost_reparam = sum([probs_init[i] * cost_fn.h_reparametrized(s, params) for i, \
                                                                                    s in enumerate(
            possible_samples)])

        assert np.allclose(cost, cost_reparam)

    @pytest.mark.parametrize("threshold", [False])
    def test_h_reparametized_example(self, dim, vgbs, params, embedding):
        """Test that h_reparametrized returns the correct value when compared to working the
        result out by hand"""
        cost_fn = train.Stochastic(h, vgbs)
        sample = np.ones(dim)
        h_reparam = cost_fn.h_reparametrized(sample, params)
        h_reparam_expected = -1.088925964188385
        assert np.allclose(h_reparam, h_reparam_expected)

    @pytest.mark.parametrize("threshold", [False])
    def test_evaluate(self, vgbs, dim, params):
        """Test that evaluate returns the expected value when the VGBS class is preloaded with a
        dataset where half of the datapoints are zeros and half of the datapoints are ones. The
        expected result of the evaluate method is then simply the average of h_reparametrized
        applied to a ones vector and a zeros vector."""
        n_samples = 10
        zeros = np.zeros((n_samples, dim))
        ones = np.ones((n_samples, dim))
        samples = np.vstack([zeros, ones])
        vgbs.add_A_init_samples(samples)

        cost_fn = train.Stochastic(h, vgbs)
        h0 = cost_fn.h_reparametrized(zeros[0], params)
        h1 = cost_fn.h_reparametrized(ones[0], params)
        eval_expected = (h0 + h1) * 0.5

        eval = cost_fn(params, 2 * n_samples)
        assert np.allclose(eval, eval_expected)

    @pytest.mark.parametrize("threshold", [True, False])
    def test_sample_difference_from_mean(self, dim, vgbs, n_mean):
        """Test that sample_difference_from_mean is correct when considering a fully connected
        adjacency matrix. With n_mean = 1, we expect each mode to have a mean number of
        clicks/photons of 0.25. If we provide a sample of all ones, then we expect the difference
        to the mean vector to be a constant vector of 0.5."""
        cost_fn = train.Stochastic(h, vgbs)
        sample = np.ones(dim)
        params = np.zeros(dim - 1)

        diff = cost_fn.sample_difference_from_mean(sample, params)
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


@pytest.mark.parametrize("dim", [4])
@pytest.mark.parametrize("n_mean", [7])
@pytest.mark.usefixtures("simple_embedding")
class TestStochasticIntegrationPNR:
    """Integration tests for the class ``train.Stochastic``. We consider the setting where the
    initial adjacency matrix is the identity, which reduces to directly sampling from
    non-interacting squeezed states without passing through an interferometer. Here,
    the trainable parameters can only control the amount of squeezing in each mode. We then use a
    mean squared error cost function that simply subtracts a fixed vector from an input sample and
    squares the result."""

    def identity_sampler(self, n_mean_by_mode, n_samples):
        """Used for quickly generating samples from a diagonal adjacency matrix. This requires
        sampling from single mode squeezed states whose photon number distribution is specified
        by the negative binomial."""
        qs = 1 / (1 + np.array(n_mean_by_mode))
        np.random.seed(0)
        return np.array([2 * np.random.negative_binomial(0.5, q, n_samples) for q in qs]).T

    def h_setup(self, objectives):
        """Mean squared error based cost function that subtracts a fixed vector from an input
        sample and squares the result"""
        def h(sample):
            return sum([(s - objectives[i]) ** 2 for i, s in enumerate(sample)])
        return h

    def test_initial_cost(self, dim, n_mean, simple_embedding):
        """Test that the cost function evaluates as expected on initial parameters of all zeros"""
        n_samples = 1000
        objectives = np.linspace(0.5, 1.5, dim)
        h = self.h_setup(objectives)
        A = np.eye(dim)
        vgbs = train.VGBS(A, n_mean, simple_embedding, threshold=False)

        n_mean_by_mode = [n_mean / dim] * dim
        samples = self.identity_sampler(n_mean_by_mode, n_samples=n_samples)
        vgbs.add_A_init_samples(samples)

        params = np.zeros(dim)
        cost_fn = train.Stochastic(h, vgbs)
        cost = cost_fn(params, n_samples=n_samples)

        # We can directly calculate the cost with respect to the samples
        expected_cost = np.mean(np.sum((samples - objectives) ** 2, axis=1))

        assert np.allclose(cost, expected_cost)

    def test_intermediate_cost(self, dim, n_mean, simple_embedding):
        """Test that the cost function evaluates as expected on non-initial parameters"""
        n_samples = 10000
        objectives = np.linspace(0.5, 1.5, dim)
        h = self.h_setup(objectives)
        A = np.eye(dim)
        vgbs = train.VGBS(A, n_mean, simple_embedding, threshold=False)

        n_mean_by_mode = [n_mean / dim] * dim
        samples = self.identity_sampler(n_mean_by_mode, n_samples=n_samples)
        vgbs.add_A_init_samples(samples)

        params = np.linspace(0, 1 / dim, dim)
        cost_fn = train.Stochastic(h, vgbs)
        cost = cost_fn(params, n_samples=n_samples)

        # We need to generate new samples and then calculate the cost with respect to them
        new_n_mean_by_mode = vgbs.mean_photons_by_mode(params)
        new_samples = self.identity_sampler(new_n_mean_by_mode, n_samples=n_samples)
        expected_cost = np.mean(np.sum((new_samples - objectives) ** 2, axis=1))

        assert np.allclose(cost, expected_cost, rtol=0.1)

    def test_gradient(self, dim, n_mean, simple_embedding):
        """Test that the gradient evaluates as expected when compared to a value calculated by
        hand"""
        n_samples = 100000  # We need a lot of shots due to the high variance in the distribution
        objectives = np.linspace(0.5, 1.5, dim)
        h = self.h_setup(objectives)
        A = np.eye(dim)
        vgbs = train.VGBS(A, n_mean, simple_embedding, threshold=False)

        n_mean_by_mode = [n_mean / dim] * dim
        samples = self.identity_sampler(n_mean_by_mode, n_samples=n_samples)
        vgbs.add_A_init_samples(samples)

        params = np.linspace(0, 1 / dim, dim)
        cost_fn = train.Stochastic(h, vgbs)

        # We want to calculate dcost_by_dn as this is available analytically, where n is the mean
        # photon number. The following calculates this using the chain rule:
        dcost_by_dtheta = cost_fn.gradient(params, n_samples=n_samples)
        dtheta_by_dw = 1 / np.diag(simple_embedding.jacobian(params))
        A_diag = np.diag(vgbs.A(params))
        A_init_diag = np.diag(vgbs.A_init)
        # Differentiate Eq. (8) of https://arxiv.org/abs/2004.04770 and invert for the next line
        dw_by_dn = (1 - A_diag ** 2) ** 2 / (2 * A_diag * A_init_diag)
        # Now use the chain rule
        dcost_by_dn = dcost_by_dtheta * dtheta_by_dw * dw_by_dn

        n_mean_by_mode = vgbs.mean_photons_by_mode(params)

        # Consider the problem with respect to a single mode. We want to calculate
        # E((s - x) ** 2) with s the number of photons in the mode, x the element of the fixed
        # vector, and with the expectation value calculated with respect to the negative binomial
        # distribution. Now, E((s - x) ** 2) = 3 * n_mean ** 2 + 2 * (1 - x) * n_mean + x ** 2,
        # with n_mean the mean number of photons in that mode. This can be differentiated to give
        # the derivative below.
        dcost_by_dn_expected = 6 * n_mean_by_mode + 2 * (1 - objectives)

        assert np.allclose(dcost_by_dn, dcost_by_dn_expected, 0.1)

    def test_converges(self, dim, n_mean, simple_embedding):
        """Test that after training the cost function is close to the analytical minimum"""
        n_samples = 10000
        objectives = np.linspace(0.5, 1.5, dim)
        h = self.h_setup(objectives)
        A = np.eye(dim)
        vgbs = train.VGBS(A, n_mean, simple_embedding, threshold=False)

        n_mean_by_mode = [n_mean / dim] * dim
        samples = self.identity_sampler(n_mean_by_mode, n_samples=n_samples)
        vgbs.add_A_init_samples(samples)

        params = np.zeros(dim)
        cost_fn = train.Stochastic(h, vgbs)
        reps = 4
        lr = 0.01

        # See test_gradient for further context. We want to minimize
        # E((s - x) ** 2) = 3 * n_mean ** 2 + 2 * (1 - x) * n_mean + x ** 2, which is minimized
        # by the n_mean given below.
        expected_n_mean = np.maximum((objectives - 1) / 3, np.zeros(dim))
        expected_cost = sum(3 * expected_n_mean ** 2 + 2 * (1 - objectives) * expected_n_mean + \
                        objectives ** 2)

        for i in range(reps):
            g = cost_fn.gradient(params, n_samples)
            params -= lr * g

        final_cost = cost_fn.evaluate(params, n_samples)
        assert np.allclose(expected_cost, final_cost, rtol=0.1, atol=2)

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

# pylint: disable=no-self-use
import numpy as np
import pytest

from strawberryfields.apps import train
from strawberryfields.apps.train import cost, embed, param

pytestmark = pytest.mark.apps

test_data = [[0, 0, 0, 0], [0, 2, 2, 2], [0, 0, 0, 2], [0, 0, 2, 2], [0, 2, 2, 0], [0, 0, 0, 4]]
n_means_data = np.mean(test_data, axis=0)
test_data = [[t[:d] for t in test_data] for d in range(2, 5)]
n_means_gbs = [[1, 1], [2 / 3, 2 / 3, 2 / 3], [1 / 2, 1 / 2, 1 / 2, 1 / 2]]
test_sum_log_probs = [6.931471805599451, 12.644620176064302, 22.377710881470353]

test_data_th = [[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 1]]
n_means_data_th = np.mean(test_data_th, axis=0)
test_data_th = [[t[:d] for t in test_data_th] for d in range(2, 5)]
n_means_gbs_th = [[1 / 2, 1 / 2], [1 / 3, 1 / 3, 1 / 3, 1 / 3], [1 / 4, 1 / 4, 1 / 4, 1 / 4]]
test_sum_log_probs_th = [1.386294361119879, 1.6218604324326575, 2.287080906458072]

params_fixed = [0, 0, 0, 0]
weights = [1, 1, 1, 1]
test_jacobian = -np.eye(4)
mean_photon_number = 2
mean_photon_number_th = 1
A_eye = np.eye(4)


@pytest.mark.parametrize("k", range(3))  # nr_modes = k + 2
class TestKL:
    """Tests for the class ``train.cost.KL"""

    def test_mean_data(self, k):
        """Tests the mean photon number per mode from hard-coded values stored in the array
        ``n_means_data``. The test iterates over different numbers of modes in the data."""
        m = k + 2
        embedding = embed.Exp(m)
        vgbs = param.VGBS(A_eye[:m, :m], mean_photon_number, embedding, threshold=False)
        kl = cost.KL(test_data[k], vgbs)
        assert np.allclose(kl.mean_n_data, n_means_data[:m])
        assert len(kl.mean_n_data) == m

    def test_grad(self, k):
        """Tests the calculation of the gradient against an explicit computation from hard-coded
        values of trainable parameters and mean photon numbers from data and model."""
        m = k + 2
        embedding = embed.Exp(m)
        vgbs = param.VGBS(A_eye[:m, :m], mean_photon_number, embedding, threshold=False)
        kl = cost.KL(test_data[k], vgbs)
        gamma = [-(n_means_data[i] - n_means_gbs[k][i]) / weights[i] for i in range(m)]
        assert np.allclose(kl.grad(params_fixed[:m]), gamma @ test_jacobian[:m, :m])

    def test_cost(self, k):
        """Tests the calculation of the Kullback-Liebler cost function against an explicit
        computation from hard-coded values of trainable parameters and mean photon numbers
        from data and model."""
        m = k + 2
        embedding = embed.Exp(m)
        vgbs = param.VGBS(A_eye[:m, :m], mean_photon_number, embedding, threshold=False)
        kl = cost.KL(test_data[k], vgbs)
        assert np.allclose(kl(params_fixed[:m]), test_sum_log_probs[k] / 6)

    def test_mean_data_threshold(self, k):
        """Tests the mean photon number per mode from hard-coded values stored in the array
        ``n_means_data``. The test iterates over different numbers of modes in the data."""
        m = k + 2
        embedding = embed.Exp(m)
        vgbs = param.VGBS(A_eye[:m, :m], mean_photon_number_th, embedding, threshold=True)
        kl = cost.KL(test_data_th[k], vgbs)
        assert np.allclose(kl.mean_n_data, n_means_data_th[:m])
        assert len(kl.mean_n_data) == m

    def test_grad_threshold(self, k):
        """Tests the calculation of the gradient against an explicit computation from hard-coded
        values of trainable parameters and mean photon numbers from data and model."""
        m = k + 2
        embedding = embed.Exp(m)
        vgbs = param.VGBS(A_eye[:m, :m], mean_photon_number_th, embedding, threshold=True)
        kl = cost.KL(test_data_th[k], vgbs)
        gamma = [-(n_means_data_th[i] - n_means_gbs_th[k][i]) / weights[i] for i in range(m)]
        assert np.allclose(kl.grad(params_fixed[:m]), gamma @ test_jacobian[:m, :m])


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
        """Test that the h_reparametrized method behaves as expected by calculating the cost
        function over a fixed set of PNR samples using both the reparametrized and
        non-reparametrized methods"""
        cost_fn = train.Stochastic(h, vgbs)
        possible_samples = list(itertools.product([0, 1, 2], repeat=dim))

        probs = [vgbs.prob_sample(params, s) for s in possible_samples]
        probs_init = [vgbs.prob_sample(np.zeros(dim - 1), s) for s in possible_samples]

        cost = sum([probs[i] * h(s) for i, s in enumerate(possible_samples)])

        cost_reparam_sum = [
            probs_init[i] * cost_fn.h_reparametrized(s, params)
            for i, s in enumerate(possible_samples)
        ]

        cost_reparam = sum(cost_reparam_sum)

        assert np.allclose(cost, cost_reparam)

    @pytest.mark.parametrize("threshold", [False])
    def test_h_reparametized_example(self, dim, vgbs, params, embedding):
        """Test that the h_reparametrized method returns the correct value when compared to
        working the result out by hand"""
        cost_fn = train.Stochastic(h, vgbs)
        sample = np.ones(dim)
        h_reparam = cost_fn.h_reparametrized(sample, params)
        h_reparam_expected = -1.088925964188385
        assert np.allclose(h_reparam, h_reparam_expected)

    @pytest.mark.parametrize("threshold", [False])
    def test_evaluate(self, vgbs, dim, params):
        """Test that the evaluate method returns the expected value when the VGBS class is
        preloaded with a dataset where half of the datapoints are zeros and half of the
        datapoints are ones. The expected result of the evaluate method is then simply the
        average of h_reparametrized applied to a ones vector and a zeros vector."""
        n_samples = 10
        zeros = np.zeros((n_samples, dim))
        ones = np.ones((n_samples, dim))
        samples = np.vstack([zeros, ones])
        vgbs.add_A_init_samples(samples)

        cost_fn = train.Stochastic(h, vgbs)
        h0 = cost_fn.h_reparametrized(zeros[0], params)
        h1 = cost_fn.h_reparametrized(ones[0], params)
        eval_expected = (h0 + h1) * 0.5

        eval = cost_fn(params, 2 * n_samples)  # Note that calling an instance of Stochastic uses
        # the evaluate method

        assert np.allclose(eval, eval_expected)

    @pytest.mark.parametrize("threshold", [True, False])
    def test_gradient_one_sample(self, vgbs, dim, params, threshold):
        """Test that the _gradient_one_sample method returns the correct values when compared to
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
        """Test that the gradient method returns the expected value when the VGBS class is
        preloaded with a dataset where half of the datapoints are zeros and half of the
        datapoints are ones. The expected result of the gradient method is then simply the
        average of _gradient_one_sample applied to a ones vector and a zeros vector."""
        n_samples = 10
        zeros = np.zeros((n_samples, dim))
        ones = np.ones((n_samples, dim))
        samples = np.vstack([zeros, ones])
        vgbs.add_A_init_samples(samples)

        cost_fn = train.Stochastic(h, vgbs)
        g0 = cost_fn._gradient_one_sample(zeros[0], params)
        g1 = cost_fn._gradient_one_sample(ones[0], params)
        grad_expected = (g0 + g1) * 0.5

        grad = cost_fn.grad(params, 2 * n_samples)
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
        ps = 1 / (1 + np.array(n_mean_by_mode))
        np.random.seed(0)
        return np.array([2 * np.random.negative_binomial(0.5, p, n_samples) for p in ps]).T

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
        """Test that the cost function evaluates as expected on non-initial parameters. This is
        done by comparing the cost function calculated using train.Stochastic with a manual
        calculation. The manual calculation involves sampling from VGBS with the non-initial
        params and averaging the cost function over the result."""
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
        hand.

        Consider the problem with respect to a single mode. We want to calculate
        E((s - x) ** 2) with s the number of photons in the mode, x the element of the fixed
        vector, and with the expectation value calculated with respect to the (twice) negative
        binomial distribution. We know that E(s) = 2 * r * (1 - q) / q and
        Var(s) = 4 * (1 - q) * r / q ** 2 in terms of the r and q parameters of the negative
        binomial distribution. Using q = 1 / (1 + n_mean) with n_mean the mean number of
        photons in that mode and r = 0.5, we can calculate
        E((s - x) ** 2) = 3 * n_mean ** 2 + 2 * (1 - x) * n_mean + x ** 2,
        This can be differentiated to give the derivative:
        d/dx E((s - x) ** 2) = 6 * n_mean + 2 * (1 - x).
        """
        n_samples = 10000  # We need a lot of shots due to the high variance in the distribution
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
        dcost_by_dtheta = cost_fn.grad(params, n_samples=n_samples)
        dtheta_by_dw = 1 / np.diag(simple_embedding.jacobian(params))
        A_diag = np.diag(vgbs.A(params))
        A_init_diag = np.diag(vgbs.A_init)
        # Differentiate Eq. (8) of https://arxiv.org/abs/2004.04770 and invert for the next line
        dw_by_dn = (1 - A_diag ** 2) ** 2 / (2 * A_diag * A_init_diag)
        # Now use the chain rule
        dcost_by_dn = dcost_by_dtheta * dtheta_by_dw * dw_by_dn

        n_mean_by_mode = vgbs.mean_photons_by_mode(params)

        dcost_by_dn_expected = 6 * n_mean_by_mode + 2 * (1 - objectives)

        assert np.allclose(dcost_by_dn, dcost_by_dn_expected, 0.1)

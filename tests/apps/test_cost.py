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
# pylint: disable=no-self-use
import numpy as np
import pytest
from strawberryfields.apps.train import cost, param, embed

pytestmark = pytest.mark.apps

test_data = [[0, 0, 0, 0], [0, 2, 2, 2], [0, 0, 0, 2], [0, 0, 2, 2], [0, 2, 2, 0], [0, 0, 0, 4]]
n_means_data = np.mean(test_data, axis=0)
test_data = [[t[:d] for t in test_data] for d in range(2, 5)]
n_means_gbs = [[1, 1], [2 / 3, 2 / 3, 2 / 3], [1 / 2, 1 / 2, 1 / 2, 1 / 2]]
params = [0, 0, 0, 0]
weights = [1, 1, 1, 1]
test_jacobian = -np.eye(4)
test_sum_log_probs = [6.931471805599451, 12.644620176064302, 22.377710881470353]
mean_photon_number = 2
A = np.eye(4)


@pytest.mark.parametrize("k", range(3))  # nr_modes = k + 2
class TestKL:
    """Tests for the class ``train.cost.KL"""

    def test_mean_data(self, k):
        """Tests the mean photon number per mode from hard-coded values stored in the array
        ``n_means_data``. The test iterates over different numbers of modes in the data."""
        m = k + 2
        embedding = embed.Exp(m)
        vgbs = param.VGBS(A[:m, :m], mean_photon_number, embedding, threshold=False)
        kl = cost.KL(test_data[k], vgbs)
        assert np.allclose(kl.mean_n_data, n_means_data[:m])
        assert len(kl.mean_n_data) == m

    def test_grad(self, k):
        """Tests the calculation of the gradient against an explicit computation from hard-coded
        values of trainable parameters and mean photon numbers from data and model."""
        m = k + 2
        embedding = embed.Exp(m)
        vgbs = param.VGBS(A[:m, :m], mean_photon_number, embedding, threshold=False)
        kl = cost.KL(test_data[k], vgbs)
        gamma = [-(n_means_data[i] - n_means_gbs[k][i]) / weights[i] for i in range(m)]
        assert np.allclose(kl.grad(params[:m]), gamma @ test_jacobian[:m, :m])

    def test_cost(self, k):
        """Tests the calculation of the Kullback-Liebler cost function against an explicit
        computation from hard-coded values of trainable parameters and mean photon numbers
        from data and model."""
        m = k + 2
        embedding = embed.Exp(m)
        vgbs = param.VGBS(A[:m, :m], mean_photon_number, embedding, threshold=False)
        kl = cost.KL(test_data[k], vgbs)
        assert np.allclose(kl.evaluate(params[:m]), test_sum_log_probs[k] / 6 - np.log(6))

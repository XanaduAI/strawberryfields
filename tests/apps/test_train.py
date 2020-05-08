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

from strawberryfields.apps import train


# TODO: REMOVE
import numpy as np
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


@pytest.mark.usefixtures("adj", "params")
@pytest.mark.parametrize("dim", [7, 9])  # used in the adj fixture to determine number of modes
@pytest.mark.parametrize("n_mean", [4, 7])
class TestVGBS:
    """Tests for the function ``train.VGBS``"""

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



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
Unit tests for strawberryfields.apps.train.embed
"""
# pylint: disable=no-self-use,unused-argument
import numpy as np
import pytest
from strawberryfields.apps.train import embed

feats = [
    [[0.1, 0.2], [0.2, 0.1]],
    [[0.1, 0.2, 0.3], [0.3, 0.1, 0.2], [0.2, 0.3, 0.1]],
    [[0.1, 0.2, 0.3, 0.4], [0.4, 0.1, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2], [0.2, 0.3, 0.4, 0.1]],
]
feats = [np.array(f) for f in feats]

ps = [[1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]
ps = np.array([np.array(p) for p in ps])

weights_f = np.array(
    [
        np.exp(-np.array([0.5, 0.4])),
        np.exp(-np.array([1.4, 1.1, 1.1])),
        np.exp(-np.array([3.0, 2.4, 2.2, 2.4])),
    ]
)
weights = np.array(
    [
        np.exp(-np.array([1.0, 2.0])),
        np.exp(-np.array([1.0, 2.0, 3.0])),
        np.exp(-np.array([1.0, 2.0, 3.0, 4.0])),
    ]
)

jacobian_f = np.array([np.zeros((d, d)) for d in range(2, 5)])
jacobian = np.array([np.zeros((d, d)) for d in range(2, 5)])

for i in range(3):
    jacobian[i] = -np.diag(weights[i])
    for j in range(i + 2):
        for k in range(i + 2):
            jacobian_f[i][j, k] = -feats[i][j, k] * weights_f[i][j]


@pytest.mark.parametrize("dim", range(2, 5))
class TestExpFeatures:
    """Tests for the class ``strawberryfields.apps.train.embed.ExpFeatures``"""

    def test_invalid_dim(self, dim):
        """Tests that a warning is issued if the feature vectors do not have the correct
        dimension"""
        with pytest.raises(ValueError, match="Dimension of parameter vector"):
            features = np.ones((dim, dim + 1))
            expf = embed.ExpFeatures(features)
            expf(np.zeros(dim))

    def test_zero_params(self, dim):
        """Tests that weights are equal to one when parameters are zero"""
        features = np.ones((dim, dim))
        expf = embed.ExpFeatures(features)
        params = np.zeros(dim)
        assert np.allclose(expf(params), np.ones(dim))

    def test_predefined(self, dim):
        """Tests that weights are computed correctly for pre-defined features and parameters"""
        k = dim - 2
        features = feats[k]
        expf = embed.ExpFeatures(features)
        assert (expf(ps[k]) - weights_f[k]).all() == 0


@pytest.mark.parametrize("dim", range(2, 5))
class TestJacobianExpFeatures:
    """Tests for the method ``strawberryfields.apps.train.embed.ExpFeatures.jacobian``"""

    def test_invalid_dim(self, dim):
        """Tests that a warning is issued if the feature vectors do not have the correct
        dimension"""
        with pytest.raises(ValueError, match="Dimension of parameter vector"):
            features = np.ones((dim, dim + 1))
            expf = embed.ExpFeatures(features)
            expf(np.zeros(dim))

    def test_jacobian_zero_params(self, dim):
        """Tests that the jacobian is equal to negative identity when parameters are zero"""
        features = np.ones((dim, dim))
        params = np.zeros(dim)
        expf = embed.ExpFeatures(features)
        g = expf.jacobian(params)
        assert (-np.ones((dim, dim)) - g).all() == 0

    def test_jacobian_predefined(self, dim):
        """Tests that the jacobian is computed correctly for pre-defined features and parameters"""
        k = dim - 2
        features = feats[k]
        expf = embed.ExpFeatures(features)
        g = expf.jacobian(ps[k])
        assert (g - jacobian_f[k]).all() == 0


@pytest.mark.parametrize("dim", range(2, 5))
class TestExp:
    """Tests for the callable class ``strawberryfields.apps.train.embed.Exp``"""

    def test_zero_params(self, dim):
        """Tests that weights are equal to one when parameters are zero"""
        exp = embed.Exp(dim)
        params = np.zeros(dim)
        assert (exp(params) - np.ones(dim)).all() == 0

    def test_predefined(self, dim):
        """Tests that weights are computed correctly for pre-defined features and parameters"""
        k = dim - 2
        exp = embed.Exp(dim)
        assert (exp(ps[k]) - weights[k]).all() == 0

    def test_identity(self, dim):
        """Tests that weights are computed correctly compared to a general calculation using an
        identity matrix of feature vectors"""
        k = dim - 2
        features = np.eye(dim)
        expf = embed.ExpFeatures(features)
        exp = embed.Exp(dim)
        assert (expf(ps[k]) - exp(ps[k])).all() == 0


@pytest.mark.parametrize("dim", range(2, 5))
class TestJacobianExp:
    """Tests for the method ``strawberryfields.apps.train.embed.Exp.jacobian``"""

    def test_jacobian_zero_params(self, dim):
        """Tests that the jacobian is equal to negative identity when parameters are zero"""
        params = np.zeros(dim)
        exp = embed.Exp(dim)
        g = exp.jacobian(params)
        assert (-np.ones((dim, dim)) - g).all() == 0

    def test_jacobian_predefined(self, dim):
        """Tests that the jacobian is computed correctly for pre-defined features and parameters"""
        k = dim - 2
        exp = embed.Exp(dim)
        g = exp.jacobian(ps[k])
        assert (g - jacobian[k]).all() == 0

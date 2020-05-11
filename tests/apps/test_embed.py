# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Unit tests for strawberryfields.apps.clique
"""
# pylint: disable=no-self-use,unused-argument
import functools

import networkx as nx
import numpy as np
import pytest

from strawberryfields.apps.train import embed

pytestmark = pytest.mark.apps

feats = [[[0.1, 0.2], [0.2, 0.1]], [[0.1, 0.2, 0.3], [0.3, 0.1, 0.2], [0.2, 0.3, 0.1]],
         [[0.1, 0.2, 0.3, 0.4], [0.4, 0.1, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2], [0.2, 0.3, 0.4, 0.1]]]
feats = np.array([np.array(f) for f in feats])

ps = [[1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]
ps = np.array([np.array(p) for p in ps])

weights_f = np.array([np.exp(-feats[i] @ ps[i]) for i in range(3)])
weights = np.array([np.exp(-ps[i]) for i in range(3)])


@pytest.mark.parametrize("dim", range(2, 5))
class TestExpFeatures:
    """Tests for the class ``strawberryfields.apps.train.embed.ExpFeatures``"""

    def test_zero_params(self, dim):
        """Tests that weights are equal to one when parameters are zero"""
        features = np.ones((dim, dim))
        expf = embed.ExpFeatures(features)
        params = np.zeros(dim)
        assert (expf(params) - np.ones(dim)).all() == 0

    def test_inf_params(self, dim):
        """Tests that weights are equal to zero when parameters are infinity"""
        features = np.ones((dim, dim))
        expf = embed.ExpFeatures(features)
        params = np.array([float("Inf") for _ in range(dim)])
        assert (expf(params) - np.zeros(dim)).all() == 0

    def test_predefined(self, dim):
        """Tests that weights are computed correctly for pre-defined features and parameters"""
        k = dim - 2
        features = feats[k]
        expf = embed.ExpFeatures(features)
        assert (expf(ps[k]) - weights_f[k]).all() == 0


@pytest.mark.parametrize("dim", range(2, 5))
class TestExp:
    """Tests for the class ``strawberryfields.apps.train.embed.Exp``"""

    def test_zero_params(self, dim):
        """Tests that weights are equal to one when parameters are zero"""
        exp = embed.Exp(dim)
        params = np.zeros(dim)
        assert (exp(params) - np.ones(dim)).all() == 0

    def test_inf_params(self, dim):
        """Tests that weights are equal to zero when parameters are infinity"""
        exp = embed.Exp(dim)
        params = np.array([float("Inf") for _ in range(dim)])
        assert (exp(params) - np.zeros(dim)).all() == 0

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


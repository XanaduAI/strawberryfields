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
from strawberryfields.apps.train import cost, param, embed


test_data = [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 2, 2], [0, 1, 2, 3], [1, 0, 0, 0]]
test_data = np.array([[np.array(t[:d]) for t in td] for d in range(2, 5)])
test_n_means = np.array([3/6, 3/6, 6/6, 7/6])
n_mean = 1


@pytest.mark.parametrize("i", range(3))
class TestKL:
    """Tests for the class ``train.cost.KL"""

    def test_mean_data(self):
        m = i + 2
        A = np.eye(m)
        params = np.zeros(m)
        embedding = embed(params)
        vgbs = param(A, n_mean, embedding, threshold=True)
        kl = cost.KL(test_data[i], vgbs)
        assert kl.mean_n_data() == test_n_means[:m]

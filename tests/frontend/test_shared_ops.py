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
r"""Unit tests for the Strawberry Fields shared_ops module"""
import pytest

pytestmark = pytest.mark.frontend

import numpy as np

import strawberryfields.backends.shared_ops as so

class TestPhaseSpaceFunctions:
    """Tests for the shared phase space operations"""

    def test_means_changebasis(self):
        """Test the change of basis function applied to vectors. This function
        converts from xp to symmetric ordering, and vice versa."""
        C = so.changebasis(3)
        means_xp = [1, 2, 3, 4, 5, 6]
        means_symmetric = [1, 4, 2, 5, 3, 6]

        assert np.all(C @ means_xp == means_symmetric)
        assert np.all(C.T @ means_symmetric == means_xp)

    def test_cov_changebasis(self):
        """Test the change of basis function applied to matrices. This function
        converts from xp to symmetric ordering, and vice versa."""
        C = so.changebasis(2)
        cov_xp = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )

        cov_symmetric = np.array(
            [[0, 2, 1, 3], [8, 10, 9, 11], [4, 6, 5, 7], [12, 14, 13, 15]]
        )

        assert np.all(C @ cov_xp @ C.T == cov_symmetric)
        assert np.all(C.T @ cov_symmetric @ C == cov_xp)

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
r"""Tests for the the point process functions"""

import pytest
import numpy as np
from strawberryfields.gbs.points import rbfkernel

pytestmark = pytest.mark.gbs


def test_rbfkernel():
    r"""Tests that correctness of the kernel matrix generated for a given set of point coordinates"""

    R = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    K = rbfkernel(R, 1.0)
    Kref = np.array(
        [
            [1.0, 0.36787944, 0.60653066, 0.60653066],
            [0.36787944, 1.0, 0.60653066, 0.60653066],
            [0.60653066, 0.60653066, 1.0, 0.36787944],
            [0.60653066, 0.60653066, 0.36787944, 1.0],
        ]
    )

    assert np.allclose(K, Kref)

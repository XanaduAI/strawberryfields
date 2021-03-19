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
"""Common shared operations that can be used by backends"""

import functools

import numpy as np

# ================================+
# Phase space shared operations  |
# ================================+


@functools.lru_cache()
def changebasis(n):
    r"""Change of basis matrix between the two Gaussian representation orderings.

    This is the matrix necessary to transform covariances matrices written
    in the (x_1,...,x_n,p_1,...,p_n) to the (x_1,p_1,...,x_n,p_n) ordering

    Args:
        n (int): number of modes
    Returns:
        array: :math:`2n\times 2n` matrix
    """
    m = np.zeros((2 * n, 2 * n))
    for i in range(n):
        m[2 * i, i] = 1
        m[2 * i + 1, i + n] = 1
    return m

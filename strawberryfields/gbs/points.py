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
Point Processes
===============

**Module name:** :mod:`strawberryfields.gbs.points`

.. currentmodule:: strawberryfields.gbs.points

This module provides functions for generating point processes.

Summary
-------

.. autosummary::
    kernel

Code details
^^^^^^^^^^^^
"""

import numpy as np
import scipy


def kernel(R: np.ndarray, sigma: float) -> np.ndarray:
    r"""Calculate the kernel matrix from a set of input points.

    We use the radial basis function (RBF) kernel which is positive semidefinite when Euclidean
    distances are used. The elements of the RBF kernel are computed as:

    .. math::
        K_{i,j} = e^{-\|\bf{r}_i-\bf{r}_j\|^2/(2\sigma^2)},

    where :math:`\bf{r}_i` are the coordinates of point :math:`i` and :math:`\sigma`
    is a kernel parameter that determines the scale of the kernel. Points that are much further
    than a distance :math: `\sigma` from each other lead to small entries of the kernel matrix,
    whereas points much closer than :math: `\sigma` generate large entries.

    **Example usage:**

    >>> R = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    >>> kernel (R, 1.0)
    array([[1., 0.36787944, 0.60653066, 0.60653066],
           [0.36787944, 1., 0.60653066, 0.60653066],
           [0.60653066, 0.60653066, 1., 0.36787944],
           [0.60653066, 0.60653066, 0.36787944, 1.,]])

    Args:
        R (array): Coordinate matrix. Rows of this array are the coordinates of the points.
        sigma (float): kernel parameter

    Returns:
        K (array): the kernel matrix
    """
    return np.exp(-(scipy.spatial.distance.cdist(R, R)) ** 2 / 2 / sigma ** 2)

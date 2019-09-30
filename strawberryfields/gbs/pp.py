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
**Module name:** :mod:`strawberryfields.gbs.pp`

.. currentmodule:: strawberryfields.gbs.pp

This module contains functions for point process utilization.

Summary
-------

.. autosummary::
    rbskernel

Code details
^^^^^^^^^^^^
"""

import numpy as np
from scipy.spatial.distance import cdist


def rbfkernel(R, sigma):
    r"""This function generates a radial basis function (RBF) kernel matrix.

    The elements of the RBF kernel are computed as:
    .. math::
    K_{i,j} = e^{-\|\bf{r}_i-\bf{r}_j\|^2/(2*\sigma^2)},
    where :math:`\bf{r}_i` is the coordinates of point :math:`i` and :math:`\sigma`
    is a constant.

    **Example usage:**
    >>> R = array([[0, 1],
                   [1, 0],
                   [0, 0],
                   [1, 1]])
    >>> sigma = 1.0
    >>> rbfkernel (R, sigma)
    array([[1.        , 0.36787944, 0.60653066, 0.60653066],
           [0.36787944, 1.        , 0.60653066, 0.60653066],
           [0.60653066, 0.60653066, 1.        , 0.36787944],
           [0.60653066, 0.60653066, 0.36787944, 1.        ]])

    Args:
        R (array): coordinates of the points.
        sigma (float): a constant.

    Returns:
        K (array): the kernel matrix.
    """
    K = np.exp(-(cdist(R, R)) ** 2 / 2 / sigma ** 2)
    return K

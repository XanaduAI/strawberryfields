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
This module contains various utility functions needed to perform quantum chemistry calculations with
Strawberry Fields.
"""
from typing import Tuple

import numpy as np
from scipy.constants import c, h, m_u, pi


def duschinsky(
    Li: np.ndarray, Lf: np.ndarray, ri: np.ndarray, rf: np.ndarray, wf: np.ndarray, m: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Generate the Duschinsky rotation matrix and displacement vector.

    **Example usage:**

    >>> Li = np.array([[-0.08727946], [ 0.00000000], [ 0.00000000],
    >>>                [ 0.95342606], [-0.00000000], [-0.00000000]])
    >>> Lf = np.array([[-0.08727946], [ 0.00000000], [ 0.00000000],
    >>>                [ 0.95342606], [-0.00000000], [-0.00000000]])
    >>> ri = np.array([-0.0236542994, 0.0000000000, 0.0000000000,
    >>>                 1.2236542994, 0.0000000000, 0.0000000000])
    >>> rf = np.array([ 0.0000000000, 0.0000000000, 0.0000000000,
    >>>                 1.4397000000, 0.0000000000, 0.0000000000])
    >>> wf = np.array([1363.210])
    >>> m = np.array([11.00931] * 3 + [1.00782] * 3)
    >>> Ud, delta = duschinsky(Li, Lf, ri, rf, wf, m)
    (array([[0.99999546]]), array([-1.1755024]))

    Args:
        Li (array): normal modes of the initial electronic state
        Lf (array): normal modes of the final electronic state
        ri (array): equilibrium molecular geometry of the initial electronic state
        rf (array): equilibrium molecular geometry of the final electronic state
        wf (array): normal mode frequencies of the final electronic state in units of
        :math:`\mbox{cm}^{-1}`
        m (array): atomic masses in units of unified atomic mass unit

    Returns:
        tuple[array, array]: Duschinsky rotation matrix :math:`Ud`, Duschinsky displacement vector
        :math:`\delta`
    """
    Ud = (Lf.T * m ** 0.5) @ (Li.T * m ** 0.5).T

    d = (ri - rf) @ (Lf.T * m).T

    l0 = np.diag((h / (wf * 100.0 * c)) ** 0.5 / 2.0 / pi) * 1.0e10 / m_u ** 0.5

    delta = np.array(d @ np.linalg.inv(l0))

    return Ud, delta

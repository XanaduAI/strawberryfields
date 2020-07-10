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
    r"""Generate the Duschinsky rotation matrix :math:`U` and displacement vector :math:`\delta`.

    The Duschinsky transformation relates the normal coordinates of the initial and
    final states in a vibronic transition, :math:`q_i` and :math:`q_f` respectively, as:

    .. math:: q_f = U q_i + d,

    where :math:`U` is the Duschinsky rotation matrix and :math:`d` is a vector giving the
    displacement between the equilibrium structures of the two states involved in the vibronic
    transition. The normal coordinates of a molecule can be represented in terms of atomic
    displacements as:

    .. math:: q = L^T \sqrt{m} (r -r_e),

    where :math:`r_e` represents the equilibrium geometry of the molecule, :math:`m` represents
    atomic masses and :math:`L` is a matrix containing the eigenvectors of the mass-weighted
    Hessian. The Duschinsky parameters :math:`U` and :math:`d` can be obtained as:

    .. math:: U = L_f^T L_i,

    .. math:: d = L_f^T \sqrt{m} (r_e^i-r_e^f).

    Note that :math:`i` and :math:`f` refer to the initial and final states, respectively. The
    parameter :math:`d` is usually represented as a dimensionless parameter :math:`\delta` as:

    .. math:: \delta = l^{-1} d,

    where :math:`l` is a diagonal matrix containing the vibrational frequencies :math:`\omega` of
    the final state:

    .. math:: l_{kk} = \left ( \frac{\hbar }{2 \pi \omega_k c} \right )^{1/2},

    where :math:`\hbar` is the reduced Planck constant and :math:`c` is the speed of light.

    The vibrational normal mode matrix for a molecule with :math:`M` vibrational modes and
    :math:`N` atoms is a :math:`3N \times M` matrix where :math:`M = 3N - 6` for nonlinear molecules and
    :math:`M = 3N - 5` for linear molecules. The Duschinsky rotation matrix of a molecule is an
    :math:`M \times M` matrix and the Duschinsky displacement vector has :math:`M` components.

    **Example usage:**

    >>> Li = np.array([[-0.28933191], [0.0], [0.0], [0.95711104], [0.0], [0.0]])
    >>> Lf = np.array([[-0.28933191], [0.0], [0.0], [0.95711104], [0.0], [0.0]])
    >>> ri = np.array([-0.0236, 0.0, 0.0, 1.2236, 0.0, 0.0])
    >>> rf = np.array([0.0, 0.0, 0.0, 1.4397, 0.0, 0.0])
    >>> wf = np.array([1363.2])
    >>> m = np.array([11.0093] * 3 + [1.0078] * 3)
    >>> U, delta = duschinsky(Li, Lf, ri, rf, wf, m)
    >>> U, delta
    (array([[0.99977449]]), array([-1.17623073]))

    Args:
        Li (array): mass-weighted normal modes of the initial electronic state
        Lf (array): mass-weighted normal modes of the final electronic state
        ri (array): equilibrium molecular geometry of the initial electronic state
        rf (array): equilibrium molecular geometry of the final electronic state
        wf (array): normal mode frequencies of the final electronic state in units of :math:`\mbox{cm}^{-1}`
        m (array): atomic masses in unified atomic mass units

    Returns:
        tuple[array, array]: Duschinsky rotation matrix :math:`U`, Duschinsky displacement vector
        :math:`\delta`
    """
    U = Lf.T @ Li

    d = Lf.T * m ** 0.5 @ (ri - rf)

    l0_inv = np.diag((h / (wf * 100.0 * c)) ** (-0.5) * 2.0 * pi) / 1.0e10 * m_u ** 0.5

    delta = np.array(d @ l0_inv)

    return U, delta

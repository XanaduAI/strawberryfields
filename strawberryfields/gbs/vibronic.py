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
Vibronic spectra
================

**Module name:** :mod:`strawberryfields.gbs.vibronic`

.. currentmodule:: strawberryfields.gbs.vibronic

This module contains functions for computing molecular vibronic spectra using GBS.
An accompanying tutorial can be found :ref:`here <>`.

Vibronic spectroscopy
---------------------

In vibronic spectroscopy, incoming light excites a molecule to higher electronic and
vibrational states simultaneously. The difference in the energies of the ground and excited states
determines the frequency of the photons that are absorbed by the molecule during the excitation.
The probability of the transition between two specific vibrational states, for an allowed
electronic transition, depends on the absolute square of the overlap integral between the
vibrational wave functions of the initial and the final states. These overlap integrals are
referred to as Franck-Condon factors (FCFs) and determine the intensity of absorption peaks in a
vibronic spectrum. The number of possible vibronic transitions in a molecule is usually large and
computing all the possible Franck-Condon factors is computationally expensive. However, a GBS
device can be programmed with quantum chemical information of a given molecule to provide output
samples that can be processed to obtain the molecular Franck-Condon profile. Theoretical background
on computing vibronic spectra using GBS can be found in :cite:`quesada2019franck` and
:cite:`huh2015boson`.

According to the Franck-Condon approximation, the probability of an electronically allowed
vibronic transition is given by a Franck-Condon factor as

.. math::
    FCF = \left | \left \langle \phi' | \phi \right \rangle \right | ^ 2,

where :math:`|\phi\rangle` and :math:`|{\phi}\rangle'` are the vibrational wave functions of the
ground and excited electronic states, respectively. The vibrational quanta states corresponding to
these ground and excited states, :math:`|\mathbf{n}\rangle` and :math:`|\mathbf{m'}\rangle`
respectively, are related to each other via a Doktorov transformation as
:math:`|\mathbf{m'}\rangle = U_{Dok} |\mathbf{n}\rangle` where :math:`U_{Dok}` is known as the
Doktorov operator. These vibrational quanta states can also be constructed from their
corresponding vacuum states by applying creation operators. Starting from the ground state of a
molecule at 0 K, which corresponds to the vacuum of ground state vibrational excitations
:math:`| \mathbf{0} \rangle`, the Franck-Condon factors can be written as:

.. math::
    FCF = \left | \left \langle \mathbf{m} |U_{Dok}| \mathbf{0} \right \rangle \right | ^ 2,

where the sate :math:`| \mathbf{m} \rangle = | m_1,...m_k \rangle` contains information to obtain
the transition energy that corresponds to the FCF through counting the number of vibrational quanta
:math:`m` in :math:`k` modes. This is equivalent to the probability of observing output patterns
of a GBS apparatus constructed according to :math:`U_{Dok}`. The Doktorov operator contains
parameters that should be obtained from quantum chemistry calculations for a given molecule.
Designing a GBS device with such molecular parameters allows obtaining the Franck-Condon profile
of the molecule which can be used to construct the molecular vibronic spectra.

The Doktorov operator can be written in terms of displacement (:math:`D_\alpha`), squeezing
(:math:`\Sigma`) and interferometer (:math:`U2, U1`) operators as

.. math::
    U_{Dok} = D_\alpha U2 \Sigma U1.

The squeezing and interferometer operators are obtained by singular value decomposition of a matrix
:math:`J` as

.. math::
    J = U2 \Sigma U1.

In order to obtain :math:`J` we need to construct diagonal matrices :math:`\Omega` and
:math:`\Omega'` from the square roots of the ground state and excited state normal mode frequencies,
:math:`\omega` and :math:`\omega'`, respectively. Then :math:`J` is obtained as

.. math::
    J = \Omega' U \Omega^{-1},

where

.. math::
    \Omega = diag (\sqrt{\omega_1},...,\sqrt{\omega_k})

    \Omega' = diag (\sqrt{\omega_1'},...,\sqrt{\omega_k'})

and :math:`U` is a molecular coordinate transformation known as the Duschinsky matrix which
relates the normal mode coordinates of the ground and excited states, :math:`q` and
:math:`q'` respectively, as

.. math::
    q' = Uq + d.

The displacement parameter :math:`d` is related to the structural changes of a molecule upon
vibronic excitation and is related to the displacement parameter :math:`\alpha` as

.. math::
    \alpha = d / \sqrt{2}.

In general, all the GBS parameters can be derived from the previously computed equilibrium
geometries, vibrational frequencies and normal coordinates of the molecule in its (electronic)
ground and excited states.


Summary
-------

.. autosummary::
    gbs_params

Code details
^^^^^^^^^^^^
"""
from typing import Tuple

import numpy as np


def gbs_params(
    w: np.ndarray, wp: np.ndarray, Ud: np.ndarray, d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Converts molecular information to GBS gate parameters.

    **Example usage:**

    >>> w = np.array([3765.2386, 3088.1826, 1825.1799, 1416.9512, 1326.4684, 1137.0490, 629.7144])
    >>> wp = np.array([3629.9472, 3064.9143, 1566.4602, 1399.6554, 1215.3421, 1190.9077, 496.2845])
    >>> delta = np.array([0.2254, 0.1469, 1.5599, -0.3784, 0.4553, -0.3439, 0.0618])
    >>> Ud = np.array([[0.9934, 0.0144, 0.0153, 0.0268, 0.0638, 0.0751, -0.0428],
    >>>               [-0.0149, 0.9931, 0.0742, 0.0769, -0.0361, -0.0025, 0.0173],
    >>>               [-0.0119, -0.0916, 0.8423, 0.1799, -0.3857, 0.3074, 0.0801],
    >>>               [0.0381, 0.0409, -0.3403, -0.5231, -0.6679, 0.3848, 0.1142],
    >>>               [-0.0413, -0.0342, -0.4004, 0.7636, -0.1036, 0.4838, 0.0941],
    >>>               [0.0908, -0.0418, -0.0907, 0.3151, -0.5900, -0.7193, 0.1304],
    >>>               [-0.0325, 0.0050, -0.0206, 0.0694, -0.2018, 0.0173, -0.9759]])
    >>> gbs_params(w, wp, Ud, delta)
    (array([[-0.07012006,  0.14489772,  0.17593463,  0.02431155, -0.63151781,
              0.61230046,  0.41087368],
            [ 0.5618538 , -0.09931968,  0.04562272,  0.02158822,  0.35700706,
              0.6614837 , -0.326946  ],
            [-0.16560687, -0.7608465 , -0.25644606, -0.54317241, -0.12822903,
              0.12809274, -0.00597384],
            [ 0.01788782,  0.60430409, -0.19831443, -0.73270964, -0.06393682,
              0.03376894, -0.23038293],
            [ 0.78640978, -0.11133936,  0.03160537, -0.09188782, -0.43483738,
             -0.4018141 ,  0.09582698],
            [-0.13664887, -0.11196486,  0.86353995, -0.19608061, -0.12313513,
             -0.08639263, -0.40251231],
            [-0.12060103, -0.01169781, -0.33937036,  0.34662981, -0.49895371,
              0.03257453, -0.70709135]]),
     array([ 0.09721339,  0.07017918,  0.02083469, -0.05974357, -0.07487845,
            -0.1119975 , -0.1866708 ]),
     array([[-0.07985219,  0.66041032, -0.19389188,  0.01340832,  0.70312675,
             -0.1208423 , -0.10352726],
            [ 0.19216669, -0.12470466, -0.81320519,  0.52045174, -0.1066017 ,
             -0.06300751, -0.00376173],
            [ 0.60838109,  0.0835063 , -0.14958816, -0.34291399,  0.06239828,
              0.68753918, -0.07955415],
            [ 0.63690134, -0.03047939,  0.46585565,  0.50545897,  0.21194805,
             -0.20422433,  0.18516987],
            [ 0.34556293,  0.22562207, -0.1999159 , -0.50280235, -0.25510781,
             -0.55793978,  0.40065893],
            [-0.03377431, -0.66280536, -0.14740447, -0.25725325,  0.6145946 ,
             -0.07128058,  0.29804963],
            [-0.24570365,  0.22402764,  0.003273  ,  0.19204683, -0.05125235,
              0.3881131 ,  0.83623564]]),
     array([ 0.15938187,  0.10387399,  1.10301587, -0.26756921,  0.32194572,
            -0.24317402,  0.0436992 ]))

    Args:
        w (array): normal mode frequencies of the electronic ground state (:math:`\text{cm}^{-1}`)
        wp (array): normal mode frequencies of the electronic excited state (:math:`\text{cm}^{-1}`)
        Ud (array): Duschinsky matrix
        d (array): Duschinsky displacement vector corrected with wp

    Returns:
        U1 (array): first interferometer unitary
        S (array): squeezing parameters
        U2 (array): second interferometer unitary
        alpha (array): displacement parameters
    """
    Wi = np.diag(w ** -0.5)
    Wp = np.diag(wp ** 0.5)
    J = Wp @ Ud @ Wi

    U2, s, U1 = np.linalg.svd(J)
    alpha = d / np.sqrt(2)

    return U1, np.log(s), U2, alpha

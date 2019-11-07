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
(:math:`\Sigma_{r}`) and interferometer (:math:`U_{2}, U_{1}`) operators as

.. math::
    U_{Dok} = D_\alpha U_{2} \Sigma_{r} U_{1}.

The squeezing and interferometer operators are obtained by singular value decomposition of a matrix
:math:`J` as

.. math::
    J = U_{2} \Sigma_{r} U_{1},

with the squeezing parameters :math:`r` determined by the natural log of the diagonal matrix.
In order to obtain :math:`J` we need to construct diagonal matrices :math:`\Omega` and
:math:`\Omega'` from the square roots of the ground state and excited state normal mode frequencies,
:math:`\omega` and :math:`\omega'`, respectively. Then :math:`J` is obtained as

.. math::
    J = \Omega' U_{d} \Omega^{-1},

where

.. math::
    \Omega = diag (\sqrt{\omega_1},...,\sqrt{\omega_k})

    \Omega' = diag (\sqrt{\omega_1'},...,\sqrt{\omega_k'})

and :math:`U_{d}` is a molecular coordinate transformation known as the Duschinsky matrix which
relates the normal mode coordinates of the ground and excited states, :math:`q` and
:math:`q'` respectively, as

.. math::
    q' = U_{d}q + d.

The displacement parameter :math:`d` is related to the structural changes of a molecule upon
vibronic excitation and is related to the displacement parameter :math:`\alpha` as

.. math::
    \alpha = d / \sqrt{2}.

In general, all the GBS parameters can be derived from the previously computed equilibrium
geometries, vibrational frequencies and normal coordinates of the molecule in its (electronic)
ground and excited states.

Finite temperature FCPs can also be obtained using a GBS setup.
First, :math:`N` two-mode squeezed vacuum states are prepared and the first :math:`N` modes
are treated with the same GBS operations explained for the zero Kelvin case. The second
:math:`N` modes are directly sent for photon number measurement. The number of photons
measured in the first and second modes, :math:`m_k` and :math:`n_k`,
respectively, are used to determine the transition frequencies and compute the finite temperature
FCP. The Gaussian gate parameters for the first :math:`N` modes are obtained for a given molecule
with the same procedure explained for the zero Kelvin case. The two-mode squeezing parameters
(:math:`t_i`) are obtained according to the distribution of phonons in the vibrational levels of the
electronic ground state according to a Boltzmann distribution with Boltzmann factor:

.. math::
    \tanh^2 (t_i) = \exp(-\hbar \omega_i/k_B T),

where :math:`\omega` is the angular frequency.

Summary
-------

.. autosummary::
    gbs_params
    energies

Code details
^^^^^^^^^^^^
"""
from typing import Tuple, Union

from scipy.constants import c, h, k

import numpy as np


def gbs_params(
    w: np.ndarray, wp: np.ndarray, Ud: np.ndarray, d: np.ndarray, T: float = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Converts molecular information to GBS gate parameters.

    **Example usage:**

    >>> w = np.array([3765.2386, 3088.1826, 1825.1799, 1416.9512, 1326.4684, 1137.0490, 629.7144])
    >>> wp = np.array([3629.9472, 3064.9143, 1566.4602, 1399.6554, 1215.3421, 1190.9077, 496.2845])
    >>> Ud = np.array(
    >>>     [
    >>>         [0.9934, 0.0144, 0.0153, 0.0268, 0.0638, 0.0751, -0.0428],
    >>>         [-0.0149, 0.9931, 0.0742, 0.0769, -0.0361, -0.0025, 0.0173],
    >>>         [-0.0119, -0.0916, 0.8423, 0.1799, -0.3857, 0.3074, 0.0801],
    >>>         [0.0381, 0.0409, -0.3403, -0.5231, -0.6679, 0.3848, 0.1142],
    >>>         [-0.0413, -0.0342, -0.4004, 0.7636, -0.1036, 0.4838, 0.0941],
    >>>         [0.0908, -0.0418, -0.0907, 0.3151, -0.5900, -0.7193, 0.1304],
    >>>         [-0.0325, 0.0050, -0.0206, 0.0694, -0.2018, 0.0173, -0.9759],
    >>>     ]
    >>> )
    >>> d = np.array([0.2254, 0.1469, 1.5599, -0.3784, 0.4553, -0.3439, 0.0618])
    >>> p = gbs_params(w, wp, Ud, d)

    Args:
        w (array): normal mode frequencies of the initial electronic state (:math:`\mbox{cm}^{-1}`)
        wp (array): normal mode frequencies of the final electronic state (:math:`\mbox{cm}^{-1}`)
        Ud (array): Duschinsky matrix
        d (array): Duschinsky displacement vector corrected with wp
        T (float): temperature (Kelvin)

    Returns:
        tuple[array, array, array, array, array]: the two-mode squeezing parameters :math:`t`,
        the first interferometer unitary matrix :math:`U_{1}`, the squeezing parameters
        :math:`r`, the second interferometer unitary matrix :math:`U_{2}`, and the displacement
        parameters :math:`\alpha`
    """
    if T < 0:
        raise ValueError("Temperature must be zero or positive")
    if T > 0:
        t = np.arctanh(np.exp(-0.5 * h * (w * c * 100) / (k * T)))
    else:
        t = np.zeros(len(w))

    U2, s, U1 = np.linalg.svd(np.diag(wp ** 0.5) @ Ud @ np.diag(w ** -0.5))
    alpha = d / np.sqrt(2)

    return t, U1, np.log(s), U2, alpha


def energies(samples: list, wp: np.ndarray) -> Union[list, float]:
    r"""Computes the energy of each GBS sample in units of :math:`\text{cm}^{-1}`.

    **Example usage:**

    >>> samples = [[1, 1, 0], [1, 0, 2]]
    >>> wp = np.array([700.0, 600.0, 500.0])
    >>> energies(samples, wp)
    [1300.0, 1700.0]

    Args:
        samples (list[list[int]] or list[int]): a list of samples from GBS, or alternatively a
            single sample
        wp (array): normal mode frequencies in units of :math:`\text{cm}^{-1}`

    Returns:
        list[float] or float: list of GBS sample energies in units of :math:`\text{cm}^{-1}`, or
        a single sample energy if only one sample is input
    """
    if not isinstance(samples[0], list):
        return np.dot(samples, wp)

    return [np.dot(s, wp) for s in samples]

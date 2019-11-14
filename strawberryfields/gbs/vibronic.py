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

This module contains functions used for computing molecular vibronic spectra with GBS. In vibronic
spectroscopy, incoming light simultaneously excites a molecule to higher electronic and
vibrational states. The difference in the energies of the ground and excited states
determines the frequency of the photons that are absorbed by the molecule. The probability of
the transition between two specific vibrational states are referred to as Franck-Condon factors
(FCFs). They determine the intensity of absorption peaks in a vibronic spectrum. A GBS
device can be programmed to provide samples that can be processed to obtain the molecular
vibronic spectra. Theoretical background on computing vibronic spectra using GBS can be found in
:cite:`huh2015boson` and :cite:`quesada2019franck`.

GBS parameters
--------------

The Franck-Condon factor is given by:

.. math::
    FCF = \left | \left \langle \mathbf{m} |U_{Dok}| \mathbf{0} \right \rangle \right | ^ 2,

where :math:`|\mathbf{m'}\rangle=|m_1, m_2, \ldots, m_k\rangle` is the state of :math:`k`
vibrational modes with :math:`m_i` excitations in the :math:`i`-th mode, and :math:`U_{Dok}` is
known as the Doktorov operator. The Doktorov operator can be written in terms of displacement
:math:`D_\alpha`, squeezing :math:`\Sigma_{r}`, and interferometer :math:`U_{2}, U_{1}` operators as

.. math::
    U_{Dok} = D_\alpha U_{2} \Sigma_{r} U_{1}.

Additionally, to account for molecules at finite temperature, the GBS algorithm employs two-mode
squeezing with parameters :math:`t` determined by the temperature of the molecule. The function
:func:`gbs_params` transforms chemical parameters, namely vibrational frequencies, displacement
vector and Duschinsky matrix, to the required GBS parameters.

.. autosummary::
    gbs_params

Energies from samples
---------------------
In the GBS algorithm for vibronic spectra, samples are identified with transition energies. The
most frequently sampled energies correspond to peaks of the vibronic spectrum. The function
:func:`energies` converts samples to energies.

.. autosummary::
    energies

Code details
^^^^^^^^^^^^
"""
from typing import Tuple, Union

from scipy.constants import c, h, k

import numpy as np


def gbs_params(
    w: np.ndarray, wp: np.ndarray, Ud: np.ndarray, delta: np.ndarray, T: float = 0
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
    >>> delta = np.array([0.2254, 0.1469, 1.5599, -0.3784, 0.4553, -0.3439, 0.0618])
    >>> p = gbs_params(w, wp, Ud, delta)

    Args:
        w (array): normal mode frequencies of the initial electronic state (:math:`\mbox{cm}^{-1}`)
        wp (array): normal mode frequencies of the final electronic state (:math:`\mbox{cm}^{-1}`)
        Ud (array): Duschinsky matrix
        delta (array): Displacement vector, with entries :math:`delta_i=\sqrt{
        \omega_i/\hbar}d_i`, and :math:`d` is the Duschinsky displacement
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
    alpha = delta / np.sqrt(2)

    return t, U1, np.log(s), U2, alpha


def energies(samples: list, w: np.ndarray, wp: np.ndarray) -> Union[list, float]:
    r"""Computes the energy :math:`E = \sum_{k=1}^{N}m_k\omega'_k - \sum_{k=N+1}^{2N}n_k\omega_k`
    of each GBS sample in units of :math:`\text{cm}^{-1}`.

    **Example usage:**

    >>> samples = [[1, 1, 0, 0, 0, 0], [1, 2, 0, 0, 1, 1]]
    >>> w  = np.array([300.0, 200.0, 100.0])
    >>> wp = np.array([700.0, 600.0, 500.0])
    >>> energies(samples, w, wp)
    [1300.0, 1600.0]

    Args:
        samples (list[list[int]] or list[int]): a list of samples from GBS, or alternatively a
            single sample
        w (array): normal mode frequencies of initial state in units of :math:`\text{cm}^{-1}`
        wp (array): normal mode frequencies of final state in units of :math:`\text{cm}^{-1}`

    Returns:
        list[float] or float: list of GBS sample energies in units of :math:`\text{cm}^{-1}`, or
        a single sample energy if only one sample is input
    """
    if not isinstance(samples[0], list):
        return np.dot(samples[:len(samples)//2], wp) - np.dot(samples[len(samples)//2:], w)

    return [np.dot(s[:len(s)//2], wp) - np.dot(s[len(s)//2:], w) for s in samples]

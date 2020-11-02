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
Functions used for computing molecular vibronic spectra with GBS.

In vibronic spectroscopy, incoming light simultaneously excites a molecule to higher electronic and
vibrational states. The difference in the energies of the initial and final states
determines the frequency of the photons that are absorbed by the molecule. The probability of
the transition between two specific vibrational states are referred to as Franck-Condon factors
(FCFs). They determine the intensity of absorption peaks in a vibronic spectrum. A GBS
device can be programmed to provide samples that can be processed to obtain the molecular
vibronic spectra. Theoretical background on computing vibronic spectra using GBS can be found in
:cite:`huh2015boson` and :cite:`quesada2019franck`.

.. seealso::

    :ref:`apps-vibronic-tutorial`

GBS parameters
--------------

The Franck-Condon factor is given by

.. math::
    FCF = \left | \left \langle \mathbf{m} |U_{Dok}| \mathbf{n} \right \rangle \right | ^ 2,

where :math:`|\mathbf{m}\rangle` and :math:`|\mathbf{n}\rangle` are the final and initial states,
respectively, and :math:`U_{Dok}` is known as the Doktorov operator which can be written in terms
of displacement :math:`D_\alpha`, squeezing :math:`\Sigma_{r}`, and interferometer :math:`U_{1},
U_{2}` operators as

.. math::
    U_{Dok} = D_\alpha U_{2} \Sigma_{r} U_{1}.

A GBS device can be programmed with the :math:`U_{Dok}` operator obtained for a given molecule. The
function :func:`gbs_params` transforms molecular parameters, namely vibrational frequencies,
displacement vector and Duschinsky matrix, to the required GBS parameters. Additionally, this
function computes two-mode squeezing parameters :math:`t`, from the molecule's temperature, which
are required by the GBS algorithm to compute vibronic spectra for molecules at finite temperature.
The :func:`sample` function then takes the computed GBS parameters and generates samples.
The :func:`~.VibronicTransition` function is an operation that can be used within the
conventional :class:`~.Program` interface for application of the Doktorov operator, allowing for
greater freedom on the choice of input state and any subsequent quantum operations.

Energies from samples
---------------------
In the GBS algorithm for vibronic spectra, samples are identified with transition energies. The
most frequently sampled energies correspond to peaks of the vibronic spectrum. The function
:func:`energies` converts samples to energies.
"""
import warnings
from typing import Tuple, Union

import numpy as np
from scipy.constants import c, h, k

import strawberryfields as sf
from strawberryfields.utils import operation


def gbs_params(
    w: np.ndarray, wp: np.ndarray, Ud: np.ndarray, delta: np.ndarray, T: float = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Converts molecular information to GBS gate parameters.

    **Example usage:**

    >>> formic = data.Formic()
    >>> w = formic.w  # ground state frequencies
    >>> wp = formic.wp  # excited state frequencies
    >>> Ud = formic.Ud  # Duschinsky matrix
    >>> delta = formic.delta  # displacement vector
    >>> T = 0  # temperature
    >>> p = gbs_params(w, wp, Ud, delta, T)

    Args:
        w (array): normal mode frequencies of the initial electronic state in units of
            :math:`\mbox{cm}^{-1}`
        wp (array): normal mode frequencies of the final electronic state in units of
            :math:`\mbox{cm}^{-1}`
        Ud (array): Duschinsky matrix
        delta (array): Displacement vector, with entries :math:`\delta_i=\sqrt{\omega'_i/\hbar}d_i`,
            and :math:`d_i` is the Duschinsky displacement
        T (float): temperature (Kelvin)

    Returns:
        tuple[array, array, array, array, array]: the two-mode squeezing parameters :math:`t`,
        the first interferometer unitary matrix :math:`U_{1}`, the squeezing parameters :math:`r`,
        the second interferometer unitary matrix :math:`U_{2}`, and the displacement
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
        return np.dot(samples[: len(samples) // 2], wp) - np.dot(samples[len(samples) // 2 :], w)

    return [np.dot(s[: len(s) // 2], wp) - np.dot(s[len(s) // 2 :], w) for s in samples]


def VibronicTransition(U1: np.ndarray, r: np.ndarray, U2: np.ndarray, alpha: np.ndarray):
    r"""An operation for applying the Doktorov operator
    :math:`\hat{U}_{\text{Dok}} = \hat{D}({\alpha}) \hat{R}(U_2) \hat{S}({r}) \hat{R}(U_1)` on a
    given state.

    The Doktorov operator describes the transformation between the initial and the final vibronic
    states of a molecule when it undergoes a vibronic transition.

    **Example usage:**

    >>> modes = 2
    >>> p = sf.Program(modes)
    >>> with p.context as q:
    ...     VibronicTransition(U1, r, U2, alpha) | q

    Args:
        U1 (array): unitary matrix for the first interferometer
        r (array): squeezing parameters
        U2 (array): unitary matrix for the second interferometer
        alpha (array): displacement parameters
    """
    # pylint: disable=expression-not-assigned
    n_modes = len(U1)

    @operation(n_modes)
    def op(q):

        sf.ops.Interferometer(U1) | q

        for i in range(n_modes):
            sf.ops.Sgate(r[i]) | q[i]

        sf.ops.Interferometer(U2) | q

        for i in range(n_modes):
            sf.ops.Dgate(np.abs(alpha[i]), np.angle(alpha[i])) | q[i]

    return op()


def sample(
    t: np.ndarray,
    U1: np.ndarray,
    r: np.ndarray,
    U2: np.ndarray,
    alpha: np.ndarray,
    n_samples: int,
    loss: float = 0.0,
) -> list:
    """Generate samples for computing vibronic spectra. The following gates are applied to input
    vacuum states:

    1. Two-mode squeezing on all :math:`2N` modes with parameters ``t``
    2. Interferometer ``U1`` on the first :math:`N` modes
    3. Squeezing on the first :math:`N` modes with parameters ``r``
    4. Interferometer ``U2`` on the first :math:`N` modes
    5. Displacement on the first :math:`N` modes with parameters ``alpha``

    A sample is generated by measuring the number of photons in each of the :math:`2N` modes. In
    the special case that all of the two-mode squeezing parameters ``t`` are zero, only :math:`N`
    modes are considered, which speeds up calculations.

    **Example usage:**

    >>> formic = data.Formic()
    >>> w = formic.w
    >>> wp = formic.wp
    >>> Ud = formic.Ud
    >>> delta = formic.delta
    >>> T = 0
    >>> t, U1, r, U2, alpha = gbs_params(w, wp, Ud, delta, T)
    >>> sample(t, U1, r, U2, alpha, 2, 0.0)
    [[0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    Args:
        t (array): two-mode squeezing parameters
        U1 (array): unitary matrix for the first interferometer
        r (array): squeezing parameters
        U2 (array): unitary matrix for the second interferometer
        alpha (array): displacement parameters
        n_samples (int): number of samples to be generated
        loss (float): loss parameter denoting the fraction of generated photons that are lost


    Returns:
        list[list[int]]: a list of samples from GBS
    """
    if n_samples < 1:
        raise ValueError("Number of samples must be at least one")
    if not 0 <= loss <= 1:
        raise ValueError("Loss parameter must take a value between zero and one")

    n_modes = len(t)

    eng = sf.LocalEngine(backend="gaussian")

    if np.any(t != 0):
        gbs = sf.Program(n_modes * 2)
    else:
        gbs = sf.Program(n_modes)

    # pylint: disable=expression-not-assigned,pointless-statement
    with gbs.context as q:

        if np.any(t != 0):
            for i in range(n_modes):
                sf.ops.S2gate(t[i]) | (q[i], q[i + n_modes])

        VibronicTransition(U1, r, U2, alpha) | q[:n_modes]

        if loss:
            for _q in q:
                sf.ops.LossChannel(1 - loss) | _q

        sf.ops.MeasureFock() | q

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Cannot simulate non-")

        s = eng.run(gbs, shots=n_samples).samples

    s = np.array(s).tolist()  # convert all generated samples to list

    if np.any(t == 0):
        s = np.pad(s, ((0, 0), (0, n_modes))).tolist()
    return s

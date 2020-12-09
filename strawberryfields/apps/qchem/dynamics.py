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
Functions used for simulating vibrational quantum dynamics of molecules.

Photonic quantum devices can be programmed with molecular data in order to simulate the quantum
dynamics of spatially-localized vibrations in molecules :cite:`sparrow2018simulating`. To that aim,
the quantum device has to be programmed to implement the transformation:

.. math::
    U(t) = U_l e^{-i\hat{H}t/\hbar} U_l^\dagger,

where :math:`\hat{H} = \sum_i \hbar \omega_i a_i^\dagger a_i` is the Hamiltonian corresponding to
the harmonic normal modes, :math:`\omega_i` is the vibrational frequency of the :math:`i`-th normal
mode, :math:`t` is time, and :math:`U_l` is a unitary matrix that relates the normal modes to a set
of new modes that are localized on specific bonds or groups in a molecule. The matrix :math:`U_l`
can be obtained by maximizing the sum of the squares of the atomic contributions to the modes
:cite:`jacob2009localizing`. Having :math:`U_l` and :math:`\omega` for a given molecule, and assuming
that it is possible to prepare the initial states of the mode, one can simulate the dynamics of
vibrational excitations in the localized basis at any given time :math:`t`. This process has three
main parts:

- Preparation of an initial vibrational state.
- Application of the dynamics transformation :math:`U(t)`.
- Generating samples and computing the probability of observing desired states.

It is noted that the initial states can be prepared in different ways. For instance, they can be
Fock states or Gaussian states such as coherent states or two-mode squeezed vacuum states.

Algorithm
---------

The algorithm for simulating the vibrational quantum dynamics in the localized basis with a photonic
device has the following form:

1. Each optical mode is assigned to a vibrational local mode and a specific initial excitation is
   created using one of the state preparation methods discussed. A list of state preparations
   methods available in Strawberry Fields is provided :doc:`here </introduction/ops>`.

2. An interferometer is configured according to the unitary :math:`U_l^\dagger` and the initial
   state is propagated through the interferometer.

3. For each mode, a rotation gate is designed as :math:`R(\theta) = \exp(i\theta \hat{a}^{\dagger}\hat{a})`
   where :math:`\theta = -\omega t`.

4. A second interferometer is configured according to the unitary :math:`U_l` and the new state
   is propagated through the interferometer.

5. The number of photons in each output mode is measured.

6. Samples are generated and the probability of obtaining a specific excitation in a given mode
   (or modes) is computed for time :math:`t`.

This module contains functions for implementing this algorithm.
"""
import warnings

import numpy as np
from scipy.constants import c, pi

import strawberryfields as sf
from strawberryfields.utils import operation


def TimeEvolution(w: np.ndarray, t: float):
    r"""An operation for performing the transformation
    :math:`e^{-i\hat{H}t/\hbar}` on a given state where :math:`\hat{H} = \sum_i \hbar \omega_i a_i^\dagger a_i`
    defines a Hamiltonian of independent quantum harmonic oscillators

    This operation can be used as part of a Strawberry Fields :class:`~.Program` just like any
    other operation from the :mod:`~.ops` module.

    **Example usage:**

    >>> modes = 2
    >>> p = sf.Program(modes)
    >>> with p.context as q:
    ...     sf.ops.Fock(1) | q[0]
    ...     sf.ops.Interferometer(Ul.T) | q
    ...     TimeEvolution(w, t) | q
    ...     sf.ops.Interferometer(Ul) | q

    Args:
        w (array): normal mode frequencies :math:`\omega` in units of :math:`\mbox{cm}^{-1}` that
            compose the Hamiltonian :math:`\hat{H} = \sum_i \hbar \omega_i a_i^\dagger a_i`
        t (float): time in femtoseconds
    """
    # pylint: disable=expression-not-assigned
    n_modes = len(w)

    @operation(n_modes)
    def op(q):

        theta = -w * 100.0 * c * 1.0e-15 * t * (2.0 * pi)

        for i in range(n_modes):
            sf.ops.Rgate(theta[i]) | q[i]

    return op()


def sample_fock(
    input_state: list,
    t: float,
    Ul: np.ndarray,
    w: np.ndarray,
    n_samples: int,
    cutoff: int,
    loss: float = 0.0,
) -> list:
    r"""Generate samples for simulating vibrational quantum dynamics with an input Fock state.

    **Example usage:**

    >>> input_state = [0, 2]
    >>> t = 10.0
    >>> Ul = np.array([[0.707106781, -0.707106781],
    ...                [0.707106781, 0.707106781]])
    >>> w = np.array([3914.92, 3787.59])
    >>> n_samples = 5
    >>> cutoff = 5
    >>> sample_fock(input_state, t, Ul, w, n_samples, cutoff)
    [[0, 2], [0, 2], [1, 1], [0, 2], [0, 2]]

    Args:
        input_state (list): input Fock state
        t (float): time in femtoseconds
        Ul (array): normal-to-local transformation matrix
        w (array): normal mode frequencies :math:`\omega` in units of :math:`\mbox{cm}^{-1}`
        n_samples (int): number of samples to be generated
        cutoff (int): cutoff dimension for each mode
        loss (float): loss parameter denoting the fraction of lost photons

    Returns:
        list[list[int]]: a list of samples
    """
    if np.any(np.iscomplex(Ul)):
        raise ValueError("The normal mode to local mode transformation matrix must be real")

    if n_samples < 1:
        raise ValueError("Number of samples must be at least one")

    if not len(input_state) == len(Ul):
        raise ValueError(
            "Number of modes in the input state and the normal-to-local transformation"
            " matrix must be equal"
        )

    if np.any(np.array(input_state) < 0):
        raise ValueError("Input state must not contain negative values")

    if max(input_state) >= cutoff:
        raise ValueError("Number of photons in each input mode must be smaller than cutoff")

    modes = len(Ul)

    s = []

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})

    prog = sf.Program(modes)

    # pylint: disable=expression-not-assigned
    with prog.context as q:

        for i in range(modes):
            sf.ops.Fock(input_state[i]) | q[i]

        sf.ops.Interferometer(Ul.T) | q

        TimeEvolution(w, t) | q

        sf.ops.Interferometer(Ul) | q

        if loss:
            for _q in q:
                sf.ops.LossChannel(1 - loss) | _q

        sf.ops.MeasureFock() | q

    for _ in range(n_samples):
        s.append(eng.run(prog).samples[0].tolist())

    return s


def sample_tmsv(
    r: list,
    t: float,
    Ul: np.ndarray,
    w: np.ndarray,
    n_samples: int,
    loss: float = 0.0,
) -> list:
    r"""Generate samples for simulating vibrational quantum dynamics with a two-mode squeezed
    vacuum input state.

    This function generates samples from a GBS device with two-mode squeezed vacuum input states.
    Given :math:`N` squeezing parameters and an :math:`N`-dimensional normal-to-local transformation
    matrix, a GBS device with :math:`2N` modes is simulated. The :func:`~.TimeEvolution` operator
    acts only on the first :math:`N` modes in the device. Samples are generated by measuring the
    number of photons in each of the :math:`2N` modes.

    **Example usage:**

    >>> r = [[0.2, 0.1], [0.8, 0.2]]
    >>> t = 10.0
    >>> Ul = np.array([[0.707106781, -0.707106781],
    ...                [0.707106781, 0.707106781]])
    >>> w = np.array([3914.92, 3787.59])
    >>> n_samples = 5
    >>> sample_tmsv(r, t, Ul, w, n_samples)
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 2, 0, 2]]

    Args:
        r (list[list[float]]): list of two-mode squeezing gate parameters given as ``[amplitude, phase]`` for all modes
        t (float): time in femtoseconds
        Ul (array): normal-to-local transformation matrix
        w (array): normal mode frequencies :math:`\omega` in units of :math:`\mbox{cm}^{-1}`
        n_samples (int): number of samples to be generated
        loss (float): loss parameter denoting the fraction of lost photons

    Returns:
        list[list[int]]: a list of samples
    """
    if np.any(np.iscomplex(Ul)):
        raise ValueError("The normal mode to local mode transformation matrix must be real")

    if n_samples < 1:
        raise ValueError("Number of samples must be at least one")

    if not len(r) == len(Ul):
        raise ValueError(
            "Number of squeezing parameters and the number of modes in the normal-to-local"
            " transformation matrix must be equal"
        )

    N = len(Ul)

    eng = sf.LocalEngine(backend="gaussian")
    prog = sf.Program(2 * N)

    # pylint: disable=expression-not-assigned
    with prog.context as q:

        for i in range(N):
            sf.ops.S2gate(r[i][0], r[i][1]) | (q[i], q[i + N])

        sf.ops.Interferometer(Ul.T) | q[:N]

        TimeEvolution(w, t) | q[:N]

        sf.ops.Interferometer(Ul) | q[:N]

        if loss:
            for _q in q:
                sf.ops.LossChannel(1 - loss) | _q

        sf.ops.MeasureFock() | q

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Cannot simulate non-")

        s = eng.run(prog, shots=n_samples).samples

    return s.tolist()


def sample_coherent(
    alpha: list,
    t: float,
    Ul: np.ndarray,
    w: np.ndarray,
    n_samples: int,
    loss: float = 0.0,
) -> list:
    r"""Generate samples for simulating vibrational quantum dynamics with an input coherent state.

    **Example usage:**

    >>> alpha = [[0.3, 0.5], [1.4, 0.1]]
    >>> t = 10.0
    >>> Ul = np.array([[0.707106781, -0.707106781],
    ...                [0.707106781, 0.707106781]])
    >>> w = np.array([3914.92, 3787.59])
    >>> n_samples = 5
    >>> sample_coherent(alpha, t, Ul, w, n_samples)
    [[0, 2], [0, 1], [0, 3], [0, 2], [0, 1]]

    Args:
        alpha (list[list[float]]): list of displacement parameters given as ``[magnitudes, angles]``
            for all modes
        t (float): time in femtoseconds
        Ul (array): normal-to-local transformation matrix
        w (array): normal mode frequencies :math:`\omega` in units of :math:`\mbox{cm}^{-1}`
        n_samples (int): number of samples to be generated
        loss (float): loss parameter denoting the fraction of lost photons

    Returns:
        list[list[int]]: a list of samples
    """
    if np.any(np.iscomplex(Ul)):
        raise ValueError("The normal mode to local mode transformation matrix must be real")

    if n_samples < 1:
        raise ValueError("Number of samples must be at least one")

    if not len(alpha) == len(Ul):
        raise ValueError(
            "Number of displacement parameters and the number of modes in the normal-to-local"
            " transformation matrix must be equal"
        )

    modes = len(Ul)

    eng = sf.LocalEngine(backend="gaussian")

    prog = sf.Program(modes)

    # pylint: disable=expression-not-assigned
    with prog.context as q:

        for i in range(modes):
            sf.ops.Dgate(alpha[i][0], alpha[i][1]) | q[i]

        sf.ops.Interferometer(Ul.T) | q

        TimeEvolution(w, t) | q

        sf.ops.Interferometer(Ul) | q

        if loss:
            for _q in q:
                sf.ops.LossChannel(1 - loss) | _q

        sf.ops.MeasureFock() | q

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Cannot simulate non-")

        s = eng.run(prog, shots=n_samples).samples

    return s.tolist()

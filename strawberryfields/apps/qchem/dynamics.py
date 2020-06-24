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

- The function :func:`~.evolution` returns a custom ``sf`` operation that contains the required
  unitary and rotation operations explained in steps 2-4 of the algorithm.

- The function :func:`~.sample_fock` generates samples for simulating vibrational quantum dynamics
  in molecules with a Fock input state.

- The function :func:`~.prob` estimates the probability of observing a desired excitation in the
  generated samples.

- The function :func:`~.marginals` generates single-mode marginal distributions from the displacement vector and
  covariance matrix of a Gaussian state.
"""
import numpy as np
from scipy.constants import c, pi
from thewalrus import quantum

import strawberryfields as sf
from strawberryfields.utils import operation


def evolution(modes: int):
    r"""Generates a custom ``sf`` operation for performing the transformation
    :math:`U(t) = U_l e^{-i\hat{H}t/\hbar} U_l^\dagger` on a given state.

    The custom operation returned by this function can be used as part of a Strawberry Fields
    :class:`~.Program` just like any other operation from the :mod:`~.ops` module. Its arguments
    are:

    - t (float): time in femtoseconds
    - Ul (array): normal-to-local transformation matrix :math:`U_l`
    - w (array): normal mode frequencies :math:`\omega` in units of :math:`\mbox{cm}^{-1}` that
      compose the Hamiltonian :math:`\hat{H} = \sum_i \hbar \omega_i a_i^\dagger a_i`

    **Example usage:**

    >>> modes = 2
    >>> transform =  evolution(modes)
    >>> p = sf.Program(modes)
    >>> with p.context as q:
    >>>     sf.ops.Fock(1) | q[0]
    >>>     sf.ops.Fock(2) | q[1]
    >>>     transform(t, Ul, w) | q

    Args:
        modes (int): number of modes

    Returns:
        an ``sf`` operation for enacting the dynamics transformation
    Return type:
        op
    """
    # pylint: disable=expression-not-assigned
    @operation(modes)
    def op(t, Ul, w, q):

        theta = -w * 100.0 * c * 1.0e-15 * t * (2.0 * pi)

        sf.ops.Interferometer(Ul.T) | q

        for i in range(modes):
            sf.ops.Rgate(theta[i]) | q[i]

        sf.ops.Interferometer(Ul) | q

    return op


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
    >>>                [0.707106781, 0.707106781]])
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
    if n_samples < 1:
        raise ValueError("Number of samples must be at least one")

    if t < 0:
        raise ValueError("Time must be zero or positive")

    if np.any(w <= 0):
        raise ValueError("Vibrational frequencies must be larger than zero")

    if np.any(np.iscomplex(Ul)):
        raise ValueError("The normal mode to local mode transformation matrix must be real")

    if not 0 <= loss <= 1:
        raise ValueError("Loss parameter must take a value between zero and one")

    if not len(input_state) == len(Ul):
        raise ValueError(
            "Number of modes in the input state and the normal-to-local transformation"
            " matrix must be equal"
        )

    if np.any(np.array(input_state) < 0):
        raise ValueError("Input state must not contain negative values")

    if max(input_state) >= cutoff:
        raise ValueError("Number of photons in each input state mode must be smaller than cutoff")

    modes = len(Ul)
    op = evolution(modes)
    s = []

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})

    prog = sf.Program(modes)

    # pylint: disable=expression-not-assigned
    with prog.context as q:

        for i in range(modes):
            sf.ops.Fock(input_state[i]) | q[i]

        op(t, Ul, w) | q

        if loss:
            for _q in q:
                sf.ops.LossChannel(1 - loss) | _q

        sf.ops.MeasureFock() | q

    for _ in range(n_samples):
        s.append(eng.run(prog).samples[0].tolist())

    return s


def prob(samples: list, excited_state: list) -> float:
    r"""Estimate probability of observing an excited state.

    The probability is estimated by calculating the relative frequency of the excited
    state among the samples.

    **Example usage:**

    >>> excited_state = [0, 2]
    >>> samples = [[0, 2], [1, 1], [0, 2], [2, 0], [1, 1], [0, 2], [1, 1], [1, 1], [1, 1]]
    >>> prob(samples, excited_state)
    0.3333333333333333

    Args:
        samples list[list[int]]: a list of samples
        excited_state (list): a Fock state

    Returns:
        float: probability of observing a Fock state in the given samples
    """
    if len(samples) == 0:
        raise ValueError("The samples list must not be empty")

    if len(excited_state) == 0:
        raise ValueError("The excited state list must not be empty")

    if not len(excited_state) == len(samples[0]):
        raise ValueError("The number of modes in the samples and the excited state must be equal")

    if np.any(np.array(excited_state) < 0):
        raise ValueError("The excited state must not contain negative values")

    return samples.count(excited_state) / len(samples)


def marginals(mu: np.ndarray, V: np.ndarray, n_max: int, hbar: float = 2.0) -> np.ndarray:
    r"""Generate single-mode marginal distributions from the displacement vector and covariance
    matrix of a Gaussian state.

    **Example usage:**

    >>> mu = np.array([0.00000000, 2.82842712, 0.00000000,
    >>>                0.00000000, 0.00000000, 0.00000000])
    >>> V = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    >>>               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    >>>               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    >>>               [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    >>>               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    >>>               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    >>> n_max = 10
    >>> marginals(mu, V, n_max)
    array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00],
           [1.35335284e-01, 2.70670567e-01, 2.70670566e-01, 1.80447044e-01,
            9.02235216e-02, 3.60894085e-02, 1.20298028e-02, 3.43708650e-03,
            8.59271622e-04, 1.90949249e-04],
           [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00]])

    Args:
        mu (array): displacement vector
        V (array): covariance matrix
        n_max (int): maximum number of vibrational quanta in the distribution
        hbar (float): the value of :math:`\hbar` in the commutation relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array[list[float]]: marginal distributions
    """
    if not V.shape[0] == V.shape[1]:
        raise ValueError("The covariance matrix must be a square matrix")

    if not len(mu) == len(V):
        raise ValueError(
            "The dimension of the displacement vector and the covariance matrix must be equal"
        )

    if n_max <= 0:
        raise ValueError("The number of vibrational states must be larger than zero")

    n_modes = len(mu) // 2

    p = np.zeros((n_modes, n_max))

    for mode in range(n_modes):
        mui, vi = quantum.reduced_gaussian(mu, V, mode)
        for i in range(n_max):
            p[mode, i] = np.real(quantum.density_matrix_element(mui, vi, [i], [i], hbar=hbar))

    return p

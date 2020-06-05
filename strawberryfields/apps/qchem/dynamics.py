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
Functions used for simulating vibrational quantum dynamics of molecules.

Photonic quantum devices can be programmed with molecular data in order to simulate the dynamics
of spatially-localized vibrations in molecules :cite:`sparrow2018simulating`. These vibrational
quantum dynamics can be simulated with a device that is programmed to implement the transformation:

.. math::
    U(t) = U_l^\dagger e^{-i\hat{H}t/\hbar} U_l,

where :math:`\hat{H} = \sum_i \hbar \omega_i a_i^\dagger a_i` is the Hamiltonian corresponding to
the harmonic normal modes, :math:`\omega_i` is the vibrational frequency of :math:`i`-th normal
mode, :math:`t` is time and :math:`U_l` is a unitary matrix that relates the normal modes to a set
of new modes that are localized on specific bonds or groups in a molecule. The matrix :math:`U_l`
can be obtained by maximizing the sum of the squares of the atomic contributions to the modes
:cite:`jacob2009localizing`. Having :math:`U` and :math:`\omega` for a given molecule, and assuming
that it is possible to prepare the initial states of the mode, one can simulate the dynamics of
vibrational excitations in the localized basis at any given time :math:`t`. This process has three
main parts. First, preparation of a vibrationally excited initial state. Second, application of the
dynamics transformation :math:`U(t)` and finally, generating samples and computing the probability
of observing desired states. It is noted that the initial states can be prepared in different ways.
For instance, they can be Fock states or Gaussian states such as coherent or two-mode squeezed
vacuum states.

Algorithm
---------

The algorithm for simulating the vibrational quantum dynamics in the localized basis with a photonic
device has the following form:

1. Each initial optical mode is assigned to a vibrational local mode and a specific excitation is
   created using one of the state preparation methods discussed.

2. An interferometer is configured according to the unitary :math:`U_l^\dagger` and the initial
   state is propagated through the interferometer.

3. For each mode, a rotation gate is designed according to the vibrational frequency of the
   corresponding normal mode (:math:`\nu`) as
   :math:`R(\theta) = \exp(i\theta \hat{a}^{\dagger}\hat{a})` where :math:`\theta = -2 \pi \nu t`
   and :math:`t` is time.

4. A second interferometer is configured according to the unitary :math:`U_l` and the new state
   is propagated through the interferometer.

5. The number of photons in each output mode is measured.

6. Samples are generated and the probability of obtaining a specific excitation in a given mode
   (or modes) is computed for time :math:`t`.

This module contains functions for implementing this algorithm. The function `dynamics_observable`
return a custom ``sf`` operation that contains the required unitary and rotation operations
explained in steps 2-4 of the algorithm.
"""
from typing import Callable

from scipy.constants import c, pi

import numpy as np

import strawberryfields as sf
from strawberryfields.utils import operation


def dynamics_observable(t: float, Ul: np.ndarray, w: np.ndarray) -> Callable:
    r"""Generates a custom ``sf`` operation for performing the transformation
    :math:`U(t) = U_l^\dagger e^{-i\hat{H}t/\hbar} U_l` on a given state.

    **Example usage:**

    >>> t = 0.0  # time
    >>> Ul = np.array([[0.707106781, -0.707106781],
    >>>                [0.707106781, 0.707106781]])  # normal to local transformation
    >>> w = np.array([3914.92, 3787.59])  # frequencies
    >>> obs =  dynamics_observable(t, Ul, w)

    Args:
        t (float): time (femtosecond)
        Ul (array): normal to local transformation matrix
        w (array): normal mode frequencies (:math:`\mbox{cm}^{-1}`)

    Returns:
        op (Callable): custom ``sf`` operation function
    """
    modes = len(Ul)
    theta = -w * 100.0 * c * t * 1e-15 * (2 * pi)

    @operation(modes)
    def op(q):
        sf.ops.Interferometer(Ul.T) | q
        for i in range(modes):
            sf.ops.Rgate(theta[i]) | q[i]
        sf.ops.Interferometer(Ul) | q

    return op

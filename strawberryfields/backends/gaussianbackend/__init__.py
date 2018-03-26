# Copyright 2018 Xanadu Quantum Technologies Inc.

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

.. _gaussian_backend:

Gaussian simulator backend
############################

**Module name:** :mod:`strawberryfields.backends.gaussianbackend`

.. currentmodule:: strawberryfields.backends.gaussianbackend

The :class:`GaussianBackend` implements a simulation of quantum optical circuits
using the Gaussian formalism. The primary component of the GaussianBackend is a
:class:`GaussianModes` object which is used to simulate a multi-mode quantum optical system.
:class:`GaussianBackend` provides the basic API-compatible interface to the simulator, while the
:class:`GaussianModes` object actually carries out the mathematical simulation.

The :class:`GaussianModes` simulators maintains an internal covariance matrix -- vector of means
representation of a multi-mode quantum optical system.

Note that unlike commonly used covariance matrix representations we encode our state in two complex
matrices :math:`N` and :math:`M` that are defined as follows
:math:`N_{i,j} = \langle a^\dagger _i a_j \rangle`
:math:`M_{i,j} = \langle a _i a_j \rangle`
and a vector of means :math:`\alpha_i =\langle a_i \rangle`


Basic quantum simulator methods
================================

.. currentmodule:: strawberryfields.backends.gaussianbackend.GaussianBackend

The :class:`GaussianBackend` simulator implements a number of state preparations, gates, and measurements (listed below).

.. autosummary::
   begin_circuit
   prepare_vacuum_state
   prepare_coherent_state
   prepare_squeezed_state
   prepare_displaced_squeezed_state
   prepare_fock_state
   prepare_ket_state
   rotation
   displacement
   squeeze
   beamsplitter
   loss
   measure_fock
   measure_homodyne
   measure_heterodyne
   is_vacuum
   del_mode
   add_mode
   get_modes
   reset
   state

Code details
~~~~~~~~~~~~

.. autoclass:: strawberryfields.backends.gaussianbackend.GaussianBackend
   :members:


Gaussian state
================

.. currentmodule:: strawberryfields.backends.gaussianbackend.GaussianState

This class which represents the quantum state
returned by a Gaussian backend. It extends :class:`~.BaseGaussianState`.

For other methods available, see the :ref:`state_class`.

.. autosummary::
   reduced_dm


Code details
~~~~~~~~~~~~

.. autoclass:: strawberryfields.backends.gaussianbackend.GaussianState
   :members:
"""

from .backend import GaussianBackend
from .states import GaussianState

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
.. _numpy_backend:

Fock simulator backend
====================================================================

**Module name:** :mod:`strawberryfields.backends.fockbackend`

.. currentmodule:: strawberryfields.backends.fockbackend

The :class:`FockBackend` object implements a simulation of quantum optical circuits using
`NumPy <http://www.numpy.org/>`_. The primary component of the FockBackend is a
:class:`Circuit` object which is used to simulate a multi-mode quantum optical system. The
:class:`FockBackend` provides the basic API-compatible interface to the simulator, while the
:class:`Circuit` object actually carries out the mathematical simulation.


The :class:`Circuit` simulator maintains an internal tensor representation of the quantum state of a multi-mode quantum optical system
using a (truncated) Fock basis representation. As its various state manipulation methods are called, the quantum state is updated
to reflect these changes. The simulator will try to keep the internal state in a pure (vector) representation
for as long as possible. Unitary gates will not change the type of representation, while state preparations and measurements will.

A number of factors determine the shape and dimensionality of the state tensor:

* the underlying state representation being used (either a ket vector or a density matrix)
* the number of modes :math:`n` actively being simulated
* the cutoff dimension :math:`D` for the Fock basis

The state tensor corresponds to a multimode quantum system. If the
representation is a pure state, the state tensor has shape
:math:`(\underbrace{D,...,D}_{n~\text{times}})`.
In a mixed state representation, the state tensor has shape
:math:`(\underbrace{D,D,...,D,D}_{2n~\text{times}})`.
Indices for the same mode appear consecutively. Hence, for a mixed state, the first two indices
are for the first mode, the second are for the second mode, etc.

.. currentmodule:: strawberryfields.backends.fockbackend.FockBackend

Basic quantum simulator methods
-------------------------------

.. autosummary::
   begin_circuit
   prepare_vacuum_state
   prepare_coherent_state
   prepare_squeezed_state
   prepare_displaced_squeezed_state
   prepare_fock_state
   prepare_ket_state
   prepare_dm_state
   rotation
   displacement
   squeeze
   beamsplitter
   kerr_interaction
   cross_kerr_interaction
   cubic_phase
   loss
   measure_fock
   measure_homodyne
   del_mode
   add_mode
   get_modes
   state

Auxiliary methods
-----------------

.. autosummary::
   reset
   get_cutoff_dim

Code details
~~~~~~~~~~~~
.. autoclass:: strawberryfields.backends.fockbackend.FockBackend
   :members:
"""

from .backend import FockBackend

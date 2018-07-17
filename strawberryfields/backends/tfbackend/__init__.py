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
.. _Tensorflow_backend:

Tensorflow simulator backend
##############################

**Module name:** :mod:`strawberryfields.backends.tfbackend`

.. currentmodule:: strawberryfields.backends.tfbackend

The :class:`TFBackend` object implements a simulation of quantum optical circuits using
`Tensorflow <http://www.Tensorflow.org/>`_. The primary component of the TFBackend is a
:class:`Circuit` object which is used to simulate a multi-mode quantum optical system. The
:class:`TFBackend` provides the basic API-compatible interface to the simulator, while the
:class:`Circuit` object actually carries out the mathematical simulation.


The :class:`Circuit` simulator maintains an internal tensor representation of the quantum state of a multi-mode quantum optical system
using a (truncated) Fock basis representation. As its various state manipulation methods are called, the quantum state is updated
to reflect these changes. The simulator will try to keep the internal state in a pure (vector) representation
for as long as possible. Unitary gates will not change the type of representation, while state preparations and measurements will.

A number of factors determine the shape and dimensionality of the state tensor:

* the underlying state representation being used (either a ket vector or a density matrix)
* the number of modes :math:`n` actively being simulated
* the cutoff dimension :math:`D` for the Fock basis
* whether the circuit is operating in *batched mode* (with batch size :math:`B`)

When not operating in batched mode, the state tensor corresponds to a single multimode quantum system. If the
representation is a pure state, the state tensor has shape
:math:`(\underbrace{D,...,D}_{n~\text{times}})`.
In a mixed state representation, the state tensor has shape
:math:`(\underbrace{D,D,...,D,D}_{2n~\text{times}})`.
Indices for the same mode appear consecutively. Hence, for a mixed state, the first two indices
are for the first mode, the second are for the second mode, etc.

In batched mode, the state tensor simultaneously encodes an ensemble of :math:`B` multimode quantum systems
(indexed using the first axis of the state tensor). Pure states thus have shape
:math:`(B,\underbrace{D,...,D}_{n~\text{times}})`,
while mixed states have shape
:math:`(B,\underbrace{D,D,...,D,D}_{2n~\text{times}})`.

.. currentmodule:: strawberryfields.backends.tfbackend.TFBackend

Basic quantum simulator methods
================================

The :class:`TFBackend` simulator implements a number of state preparations, gates, and measurements (listed below).
The parameters supplied for these operations can be either numeric (float, complex) values
or Tensorflow :class:`Variables`/:class:`Tensors`. The Tensorflow objects can either be scalars or vectors. For
vectors, they must have the same dimension as the declared batch size of the underlying circuit.

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
   cubic_phase
   kerr_interaction
   cross_kerr_interaction
   loss
   measure_fock
   measure_homodyne
   del_mode
   add_mode
   get_modes
   state

Auxiliary methods
==================

.. autosummary::
   reset
   get_cutoff_dim
   graph

Code details
~~~~~~~~~~~~
.. autoclass:: strawberryfields.backends.tfbackend.TFBackend
   :members:

FockStateTF
================

.. currentmodule:: strawberryfields.backends.tfbackend.states.FockStateTF

This class represents the quantum state
returned by the Tensorflow backend. It extends :class:`~.BaseFockState` with additional functionality
unique to the Tensorflow backend. The primary difference between this class and the Base Fock state
is that its methods and attributes can return either numerical or symbolic values.

The default representation (numerical or symbolic)
is set when creating the state: :code:`state = eng.run('backend', eval=True/False)`. The representation can also be specified on a per-use basis when calling a method, e.g., :code:`state.mean_photon(eval=True/False)`. Along with the boolean :code:`eval`, acceptable keyword arguments are :code:`session` (a Tensorflow Session object) and :code:`feed_dict` (a dictionary mapping Tensorflow objects to numerical values). These will be used when evaluating any Tensors.

.. currentmodule:: strawberryfields.backends.tfbackend.states.FockStateTF

.. autosummary::
   ket
   dm
   reduced_dm
   trace
   fock_prob
   all_fock_probs
   fidelity
   fidelity_coherent
   fidelity_vacuum
   is_vacuum
   quad_expectation
   mean_photon
   batched
   cutoff_dim
   graph


Code details
~~~~~~~~~~~~

.. autoclass:: strawberryfields.backends.tfbackend.states.FockStateTF
   :members:

"""

from .backend import TFBackend
from .ops import def_type as tf_complex_type

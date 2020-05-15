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
.. _TensorFlow_backend:

TensorFlow simulator backend
##############################

**Module name:** :mod:`strawberryfields.backends.tfbackend`

.. currentmodule:: strawberryfields.backends.tfbackend

The :class:`TFBackend` object implements a simulation of quantum optical circuits using
`TensorFlow <http://www.TensorFlow.org/>`_. The primary component of the TFBackend is a
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

TFBackend methods
=================

The parameters supplied for these operations can be either numeric (float, complex) values
or TensorFlow :class:`Variables`/:class:`Tensors`. The TensorFlow objects can either be scalars or vectors. For
vectors, they must have the same dimension as the declared batch size of the underlying circuit.

.. autosummary::
   supports
   begin_circuit
   add_mode
   del_mode
   get_modes
   get_cutoff_dim
   reset
   state
   is_vacuum
   prepare_vacuum_state
   prepare_coherent_state
   prepare_squeezed_state
   prepare_displaced_squeezed_state
   prepare_thermal_state
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
   thermal_loss
   measure_homodyne
   measure_fock


Code details
~~~~~~~~~~~~
.. autoclass:: strawberryfields.backends.tfbackend.TFBackend
   :members:

FockStateTF
================

.. currentmodule:: strawberryfields.backends.tfbackend.states.FockStateTF

This class represents the quantum state
returned by the TensorFlow backend. It extends :class:`~.BaseFockState` with additional functionality
unique to the TensorFlow backend. The primary difference between this class and the Base Fock state
is that it returns TensorFlow ``Tensor`` objects

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


Code details
~~~~~~~~~~~~

.. autoclass:: strawberryfields.backends.tfbackend.states.FockStateTF
   :members:

"""
import sys

try:
    import tensorflow
except ImportError:
    tf_available = False
    tf_version = None
else:
    tf_available = True
    tf_version = tensorflow.__version__


tf_info = """\
To use Strawberry Fields with TensorFlow support, version 2.x of TensorFlow is required.
This can be installed as follows:

pip install tensorflow
"""


def excepthook(type, value, traceback):
    """Exception hook to suppress superfluous exceptions"""
    # pylint: disable=unused-argument
    print(value)


if not (tf_available and tf_version[:2] == "2."):
    sys.excepthook = excepthook

    raise ImportError(tf_info)


from .backend import TFBackend
from .ops import def_type as tf_complex_type

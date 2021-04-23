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
"""This package contains the modules that make up the
Strawberry Fields backends. This includes photonic simulators,
shared numerical operations, and states objects returned by
statevector simulators.

Local statevector simulators
----------------------------

Below are all available local statevector backends currently
provided by Strawberry Fields. These simulators all run locally,
provide access to the state after simulation, and the state is
preserved between engine runs.

.. currentmodule:: strawberryfields.backends
.. autosummary::
    :toctree: api

    FockBackend
    GaussianBackend
    ~tfbackend.TFBackend
    BosonicBackend

.. raw:: html

    <div style="display: none;">

.. currentmodule:: strawberryfields.backends
.. autosummary::
    :toctree: api

    BaseFockState
    BaseGaussianState
    ~tfbackend.states.FockStateTF
    BaseBosonicState

.. raw:: html

    </div>

Backend API
-----------

A list of the abstract base classes that define the
statevector backend API

.. currentmodule:: strawberryfields.backends
.. autosummary::
    :toctree: api

    BaseState
    BaseBackend
    BaseFock
    BaseGaussian
    BaseBosonic

"""

from .base import BaseBackend, BaseFock, BaseGaussian, BaseBosonic, ModeMap
from .gaussianbackend import GaussianBackend
from .fockbackend import FockBackend
from .bosonicbackend import BosonicBackend
from .states import BaseState, BaseGaussianState, BaseFockState, BaseBosonicState

# There is no import for the TFBackend to avoid TensorFlow being a direct
# requirement of SF through a chain of imports

__all__ = [
    "BaseBackend",
    "BaseFock",
    "BaseGaussian",
    "BaseBosonic",
    "FockBackend",
    "GaussianBackend",
    "BosonicBackend",
    "TFBackend",
    "BaseState",
    "BaseFockState",
    "BaseGaussianState",
    "BaseBosonicState",
]


virtual_backends = ["X8_01"]

local_backends = {
    b.short_name: b for b in (BaseBackend, GaussianBackend, FockBackend, BosonicBackend)
}


def load_backend(name):
    """Loads the specified backend by mapping a string
    to the backend type, via the ``local_backends``
    dictionary. Note that this function is used by the
    frontend only, and should not be user-facing.
    """
    if name == "tf":
        # treat the tensorflow backend differently, to
        # isolate the import of TensorFlow
        from .tfbackend import TFBackend  # pylint: disable=import-outside-toplevel

        return TFBackend()

    if name in virtual_backends:
        # Backend is a remote device/simulator, that has a
        # defined circuit spec, but no local backend class.
        # By convention, the short name and corresponding
        # circuit spec are the same.
        backend_attrs = {"short_name": name, "circuit_spec": name}
        backend_class = type(name, (BaseBackend,), backend_attrs)
        return backend_class()

    if name in local_backends:
        backend = local_backends[name]()
        return backend

    raise ValueError("Backend '{}' is not supported.".format(name))

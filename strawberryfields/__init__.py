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
"""
.. _code:

Overview
==================

.. image:: ../_static/sfcomponents.svg
    :align: center
    :width: 90%
    :target: javascript:void(0);

|

The Strawberry Fields codebase includes a number of complementary components.
These can be separated into frontend components
and backend components (all found within the :file:`strawberryfields.backends` submodule).

Software components
-------------------

**Frontend:**

* Quantum programs: :mod:`strawberryfields.program`
* Quantum compiler engine: :mod:`strawberryfields.engine`
* Quantum operations: :mod:`strawberryfields.ops`
* Utilities: :mod:`strawberryfields.utils`
* Circuit drawer: :mod:`strawberryfields.circuitdrawer`

**Backend:**

* Backend API: :mod:`strawberryfields.backends.base`
* Quantum states: :mod:`strawberryfields.backends.states`
* Fock simulator backend: :mod:`strawberryfields.backends.fockbackend`
* Gaussian simulator backend: :mod:`strawberryfields.backends.gaussianbackend`
* Tensorflow simulator backend: :mod:`strawberryfields.backends.tfbackend`

Top-level functions
-------------------

.. autosummary::
   convert
   version

Code details
~~~~~~~~~~~~
"""
from .engine import Engine
from .io import save, load
from .program import Program, _convert as convert
from ._version import __version__

__all__ = ["Engine", "Program", "convert", "version", "save", "load"]


def version():
    r"""
    Version number of Strawberry Fields.

    Returns:
      str: package version number
    """
    return __version__


#: float: numerical value of hbar for the frontend (in the implicit units of position * momentum)
hbar = 2

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
"""
.. _code:

Overview
==================

The Strawberry Fields codebase includes a number of complementary components.
These can be separated into frontend components (:file:`engine.py`, :file:`ops.py`,
and :file:`utils.py`) and backend components (all found within the :file:`strawberryfields.backends` submodule).

Software components
-------------------

**Frontend:**

* Quantum compiler engine: :mod:`strawberryfields.engine`
* Quantum operations: :mod:`strawberryfields.ops`
* Utilities: :mod:`strawberryfields.utils`

**Backend:**

* Backend API: :mod:`strawberryfields.backends.base`
* Quantum states: :mod:`strawberryfields.backends.states`
* Fock simulator backend: :mod:`strawberryfields.backends.fockbackend`
* Gaussian simulator backend: :mod:`strawberryfields.backends.gaussianbackend`
* Tensorflow simulator backend: :mod:`strawberryfields.backends.tfbackend`

Top-level functions
-------------------

.. autosummary::
   Engine
   convert
   version

Code details
~~~~~~~~~~~~
"""

from __future__ import unicode_literals
from ._version import __version__
from .engine import Engine as _Engine
from .engine import _convert as convert

__all__ = ['Engine', 'convert', 'version']


def version():
    r"""
    Version number of Strawberry Fields.

    Returns:
      str: package version number
    """
    return __version__


def Engine(num_subsystems, **kwargs):
    r"""
    Helper function for creating an engine and associated quantum register.

    Args:
      num_subsystems (int): number of subsystems in the quantum register
    Keyword Args:
      hbar (float): The value of :math:`\hbar` to initialise the engine with, depending on the
        conventions followed. By default, :math:`\hbar=2`. See
        :ref:`conventions` for more details.

    Returns:
        (strawberryFields.engine.Engine, tuple[RegRef]): tuple containing (i) a Strawberry Fields Engine object, and (ii) a tuple of quantum register references

    """
    eng = _Engine(num_subsystems, **kwargs)
    return eng, eng.register

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
Backend specifications
======================

**Module name:** :mod:`strawberryfields.circuitspecs`

.. currentmodule:: strawberryfields.circuitspecs

This module implements the :class:`~.CircuitSpecs` class, an abstract base
data class used to store details, specifications, and compilation rules for various
families of Strawberry Fields circuits, e.g., the structure of circuits which can
be executed on particular hardware or simulator backends.

These details are used by the :class:`~.Program` class when validating and
compiling quantum programs. By querying the data class corresponding to a
requested device/backend target, the :class:`~.Program` will be able to:

1. **Validate** that the program has the correct number of modes, and consists
   of valid (or decomposable) quantum operations for that circuit class.

2. **Compile** the program to match the topology or operations supported by
   the specified circuit class, making use of allowed decompositions along the way.

To access the correct specifications dataclass, :attr:`~.backend_specs` provides
a dictionary mapping the circuit family shortname to the correct dataclass.


CircuitSpecs methods
-------------------

.. currentmodule:: strawberryfields.circuitspecs.CircuitSpecs

.. autosummary::
   modes
   local
   remote
   interactive
   primitives
   decompositions
   parameter_ranges
   graph
   circuit

Code details
^^^^^^^^^^^^
"""
from .circuit_specs import CircuitSpecs
from .base import BaseSpecs
from .chip0 import Chip0Specs
from .fock import FockSpecs
from .gaussian import GaussianSpecs
from .gbs import GBSSpecs
from .tensorflow import TFSpecs


specs = (BaseSpecs, Chip0Specs, FockSpecs, GaussianSpecs, GBSSpecs, TFSpecs)

backend_specs = {c.short_name: c for c in specs}
"""dict[str, CircuitSpecs]: dictionary mapping circuit family short_name to the corresponding class."""

__all__ = ["backend_specs", "CircuitSpecs"]

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

**Module name:** :mod:`strawberryfields.devicespecs`

.. currentmodule:: strawberryfields.devicespecs

This module implements the :class:`~.DeviceSpecs` class, an abstract base
data class used to store details and specifications of the Strawberry Fields
backend and devices.

These details are used by the :class:`~.Program` class when validating and
compiling quantum programs. By querying the data class corresponding to the
requested device/backend, the :class:`~.Program` will be able to:

1. **Validate** that the program has the correct number of modes, and consists
   of valid quantum operations for that device.

2. **Compile** the program to match the backend topology, making use
   of allowed decompositions along the way.

To access the correct specifications dataclass, :attr:`~.backend_specs` provides
a dictionary mapping the backend shortname to the correct dataclass.


DeviceSpecs methods
-------------------

.. currentmodule:: strawberryfields.devicespecs.DeviceSpecs

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
from .device_specs import DeviceSpecs
from .base import BaseSpecs
from .chip0 import Chip0Specs
from .fock import FockSpecs
from .gaussian import GaussianSpecs
from .gbs import GBSSpecs
from .tensorflow import TFSpecs


devices = (BaseSpecs, Chip0Specs, FockSpecs, GaussianSpecs, GBSSpecs, TFSpecs)

backend_specs = {c.short_name: c for c in devices}
"""dict[str, DeviceSpecs]: dictionary mapping device short_name to the corresponding class."""

__all__ = ["backend_specs", "DeviceSpecs"]

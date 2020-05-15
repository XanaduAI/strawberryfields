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
This subpackage implements the :class:`CircuitSpecs` class, an abstract base class
used to define classes or families of quantum circuits, e.g., circuits that can be executed on particular
hardware or simulator backends.

The information in the :class:`CircuitSpecs` instances is used by :meth:`.Program.compile` to validate and
compile quantum programs. By querying the :class:`CircuitSpecs` class representing the requested compilation
target, :meth:`.Program.compile` can

1. **Validate** that the Program has the correct number of modes, and consists
   of valid quantum operations in the correct topology for the targeted circuit class.

2. **Compile** the Program into an :term:`equivalent circuit` that has the topology required by the
   targeted circuit class, decomposing circuit operations as required.

Note that the compilation process is not perfect and can provide false negatives: it can admit
failure by raising a :class:`.CircuitError` even if the Program theoretically is equivalent to a
circuit that belongs in the target circuit class.

The circuit class database :attr:`circuit_db` is a dictionary mapping the circuit family
short name to the corresponding CircuitSpecs instance.
In particular, for each backend supported by Strawberry Fields the database contains a
corresponding CircuitSpecs instance with the same short name, used to validate Programs to be
executed on that backend.
"""
from .circuit_specs import CircuitSpecs
from .X8 import X8Specs, X8_01
from .X12 import X12Specs, X12_01, X12_02
from .fock import FockSpecs
from .gaussian import GaussianSpecs
from .gbs import GBSSpecs
from .tensorflow import TFSpecs
from .gaussian_unitary import GaussianUnitary

specs = (
    X8Specs,
    X8_01,
    X12Specs,
    X12_01,
    X12_02,
    FockSpecs,
    GaussianSpecs,
    GBSSpecs,
    TFSpecs,
    GaussianUnitary,
)

circuit_db = {c.short_name: c for c in specs}
"""dict[str, ~strawberryfields.circuitspecs.CircuitSpecs]: Map from circuit
family short name to the corresponding class."""

__all__ = ["circuit_db", "CircuitSpecs"] + [i.__name__ for i in specs]

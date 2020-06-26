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
This subpackage implements the :class:`Compiler` class, an abstract base class
used to define classes or families of quantum circuits, e.g., circuits that can be executed on particular
hardware or simulator backends.

The information in the :class:`Compiler` instances is used by :meth:`.Program.compile` to validate and
compile quantum programs. By querying the :class:`Compiler` class representing the requested compilation
target, :meth:`.Program.compile` can

1. **Validate** that the Program has the correct number of modes, and consists
   of valid quantum operations in the correct topology for the targeted circuit class.

2. **Compile** the Program into an :term:`equivalent circuit` that has the topology required by the
   targeted circuit class, decomposing circuit operations as required.

Note that the compilation process is not perfect and can provide false negatives: it can admit
failure by raising a :class:`.CircuitError` even if the Program theoretically is equivalent to a
circuit that belongs in the target circuit class.

The circuit class database :attr:`circuit_db` is a dictionary mapping the circuit family
short name to the corresponding Compiler instance.
In particular, for each backend supported by Strawberry Fields the database contains a
corresponding Compiler instance with the same short name, used to validate Programs to be
executed on that backend.
"""
from .compiler import Compiler, Ranges
from .xcov import Xcov
from .xunitary import Xunitary
from .fock import FockSpecs
from .gaussian import GaussianSpecs
from .gbs import GBSSpecs
from .tensorflow import TFSpecs
from .gaussian_unitary import GaussianUnitary

specs = (
    FockSpecs,
    GaussianSpecs,
    GBSSpecs,
    TFSpecs,
    GaussianUnitary,
    Xcov,
    Xunitary,
)

circuit_db = {c.short_name: c for c in specs}
"""dict[str, ~strawberryfields.compilers.Compiler]: Map from circuit
family short name to the corresponding class."""

__all__ = ["circuit_db", "Compiler", "Ranges"] + [i.__name__ for i in specs]

#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import operation


@operation(4)
def prepare_squeezing(q):
    """This operation prepares modes 0 and 1
    as squeezed states with r = -1.

    Args:
        q (register): the qumode register.
    """
    S = Sgate(-1)
    S | q[0]
    S | q[1]


@operation(3)
def circuit_op(v1, v2, q):
    """Some gates that are groups into a custom operation.

    Args:
        v1 (float): parameter for CZgate
        v2 (float): parameter for the cubic phase gate
        q (register): the qumode register
    """
    CZgate(v1) | (q[0], q[1])
    Vgate(v2) | q[1]


# initialize engine and program objects
eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
circuit = sf.Program(4)

with circuit.context as q:
    # The following operation takes no arguments
    prepare_squeezing() | q

    # another operation with 2 parameters that operates on three registers: 0, 1
    circuit_op(0.5719, 2.0603) | (q[0], q[1], q[3])

# run the engine
results = eng.run(circuit)

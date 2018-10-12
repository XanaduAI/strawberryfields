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
from strawberryfields.utils import operator


@operator(4)
def prepare_squeezing(q):
    """This operator prepares modes 0 to 4
    as squeezed states with r = -1.

    Args:
        q (register): the qumode register.
    """
    S = Sgate(-1)
    S | q[0]
    S | q[1]
    S | q[2]
    S | q[3]


@operator(3)
def circuit_op(v1, v2, q):
    """Some gates that are groups into a custom operator.

    Args:
        v1 (float): parameter for CZgate
        v2 (float): parameter for the cubic phase gate
        q (register): the qumode register.
    """
    CZgate(v1) | (q[0], q[1])
    Vgate(v2) | q[2]


# initialise the engine and register
eng, q = sf.Engine(4)

with eng:
    # The following operator takes no arguments
    prepare_squeezing() | q

    # another operator with 2 parameters that operates on three registers: 0, 1, 3
    circuit_op(0.5719, 2.0603) | (q[0], q[1], q[3])

# run the engine
state = eng.run("fock", cutoff_dim=5)

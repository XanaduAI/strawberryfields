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


@operator(ns=4)
def prepare_op(q):
    """
    :param q: This operator is applied to 4 registers
    """
    S = Sgate(-1)
    S | q[0]
    S | q[1]
    S | q[2]
    S | q[3]


@operator(ns=3)
def circuit_op(v1, v2, q):
    """
    Some gates that are groups into a custom operator
    :param v1: parameter for CZgate
    :param v2: parameter for Vgate
    :param q: it requires three registers to be passed
    """
    CZgate(v1) | (q[0], q[1])
    Vgate(v2) | q[2]


# initialise the engine and register
eng, q = sf.Engine(4)

with eng:
    # operator without arguments
    prepare_op() | q

    # another operator with 2 parameters that operates on three registers: 0, 1, 3
    circuit_op(0.5719, 2.0603) | (q[0], q[1], q[3])

# run the engine
eng.run("fock", cutoff_dim=5)

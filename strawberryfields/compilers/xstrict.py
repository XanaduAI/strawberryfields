# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Strict compiler for the X class of circuits."""
from .compiler import Compiler
from .gbs import GBS


class Xstrict(Compiler):
    """Strict compiler for the X class of circuits.

    Ensures that the program exactly matches the devices topology.
    As a result, this compiler only accepts :class:`~.ops.S2gate`, :class:`~.ops.MZgate`,
    :class:`~.ops.Rgate`, and :class:`~.ops.MeasureFock` operations.

    This compiler must be used with an X series device specification.

    **Example**

    >>> eng = sf.RemoteEngine("X8")
    >>> spec = eng.device_spec
    >>> prog.compile(device=spec, compiler="Xstrict")
    """

    short_name = "Xstrict"
    interactive = False

    primitives = {"S2gate", "MeasureFock", "MZgate", "Rgate"}

    decompositions = {}

    def compile(self, seq, registers):
        # Call the GBS compiler to do basic measurement validation.
        # The GBS compiler also merges multiple measurement commands
        # into a single MeasureFock command at the end of the circuit.
        return GBS().compile(seq, registers)

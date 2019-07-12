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
"""Circuit class specification for the chip0 class of circuits."""
import textwrap

import numpy as np

from strawberryfields.program_utils import CircuitError, Command, group_operations

from .circuit_specs import CircuitSpecs
from .gbs import GBSspecs


class Chip0Specs(CircuitSpecs):
    """Circuit specifications for the chip0 class of circuits."""

    short_name = 'chip0'
    modes = 4
    remote = True
    local = True
    interactive = True

    primitives = {"S2gate", "MeasureFock", "Rgate", "BSgate"}
    decompositions = {"Interferometer": {"mesh": "rectangular_symmetric"}, "MZgate": {}}

    circuit = textwrap.dedent(
        """\
        name template_2x2_chip0
        version 1.0
        target chip0 (shots=10)

        # for n spatial degrees, first n signal modes, then n idler modes, phase zero
        S2gate({squeezing_amplitude_0}, 0.0) | [0, 2]
        S2gate({squeezing_amplitude_1}, 0.0) | [1, 3]

        # standard 2x2 interferometer for the signal modes (the lower ones in frequency)
        Rgate({external_phase_0}) | [0]
        BSgate(pi/4, pi/2) | [0, 1]
        Rgate({internal_phase_0}) | [0]
        BSgate(pi/4, pi/2) | [0, 1]

        #duplicate the interferometer for the idler modes (the higher ones in frequency)
        Rgate({external_phase_0}) | [2]
        BSgate(pi/4, pi/2) | [2, 3]
        Rgate({internal_phase_0}) | [2]
        BSgate(pi/4, pi/2) | [2, 3]

        # final local phases
        Rgate({phi0}) | 0
        Rgate({phi0}) | 1
        Rgate({phi0}) | 2
        Rgate({phi0}) | 3

        # Measurement in Fock basis
        MeasureFock() | [0, 1, 2, 3]
        """
    )

    def compile(self, seq):
        # first do general GBS compilation to make sure
        # Fock measurements are correct
        seq = GBSSpecs().compile(seq)

        # next check for S2gates
        A, B, C = group_operations(seq, lambda x: isinstance(x, ops.S2gate))

        if not A:
            raise CircuitError('Chip0 circuits must start with S2gates.')

        # finally, combine and then decompose all unitaries
        A, B, C = group_operations(seq, lambda x: isinstance(x, (ops.Rgate, ops.BSgate)))

        U_list = []

        for cmd in B:
            if isinstance(cmd.op, ops.Rgate):
                U = 0 # expression for Rgate unitary acting on system

            elif isinstance(cmd.op, ops.BSgate):
                U = 0 # expression for BSgate unitary acting on system

            U_list.append(U)

        # finally, make sure it matches the circuit template
        seq = super().compile(seq)
        return seq

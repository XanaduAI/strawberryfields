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
from numpy.linalg import multi_dot

from strawberryfields.program_utils import CircuitError, Command, group_operations
import strawberryfields.ops as ops

from .circuit_specs import CircuitSpecs
from .gbs import GBSSpecs


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
        Rgate({phi1}) | 1
        Rgate({phi2}) | 2
        Rgate({phi3}) | 3

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

        if A:
            raise CircuitError('Chip0 circuits must start with S2gates.')

        # finally, combine and then decompose all unitaries
        A, B, C = group_operations(seq, lambda x: isinstance(x, (ops.Rgate, ops.BSgate)))

        U_list = []
        regrefs = set()

        for cmd in B:
            modes = [i.ind for i in cmd.reg]
            params = [i.x for i in cmd.op.p]
            regrefs |= set(cmd.reg)
            U = np.identity(self.modes, dtype=np.complex128)

            if isinstance(cmd.op, ops.Rgate):
                U[modes[0], modes[0]] = np.exp(1j*params[0])

            elif isinstance(cmd.op, ops.BSgate):
                t = np.cos(params[0])
                r = np.exp(1j*params[1])*np.sin(params[0])
                U[modes[0], modes[0]] = t
                U[modes[0], modes[1]] = -np.conj(r)
                U[modes[1], modes[0]] = r
                U[modes[1], modes[1]] = t

            U_list.append(U)

        U = multi_dot(U_list)

        # replace B with an interferometer
        Ucmd = Command(ops.Interferometer(U, mesh="rectangular_symmetric"), list(regrefs))
        # decompose the interferometer
        B = self.compile_sequence(B)

        # finally, make sure it matches the circuit template
        seq = super().compile(A + B + C)
        return seq

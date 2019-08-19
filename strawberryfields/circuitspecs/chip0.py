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
from scipy.linalg import block_diag

from strawberryfields.program_utils import CircuitError, Command, group_operations
import strawberryfields.ops as ops

from .circuit_specs import CircuitSpecs
from .gbs import GBSSpecs


class Chip0Specs(CircuitSpecs):
    """Circuit specifications for the chip0 class of circuits."""

    short_name = "chip0"
    modes = 4
    remote = True
    local = True
    interactive = True

    primitives = {"S2gate", "MeasureFock", "Rgate", "BSgate"}
    decompositions = {
        "Interferometer": {"mesh": "rectangular_symmetric", "drop_identity": False},
        "BipartiteGraphEmbed": {"mesh": "rectangular_symmetric", "drop_identity": False},
        "MZgate": {},
    }

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
        Rgate({final_phase_0}) | 0
        Rgate({final_phase_1}) | 1
        Rgate({final_phase_2}) | 2
        Rgate({final_phase_3}) | 3

        # Measurement in Fock basis
        MeasureFock() | [0, 1, 2, 3]
        """
    )

    def compile(self, seq, registers):
        """Try to arrange a quantum circuit into a form suitable for Chip0.

        Args:
            seq (Sequence[Command]): quantum circuit to modify
            registers (Sequence[RegRefs]): quantum registers
        Returns:
            List[Command]: modified circuit
        Raises:
            CircuitError: the circuit does not correspond to Chip0
        """
        # pylint: disable=too-many-statements,too-many-branches
        # First, check if provided sequence matches the circuit template.
        # This will avoid superfluous compilation if the user is using the
        # template directly.
        try:
            seq = super().compile(seq, registers)
        except CircuitError:
            # failed topology check. Continue to more general
            # compilation below.
            pass
        else:
            return seq

        # first do general GBS compilation to make sure
        # Fock measurements are correct
        # ---------------------------------------------
        seq = GBSSpecs().compile(seq, registers)
        A, B, C = group_operations(seq, lambda x: isinstance(x, ops.MeasureFock))

        if len(B[0].reg) != self.modes:
            raise CircuitError("All modes must be measured.")

        # Check circuit begins with two mode squeezers
        # --------------------------------------------
        A, B, C = group_operations(seq, lambda x: isinstance(x, ops.S2gate))

        if A:
            raise CircuitError("Circuits must start with two S2gates.")

        # get circuit registers
        regrefs = {q for cmd in B for q in cmd.reg}

        if len(regrefs) != self.modes:
            raise CircuitError("S2gates do not appear on the correct modes.")

        # Compile the unitary: combine and then decompose all unitaries
        # -------------------------------------------------------------
        A, B, C = group_operations(seq, lambda x: isinstance(x, (ops.Rgate, ops.BSgate)))

        # begin unitary lists for mode [0, 1] and modes [2, 3] with
        # two identity matrices. This is because multi_dot requires
        # at least two matrices in the list.
        U_list01 = [np.identity(self.modes // 2, dtype=np.complex128)] * 2
        U_list23 = [np.identity(self.modes // 2, dtype=np.complex128)] * 2

        if not B:
            # no interferometer was applied
            A, B, C = group_operations(seq, lambda x: isinstance(x, ops.S2gate))
            A = B  # move the S2gates to A
        else:
            for cmd in B:
                # calculate the unitary matrix representing each
                # rotation gate and each beamsplitter
                # Note: this is done separately on modes [0, 1]
                # and modes [2, 3]
                modes = [i.ind for i in cmd.reg]
                params = [i.x for i in cmd.op.p]
                U = np.identity(self.modes // 2, dtype=np.complex128)

                if isinstance(cmd.op, ops.Rgate):
                    m = modes[0]
                    U[m % 2, m % 2] = np.exp(1j * params[0])

                elif isinstance(cmd.op, ops.BSgate):
                    m, n = modes

                    t = np.cos(params[0])
                    r = np.exp(1j * params[1]) * np.sin(params[0])

                    U[m % 2, m % 2] = t
                    U[m % 2, n % 2] = -np.conj(r)
                    U[n % 2, m % 2] = r
                    U[n % 2, n % 2] = t

                if set(modes).issubset({0, 1}):
                    U_list01.insert(0, U)
                elif set(modes).issubset({2, 3}):
                    U_list23.insert(0, U)
                else:
                    raise CircuitError(
                        "Unitary must be applied separately to modes [0, 1] and modes [2, 3]."
                    )

        # multiply all unitaries together, to get the final
        # unitary representation on modes [0, 1] and [2, 3].
        U01 = multi_dot(U_list01)
        U23 = multi_dot(U_list23)

        # check unitaries are equal
        if not np.allclose(U01, U23):
            raise CircuitError(
                "Interferometer on modes [0, 1] must be identical to interferometer on modes [2, 3]."
            )

        U = block_diag(U01, U23)

        # replace B with an interferometer
        B = [
            Command(ops.Interferometer(U01), registers[:2]),
            Command(ops.Interferometer(U23), registers[2:]),
        ]

        # decompose the interferometer, using Mach-Zehnder interferometers
        B = self.decompose(B)

        # Do a final circuit topology check
        # ---------------------------------
        seq = super().compile(A + B + C, registers)
        return seq

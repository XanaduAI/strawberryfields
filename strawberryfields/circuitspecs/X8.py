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
"""Circuit class specification for the X8 class of circuits."""
import textwrap

import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import block_diag

from strawberryfields.decompositions import mach_zehnder
from strawberryfields.program_utils import CircuitError, Command, group_operations
from strawberryfields.parameters import par_evaluate
import strawberryfields.ops as ops

from .circuit_specs import CircuitSpecs
from .gbs import GBSSpecs


# Supporting multiple string formatting, such that the target can be replaced
# first followed by squeezing amplitude and phase values
X8_CIRCUIT = textwrap.dedent(
    """\
    name template_4x2_X8
    version 1.0
    target {target} (shots=1)

    # for n spatial degrees, first n signal modes, then n idler modes, all phases zero
    S2gate({{squeezing_amplitude_0}}, 0.0) | [0, 4]
    S2gate({{squeezing_amplitude_1}}, 0.0) | [1, 5]
    S2gate({{squeezing_amplitude_2}}, 0.0) | [2, 6]
    S2gate({{squeezing_amplitude_3}}, 0.0) | [3, 7]

    # standard 4x4 interferometer for the signal modes (the lower ones in frequency)
    # even phase indices correspond to internal Mach-Zehnder interferometer phases
    # odd phase indices correspond to external Mach-Zehnder interferometer phases
    MZgate({{phase_0}}, {{phase_1}}) | [0, 1]
    MZgate({{phase_2}}, {{phase_3}}) | [2, 3]
    MZgate({{phase_4}}, {{phase_5}}) | [1, 2]
    MZgate({{phase_6}}, {{phase_7}}) | [0, 1]
    MZgate({{phase_8}}, {{phase_9}}) | [2, 3]
    MZgate({{phase_10}}, {{phase_11}}) | [1, 2]

    # duplicate the interferometer for the idler modes (the higher ones in frequency)
    MZgate({{phase_0}}, {{phase_1}}) | [4, 5]
    MZgate({{phase_2}}, {{phase_3}}) | [6, 7]
    MZgate({{phase_4}}, {{phase_5}}) | [5, 6]
    MZgate({{phase_6}}, {{phase_7}}) | [4, 5]
    MZgate({{phase_8}}, {{phase_9}}) | [6, 7]
    MZgate({{phase_10}}, {{phase_11}}) | [5, 6]

    # add final dummy phases to allow mapping any unitary to this template (these do not
    # affect the photon number measurement)
    Rgate({{final_phase_0}}) | [0]
    Rgate({{final_phase_1}}) | [1]
    Rgate({{final_phase_2}}) | [2]
    Rgate({{final_phase_3}}) | [3]
    Rgate({{final_phase_4}}) | [4]
    Rgate({{final_phase_5}}) | [5]
    Rgate({{final_phase_6}}) | [6]
    Rgate({{final_phase_7}}) | [7]

    # measurement in Fock basis
    MeasureFock() | [0, 1, 2, 3, 4, 5, 6, 7]
    """
)


class X8Specs(CircuitSpecs):
    """Circuit specifications for the X8 class of circuits."""

    short_name = "X8"
    modes = 8
    remote = True
    local = True
    interactive = False

    sq_amplitude = 1.0

    primitives = {"S2gate", "MeasureFock", "Rgate", "BSgate", "MZgate"}
    decompositions = {
        "Interferometer": {"mesh": "rectangular_symmetric", "drop_identity": False},
        "BipartiteGraphEmbed": {"mesh": "rectangular_symmetric", "drop_identity": False},
    }

    circuit = X8_CIRCUIT.format(target=short_name)

    def compile(self, seq, registers):
        """Try to arrange a quantum circuit into a form suitable for X8.

        Args:
            seq (Sequence[Command]): quantum circuit to modify
            registers (Sequence[RegRefs]): quantum registers
        Returns:
            List[Command]: modified circuit
        Raises:
            CircuitError: the circuit does not correspond to X8
        """
        # pylint: disable=too-many-statements,too-many-branches

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

        regrefs = set()

        if B:
            # get set of circuit registers as a tuple for each S2gate
            regrefs = {(cmd.reg[0].ind, cmd.reg[1].ind) for cmd in B}

        # the set of allowed mode-tuples the S2gates must have
        allowed_modes = set(zip(range(0, 4), range(4, 8)))

        if not regrefs.issubset(allowed_modes):
            raise CircuitError("S2gates do not appear on the correct modes.")

        # ensure provided S2gates all have the allowed squeezing values
        allowed_sq_value = {(0.0, 0.0), (self.sq_amplitude, 0.0)}
        sq_params = {(float(np.round(cmd.op.p[0], 3)), float(cmd.op.p[1])) for cmd in B}

        if not sq_params.issubset(allowed_sq_value):
            wrong_params = sq_params - allowed_sq_value
            raise CircuitError(
                "Incorrect squeezing value(s) (r, phi)={}. Allowed squeezing "
                "value(s) are (r, phi)={}.".format(wrong_params, allowed_sq_value)
            )

        # determine which modes do not have input S2gates specified
        missing = allowed_modes - regrefs

        for i, j in missing:
            # insert S2gates with 0 squeezing
            seq.insert(0, Command(ops.S2gate(0, 0), [registers[i], registers[j]]))

        # Check if matches the circuit template
        # --------------------------------------------
        # This will avoid superfluous unitary compilation.
        try:
            seq = super().compile(seq, registers)
        except CircuitError:
            # failed topology check. Continue to more general
            # compilation below.
            pass
        else:
            return seq

        # Compile the unitary: combine and then decompose all unitaries
        # -------------------------------------------------------------
        A, B, C = group_operations(
            seq, lambda x: isinstance(x, (ops.Rgate, ops.BSgate, ops.MZgate))
        )

        # begin unitary lists for mode [0, 1, 2, 3] and modes [4, 5, 6, 7] with
        # two identity matrices. This is because multi_dot requires
        # at least two matrices in the list.
        U_list0 = [np.identity(self.modes // 2, dtype=np.complex128)] * 2
        U_list4 = [np.identity(self.modes // 2, dtype=np.complex128)] * 2

        if not B:
            # no interferometer was applied
            A, B, C = group_operations(seq, lambda x: isinstance(x, ops.S2gate))
            A = B  # move the S2gates to A
        else:
            for cmd in B:
                # calculate the unitary matrix representing each
                # rotation gate and each beamsplitter
                modes = [i.ind for i in cmd.reg]
                params = par_evaluate(cmd.op.p)
                U = np.identity(self.modes // 2, dtype=np.complex128)

                if isinstance(cmd.op, ops.Rgate):
                    m = modes[0]
                    U[m % 4, m % 4] = np.exp(1j * params[0])

                elif isinstance(cmd.op, ops.MZgate):
                    m, n = modes
                    U = mach_zehnder(m % 4, n % 4, params[0], params[1], self.modes // 2)

                elif isinstance(cmd.op, ops.BSgate):
                    m, n = modes

                    t = np.cos(params[0])
                    r = np.exp(1j * params[1]) * np.sin(params[0])

                    U[m % 4, m % 4] = t
                    U[m % 4, n % 4] = -np.conj(r)
                    U[n % 4, m % 4] = r
                    U[n % 4, n % 4] = t

                if set(modes).issubset({0, 1, 2, 3}):
                    U_list0.insert(0, U)
                elif set(modes).issubset({4, 5, 6, 7}):
                    U_list4.insert(0, U)
                else:
                    raise CircuitError(
                        "Unitary must be applied separately to modes [0, 1, 2, 3] and modes [4, 5, 6, 7]."
                    )

        # multiply all unitaries together, to get the final
        # unitary representation on modes [0, 1] and [2, 3].
        U0 = multi_dot(U_list0)
        U4 = multi_dot(U_list4)

        # check unitaries are equal
        if not np.allclose(U0, U4):
            raise CircuitError(
                "Interferometer on modes [0, 1, 2, 3] must be identical to interferometer on modes [4, 5, 6, 7]."
            )

        U = block_diag(U0, U4)

        # replace B with an interferometer
        B = [
            Command(ops.Interferometer(U0), registers[:4]),
            Command(ops.Interferometer(U4), registers[4:]),
        ]

        # decompose the interferometer, using Mach-Zehnder interferometers
        B = self.decompose(B)

        # Do a final circuit topology check
        # ---------------------------------
        seq = super().compile(A + B + C, registers)
        return seq


class X8_01(X8Specs):
    """Circuit specifications for the X8_01 class of circuits."""

    short_name = "X8_01"
    circuit = X8_CIRCUIT.format(target=short_name)

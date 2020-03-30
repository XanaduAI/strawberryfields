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
"""Circuit class specification for the X12 class of circuits."""
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
X12_CIRCUIT = textwrap.dedent(
    """\
    name template_6x2_X12
    version 1.0
    target {target} (shots=1)

    # for n spatial degrees, first n signal modes, then n idler modes, all phases zero
    S2gate({{squeezing_amplitude_0}}, 0.0) | [0, 6]
    S2gate({{squeezing_amplitude_1}}, 0.0) | [1, 7]
    S2gate({{squeezing_amplitude_2}}, 0.0) | [2, 8]
    S2gate({{squeezing_amplitude_3}}, 0.0) | [3, 9]
    S2gate({{squeezing_amplitude_4}}, 0.0) | [4, 10]
    S2gate({{squeezing_amplitude_5}}, 0.0) | [5, 11]

    # standard 6x6 interferometer for the signal modes (the lower ones in frequency)
    # even phase indices correspond to internal Mach-Zehnder interferometer phases
    # odd phase indices correspond to external Mach-Zehnder interferometer phases
    MZgate({{phase_0}}, {{phase_1}}) | [0, 1]
    MZgate({{phase_2}}, {{phase_3}}) | [2, 3]
    MZgate({{phase_4}}, {{phase_5}}) | [4, 5]
    MZgate({{phase_6}}, {{phase_7}}) | [1, 2]
    MZgate({{phase_8}}, {{phase_9}}) | [3, 4]

    MZgate({{phase_10}}, {{phase_11}}) | [0, 1]

    MZgate({{phase_12}}, {{phase_13}}) | [2, 3]
    MZgate({{phase_14}}, {{phase_15}}) | [4, 5]
    MZgate({{phase_16}}, {{phase_17}}) | [1, 2]
    MZgate({{phase_18}}, {{phase_19}}) | [3, 4]

    MZgate({{phase_20}}, {{phase_21}}) | [0, 1]
    MZgate({{phase_22}}, {{phase_23}}) | [2, 3]
    MZgate({{phase_24}}, {{phase_25}}) | [4, 5]
    MZgate({{phase_26}}, {{phase_27}}) | [1, 2]
    MZgate({{phase_28}}, {{phase_29}}) | [3, 4]

    # duplicate the interferometer for the idler modes (the higher ones in frequency)
    MZgate({{phase_0}}, {{phase_1}}) | [6, 7]
    MZgate({{phase_2}}, {{phase_3}}) | [8, 9]
    MZgate({{phase_4}}, {{phase_5}}) | [10, 11]
    MZgate({{phase_6}}, {{phase_7}}) | [7, 8]
    MZgate({{phase_8}}, {{phase_9}}) | [9, 10]

    MZgate({{phase_10}}, {{phase_11}}) | [6, 7]
    MZgate({{phase_12}}, {{phase_13}}) | [8, 9]
    MZgate({{phase_14}}, {{phase_15}}) | [10, 11]
    MZgate({{phase_16}}, {{phase_17}}) | [7, 8]
    MZgate({{phase_18}}, {{phase_19}}) | [9, 10]

    MZgate({{phase_20}}, {{phase_21}}) | [6, 7]
    MZgate({{phase_22}}, {{phase_23}}) | [8, 9]
    MZgate({{phase_24}}, {{phase_25}}) | [10, 11]
    MZgate({{phase_26}}, {{phase_27}}) | [7, 8]
    MZgate({{phase_28}}, {{phase_29}}) | [9, 10]

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
    Rgate({{final_phase_8}}) | [8]
    Rgate({{final_phase_9}}) | [9]
    Rgate({{final_phase_10}}) | [10]
    Rgate({{final_phase_11}}) | [11]

    # measurement in Fock basis
    MeasureFock() | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    """
)


class X12Specs(CircuitSpecs):
    """Circuit specifications for the X12 class of circuits."""

    short_name = "X12"
    modes = 12
    remote = True
    local = True
    interactive = False
    circuit = X12_CIRCUIT.format(target=short_name)

    sq_amplitude = 1.0

    primitives = {"S2gate", "MeasureFock", "Rgate", "BSgate", "MZgate"}
    decompositions = {
        "Interferometer": {"mesh": "rectangular_symmetric", "drop_identity": False},
        "BipartiteGraphEmbed": {"mesh": "rectangular_symmetric", "drop_identity": False},
    }

    def compile(self, seq, registers):
        """Try to arrange a quantum circuit into a form suitable for X12.

        Args:
            seq (Sequence[Command]): quantum circuit to modify
            registers (Sequence[RegRefs]): quantum registers
        Returns:
            List[Command]: modified circuit
        Raises:
            CircuitError: the circuit does not correspond to X12
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
        allowed_modes = set(zip(range(0, 6), range(6, 12)))

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

        # begin unitary lists for mode [0, 1, 2, 3, 4, 5] and modes [6, 7, 8, 9, 10, 11]
        # with two identity matrices. This is because multi_dot requires at
        # least two matrices in the list.
        U_list0 = [np.identity(self.modes // 2, dtype=np.complex128)] * 2
        U_list6 = [np.identity(self.modes // 2, dtype=np.complex128)] * 2

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
                    U[m % 6, m % 6] = np.exp(1j * params[0])

                elif isinstance(cmd.op, ops.MZgate):
                    m, n = modes
                    U = mach_zehnder(m % 6, n % 6, params[0], params[1], self.modes // 2)

                elif isinstance(cmd.op, ops.BSgate):
                    m, n = modes

                    t = np.cos(params[0])
                    r = np.exp(1j * params[1]) * np.sin(params[0])

                    U[m % 6, m % 6] = t
                    U[m % 6, n % 6] = -np.conj(r)
                    U[n % 6, m % 6] = r
                    U[n % 6, n % 6] = t

                if set(modes).issubset({0, 1, 2, 3, 4, 5}):
                    U_list0.insert(0, U)
                elif set(modes).issubset({6, 7, 8, 9, 10, 11}):
                    U_list6.insert(0, U)
                else:
                    raise CircuitError(
                        "Unitary must be applied separately to modes [0, 1, 2, 3, 4, 5] and modes [6, 7, 8, 9, 10, 11]."
                    )

        # multiply all unitaries together, to get the final
        # unitary representation on modes [0, 1] and [2, 3].
        U0 = multi_dot(U_list0)
        U6 = multi_dot(U_list6)

        # check unitaries are equal
        if not np.allclose(U0, U6):
            raise CircuitError(
                "Interferometer on modes [0, 1, 2, 3, 4, 5] must be identical to interferometer on modes [6, 7, 8, 9, 10, 11]."
            )

        U = block_diag(U0, U6)

        # replace B with an interferometer
        B = [
            Command(ops.Interferometer(U0), registers[:6]),
            Command(ops.Interferometer(U6), registers[6:]),
        ]

        # decompose the interferometer, using Mach-Zehnder interferometers
        B = self.decompose(B)

        # Do a final circuit topology check
        # ---------------------------------
        seq = super().compile(A + B + C, registers)
        return seq


class X12_01(X12Specs):
    """Circuit specifications for the first X12 chip."""

    short_name = "X12_01"
    circuit = X12_CIRCUIT.format(target=short_name)


class X12_02(X12Specs):
    """Circuit specifications for the second X12 chip."""

    short_name = "X12_02"
    circuit = X12_CIRCUIT.format(target=short_name)

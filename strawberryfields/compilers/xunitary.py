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
"""Circuit class specification for the X class of circuits."""

import copy

import numpy as np
from thewalrus.symplectic import expand

from strawberryfields.program_utils import CircuitError, Command, group_operations

import strawberryfields.ops as ops

from .compiler import Compiler, Ranges
from .gbs import GBSSpecs
from .gaussian_unitary import GaussianUnitary


class Xunitary(Compiler):
    """Circuit specifications for the X class of circuits."""

    short_name = "Xunitary"
    modes = None
    remote = True
    local = True
    interactive = False
    allowed_sq_ranges = Ranges([0], [1.0], variable_name="r")
    sq_amplitude = 1.0

    primitives = {
        "S2gate",
        "Sgate",
        "MeasureFock",
        "Rgate",
        "BSgate",
        "MZgate",
        "Interferometer",
    }

    decompositions = {
        "BipartiteGraphEmbed": {"mesh": "rectangular_symmetric", "drop_identity": False,},
    }

    def compile(self, seq, registers):
        # the number of modes in the provided program
        n_modes = len(registers)

        # Number of modes must be even
        if n_modes % 2 != 0:
            raise CircuitError("The X series only supports programs with an even number of modes.")
        half_n_modes = n_modes // 2
        # Call the GBS compiler to do basic measurement validation.
        # The GBS compiler also merges multiple measurement commands
        # into a single MeasureFock command at the end of the circuit.
        seq = GBSSpecs().compile(seq, registers)

        # ensure that all modes are measured
        if len(seq[-1].reg) != n_modes:
            raise CircuitError("All modes must be measured.")

        # Check circuit begins with two-mode squeezers
        # --------------------------------------------
        A, B, C = group_operations(seq, lambda x: isinstance(x, ops.S2gate))
        # If there are no two-mode squeezers add squeezers at the beginning with squeezing param equal to zero.
        if B == []:
            initS2 = [
                Command(ops.S2gate(0, 0), [registers[i], registers[i + half_n_modes]])
                for i in range(half_n_modes)
            ]
            seq = initS2 + seq
            A, B, C = group_operations(seq, lambda x: isinstance(x, ops.S2gate))

        if A != []:
            raise CircuitError("There can be no operations before the S2gates.")

        regrefs = set()

        if B:
            # get set of circuit registers as a tuple for each S2gate
            regrefs = {(cmd.reg[0].ind, cmd.reg[1].ind) for cmd in B}

        # the set of allowed mode-tuples the S2gates must have
        allowed_modes = set(zip(range(0, half_n_modes), range(half_n_modes, n_modes)))

        if not regrefs.issubset(allowed_modes):
            raise CircuitError("S2gates do not appear on the correct modes.")

        # determine which modes do not have input S2gates specified
        missing = allowed_modes - regrefs

        for i, j in missing:
            # insert S2gates with 0 squeezing
            B.insert(0, Command(ops.S2gate(0, 0), [registers[i], registers[j]]))

        sqs = [cmd.op.p[0] for cmd in B]

        # ensure provided S2gates all have the allowed squeezing values
        if not all(s in self.allowed_sq_ranges for s in sqs):
            wrong_sq_values = [np.round(s, 4) for s in sqs if s not in self.allowed_sq_ranges]
            raise CircuitError(
                "Incorrect squeezing value(s) r={}. Allowed squeezing "
                "value(s) are {}.".format(wrong_sq_values, self.allowed_sq_ranges)
            )
        # This could in principle be changed
        phases = [cmd.op.p[1] for cmd in B]
        if not np.allclose(phases, 0):
            raise CircuitError(
                "Incorrect phase value(s) phi={}. Allowed squeezing "
                "value(s) are 0.0.".format(phases)
            )

        meas_seq = [C[-1]]
        seq = GaussianUnitary().compile(C[:-1], registers)

        # extract the compiled symplectic matrix
        if seq == []:
            S = np.identity(2 * n_modes)
            used_modes = list(range(n_modes))
        else:
            S = seq[0].op.p[0]
            # determine the modes that are acted on by the symplectic transformation
            used_modes = [x.ind for x in seq[0].reg]

        if not np.allclose(S @ S.T, np.identity(len(S))):
            raise CircuitError(
                "The operations after squeezing do not correspond to an interferometer."
            )

        if len(used_modes) != n_modes:
            # The symplectic transformation acts on a subset of
            # the programs registers. We must expand the symplectic
            # matrix to one that acts on all registers.
            # simply extract the computed symplectic matrix
            S = expand(seq[0].op.p[0], used_modes, n_modes)

        U = S[:n_modes, :n_modes] - 1j * S[:n_modes, n_modes:]
        U11 = U[:half_n_modes, :half_n_modes]
        U12 = U[:half_n_modes, half_n_modes:]
        U21 = U[half_n_modes:, :half_n_modes]
        U22 = U[half_n_modes:, half_n_modes:]
        if not np.allclose(U12, 0) or not np.allclose(U21, 0):
            # Not a bipartite graph
            raise CircuitError(
                "The applied unitary cannot mix between the modes {}-{} and modes {}-{}.".format(
                    0, half_n_modes - 1, half_n_modes, n_modes - 1
                )
            )

        if not np.allclose(U11, U22):
            # Not a symmetric bipartite graph
            raise CircuitError(
                "The applied unitary on modes {}-{} must be identical to the applied unitary on modes {}-{}.".format(
                    0, half_n_modes - 1, half_n_modes, n_modes - 1
                )
            )
        U1 = ops.Interferometer(U11, mesh="rectangular_symmetric", drop_identity=False)._decompose(
            registers[:half_n_modes]
        )
        U2 = copy.deepcopy(U1)

        for Ui in U2:
            Ui.reg = [registers[r.ind + half_n_modes] for r in Ui.reg]

        return B + U1 + U2 + meas_seq

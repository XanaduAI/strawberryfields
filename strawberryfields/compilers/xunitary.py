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
# pylint: disable=too-many-branches,too-many-statements
"""General interferometer compiler for the X class of circuits."""

from collections import defaultdict
import copy

import numpy as np
from thewalrus.symplectic import expand

from strawberryfields.program_utils import CircuitError, Command, group_operations

import strawberryfields.ops as ops

from .compiler import Compiler
from .gbs import GBS
from .gaussian_unitary import GaussianUnitary


def list_duplicates(seq):
    """Returns a generator representing the duplicated values in the sequence
    mapped to the indices they appear at."""
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


class Xunitary(Compiler):
    r"""General interferometer or unitary compiler for the X class of circuits.

    This compilation routine is performed at the interferometer/unitary matrix level.

    This compiler accepts the following gates, decompositions, and measurements:

    * :class:`~.ops.S2gate`
    * :class:`~.ops.Rgate`
    * :class:`~.ops.BSgate`
    * :class:`~.ops.MZgate`
    * :class:`~.ops.Interferometer`
    * :class:`~.ops.BipartiteGraphEmbed`
    * :class:`~.ops.MeasureFock`

    All ``S2gate`` operations, if present, **must** be placed at the start of the program,
    and match the X-series topology. That is, for a device with :math:`2N` modes, the
    ``S2gate`` operations can only be applied to modes :math:`(m, m+N)`.

    Subsequent operations represent the interferometer, and may consist of any combination of
    :class:`~.ops.BSgate`, :class:`~.ops.MZgate`,
    :class:`~.ops.Interferometer`, :class:`~.ops.BipartiteGraphEmbed`, as long as the unitary
    on modes :math:`(0, 1,\dots, N-1)` is repeated on modes :math:`(N, N+1, \dots, 2N-1)`.
    The unitary will automatically be compiled to match the topology of the X-series.

    Finally, the circuit must complete with Fock measurements.

    **Example**

    The compiler may be used on its own:

    >>> prog.compile(compiler="Xunitary")

    Alternatively, it can be combined with an X series device specification to include additional
    information, such as allowed parameter ranges.

    >>> eng = sf.RemoteEngine("X8")
    >>> spec = eng.device_spec
    >>> prog.compile(device=spec, compiler="Xunitary")
    """
    short_name = "Xunitary"
    interactive = False

    primitives = {
        "S2gate",
        "MeasureFock",
        "Rgate",
        "BSgate",
        "MZgate",
        "Interferometer",
    }

    decompositions = {
        "BipartiteGraphEmbed": {
            "mesh": "rectangular_symmetric",
            "drop_identity": False,
        },
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
        seq = GBS().compile(seq, registers)

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

        # get list of circuit registers as a tuple for each S2gate
        regrefs = [(cmd.reg[0].ind, cmd.reg[1].ind) for cmd in B]

        # merge S2gates
        if len(regrefs) > half_n_modes:
            for mode, indices in list_duplicates(regrefs):
                r = 0
                phi = 0

                for k, i in enumerate(sorted(indices, reverse=True)):
                    removed_cmd = B.pop(i)
                    r += removed_cmd.op.p[0]
                    phi_new = removed_cmd.op.p[1]

                    if k > 0 and phi_new != phi:
                        raise CircuitError("Cannot merge S2gates with different phase values.")

                    phi = phi_new

                i, j = mode
                B.insert(indices[0], Command(ops.S2gate(r, phi), [registers[i], registers[j]]))

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

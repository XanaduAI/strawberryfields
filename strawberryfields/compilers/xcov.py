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
"""General state compiler for the X class of circuits."""

import copy

import numpy as np
from thewalrus.quantum import Amat
from thewalrus.symplectic import expand

import strawberryfields as sf
from strawberryfields.decompositions import takagi
from strawberryfields.program_utils import CircuitError, Command
import strawberryfields.ops as ops

from .compiler import Compiler
from .gbs import GBS
from .gaussian_unitary import GaussianUnitary


class Xcov(Compiler):
    r"""General state compiler for the X class of circuits.

    An important property of this compilation routine is that it is done at the covariance matrix
    level. This implies that one should not use it to compare the interferometers of a given circuit
    since they may differ by permutations in the unitary and the squeezing parameters.

    This compiler accepts the following gates, decompositions, and measurements:

    * :class:`~.ops.S2gate`
    * :class:`~.ops.Sgate`
    * :class:`~.ops.Rgate`
    * :class:`~.ops.BSgate`
    * :class:`~.ops.MZgate`
    * :class:`~.ops.Interferometer`
    * :class:`~.ops.BipartiteGraphEmbed`
    * :class:`~.ops.MeasureFock`

    The operations above may be provided in any combination and order, provided that the unitary is
    identical between the modes :math:`(0, 1,\dots, N-1)`and :math:`(N, N+1, \dots, 2N-1)`, and does
    not mix between these two sets of modes.

    Finally, the circuit must complete with Fock measurements.

    **Example**

    The compiler may be used on its own:

    >>> prog.compile(compiler="Xcov")

    Alternatively, it can be combined with an X series device specification to include additional
    information, such as allowed parameter ranges.

    >>> eng = sf.RemoteEngine("X8")
    >>> spec = eng.device_spec
    >>> prog.compile(device=spec, compiler="Xcov")
    """

    short_name = "Xcov"
    interactive = False

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
        "BipartiteGraphEmbed": {"mesh": "rectangular_symmetric", "drop_identity": False},
    }

    def compile(self, seq, registers):
        # the number of modes in the provided program
        n_modes = len(registers)
        half_n_modes = n_modes // 2

        # Number of modes must be even
        if n_modes % 2 != 0:
            raise CircuitError("The X series only supports programs with an even number of modes.")

        # Call the GBS compiler to do basic measurement validation.
        # The GBS compiler also merges multiple measurement commands
        # into a single MeasureFock command at the end of the circuit.
        seq = GBS().compile(seq, registers)

        # ensure that all modes are measured
        if len(seq[-1].reg) != n_modes:
            raise CircuitError("All modes must be measured.")

        meas_seq = [seq[-1]]

        if len(seq) == 1:
            # program consists only of measurements
            sq_seq = [
                Command(ops.S2gate(0), [registers[i], registers[i + half_n_modes]])
                for i in range(half_n_modes)
            ]

            U = np.identity(half_n_modes)
        else:
            # Use the GaussianUnitary compiler to compute the symplectic
            # matrix representing the Gaussian operations.
            # Note that the Gaussian unitary compiler does not accept measurements,
            # so we append the measurement separately.
            seq = GaussianUnitary().compile(seq[:-1], registers) + meas_seq

            # determine the modes that are acted on by the symplectic transformation
            used_modes = [x.ind for x in seq[0].reg]

            # Since this compiler does not allow for displacements
            # when its parameters are passed to the GaussianUnitary compiler,
            # the latter either returns a GaussianTransform + MeasureFock
            # or just MeasureFock. This is because the GaussianUnitary checks
            # if the symplectic matrix is just the identity; if so, it simply elides it

            # extract the compiled symplectic matrix
            if isinstance(seq[0].op, ops.MeasureFock):
                S = np.identity(2 * n_modes)
            else:
                S = seq[0].op.p[0]

            if len(used_modes) != n_modes:
                # The symplectic transformation acts on a subset of
                # the programs registers. We must expand the symplectic
                # matrix to one that acts on all registers.
                # simply extract the computed symplectic matrix
                S = expand(seq[0].op.p[0], used_modes, n_modes)

            # Construct the covariance matrix of the state.
            # Note that hbar is a global variable that is set by the user
            cov = (sf.hbar / 2) * S @ S.T

            # Construct the A matrix
            A = Amat(cov, hbar=sf.hbar)

            # Construct the adjacency matrix represented by the A matrix.
            # This must be an weighted, undirected bipartite graph. That is,
            # B00 = B11 = 0 (no edges between the two vertex sets 0 and 1),
            # and B01 == B10.T (undirected edges between the two vertex sets).
            B = A[:n_modes, :n_modes]
            B00 = B[:half_n_modes, :half_n_modes]
            B01 = B[:half_n_modes, half_n_modes:]
            B10 = B[half_n_modes:, :half_n_modes]
            B11 = B[half_n_modes:, half_n_modes:]

            # Perform unitary validation to ensure that the
            # applied unitary is valid.

            if not np.allclose(B00, 0) or not np.allclose(B11, 0):
                # Not a bipartite graph
                raise CircuitError(
                    "The applied unitary cannot mix between the modes {}-{} and modes {}-{}.".format(
                        0, half_n_modes - 1, half_n_modes, n_modes - 1
                    )
                )

            if not np.allclose(B01, B10):
                # Not a symmetric bipartite graph
                raise CircuitError(
                    "The applied unitary on modes {}-{} must be identical to the applied unitary on modes {}-{}.".format(
                        0, half_n_modes - 1, half_n_modes, n_modes - 1
                    )
                )

            # Now that the unitary has been validated, perform the Takagi decomposition
            # to determine the constituent two-mode squeezing and interferometer
            # parameters.
            sqs, U = takagi(B01)
            sqs = np.arctanh(sqs)

            # Convert the squeezing values into a sequence of S2gate commands
            sq_seq = [
                Command(ops.S2gate(sqs[i]), [registers[i], registers[i + half_n_modes]])
                for i in range(half_n_modes)
            ]

        # NOTE: at some point, it might make sense to add a keyword argument to this method,
        # to allow the user to specify if they want the interferometers decomposed or not.

        # Convert the unitary into a sequence of MZgate and Rgate commands on the signal modes
        U1 = ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False)._decompose(
            registers[:half_n_modes]
        )
        U2 = copy.deepcopy(U1)

        for Ui in U2:
            Ui.reg = [registers[r.ind + half_n_modes] for r in Ui.reg]

        return sq_seq + U1 + U2 + meas_seq

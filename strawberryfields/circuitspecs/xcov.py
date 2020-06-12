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
from thewalrus.quantum import Amat
from thewalrus.symplectic import expand

import strawberryfields as sf
from strawberryfields.decompositions import takagi
from strawberryfields.program_utils import CircuitError, Command
import strawberryfields.ops as ops

from .circuit_specs import CircuitSpecs, Ranges
from .gbs import GBSSpecs
from .gaussian_unitary import GaussianUnitary


class Xcov(CircuitSpecs):
    """Circuit specifications for the X class of circuits.

    An important property of this compilation routine is that it is done at the covariance matrix level.
    This implies that one should not use it to compare the interferometers of a given circuit since they may
    differ by permutations in the unitary and the squeezing parameters.
    """

    short_name = "Xcov"
    modes = None
    remote = True
    local = True
    interactive = False
    allowed_sq_ranges = Ranges([0], [1.0], variable_name="r")

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

    def compile(self, seq, registers, allow_imperfections=False):
        # the number of modes in the provided program
        n_modes = len(registers)

        # Number of modes must be even
        if n_modes % 2 != 0:
            raise CircuitError("The X series only supports programs with an even number of modes.")

        # Call the GBS compiler to do basic measurement validation.
        # The GBS compiler also merges multiple measurement commands
        # into a single MeasureFock command at the end of the circuit.
        seq = GBSSpecs().compile(seq, registers)

        # ensure that all modes are measured
        if len(seq[-1].reg) != n_modes:
            raise CircuitError("All modes must be measured.")

        # Use the GaussianUnitary compiler to compute the symplectic
        # matrix representing the Gaussian operations.
        # Note that the Gaussian unitary compiler does not accept measurements,
        # so we append the measurement separately.
        meas_seq = [seq[-1]]
        seq = GaussianUnitary().compile(seq[:-1], registers) + meas_seq

        # determine the modes that are acted on by the symplectic transformation
        used_modes = [x.ind for x in seq[0].reg]

        # extract the compiled symplectic matrix
        S = seq[0].op.p[0]

        if len(used_modes) != n_modes:
            # The symplectic transformation acts on a subset of
            # the programs registers. We must expand the symplectic
            # matrix to one that acts on all registers.
            # simply extract the computed symplectic matrix
            S = expand(seq[0].op.p[0], used_modes, n_modes)

        half_n_modes = n_modes // 2

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

        # ensure provided S2gates all have the allowed squeezing values
        if not all(s in self.allowed_sq_ranges for s in sqs):
            wrong_sq_values = [np.round(s, 4) for s in sqs if s not in self.allowed_sq_ranges]
            raise CircuitError(
                "Incorrect squeezing value(s) r={}. Allowed squeezing "
                "value(s) are {}.".format(wrong_sq_values, self.allowed_sq_ranges)
            )

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

        if allow_imperfections is False:
        	return sq_seq + U1 + U2 + meas_seq

        dark_counts = 0.01
        end_to_end_transmission = 0.5
        loss_seq = [
            Command(ops.LossChannel(end_to_end_transmission), [registers[i]])
            for i in range(n_modes)
        ]
        meas_seq_dc = [
            Command(ops.MeasureFock(dark_counts=[dark_counts] * n_modes), [registers[i] for i in range(n_modes)])
        ]
        return sq_seq + U1 + U2 + loss_seq + meas_seq_dc
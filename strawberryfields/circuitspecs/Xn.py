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
"""Circuit class specification for the Xn class of circuits."""

import numpy as np
from thewalrus.quantum import Amat
from strawberryfields.decompositions import takagi
from strawberryfields.program_utils import CircuitError, Command, group_operations
import strawberryfields.ops as ops
from .circuit_specs import CircuitSpecs
from .gbs import GBSSpecs
from .gaussian_unitary import GaussianUnitary


class XnSpecs(CircuitSpecs):
    """Circuit specifications for the X8 class of circuits."""

    short_name = "Xn"
    modes = 8
    remote = True
    local = True
    interactive = False
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
    # This could be all Gaussian operations except displacement, but OK.
    decompositions = {
        "BipartiteGraphEmbed": {
            "mesh": "rectangular_symmetric",
            "drop_identity": False,
        },
    }

    def compile(self, seq, registers):
        seq = GBSSpecs().compile(seq, registers)  # wonder if this is necessary
        A, B, C = group_operations(seq, lambda x: isinstance(x, ops.MeasureFock))
        if C != []:
            raise ValueError("There should be no operations after MeasureFock")

        if len(B[0].reg) != self.modes:
            raise CircuitError("All modes must be measured.")
        tmp_seq = seq[:-1]  # This must be the measurements
        meas_seq = [seq[-1]]
        seq = GaussianUnitary().compile(tmp_seq, registers) + meas_seq
        S = seq[0].op.p[0]
        n_modes = len(S) // 2
        half_n_modes = n_modes // 2
        hbar = 2
        cov = (hbar / 2) * S @ S.T
        A = Amat(cov, hbar=hbar)
        B = A[:n_modes, :n_modes]
        B00 = B[:half_n_modes, :half_n_modes]
        B01 = B[:half_n_modes, half_n_modes:]
        B10 = B[half_n_modes:, :half_n_modes]
        B11 = B[half_n_modes:, half_n_modes:]

        if not np.allclose(B00, 0) or not np.allclose(B11, 0):
            raise ValueError(
                "The Gaussian state being prepared does not correspond to a bipartite graph"
            )
        if not np.allclose(B01, B10):
            raise ValueError(
                "The Gaussian state being prepared does not correspond to a symmetric bipartite graph"
            )
        sqs, U = takagi(B01)
        sqs = np.arctanh(sqs)
        atol = 1e-3
        for s in sqs:
            if not np.allclose(s, 0, atol=atol, rtol=0) and not np.allclose(
                s, self.sq_amplitude, atol=atol, rtol=0
            ):
                raise ValueError(
                    "The squeezing parameters necessary for state preparation are outside the range"
                )

        # logic to convert back to SF commands that match the chip
        sq_seq = [
            Command(ops.S2gate(sqs[i]), [registers[i], registers[i + half_n_modes]])
            for i in range(half_n_modes)
        ]
        unitary_seq = [
            Command(
                ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=True),
                registers[:half_n_modes],
            ),
            Command(
                ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=True),
                registers[half_n_modes:],
            ),
        ]
        return sq_seq + unitary_seq + meas_seq

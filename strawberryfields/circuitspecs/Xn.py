from thewalrus.quantum import Amat
from thewalrus.symplectic import expand
from strawberryfields.decompositions import takagi
from .circuit_specs import CircuitSpecs
from strawberryfields.program_utils import CircuitError, Command, group_operations
import strawberryfields.ops as ops
from .gbs import GBSSpecs
from .gaussian_unitary import GaussianUnitary
import numpy as np


class XnSpecs(CircuitSpecs):
    """Circuit specifications for the X8 class of circuits."""

    short_name = "Xn"
    modes = 8
    remote = True
    local = True
    interactive = False
    sq_amplitude = 1.0

    primitives = {"S2gate", "Sgate", "MeasureFock", "Rgate", "BSgate", "MZgate", "Interferometer"}

    decompositions = {
        "BipartiteGraphEmbed": {"mesh": "rectangular_symmetric", "drop_identity": False},
    }

    def compile(self, seq, registers):
        n_modes = len(registers)
        seq = GBSSpecs().compile(seq, registers)
        A, B, C = group_operations(seq, lambda x: isinstance(x, ops.MeasureFock))

        if len(B[0].reg) != self.modes:
            raise CircuitError("All modes must be measured.")
        tmp_seq = seq[:-1]  # This must be the measurements
        meas_seq = [seq[-1]]
        seq = GaussianUnitary().compile(tmp_seq, registers) + meas_seq
        used_modes = [x.ind for x in seq[0].reg]
        if len(used_modes) == n_modes:
            S = seq[0].op.p[0]
        else:
            S = expand(seq[0].op.p[0], used_modes, n_modes)

        half_n_modes = n_modes // 2
        hbar = 2
        cov = (hbar / 2) * S @ S.T
        A = Amat(cov, hbar=hbar)
        B = A[:n_modes, :n_modes]
        B00 = B[:half_n_modes, :half_n_modes]
        B01 = B[:half_n_modes, half_n_modes:]
        B10 = B[half_n_modes:, :half_n_modes]
        B11 = B[half_n_modes:, half_n_modes:]
        # print(B01)
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
        print(sq_seq + unitary_seq + meas_seq)
        return sq_seq + unitary_seq + meas_seq

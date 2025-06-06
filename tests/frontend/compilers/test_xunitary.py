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
r"""Unit tests for the Xunitary compiler"""
import textwrap

import pytest
import numpy as np
import networkx as nx

import blackbird

import strawberryfields as sf
import strawberryfields.ops as ops

from strawberryfields.program_utils import CircuitError, program_equivalence
from strawberryfields.io import to_program
from strawberryfields.utils import random_interferometer
from strawberryfields.compilers import Compiler

from thewalrus.symplectic import two_mode_squeezing, expand

pytestmark = pytest.mark.frontend


SQ_AMPLITUDE = 1
"""float: the allowed squeezing amplitude"""


class DummyCircuit(Compiler):
    """Dummy circuit used to instantiate
    the abstract base class"""

    modes = 8
    remote = False
    local = True
    interactive = True
    primitives = {"S2gate", "MeasureFock", "Rgate", "BSgate", "MZgate"}
    decompositions = {"Interferometer": {}}


X8_CIRCUIT = textwrap.dedent(
    """\
    name template_4x2_X8
    version 1.0
    target X8_01 (shots=1)
    # for n spatial degrees, first n signal modes, then n idler modes, all phases zero
    S2gate({squeezing_amplitude_0}, 0.0) | [0, 4]
    S2gate({squeezing_amplitude_1}, 0.0) | [1, 5]
    S2gate({squeezing_amplitude_2}, 0.0) | [2, 6]
    S2gate({squeezing_amplitude_3}, 0.0) | [3, 7]
    # standard 4x4 interferometer for the signal modes (the lower ones in frequency)
    # even phase indices correspond to internal Mach-Zehnder interferometer phases
    # odd phase indices correspond to external Mach-Zehnder interferometer phases
    MZgate({phase_0}, {phase_1}) | [0, 1]
    MZgate({phase_2}, {phase_3}) | [2, 3]
    MZgate({phase_4}, {phase_5}) | [1, 2]
    MZgate({phase_6}, {phase_7}) | [0, 1]
    MZgate({phase_8}, {phase_9}) | [2, 3]
    MZgate({phase_10}, {phase_11}) | [1, 2]
    # duplicate the interferometer for the idler modes (the higher ones in frequency)
    MZgate({phase_0}, {phase_1}) | [4, 5]
    MZgate({phase_2}, {phase_3}) | [6, 7]
    MZgate({phase_4}, {phase_5}) | [5, 6]
    MZgate({phase_6}, {phase_7}) | [4, 5]
    MZgate({phase_8}, {phase_9}) | [6, 7]
    MZgate({phase_10}, {phase_11}) | [5, 6]
    # add final dummy phases to allow mapping any unitary to this template (these do not
    # affect the photon number measurement)
    Rgate({final_phase_0}) | [0]
    Rgate({final_phase_1}) | [1]
    Rgate({final_phase_2}) | [2]
    Rgate({final_phase_3}) | [3]
    Rgate({final_phase_4}) | [4]
    Rgate({final_phase_5}) | [5]
    Rgate({final_phase_6}) | [6]
    Rgate({final_phase_7}) | [7]
    # measurement in Fock basis
    MeasureFock() | [0, 1, 2, 3, 4, 5, 6, 7]
    """
)


class TestXCompilation:
    """Tests for compilation using the X8_01 circuit specification"""

    def test_exact_template(self, tol):
        """Test compilation works for the exact circuit"""
        bb = blackbird.loads(X8_CIRCUIT)
        bb = bb(
            squeezing_amplitude_0=SQ_AMPLITUDE,
            squeezing_amplitude_1=SQ_AMPLITUDE,
            squeezing_amplitude_2=SQ_AMPLITUDE,
            squeezing_amplitude_3=SQ_AMPLITUDE,
            phase_0=0,
            phase_1=1,
            phase_2=2,
            phase_3=3,
            phase_4=4,
            phase_5=5,
            phase_6=6,
            phase_7=7,
            phase_8=8,
            phase_9=9,
            phase_10=10,
            phase_11=11,
            final_phase_0=1.24,
            final_phase_1=-0.54,
            final_phase_2=4.12,
            final_phase_3=0,
            final_phase_4=1.24,
            final_phase_5=-0.54,
            final_phase_6=4.12,
            final_phase_7=0,
        )

        expected = to_program(bb)
        res = expected.compile(compiler="Xunitary")
        assert program_equivalence(res, expected, atol=tol, compare_params=False)

    @pytest.mark.parametrize("num_pairs", [4, 5, 6, 7])
    def test_not_all_modes_measured(self, num_pairs):
        """Test exceptions raised if not all modes are measured"""
        prog = sf.Program(2 * num_pairs)
        U = random_interferometer(num_pairs)
        with prog.context as q:
            for i in range(num_pairs):
                ops.S2gate(SQ_AMPLITUDE) | (q[i], q[i + num_pairs])
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs))
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs, 2 * num_pairs))
            ops.MeasureFock() | (q[0], q[num_pairs])
        with pytest.raises(CircuitError, match="All modes must be measured"):
            prog.compile(compiler="Xunitary")

    @pytest.mark.parametrize("num_pairs", [4, 5, 6, 7])
    def test_no_s2gates(self, num_pairs, tol):
        """Test identity S2gates are inserted when no S2gates
        are provided."""
        prog = sf.Program(2 * num_pairs)
        U = random_interferometer(num_pairs)

        with prog.context as q:
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs))
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs, 2 * num_pairs))
            ops.MeasureFock() | q

        expected = sf.Program(2 * num_pairs)

        with expected.context as q:
            for i in range(num_pairs):
                ops.S2gate(0) | (q[i], q[i + num_pairs])
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs))
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs, 2 * num_pairs))
            ops.MeasureFock() | q

        # with pytest.raises(CircuitError, match="There can be no operations before the S2gates."):
        res = prog.compile(compiler="Xunitary")
        # with pytest.raises(CircuitError, match="There can be no operations before the S2gates."):
        expected = expected.compile(compiler="Xunitary")
        assert program_equivalence(res, expected, atol=tol)

    @pytest.mark.parametrize("num_pairs", [4, 5, 6, 7])
    def test_missing_s2gates(self, num_pairs, tol):
        """Test identity S2gates are inserted when some (but not all)
        S2gates are included."""
        prog = sf.Program(2 * num_pairs)
        U = random_interferometer(num_pairs)
        assert num_pairs > 3
        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[num_pairs + 1])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[num_pairs + 3])
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs))
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs, 2 * num_pairs))
            ops.MeasureFock() | q

        expected = sf.Program(2 * num_pairs)

        with expected.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[num_pairs + 1])
            ops.S2gate(0) | (q[0], q[num_pairs + 0])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[num_pairs + 3])
            ops.S2gate(0) | (q[2], q[num_pairs + 2])
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs))
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs, 2 * num_pairs))
            ops.MeasureFock() | q

        res = prog.compile(compiler="Xunitary")
        expected = expected.compile(compiler="Xunitary")
        assert program_equivalence(res, expected, atol=tol)

    @pytest.mark.parametrize("num_pairs", [4, 5, 6, 7])
    def test_incorrect_s2gate_modes(self, num_pairs):
        """Test exceptions raised if S2gates do not appear on correct modes"""
        prog = sf.Program(2 * num_pairs)
        U = random_interferometer(num_pairs)
        n_modes = 2 * num_pairs
        half_n_modes = n_modes // 2
        with prog.context as q:
            for i in range(num_pairs):
                ops.S2gate(SQ_AMPLITUDE) | (q[2 * i], q[2 * i + 1])
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs))
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs, 2 * num_pairs))
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="S2gates do not appear on the correct modes."):
            res = prog.compile(compiler="Xunitary")

    @pytest.mark.parametrize("num_pairs", [4, 5, 6, 7])
    def test_s2gate_repeated_modes_half_squeezing(self, num_pairs):
        """Test that squeezing gates are correctly merged"""
        prog = sf.Program(2 * num_pairs)
        U = random_interferometer(num_pairs)

        with prog.context as q:

            ops.S2gate(SQ_AMPLITUDE / 2) | (q[0], q[0 + num_pairs])
            for i in range(1, num_pairs):
                ops.S2gate(SQ_AMPLITUDE) | (q[i], q[i + num_pairs])
            ops.S2gate(SQ_AMPLITUDE / 2) | (q[0], q[0 + num_pairs])
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs))
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs, 2 * num_pairs))
            ops.MeasureFock() | q

        res = prog.compile(compiler="Xunitary")
        assert np.allclose(res.circuit[0].op.p[0], SQ_AMPLITUDE)

    @pytest.mark.parametrize("num_pairs", [4, 5, 6, 7])
    def test_s2gate_repeated_modes_wrong_phase(self, num_pairs):
        """Test that an error is raised if there is an attempt to merge squeezing gates
        with different phase values"""
        prog = sf.Program(2 * num_pairs)
        U = random_interferometer(num_pairs)

        with prog.context as q:

            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[0 + num_pairs])
            for i in range(1, num_pairs):
                ops.S2gate(SQ_AMPLITUDE) | (q[i], q[i + num_pairs])
            ops.S2gate(SQ_AMPLITUDE / 2, 0.5) | (q[0], q[0 + num_pairs])
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs))
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs, 2 * num_pairs))
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="Cannot merge S2gates with different phase values"):
            prog.compile(compiler="Xunitary")

    def test_gates_compile(self):
        """Test that combinations of MZgates, Rgates, and BSgates
        correctly compile."""
        prog = sf.Program(8)

        def unitary(q):
            ops.MZgate(0.5, 0.1) | (q[0], q[1])
            ops.BSgate(0.1, 0.2) | (q[1], q[2])
            ops.Rgate(0.4) | q[0]

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[7])

            unitary(q[:4])
            unitary(q[4:])
            ops.MeasureFock() | q

        prog.compile(compiler="Xunitary")

    def test_no_unitary(self, tol):
        """Test compilation works with no unitary provided"""
        prog = sf.Program(8)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[7])
            ops.MeasureFock() | q

        res = prog.compile(compiler="Xunitary")
        expected = sf.Program(8)

        with expected.context as q:
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[3], q[7])

            # corresponds to an identity on modes [0, 1, 2, 3]
            # This can be easily seen from below by noting that:
            # MZ(pi, pi) = R(0) = I
            # MZ(pi, 0) @ MZ(pi, 0) = I
            # [R(pi) \otimes I] @ MZ(pi, 0) = I
            ops.MZgate(np.pi, 0) | (q[0], q[1])
            ops.MZgate(np.pi, 0) | (q[2], q[3])
            ops.MZgate(np.pi, np.pi) | (q[1], q[2])
            ops.MZgate(np.pi, np.pi) | (q[0], q[1])
            ops.MZgate(np.pi, 0) | (q[2], q[3])
            ops.MZgate(np.pi, np.pi) | (q[1], q[2])
            ops.Rgate(np.pi) | (q[0])
            ops.Rgate(0) | (q[1])
            ops.Rgate(0) | (q[2])
            ops.Rgate(0) | (q[3])

            # corresponds to an identity on modes [4, 5, 6, 7]
            ops.MZgate(np.pi, 0) | (q[4], q[5])
            ops.MZgate(np.pi, 0) | (q[6], q[7])
            ops.MZgate(np.pi, np.pi) | (q[5], q[6])
            ops.MZgate(np.pi, np.pi) | (q[4], q[5])
            ops.MZgate(np.pi, 0) | (q[6], q[7])
            ops.MZgate(np.pi, np.pi) | (q[5], q[6])
            ops.Rgate(np.pi) | (q[4])
            ops.Rgate(0) | (q[5])
            ops.Rgate(0) | (q[6])
            ops.Rgate(0) | (q[7])

            ops.MeasureFock() | q

        assert program_equivalence(res, expected, atol=tol, compare_params=False)

        # double check that the applied symplectic is correct

        # remove the Fock measurements
        res.circuit = res.circuit[:-1]

        # extract the Gaussian symplectic matrix
        O = res.compile(compiler="gaussian_unitary").circuit[0].op.p[0]

        # construct the expected symplectic matrix corresponding
        # to just the initial two mode squeeze gates
        S = two_mode_squeezing(SQ_AMPLITUDE, 0)
        num_modes = 8
        expected = np.identity(2 * num_modes)
        for i in range(num_modes // 2):
            expected = expand(S, [i, i + num_modes // 2], num_modes) @ expected
        # Note that the comparison has to be made at the level of covariance matrices
        # Not at the level of symplectic matrices
        assert np.allclose(O @ O.T, expected @ expected.T, atol=tol)

    @pytest.mark.parametrize("num_pairs", [4])
    def test_interferometers(self, num_pairs, tol):
        """Test that the compilation correctly decomposes the interferometer using
        the rectangular_symmetric mesh"""
        prog = sf.Program(2 * num_pairs)
        U = random_interferometer(num_pairs)

        with prog.context as q:
            for i in range(0, num_pairs):
                ops.S2gate(SQ_AMPLITUDE) | (q[i], q[i + num_pairs])
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs))
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs, 2 * num_pairs))
            ops.MeasureFock() | q

        res = prog.compile(compiler="Xunitary")

        expected = sf.Program(2 * num_pairs)

        with expected.context as q:
            for i in range(0, num_pairs):
                ops.S2gate(SQ_AMPLITUDE) | (q[i], q[i + num_pairs])
            ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False) | tuple(
                q[i] for i in range(num_pairs)
            )
            ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False) | tuple(
                q[i] for i in range(num_pairs, 2 * num_pairs)
            )
            ops.MeasureFock() | q

        expected = expected.compile(compiler=DummyCircuit())
        # Note that since DummyCircuit() has a hard coded limit of 8 modes we only check for this number
        assert program_equivalence(res, expected, atol=tol, compare_params=False)

    @pytest.mark.parametrize("num_pairs", [4, 5, 6, 7])
    def test_unitaries_do_not_match(self, num_pairs):
        """Test exception raised if the unitary applied to modes [0, 1, 2, 3] is
        different to the unitary applied to modes [4, 5, 6, 7]"""
        prog = sf.Program(2 * num_pairs)
        U = random_interferometer(num_pairs)

        with prog.context as q:
            for i in range(0, num_pairs):
                ops.S2gate(SQ_AMPLITUDE) | (q[i], q[i + num_pairs])
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs))
            ops.Interferometer(U) | tuple(q[i] for i in range(num_pairs, 2 * num_pairs))
            ops.BSgate() | (q[2], q[3])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="The applied unitary on modes"):
            prog.compile(compiler="Xunitary")

    @pytest.mark.parametrize("num_pairs", [4, 5, 6, 7])
    def test_unitary_too_large(self, num_pairs):
        """Test exception raised if the unitary is applied to more
        than just modes [0, 1, 2, 3, ..., num_pairs-1] and [num_pairs, num_pairs+1, ..., 2*num_pairs-1].
        """
        prog = sf.Program(2 * num_pairs)
        U = random_interferometer(2 * num_pairs)

        with prog.context as q:
            for i in range(0, num_pairs):
                ops.S2gate(SQ_AMPLITUDE) | (q[i], q[i + num_pairs])
            ops.Interferometer(U) | q
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="The applied unitary cannot mix between the modes"):
            res = prog.compile(compiler="Xunitary")

    def test_odd_number_of_modes(self):
        """Test error is raised when xstrict is called with odd number of modes"""
        prog = sf.Program(1)

        with pytest.raises(
            CircuitError, match="The X series only supports programs with an even number of modes."
        ):
            res = prog.compile(compiler="Xunitary")

    def test_operation_before_squeezing(self):
        """Test error is raised when an operation is passed before the S2gates"""
        prog = sf.Program(2)
        with prog.context as q:
            ops.BSgate() | (q[0], q[1])
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[1])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="There can be no operations before the S2gates."):
            prog.compile(compiler="Xunitary")

    def test_identity_program(self, tol):
        """Test that compilation correctly works if the gate consists only of measurements"""
        prog = sf.Program(4)

        with prog.context as q:
            ops.MeasureFock() | q

        res = prog.compile(compiler="Xunitary")

        # remove the Fock measurements
        res.circuit = res.circuit[:-1]

        # extract the Gaussian symplectic matrix
        res = res.compile(compiler="gaussian_unitary")
        assert not res.circuit

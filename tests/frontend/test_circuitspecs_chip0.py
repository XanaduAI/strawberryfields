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
r"""Unit tests for the CircuitSpec class"""
import textwrap

import pytest
import numpy as np

import blackbird

import strawberryfields as sf
from strawberryfields.program_utils import CircuitError
import strawberryfields.ops as ops
from strawberryfields.io import to_program
from strawberryfields.utils import random_interferometer
from strawberryfields.circuitspecs.chip0 import Chip0Specs, CircuitSpecs


pytestmark = pytest.mark.frontend

np.random.seed(42)


class DummyCircuit(CircuitSpecs):
    """Dummy circuit used to instantiate
    the abstract base class"""
    modes = 4
    remote = False
    local = True
    interactive = True
    primitives = {"S2gate", "MeasureFock", "Rgate", "BSgate"}
    decompositions = {"Interferometer": {}, "MZgate": {}}


class TestChip0Compilation:
    """Tests for compilation using the Chip0 circuit specification"""

    def test_exact_template(self):
        """Test compilation works for the exact circuit"""
        bb = blackbird.loads(Chip0Specs.circuit)
        bb = bb(
            squeezing_amplitude_0=0.43,
            squeezing_amplitude_1=0.65,
            external_phase_0=0.54,
            internal_phase_0=-0.23,
            phi0=1.24,
            phi1=-0.54,
            phi2=4.12,
            phi3=0,
        )

        expected = to_program(bb)
        res = expected.compile("chip0")

        for cmd1, cmd2 in zip(res.circuit, expected.circuit):
            assert cmd1.op.__class__ == cmd2.op.__class__
            assert [p.x for p in cmd1.op.p] == [p.x for p in cmd2.op.p]
            assert [i.ind for i in cmd1.reg] == [i.ind for i in cmd2.reg]

    def test_not_all_modes_measured(self):
        """Test exceptions raised if not all modes are measured"""
        prog = sf.Program(4)
        U = random_interferometer(2)

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.S2gate(0.5) | (q[1], q[3])
            ops.Interferometer(U) | (q[0], q[1])
            ops.Interferometer(U) | (q[2], q[3])
            ops.MeasureFock() | (q[0], q[1])

        with pytest.raises(CircuitError, match="All modes must be measured"):
            res = prog.compile("chip0")

    def test_no_s2gates(self):
        """Test exceptions raised if no S2gates are present"""
        prog = sf.Program(4)
        U = random_interferometer(2)

        with prog.context as q:
            ops.Interferometer(U) | (q[0], q[1])
            ops.Interferometer(U) | (q[2], q[3])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="must start with two S2gates"):
            res = prog.compile("chip0")

    def test_incorrect_s2gates(self):
        """Test exceptions raised if S2gates are on incorrect modes"""
        prog = sf.Program(4)
        U = random_interferometer(2)

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.Interferometer(U) | (q[0], q[1])
            ops.Interferometer(U) | (q[2], q[3])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="S2gates placed on the incorrect modes"):
            res = prog.compile("chip0")

    def test_no_unitary(self):
        """Test compilation works with no unitary provided"""
        prog = sf.Program(4)

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.S2gate(0.5) | (q[1], q[3])
            ops.MeasureFock() | q

        res = prog.compile("chip0")

        expected = sf.Program(4)

        with expected.context as q:
            ops.S2gate(0.5, 0) | (q[0], q[2])
            ops.S2gate(0.5, 0) | (q[1], q[3])

            # corresponds to an identity on modes [0, 1]
            ops.Rgate(0) | q[0]
            ops.BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            ops.Rgate(np.pi) | q[0]
            ops.BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            ops.Rgate(np.pi) | q[0]
            ops.Rgate(0) | q[1]

            # corresponds to an identity on modes [2, 3]
            ops.Rgate(0) | q[2]
            ops.BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            ops.Rgate(np.pi) | q[2]
            ops.BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            ops.Rgate(np.pi) | q[2]
            ops.Rgate(0) | q[3]

            ops.MeasureFock() | (q[0], q[3], q[1], q[2])

        for cmd1, cmd2 in zip(res.circuit, expected.circuit):
            assert cmd1.op.__class__ == cmd2.op.__class__
            assert [p.x for p in cmd1.op.p] == [p.x for p in cmd2.op.p]
            assert [i.ind for i in cmd1.reg] == [i.ind for i in cmd2.reg]

    def test_interferometers(self):
        """Test interferometers correctly decompose to MZ gates"""
        prog = sf.Program(4)
        U = random_interferometer(2)

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.S2gate(0.5) | (q[1], q[3])
            ops.Interferometer(U) | (q[0], q[1])
            ops.Interferometer(U) | (q[2], q[3])
            ops.MeasureFock() | q

        res = prog.compile("chip0")

        prog = sf.Program(4)

        with prog.context as q:
            ops.S2gate(0.5, 0) | (q[0], q[2])
            ops.S2gate(0.5, 0) | (q[1], q[3])
            ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False) | (q[0], q[1])
            ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False) | (q[2], q[3])
            ops.MeasureFock() | q

        expected = prog.compile(DummyCircuit())

        for cmd1, cmd2 in zip(res.circuit, expected.circuit):
            assert cmd1.op.__class__ == cmd2.op.__class__
            assert [p.x for p in cmd1.op.p] == [p.x for p in cmd2.op.p]
            assert [i.ind for i in cmd1.reg] == [i.ind for i in cmd2.reg]

    def test_unitaries_do_not_match(self):
        """Test exception raised if the unitary applied to modes [0, 1] is
        different to the unitary applied to modes [2, 3]"""
        prog = sf.Program(4)
        U = random_interferometer(2)

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.S2gate(0.5) | (q[1], q[3])
            ops.Interferometer(U) | (q[0], q[1])
            ops.Interferometer(U) | (q[2], q[3])
            ops.BSgate() | (q[2], q[3])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="must be identical to interferometer"):
            res = prog.compile("chip0")

    def test_unitary_too_large(self):
        """Test exception raised if the unitary is applied to more
        than just modes [0, 1] and [2, 3]."""
        prog = sf.Program(4)
        U = random_interferometer(4)

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.S2gate(0.5) | (q[1], q[3])
            ops.Interferometer(U) | q
            ops.BSgate() | (q[2], q[3])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="must be applied separately"):
            res = prog.compile("chip0")

    def test_mach_zehnder_interferometers(self):
        """Test Mach-Zehnder gates correctly compile"""
        prog = sf.Program(4)
        phi = 0.543
        theta = -1.654

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.S2gate(0.5) | (q[1], q[3])
            ops.MZgate(phi, theta) | (q[0], q[1])
            ops.MZgate(phi, theta) | (q[2], q[3])
            ops.MeasureFock() | q

        res = prog.compile("chip0")

        expected = sf.Program(4)

        with expected.context as q:
            ops.S2gate(0.5, 0) | (q[0], q[2])
            ops.S2gate(0.5, 0) | (q[1], q[3])

            # corresponds to MZgate(phi, theta) on modes [0, 1]
            ops.Rgate(phi) | q[0]
            ops.BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            ops.Rgate(theta+2*np.pi) | q[0]
            ops.BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            ops.Rgate(0) | q[0]
            ops.Rgate(0) | q[1]

            # corresponds to MZgate(phi, theta) on modes [2, 3]
            ops.Rgate(phi) | q[2]
            ops.BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            ops.Rgate(theta+2*np.pi) | q[2]
            ops.BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            ops.Rgate(0) | q[2]
            ops.Rgate(0) | q[3]

            ops.MeasureFock() | (q[0], q[3], q[1], q[2])

        for cmd1, cmd2 in zip(res.circuit, expected.circuit):
            assert cmd1.op.__class__ == cmd2.op.__class__
            assert np.allclose([p.x for p in cmd1.op.p], [p.x for p in cmd2.op.p])
            assert [i.ind for i in cmd1.reg] == [i.ind for i in cmd2.reg]

    def test_50_50_BSgate(self):
        """Test 50-50 BSgates correctly compile"""
        prog = sf.Program(4)

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.S2gate(0.5) | (q[1], q[3])
            ops.BSgate() | (q[0], q[1])
            ops.BSgate() | (q[2], q[3])
            ops.MeasureFock() | q

        res = prog.compile("chip0")

        expected = sf.Program(4)

        with expected.context as q:
            ops.S2gate(0.5, 0) | (q[0], q[2])
            ops.S2gate(0.5, 0) | (q[1], q[3])

            # corresponds to BSgate() on modes [0, 1]
            ops.Rgate(0) | (q[0])
            ops.BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            ops.Rgate(3*np.pi/2) | (q[0])
            ops.BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            ops.Rgate(3*np.pi/4) | (q[0])
            ops.Rgate(-np.pi/4) | (q[1])

            # corresponds to BSgate() on modes [2, 3]
            ops.Rgate(0) | (q[2])
            ops.BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            ops.Rgate(3*np.pi/2) | (q[2])
            ops.BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            ops.Rgate(3*np.pi/4) | (q[2])
            ops.Rgate(-np.pi/4) | (q[3])

            ops.MeasureFock() | (q[0], q[3], q[1], q[2])

        for cmd1, cmd2 in zip(res.circuit, expected.circuit):
            assert cmd1.op.__class__ == cmd2.op.__class__
            assert [p.x for p in cmd1.op.p] == [p.x for p in cmd2.op.p]
            assert [i.ind for i in cmd1.reg] == [i.ind for i in cmd2.reg]

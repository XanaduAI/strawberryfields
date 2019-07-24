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
import networkx as nx

import blackbird

import strawberryfields as sf
import strawberryfields.ops as ops

from strawberryfields.program_utils import CircuitError, list_to_DAG
from strawberryfields.io import to_program
from strawberryfields.utils import random_interferometer
from strawberryfields.circuitspecs.chip0 import Chip0Specs, CircuitSpecs


pytestmark = pytest.mark.frontend

np.random.seed(42)


def program_equivalence(prog1, prog2, compare_params=True, atol=1e-6, rtol=0):
    r"""Checks if two programs are equivalent.

    This function converts the program lists into directed acyclic graphs,
    and runs the NetworkX `is_isomorphic` graph function in order
    to determine if the two programs are equivalent.

    Note: when checking for parameter equality between two parameters
    :math:`a` and :math:`b`, we use the following formula:

    .. math:: |a - b| \leq (\texttt{atol} + \texttt{rtol}\times|b|)

    Args:
        prog1 (strawberryfields.program.Program): quantum program
        prog2 (strawberryfields.program.Program): quantum program
        compare_params (bool): Set to ``False`` to turn of comparing
            program parameters; equivalency will only take into
            account the operation order.
        atol (float): the absolute tolerance parameter for checking
            quantum operation parameter equality
        rtol (float): the relative tolerance parameter for checking
            quantum operation parameter equality

    Returns:
        bool: returns ``True`` if two quantum programs are equivalent
    """
    DAG1 = list_to_DAG(prog1.circuit)
    DAG2 = list_to_DAG(prog2.circuit)

    circuit = []
    for G in [DAG1, DAG2]:
        # relabel the DAG nodes to integers
        circuit.append(nx.convert_node_labels_to_integers(G))

        # add node attributes to store the operation name and parameters
        name_mapping = {i: n.op.__class__.__name__ for i, n in enumerate(G.nodes())}
        parameter_mapping = {i: [j.x for j in n.op.p] for i, n in enumerate(G.nodes())}

        # CXgate and BSgate are not symmetric wrt to permuting the order of the two
        # modes it acts on; i.e., the order of the wires matter
        wire_mapping = {}
        for i, n in enumerate(G.nodes()):
            if n.op.__class__.__name__ == "CXgate":
                if np.allclose(n.op.p[0], 0):
                    # if the CXgate parameter is 0, wire order doesn't matter
                    wire_mapping[i] = 0
                else:
                    # if the CXgate parameter is not 0, order matters
                    wire_mapping[i] = [j.ind for j in n.reg]

            elif n.op.__class__.__name__ == "BSgate":
                if np.allclose([j.x % np.pi for j in n.op.p], [np.pi/4, np.pi/2]):
                    # if the beamsplitter is *symmetric*, then the order of the
                    # wires does not matter.
                    wire_mapping[i] = 0
                else:
                    # beamsplitter is not symmetric, order matters
                    wire_mapping[i] = [j.ind for j in n.reg]

            else:
                # not a CXgate or a BSgate, order of wires doesn't matter
                wire_mapping[i] = 0

        # TODO: at the moment, we do not check for whether an empty
        # wire will match an operation with trivial parameters.
        # Maybe we can do this in future, but this is a subgraph
        # isomorphism problem and much harder.

        nx.set_node_attributes(circuit[-1], name_mapping, name="name")
        nx.set_node_attributes(circuit[-1], parameter_mapping, name="p")
        nx.set_node_attributes(circuit[-1], wire_mapping, name="w")

    def node_match(n1, n2):
        """Returns True if both nodes have the same name and
        same parameters, within a certain tolerance"""
        name_match = n1["name"] == n2["name"]
        p_match = np.allclose(n1["p"], n2["p"], atol=atol, rtol=rtol)
        wire_match = n1["w"] == n2["w"]

        if compare_params:
            return name_match and p_match and wire_match

        return name_match and wire_match

    # check if circuits are equivalent
    return nx.is_isomorphic(circuit[0], circuit[1], node_match)


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

    def test_exact_template(self, tol):
        """Test compilation works for the exact circuit"""
        bb = blackbird.loads(Chip0Specs.circuit)
        bb = bb(
            squeezing_amplitude_0=0.43,
            squeezing_amplitude_1=0.65,
            external_phase_0=0.54,
            internal_phase_0=-0.23,
            final_phase_0=1.24,
            final_phase_1=-0.54,
            final_phase_2=4.12,
            final_phase_3=0,
        )

        expected = to_program(bb)
        res = expected.compile("chip0")

        assert program_equivalence(res, expected, atol=tol)

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
        """Test exceptions raised if S2gates do not appear on correct modes"""
        prog = sf.Program(4)
        U = random_interferometer(2)

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.Interferometer(U) | (q[0], q[1])
            ops.Interferometer(U) | (q[2], q[3])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="S2gates do not appear on the correct modes"):
            res = prog.compile("chip0")

    def test_no_unitary(self, tol):
        """Test compilation works with no unitary provided"""
        prog = sf.Program(4)

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.S2gate(0.5) | (q[1], q[3])
            ops.MeasureFock() | q

        res = prog.compile("chip0")

        for cmd in res.circuit:
            print(cmd)

        expected = sf.Program(4)

        with expected.context as q:
            ops.S2gate(0.5, 0) | (q[0], q[2])
            ops.S2gate(0.5, 0) | (q[1], q[3])

            # corresponds to an identity on modes [0, 1]
            ops.Rgate(0) | q[0]
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
            ops.Rgate(np.pi) | q[0]
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
            ops.Rgate(np.pi) | q[0]
            ops.Rgate(0) | q[1]

            # corresponds to an identity on modes [2, 3]
            ops.Rgate(0) | q[2]
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[2], q[3])
            ops.Rgate(np.pi) | q[2]
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[3], q[2])
            ops.Rgate(np.pi) | q[2]
            ops.Rgate(0) | q[3]

            ops.MeasureFock() | q

        assert program_equivalence(res, expected, atol=tol)

    def test_interferometers(self, tol):
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

        expected = sf.Program(4)

        with expected.context as q:
            ops.S2gate(0.5, 0) | (q[0], q[2])
            ops.S2gate(0.5, 0) | (q[1], q[3])
            ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False) | (q[0], q[1])
            ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False) | (q[2], q[3])
            ops.MeasureFock() | q

        expected = expected.compile(DummyCircuit())

        assert program_equivalence(res, expected, atol=tol)

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

    def test_mach_zehnder_interferometers(self, tol):
        """Test Mach-Zehnder gates correctly compile"""
        prog = sf.Program(4)
        phi = 0.543
        theta = -1.654

        with prog.context as q:
            ops.S2gate(0.5) | (q[0], q[2])
            ops.S2gate(0.5) | (q[3], q[1])
            ops.MZgate(phi, theta) | (q[0], q[1])
            ops.MZgate(phi, theta) | (q[2], q[3])
            ops.MeasureFock() | q

        res = prog.compile("chip0")

        expected = sf.Program(4)

        with expected.context as q:
            ops.S2gate(0.5, 0) | (q[0], q[2])
            ops.S2gate(0.5, 0) | (q[1], q[3])

            # corresponds to MZgate(phi, theta) on modes [0, 1]
            ops.Rgate(np.mod(phi, 2*np.pi)) | q[0]
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
            ops.Rgate(np.mod(theta, 2*np.pi)) | q[0]
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
            ops.Rgate(0) | q[0]
            ops.Rgate(0) | q[1]

            # corresponds to MZgate(phi, theta) on modes [2, 3]
            ops.Rgate(np.mod(phi, 2*np.pi)) | q[2]
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[2], q[3])
            ops.Rgate(np.mod(theta, 2*np.pi)) | q[2]
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[2], q[3])
            ops.Rgate(0) | q[2]
            ops.Rgate(0) | q[3]

            ops.MeasureFock() | (q[0], q[3], q[1], q[2])

        assert program_equivalence(res, expected, atol=tol)

    def test_50_50_BSgate(self, tol):
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
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
            ops.Rgate(3 * np.pi / 2) | (q[0])
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
            ops.Rgate(3 * np.pi / 4) | (q[0])
            ops.Rgate(-np.pi / 4) | (q[1])

            # corresponds to BSgate() on modes [2, 3]
            ops.Rgate(0) | (q[2])
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[2], q[3])
            ops.Rgate(3 * np.pi / 2) | (q[2])
            ops.BSgate(np.pi / 4, np.pi / 2) | (q[2], q[3])
            ops.Rgate(3 * np.pi / 4) | (q[2])
            ops.Rgate(-np.pi / 4) | (q[3])

            ops.MeasureFock() | q

        assert program_equivalence(res, expected, atol=tol)

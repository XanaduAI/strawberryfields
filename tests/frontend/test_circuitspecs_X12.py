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
r"""Unit tests for the CircuitSpec class"""
import textwrap

import pytest
import numpy as np
import networkx as nx
from scipy.linalg import block_diag

import blackbird

import strawberryfields as sf
import strawberryfields.ops as ops

from strawberryfields.parameters import par_evaluate
from strawberryfields.program_utils import CircuitError, list_to_DAG
from strawberryfields.io import to_program
from strawberryfields.utils import random_interferometer
from strawberryfields.circuitspecs.X12 import X12_01, X12_02, CircuitSpecs
from strawberryfields.circuitspecs.gaussian_unitary import GaussianUnitary


pytestmark = pytest.mark.frontend

np.random.seed(42)

SQ_AMPLITUDE = 1
"""float: the allowed squeezing amplitude"""


def TMS(r, phi):
    """Two-mode squeezing.

    Args:
        r (float): squeezing magnitude
        phi (float): rotation parameter

    Returns:
        array: symplectic transformation matrix
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ch = np.cosh(r)
    sh = np.sinh(r)

    S = np.array(
        [
            [ch, cp * sh, 0, sp * sh],
            [cp * sh, ch, sp * sh, 0],
            [0, sp * sh, ch, -cp * sh],
            [sp * sh, 0, -cp * sh, ch],
        ]
    )

    return S


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
        parameter_mapping = {i: par_evaluate(n.op.p) for i, n in enumerate(G.nodes())}

        # CXgate and BSgate are not symmetric wrt permuting the order of the two
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
                if np.allclose([j % np.pi for j in par_evaluate(n.op.p)], [np.pi/4, np.pi/2]):
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

    modes = 12
    remote = False
    local = True
    interactive = True
    primitives = {"S2gate", "MeasureFock", "Rgate", "BSgate", "MZgate"}
    decompositions = {"Interferometer": {}}

@pytest.mark.parametrize("chip", [X12_01, X12_02])
class TestX12Compilation:
    """Tests for compilation using the Chip2 circuit specification"""

    def test_exact_template(self, chip, tol):
        """Test compilation works for the exact circuit"""
        bb = blackbird.loads(chip.circuit)
        bb = bb(
            squeezing_amplitude_0=SQ_AMPLITUDE,
            squeezing_amplitude_1=SQ_AMPLITUDE,
            squeezing_amplitude_2=SQ_AMPLITUDE,
            squeezing_amplitude_3=SQ_AMPLITUDE,
            squeezing_amplitude_4=SQ_AMPLITUDE,
            squeezing_amplitude_5=SQ_AMPLITUDE,
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
            phase_12=12,
            phase_13=13,
            phase_14=14,
            phase_15=15,
            phase_16=16,
            phase_17=17,
            phase_18=18,
            phase_19=19,
            phase_20=20,
            phase_21=21,
            phase_22=22,
            phase_23=23,
            phase_24=24,
            phase_25=25,
            phase_26=26,
            phase_27=27,
            phase_28=28,
            phase_29=29,
            final_phase_0=1.24,
            final_phase_1=-0.54,
            final_phase_2=4.12,
            final_phase_3=0,
            final_phase_4=1.24,
            final_phase_5=-0.54,
            final_phase_6=4.12,
            final_phase_7=0,
            final_phase_8=1.24,
            final_phase_9=-0.54,
            final_phase_10=4.12,
            final_phase_11=0,

        )

        expected = to_program(bb)
        res = expected.compile(chip.short_name)

        assert program_equivalence(res, expected, atol=tol)

    def test_not_all_modes_measured(self, chip):
        """Test exceptions raised if not all modes are measured"""
        prog = sf.Program(12)
        U = random_interferometer(6)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[6])
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[7])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[8])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[9])
            ops.S2gate(SQ_AMPLITUDE) | (q[4], q[10])
            ops.S2gate(SQ_AMPLITUDE) | (q[5], q[11])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3], q[4], q[5])
            ops.Interferometer(U) | (q[6], q[7], q[8], q[9], q[10], q[11])
            ops.MeasureFock() | (q[0], q[1])

        with pytest.raises(CircuitError, match="All modes must be measured"):
            res = prog.compile(chip.short_name)

    def test_no_s2gates(self, chip, tol):
        """Test identity S2gates are inserted when no S2gates
        are provided."""
        prog = sf.Program(12)
        U = random_interferometer(6)

        with prog.context as q:
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3], q[4], q[5])
            ops.Interferometer(U) | (q[6], q[7], q[8], q[9], q[10], q[11])
            ops.MeasureFock() | q

        expected = sf.Program(12)

        with expected.context as q:
            ops.S2gate(0) | (q[0], q[6])
            ops.S2gate(0) | (q[1], q[7])
            ops.S2gate(0) | (q[2], q[8])
            ops.S2gate(0) | (q[3], q[9])
            ops.S2gate(0) | (q[4], q[10])
            ops.S2gate(0) | (q[5], q[11])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3], q[4], q[5])
            ops.Interferometer(U) | (q[6], q[7], q[8], q[9], q[10], q[11])
            ops.MeasureFock() | q

        res = prog.compile(chip.short_name)
        expected = expected.compile(chip.short_name)
        assert program_equivalence(res, expected, atol=tol)

    def test_missing_s2gates(self, chip, tol):
        """Test identity S2gates are inserted when some (but not all)
        S2gates are included."""
        prog = sf.Program(12)
        U = random_interferometer(6)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[7])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[9])
            ops.S2gate(SQ_AMPLITUDE) | (q[5], q[11])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3], q[4], q[5])
            ops.Interferometer(U) | (q[6], q[7], q[8], q[9], q[10], q[11])
            ops.MeasureFock() | q

        expected = sf.Program(12)

        with expected.context as q:
            ops.S2gate(0) | (q[0], q[6])
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[7])
            ops.S2gate(0) | (q[2], q[8])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[9])
            ops.S2gate(0) | (q[4], q[10])
            ops.S2gate(SQ_AMPLITUDE) | (q[5], q[11])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3], q[4], q[5])
            ops.Interferometer(U) | (q[6], q[7], q[8], q[9], q[10], q[11])
            ops.MeasureFock() | q

        res = prog.compile(chip.short_name)
        expected = expected.compile(chip.short_name)
        assert program_equivalence(res, expected, atol=tol)

    def test_incorrect_s2gate_modes(self, chip):
        """Test exceptions raised if S2gates do not appear on correct modes"""
        prog = sf.Program(12)
        U = random_interferometer(6)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[1])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[3])
            ops.S2gate(SQ_AMPLITUDE) | (q[4], q[8])
            ops.S2gate(SQ_AMPLITUDE) | (q[6], q[9])
            ops.S2gate(SQ_AMPLITUDE) | (q[8], q[10])
            ops.S2gate(SQ_AMPLITUDE) | (q[10], q[11])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3], q[4], q[5])
            ops.Interferometer(U) | (q[6], q[7], q[8], q[9], q[10], q[11])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="S2gates do not appear on the correct modes"):
            res = prog.compile(chip.short_name)

    def test_incorrect_s2gate_params(self, chip):
        """Test exceptions raised if S2gates have illegal parameters"""
        prog = sf.Program(12)
        U = random_interferometer(6)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[6])
            ops.S2gate(0) | (q[1], q[7])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[8])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[9])
            ops.S2gate(SQ_AMPLITUDE) | (q[4], q[10])
            ops.S2gate(SQ_AMPLITUDE+0.1) | (q[5], q[11])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3], q[4], q[5])
            ops.Interferometer(U) | (q[6], q[7], q[8], q[9], q[10], q[11])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match=r"Incorrect squeezing value\(s\) \(r, phi\)={\(1.1, 0.0\)}"):
            res = prog.compile(chip.short_name)

    def test_s2gate_repeated_modes(self, chip):
        """Test exceptions raised if S2gates are repeated"""
        prog = sf.Program(12)
        U = random_interferometer(6)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[6])
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[6])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3], q[4], q[5])
            ops.Interferometer(U) | (q[6], q[7], q[8], q[9], q[10], q[11])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="incompatible topology."):
            res = prog.compile(chip.short_name)

    def test_gates_compile(self, chip):
        """Test that combinations of MZgates, Rgates, and BSgates
        correctly compile."""
        prog = sf.Program(12)
        U = random_interferometer(6)

        def unitary(q):
            ops.MZgate(0.5, 0.1) | (q[0], q[1])
            ops.BSgate(0.1, 0.2) | (q[1], q[2])
            ops.Rgate(0.4) | q[0]

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[6])
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[7])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[8])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[9])
            ops.S2gate(SQ_AMPLITUDE) | (q[4], q[10])
            ops.S2gate(SQ_AMPLITUDE) | (q[5], q[11])
            unitary(q[:6])
            unitary(q[6:])
            ops.MeasureFock() | q

        res = prog.compile(chip.short_name)

    def test_no_unitary(self, chip, tol):
        """Test compilation works with no unitary provided"""
        prog = sf.Program(12)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[6])
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[7])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[8])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[9])
            ops.S2gate(SQ_AMPLITUDE) | (q[4], q[10])
            ops.S2gate(SQ_AMPLITUDE) | (q[5], q[11])
            ops.MeasureFock() | q

        res = prog.compile(chip.short_name)
        expected = sf.Program(12)

        with expected.context as q:
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[0], q[6])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[1], q[7])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[2], q[8])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[3], q[9])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[4], q[10])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[5], q[11])

            # corresponds to an identity on modes [0, 1, 2, 3, 4, 5]
            # This can be easily seen from below by noting that:
            # MZ(pi, pi) = R(0) = I
            # MZ(pi, 0) @ MZ(pi, 0) = I
            # [R(pi) \otimes I] @ MZ(pi, 0) = I
            ops.MZgate(np.pi, 0) | (q[0], q[1])
            ops.MZgate(np.pi, 0) | (q[2], q[3])
            ops.MZgate(np.pi, 0) | (q[4], q[5])
            ops.MZgate(np.pi, np.pi) | (q[1], q[2])
            ops.MZgate(np.pi, np.pi) | (q[3], q[4])

            ops.MZgate(np.pi, 0) | (q[0], q[1])
            ops.MZgate(np.pi, 0) | (q[2], q[3])
            ops.MZgate(np.pi, 0) | (q[4], q[5])
            ops.MZgate(np.pi, np.pi) | (q[1], q[2])
            ops.MZgate(np.pi, np.pi) | (q[3], q[4])

            ops.MZgate(np.pi, 0) | (q[0], q[1])
            ops.MZgate(np.pi, np.pi) | (q[2], q[3])
            ops.MZgate(np.pi, np.pi) | (q[4], q[5])
            ops.MZgate(np.pi, np.pi) | (q[1], q[2])
            ops.MZgate(np.pi, np.pi) | (q[3], q[4])

            ops.Rgate(np.pi) | (q[0])
            ops.Rgate(0) | (q[1])
            ops.Rgate(0) | (q[2])
            ops.Rgate(0) | (q[3])
            ops.Rgate(0) | (q[4])
            ops.Rgate(0) | (q[5])

            # corresponds to an identity on modes [6, 7, 8, 9, 10, 11]
            ops.MZgate(np.pi, 0) | (q[6], q[7])
            ops.MZgate(np.pi, 0) | (q[8], q[9])
            ops.MZgate(np.pi, 0) | (q[10], q[11])
            ops.MZgate(np.pi, np.pi) | (q[7], q[8])
            ops.MZgate(np.pi, np.pi) | (q[9], q[10])

            ops.MZgate(np.pi, 0) | (q[6], q[7])
            ops.MZgate(np.pi, 0) | (q[8], q[9])
            ops.MZgate(np.pi, 0) | (q[10], q[11])
            ops.MZgate(np.pi, np.pi) | (q[7], q[8])
            ops.MZgate(np.pi, np.pi) | (q[9], q[10])

            ops.MZgate(np.pi, 0) | (q[6], q[7])
            ops.MZgate(np.pi, np.pi) | (q[8], q[9])
            ops.MZgate(np.pi, np.pi) | (q[10], q[11])
            ops.MZgate(np.pi, np.pi) | (q[7], q[8])
            ops.MZgate(np.pi, np.pi) | (q[9], q[10])

            ops.Rgate(np.pi) | (q[6])
            ops.Rgate(0) | (q[7])
            ops.Rgate(0) | (q[8])
            ops.Rgate(0) | (q[9])
            ops.Rgate(0) | (q[10])
            ops.Rgate(0) | (q[11])

            ops.MeasureFock() | q

        # Check that the applied symplectic is correct

        # remove the Fock measurements
        res.circuit = res.circuit[:-1]

        # extract the Gaussian symplectic matrix
        O = res.compile("gaussian_unitary").circuit[0].op.p[0]

        # construct the expected symplectic matrix corresponding
        # to just the initial two mode squeeze gates
        S = TMS(SQ_AMPLITUDE, 0)

        expected = np.zeros([2*12, 2*12])
        l = 12 // 2
        ch = np.cosh(SQ_AMPLITUDE) * np.identity(l)
        sh = np.sinh(SQ_AMPLITUDE) * np.identity(l)
        zh = np.zeros([l, l])
        expected = np.block([[ch, sh, zh, zh], [sh, ch, zh, zh], [zh, zh, ch, -sh], [zh, zh, -sh, ch]])

        assert np.allclose(O, expected, atol=tol)

    def test_interferometers(self, chip, tol):
        """Test that the compilation correctly decomposes the interferometer using
        the rectangular_symmetric mesh"""
        prog = sf.Program(12)
        U = random_interferometer(6)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[0], q[6])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[1], q[7])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[2], q[8])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[3], q[9])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[4], q[10])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[5], q[11])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3], q[4], q[5])
            ops.Interferometer(U) | (q[6], q[7], q[8], q[9], q[10], q[11])
            ops.MeasureFock() | q

        res = prog.compile(chip.short_name)

        expected = sf.Program(12)

        with expected.context as q:
            for i, j in np.arange(12).reshape(2, -1).T:
                ops.S2gate(SQ_AMPLITUDE, 0) | (q[i], q[j])

            ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False) | (q[0], q[1], q[2], q[3], q[4], q[5])
            ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False) | (q[6], q[7], q[8], q[9], q[10], q[11])
            ops.MeasureFock() | q

        expected = expected.compile(DummyCircuit())

        assert program_equivalence(res, expected, atol=tol)

    def test_unitaries_do_not_match(self, chip):
        """Test exception raised if the unitary applied to modes [0, 1, 2, 3] is
        different to the unitary applied to modes [4, 5, 6, 7]"""
        prog = sf.Program(12)
        U = random_interferometer(6)

        with prog.context as q:
            for i, j in np.arange(12).reshape(2, -1).T:
                ops.S2gate(SQ_AMPLITUDE, 0) | (q[i], q[j])

            ops.Interferometer(U) | q[:6]
            ops.Interferometer(U) | q[6:]
            ops.BSgate() | (q[2], q[3])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="must be identical to interferometer"):
            res = prog.compile(chip.short_name)

    def test_unitary_too_large(self, chip):
        """Test exception raised if the unitary is applied to more
        than just modes [0, 1, 2, 3] and [4, 5, 6, 7]."""
        prog = sf.Program(12)
        U = random_interferometer(12)

        with prog.context as q:
            for i, j in np.arange(12).reshape(2, -1).T:
                ops.S2gate(SQ_AMPLITUDE, 0) | (q[i], q[j])
            ops.Interferometer(U) | q
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="must be applied separately"):
            res = prog.compile(chip.short_name)

    def test_allow_imperfections_warning(self):
        """Test the X12 compile method warns when allowing imperfections"""
        prog = sf.Program(2)
        with prog.context as q:
            ops.MeasureFock() | q

        with pytest.warns(UserWarning, match="compile method does not currently support imperfections"):
            prog.compile("X12", allow_imperfections=True)
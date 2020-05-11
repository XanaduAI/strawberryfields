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
from strawberryfields.circuitspecs.X8 import X8_01, CircuitSpecs
from strawberryfields.circuitspecs.x import Ranges

from thewalrus.symplectic import two_mode_squeezing, expand

pytestmark = pytest.mark.frontend

np.random.seed(42)

SQ_AMPLITUDE = 1
"""float: the allowed squeezing amplitude"""


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

    modes = 8
    remote = False
    local = True
    interactive = True
    primitives = {"S2gate", "MeasureFock", "Rgate", "BSgate", "MZgate"}
    decompositions = {"Interferometer": {}}


class TestXCompilation:
    """Tests for compilation using the X8_01 circuit specification"""

    def test_exact_template(self, tol):
        """Test compilation works for the exact circuit"""
        bb = blackbird.loads(X8_01.circuit)
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
        res = expected.compile("X")

        assert program_equivalence(res, expected, atol=tol, compare_params=False)

    def test_not_all_modes_measured(self):
        """Test exceptions raised if not all modes are measured"""
        prog = sf.Program(8)
        U = random_interferometer(4)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[7])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | (q[0], q[1])

        with pytest.raises(CircuitError, match="All modes must be measured"):
            res = prog.compile("X")

    def test_no_s2gates(self, tol):
        """Test identity S2gates are inserted when no S2gates
        are provided."""
        prog = sf.Program(8)
        U = random_interferometer(4)

        with prog.context as q:
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | q

        expected = sf.Program(8)

        with expected.context as q:
            ops.S2gate(0) | (q[0], q[4])
            ops.S2gate(0) | (q[1], q[5])
            ops.S2gate(0) | (q[2], q[6])
            ops.S2gate(0) | (q[3], q[7])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | q

        res = prog.compile("X")
        expected = expected.compile("X")
        assert program_equivalence(res, expected, atol=tol)

    def test_missing_s2gates(self, tol):
        """Test identity S2gates are inserted when some (but not all)
        S2gates are included."""
        prog = sf.Program(8)
        U = random_interferometer(4)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[7])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | q

        expected = sf.Program(8)

        with expected.context as q:
            ops.S2gate(0) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[5])
            ops.S2gate(0) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[7])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | q

        res = prog.compile("X")
        expected = expected.compile("X")
        assert program_equivalence(res, expected, atol=tol)

    def test_incorrect_s2gate_modes(self):
        """Test exceptions raised if S2gates do not appear on correct modes"""
        prog = sf.Program(8)
        U = random_interferometer(4)
        n_modes = 8
        half_n_modes = n_modes // 2
        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[1])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[3])
            ops.S2gate(SQ_AMPLITUDE) | (q[4], q[5])
            ops.S2gate(SQ_AMPLITUDE) | (q[7], q[6])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="The applied unitary cannot mix between the modes {}-{} and modes {}-{}.".format(
                    0, half_n_modes - 1, half_n_modes, n_modes - 1
                )):
            res = prog.compile("X")

    def test_incorrect_s2gate_params(self):
        """Test exceptions raised if S2gates have illegal parameters"""
        prog = sf.Program(8)
        U = random_interferometer(4)
        wrong_sq_value = [np.round(SQ_AMPLITUDE+0.1, 4)]
        allowed_sq_ranges = Ranges([0], [1.0], variable_name="r")
        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[4])
            ops.S2gate(0) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE+0.1) | (q[3], q[7])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match=r"Incorrect squeezing val"):
            res = prog.compile("X")

    def test_s2gate_repeated_modes(self):
        """Test exceptions raised if S2gates are repeated"""
        prog = sf.Program(8)
        U = random_interferometer(4)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[4])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match=r"Incorrect squeezing val"):
            res = prog.compile("X")

    def test_s2gate_repeated_modes_half_squeezing(self):
        """Test exceptions raised if S2gates are repeated"""
        prog = sf.Program(8)
        U = random_interferometer(4)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE/2) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE/2) | (q[0], q[4])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | q

        res = prog.compile("X")

    def test_gates_compile(self):
        """Test that combinations of MZgates, Rgates, and BSgates
        correctly compile."""
        prog = sf.Program(8)
        U = random_interferometer(4)

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

        res = prog.compile("X")

    def test_no_unitary(self, tol):
        """Test compilation works with no unitary provided"""
        prog = sf.Program(8)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE) | (q[3], q[7])
            ops.MeasureFock() | q

        res = prog.compile("X")
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
        O = res.compile("gaussian_unitary").circuit[0].op.p[0]

        # construct the expected symplectic matrix corresponding
        # to just the initial two mode squeeze gates
        S = two_mode_squeezing(SQ_AMPLITUDE, 0)
        num_modes = 8
        expected = np.identity(2*num_modes)
        for i in range(num_modes//2):
            expected =  expand(S, [i, i+num_modes//2], num_modes) @ expected
        # Note that the comparison has to be made at the level of covariance matrices
        # Not at the level of symplectic matrices
        assert np.allclose(O @ O.T, expected @ expected.T, atol=tol)


    def test_interferometers(self, tol):
        """Test that the compilation correctly decomposes the interferometer using
        the rectangular_symmetric mesh"""
        prog = sf.Program(8)
        U = random_interferometer(4)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[3], q[7])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | q

        res = prog.compile("X")

        expected = sf.Program(8)

        with expected.context as q:
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[3], q[7])
            ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U, mesh="rectangular_symmetric", drop_identity=False) | (q[4], q[5], q[6], q[7])
            ops.MeasureFock() | q

        expected = expected.compile(DummyCircuit())

        assert program_equivalence(res, expected, atol=tol, compare_params=False)

    def test_unitaries_do_not_match(self):
        """Test exception raised if the unitary applied to modes [0, 1, 2, 3] is
        different to the unitary applied to modes [4, 5, 6, 7]"""
        prog = sf.Program(8)
        U = random_interferometer(4)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[3], q[7])
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.BSgate() | (q[2], q[3])
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="The applied unitary on modes"):
            res = prog.compile("X")

    def test_unitary_too_large(self):
        """Test exception raised if the unitary is applied to more
        than just modes [0, 1, 2, 3] and [4, 5, 6, 7]."""
        prog = sf.Program(8)
        U = random_interferometer(8)

        with prog.context as q:
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[0], q[4])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[1], q[5])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[2], q[6])
            ops.S2gate(SQ_AMPLITUDE, 0) | (q[3], q[7])
            ops.Interferometer(U) | q
            ops.MeasureFock() | q

        with pytest.raises(CircuitError, match="The applied unitary cannot mix between the modes"):
            res = prog.compile("X")

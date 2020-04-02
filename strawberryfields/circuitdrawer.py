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
r"""
A Strawberry Fields module that provides an object-oriented interface for building
quantum circuit representations of continuous-variable circuits using the
:math:`\LaTeX` `Qcircuit package <https://ctan.org/pkg/qcircuit>`_.
"""
import datetime
import os

# string constants used by :class:`~strawberryfields.circuitdrawer.Circuit`
# to transform Strawberry Fields operators to latex code.

DOCUMENT_CLASS = r"\documentclass{article}"
EMPTY_PAGESTYLE = r"\pagestyle{empty}"
QCIRCUIT_PACKAGE = r"\usepackage{qcircuit}"
BEGIN_DOCUMENT = r"\begin{document}"
DOCUMENT_END = r"\end{document}"
CIRCUIT_START = r"\Qcircuit"
COLUMN_SPACING = "@C={0}"  # spacing i.e. "1em"
ROW_SPACING = "@R={0}"  # spacing
UNIFORM_ROW_SPACING = "@!R"
UNIFORM_COLUMN_SPACING = "@!C"
UNIFORM_ELEMENT_SPACING = "@!"
QUANTUM_WIRE = r"\qw"
MULTI_QUANTUM_WIRE = r"\qw[{0}]"  # length vector i.e. "-1"
VERTICAL_QUANTUM_WIRE = r"\qwx[{0}]"  # length vector
WIRE_END = r"\qwa[{0}]"  # length vector
CLASSICAL_WIRE = r"\cw[{0}]"  # length vector
CLASSICAL_WIRE_END = r"\cwa[{0}]"  # length vector
VERTICAL_CLASSICAL_WIRE = r"\cwx[{0}]"  # length vector
LABELLED_GATE = r"\gate{{{0}}}"  # label i.e. "X"
TARGET = r"\targ"
SWAP = r"\qswap"
MULTIGATE = r"\multigate{{{0}}}{{{1}}}"  # depth i.e. "2", label
NON_ADJACENT_MULTIGATE = r"\sgate{{{0}}}{{{1}}}"  # gate i.e. "X", second wire vector i.e. "1"
GHOST = r"\ghost{{{0}}}"  # gate
CLASSICAL_GHOST = r"\cghost{{{0}}}"  # gate
NO_GHOST = r"\nghost{{{0}}}"  # gate
CONTROL = r"\ctrl{{{0}}}"  # length vector
CONTROL_ON_ZERO = r"\ctrlo{{{0}}}"  # length vector
CLASSICAL_CONTROL = r"\cctrl{{{0}}}"  # length vector
CLASSICAL_CONTROL_ON_ZERO = r"\cctrlo{{{0}}}"  # length vector
ISOLATED_CONTROL = r"\control"
ISOLATED_CONTROL_ON_ZERO = r"\controlo"
METER = r"\meter"
BASIS_METER = r"\meterB{{{0}}}"  # basis i.e. \ket{\xi_\pm}
SPLIT_BASIS_METER = r"\smeterB{{{0}}}{{{1}}}"  # basis, second wire vector
MEASURE = r"\measuretab{{{0}}}"  # label
MULTIMEASURE = r"\multimeasure{{{0}}}{{{1}}}"  # depth, label
LEFT_WIRE_LABEL = r"\lstick{{{0}}}"  # label
RIGHT_WIRE_LABEL = r"\rstick{{{0}}}"  # label
BRA = r"\bra{{{0}}}"  # state i.e. "\psi"
KET = r"\ket{{{0}}}"  # state

HADAMARD_COMP = LABELLED_GATE.format("H")
PAULI_X_COMP = LABELLED_GATE.format("X")
PAULI_Y_COMP = LABELLED_GATE.format("Y")
PAULI_Z_COMP = LABELLED_GATE.format("Z")
D_COMP = LABELLED_GATE.format("D")
S_COMP = LABELLED_GATE.format("S")
R_COMP = LABELLED_GATE.format("R")
P_COMP = LABELLED_GATE.format("P")
V_COMP = LABELLED_GATE.format("V")
K_COMP = LABELLED_GATE.format("K")
FOURIER_COMP = LABELLED_GATE.format("F")

BS_MULTI_COMP = "BS"
S_MULTI_COMP = "S"

WIRE_OPERATION = "& {0}"
WIRE_TERMINATOR = r"\\" + "\n"
CIRCUIT_BODY_TERMINATOR = "}\n"
CIRCUIT_BODY_START = " {" + "\n"
INIT_DOCUMENT = (
    DOCUMENT_CLASS
    + "\n"
    + EMPTY_PAGESTYLE
    + "\n"
    + QCIRCUIT_PACKAGE
    + "\n"
    + BEGIN_DOCUMENT
    + "\n"
    + CIRCUIT_START
)

PIPE = "|"
LINE_RETURN = "\n"


class NotDrawableException(Exception):
    """Exception raised when a circuit is not drawable.

    This class corresponds to the exception raised by :meth:`~.parse_op`
    when a circuit is deemed impossible to effectively render using qcircuit.
    """

    pass


class ModeMismatchException(Exception):
    """Exception raised when parsing a Gate object.

    This class corresponds to the exception raised by :meth:`~.parse_op`
    when an operator is interpreted as an n-mode gate but is applied to a number of modes != n.
    """

    pass


class UnsupportedGateException(Exception):
    """Exception raised when attempting to add an unsupported operator.

    This class corresponds to the exception raised by :meth:`~.parse_op` when it is attempted to add an
    unsupported operator to the circuit.
    """

    pass


class Circuit:
    """Represents a quantum circuit that can be compiled to tex format.

    Args:
        wires (int): the number of quantum wires or subsystems to use in the
            circuit diagram.
    """

    _circuit_matrix = []

    def __init__(self, wires):
        self._document = ""
        self._circuit_matrix = [[QUANTUM_WIRE.format(1)] for wire in range(wires)]
        self._column_spacing = None
        self._row_spacing = None

        self.single_mode_gates = {
            "Xgate": self._x,
            "Zgate": self._z,
            "Dgate": self._d,
            "Sgate": self._s,
            "Rgate": self._r,
            "Pgate": self._p,
            "Vgate": self._v,
            "Kgate": self._k,
            "Fourier": self._fourier,
        }

        self.two_mode_gates = {
            "CXgate": self._cx,
            "CZgate": self._cz,
            "CKgate": self._ck,
            "BSgate": self._bs,
            "S2gate": self._s2,
        }

    def _gate_from_operator(self, op):
        """Infers the number of modes and callable Circuit class method that correspond
        with a Strawberry Fields operator object.

        Args:
            op (strawberryfields.ops.Gate): the Strawberry Fields operator object.

        Returns:
            method (function): callable method that adds the given operator to the latex circuit.
            mode (int): the number of modes affected by the operator gate.
        """
        operator = str(op).split(PIPE)[0]
        method = None
        mode = None

        for two_mode_gate in self.two_mode_gates:
            if two_mode_gate in operator:
                method = self.two_mode_gates[two_mode_gate]
                mode = 2

        if method is None:
            for single_mode_gate in self.single_mode_gates:
                if single_mode_gate in operator:
                    method = self.single_mode_gates[single_mode_gate]
                    mode = 1

        return method, mode

    def parse_op(self, op):
        """Transforms a Strawberry Fields operator object to a latex qcircuit gate.

        Args:
            op (strawberryfields.ops.Gate): the Strawberry Fields operator object.

        Raises:
            UnsupportedGateException: if the operator is not supported by the circuit drawer module.
            ModeMismatchException: if the operator is interpreted as an n-mode gate but is
                applied to a number of modes != n.

        """
        if not op.__class__.__name__ == "Command":
            return

        method, mode = self._gate_from_operator(op)
        wires = list(map(lambda register: register.ind, op.reg))

        if method is None:
            raise UnsupportedGateException(
                "Unsupported operation {0} not printable by circuit builder!".format(str(op))
            )
        if mode == len(wires):
            method(*wires)
        else:
            raise ModeMismatchException(
                "{0} mode gate applied to {1} wires!".format(mode, len(wires))
            )

    def _x(self, wire):
        """Adds a position displacement operator to the circuit.

        Args:
            wire (int): the subsystem wire to apply the operator to.
        """
        self._single_mode_gate(wire, PAULI_X_COMP)

    def _z(self, wire):
        """Adds a momentum displacement operator to the circuit.

        Args:
            wire (int): the subsystem wire to apply the operator to.
        """
        self._single_mode_gate(wire, PAULI_Z_COMP)

    def _s(self, wire):
        """Adds a squeezing operator to the circuit.

        Args:
            wire (int): the subsystem wire to apply the operator to.
        """
        self._single_mode_gate(wire, S_COMP)

    def _d(self, wire):
        """Adds a displacement operator to the circuit.

        Args:
            wire (int): the subsystem wire to apply the operator to.
        """
        self._single_mode_gate(wire, D_COMP)

    def _r(self, wire):
        """Adds a rotation operator to the circuit.

        Args:
            wire (int): the subsystem wire to apply the operator to.
        """
        self._single_mode_gate(wire, R_COMP)

    def _p(self, wire):
        """Adds a quadratic phase shift operator to the circuit.

        Args:
            wire (int): the subsystem wire to apply the operator to.
        """
        self._single_mode_gate(wire, P_COMP)

    def _v(self, wire):
        """Adds a cubic phase shift operator to the circuit.

        Args:
            wire (int): the subsystem wire to apply the operator to.
        """
        self._single_mode_gate(wire, V_COMP)

    def _k(self, wire):
        """Adds a Kerr operator to the circuit.

        Args:
            wire (int): the subsystem wire to apply the operator to.
        """
        self._single_mode_gate(wire, K_COMP)

    def _fourier(self, wire):
        """Adds a Fourier transform operator to the circuit.

        Args:
            wire (int): the subsystem wire to apply the operator to.
        """
        self._single_mode_gate(wire, FOURIER_COMP)

    def _cx(self, source_wire, target_wire):
        """Adds a controlled position displacement operator to the circuit.

        Args:
            source_wire (int): the controlling subsystem wire.
            target_wire (int): the controlled subsystem wire.
        """
        self._controlled_mode_gate(source_wire, target_wire, TARGET)

    def _cz(self, source_wire, target_wire):
        """Adds a controlled phase operator to the circuit.

        Args:
            source_wire (int): the controlling subsystem wire.
            target_wire (int): the controlled subsystem wire.
        """
        self._controlled_mode_gate(source_wire, target_wire, PAULI_Z_COMP)

    def _ck(self, source_wire, target_wire):
        """Adds a controlled Kerr operator to the circuit.

        Args:
            source_wire (int): the controlling subsystem wire.
            target_wire (int): the controlled subsystem wire.
        """
        self._controlled_mode_gate(source_wire, target_wire, K_COMP)

    def _bs(self, first_wire, second_wire):
        """Adds a beams plitter operator to the circuit.

        Args:
            first_wire (int): the first subsystem wire to apply the operator to.
            second_wire (int): the second subsystem wire to apply the operator to.
        """
        self._multi_mode_gate(BS_MULTI_COMP, [first_wire, second_wire])

    def _s2(self, first_wire, second_wire):
        """Adds an two mode squeezing operator to the circuit.

        Args:
            first_wire (int): the first subsystem wire to apply the operator to.
            second_wire (int): the second subsystem wire to apply the operator to.
        """
        self._multi_mode_gate(S_MULTI_COMP, [first_wire, second_wire])

    # operation types

    def _single_mode_gate(self, wire, circuit_op):
        """Adds a single-mode operator gate to the circuit.

        Args:
            circuit_op (str): the latex code for the operator.
            wires (list[int]): a list of the indeces of subsystem wires to apply the multi-mode gate to.
        """
        matrix = self._circuit_matrix
        wire_ops = matrix[wire]

        if Circuit._is_empty(wire_ops[-1]):
            wire_ops[-1] = circuit_op
        else:
            wire_ops.append(circuit_op)
            for prev_wire in matrix[:wire]:
                prev_wire.append(QUANTUM_WIRE.format(1))
            for post_wire in matrix[wire + 1 :]:
                post_wire.append(QUANTUM_WIRE.format(1))

    def _multi_mode_gate(self, circuit_op, wires):
        """Adds a multi-mode operator to the circuit.

        Args:
            circuit_op (str): the latex code for the operator.
            wires (list[int]): a list of the indeces of subsystem wires to apply the gate to.
        Raises:
            ModeMismatchException: if the operator is applied to non-adjacent wires.
        """
        matrix = self._circuit_matrix

        if not self._on_empty_column():
            self._add_column()

        wires.sort()

        first_wire = wires.pop(0)
        wire_ops = matrix[first_wire]
        wire_ops[-1] = MULTIGATE.format(1, circuit_op)
        matrix[first_wire] = wire_ops
        previous_wire = first_wire

        for wire in wires:
            if not previous_wire == wire - 1:
                raise NotDrawableException(
                    "{0} multi-mode gate applied to non-adjacent wires!".format(circuit_op)
                )
            wire_ops = matrix[wire]
            wire_ops[-1] = GHOST.format(circuit_op)
            matrix[wire] = wire_ops
            previous_wire = wire

        self._circuit_matrix = matrix

    def _controlled_mode_gate(self, source_wire, target_wire, circuit_op):
        """Adds a controlled operator gate to the circuit.

        Args:
            source wire (int): the index of the controlling subsystem.
            target_wire (int): the index of the controlled subsystem.
            circuit_op (str): the latex code for the operator.
        """
        matrix = self._circuit_matrix
        source_ops = matrix[source_wire]
        target_ops = matrix[target_wire]
        distance = target_wire - source_wire

        if Circuit._is_empty(source_ops[-1]) and Circuit._is_empty(target_ops[-1]):
            source_ops[-1] = CONTROL.format(distance)
            target_ops[-1] = circuit_op
        else:
            for index, wire_ops in enumerate(matrix):
                if index == source_wire:
                    wire_ops.append(CONTROL.format(distance))
                elif index == target_wire:
                    wire_ops.append(circuit_op)
                else:
                    wire_ops.append(QUANTUM_WIRE.format(1))

    # helpers

    def _on_empty_column(self):
        """Checks if the right-most wires for each subsystem in the circuit are all empty

        Returns:
            bool: whether the right-most wires for each subsystem in the circuit are all empty
        """
        matrix = self._circuit_matrix

        empty_column = True
        for wire in enumerate(matrix):
            wire_ops = wire[1]
            if not Circuit._is_empty(wire_ops[-1]):
                empty_column = False
                break

        return empty_column

    def _add_column(self):
        """Adds a unit of quantum wire to each subsystem in the circuit."""
        for wire in self._circuit_matrix:
            wire.append(QUANTUM_WIRE.format(1))

    @staticmethod
    def _is_empty(op):
        """Checks for a NOP, a quantum wire location without an operator.

        Args:
            op (str): latex code for either an operator, or empty quantum wire.

        Returns:
            bool: whether the argument is an empty quantum wire.
        """
        return op == QUANTUM_WIRE.format(1)

    # cosmetic

    def _set_column_spacing(self, spacing):
        """Sets visual spacing between operators in quantum circuit.

        Args:
            spacing (int): spacing between operators.
        """
        self._column_spacing = spacing

    def _set_row_spacing(self, spacing):
        """Sets visual spacing of wires in quantum circuit.

        Args:
            spacing (int): spacing between wires.
        """
        self._row_spacing = spacing

    @staticmethod
    def _pad_with_spaces(string):
        """Pads string with spaces.

        Args:
            string (str): string to pad.

        Returns:
            str: string with space added to either side.
        """
        return " " + string + " "

    # latex translation

    def dump_to_document(self):
        """Writes current circuit to document.

        Returns:
            str: latex document string.
        """
        self._init_document()
        self._apply_spacing()
        self._begin_circuit()

        self._add_column()

        for wire_ops in enumerate(self._circuit_matrix):
            for wire_op in wire_ops[1]:
                self._write_operation_to_document(wire_op)
            self._end_wire()

        self._end_circuit()
        self._end_document()

        return self._document

    def compile_document(self, tex_dir="./circuit_tex"):
        """Compiles latex documents.

        Args:
            tex_dir (str): relative directory for latex document output.

        Returns:
            str: the file path of the resulting latex document.
        """
        tex_dir = os.path.abspath(tex_dir)

        if not os.path.isdir(tex_dir):
            os.mkdir(tex_dir)

        file_name = "output_{0}".format(datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p"))
        file_path = "{0}/{1}.tex".format(tex_dir, file_name)

        with open(file_path, "w+") as output_file:
            output_file.write(self._document)

        return file_path

    def _init_document(self):
        """Adds the required latex headers to the document."""
        self._document = INIT_DOCUMENT

    def _end_document(self):
        """Appends latex EOD code to the document."""
        self._document += DOCUMENT_END

    def _begin_circuit(self):
        """Prepares document for latex circuit content."""
        self._document += CIRCUIT_BODY_START

    def _end_circuit(self):
        """Ends the latex circuit content."""
        self._document += CIRCUIT_BODY_TERMINATOR

    def _end_wire(self):
        """Ends a wire within the latex circuit."""
        self._document += WIRE_TERMINATOR

    def _apply_spacing(self):
        """Applies wire and operator visual spacing."""
        if self._column_spacing is not None:
            self._document += Circuit._pad_with_spaces(COLUMN_SPACING.format(self._column_spacing))
        if self._row_spacing is not None:
            self._document += Circuit._pad_with_spaces(ROW_SPACING.format(self._row_spacing))

    def _write_operation_to_document(self, operation):
        """Appends operation latex code to circuit in latex document.

        Args:
            operation (str): the latex code for the quantum operation to be applied.
        """
        self._document += Circuit._pad_with_spaces(WIRE_OPERATION.format(operation))

    def __str__(self):
        """String representation of the Circuit class."""
        return self._document

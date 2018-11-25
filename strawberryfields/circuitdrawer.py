"""
Python Qcircuit Builder
====================
A Python module that provides an object-oriented interface for building Latex quantum circuit representations
using the Qcircuit package (https://ctan.org/pkg/qcircuit).

The following features of Qcircuit are currently supported:

Loading Q-circuit
        \input{Qcircuit}
Making Circuits
        \Qcircuit
Spacing
        @C=#1
        @R=#1
Wires
        \qw[#1]
Gates
        \gate {#1}
        \targ
        \qswap
Control
        \ctrl{#1}
"""

from .qcircuit_strings import HADAMARD_COMP, QUANTUM_WIRE, PAULI_X_COMP, PAULI_Y_COMP, PAULI_Z_COMP, CONTROL, \
    TARGET, SWAP, COLUMN_SPACING, ROW_SPACING, DOCUMENT_END, WIRE_OPERATION, WIRE_TERMINATOR, CIRCUIT_BODY_TERMINATOR, \
    CIRCUIT_BODY_START, INIT_DOCUMENT
import numpy as np
import datetime
import subprocess


class Circuit:

    _circuit_matrix = []

    def __init__(self, wires):
        self._document = ''
        self._circuit_matrix = np.array([[QUANTUM_WIRE] for wire in range(wires)])
        self._column_spacing = None
        self._row_spacing = None

    # operations

    def parse_op(self, op):
        r"""
        will be responsible for mapping operations of form
        BSgate(pi / 4, 0) | (q[0], q[1]) to drawable gates
        """
        pass

    def h(self, wire):
        self.single_qubit_gate(wire, HADAMARD_COMP)

    def x(self, wire):
        self.single_qubit_gate(wire, PAULI_X_COMP)

    def y(self, wire):
        self.single_qubit_gate(wire, PAULI_Y_COMP)

    def z(self, wire):
        self.single_qubit_gate(wire, PAULI_Z_COMP)

    def cnot(self, source_wire, target_wire):
        self.controlled_qubit_gate(source_wire, target_wire, TARGET)

    def cz(self, source_wire, target_wire):
        self.controlled_qubit_gate(source_wire, target_wire, PAULI_Z_COMP)

    def swap(self, source_wire, target_wire):
        self.multi_qubit_gate(SWAP, [source_wire, target_wire])

    # operation types

    def single_qubit_gate(self, wire, circuit_op):
        matrix = self._circuit_matrix
        wire_ops = matrix[wire]

        if Circuit.is_empty(wire_ops[-1]):
            wire_ops[-1] = circuit_op
        else:
            wire_ops.append(circuit_op)
            for prev_wire in matrix[:wire]:
                prev_wire.append(QUANTUM_WIRE)
            for post_wire in matrix[wire + 1:]:
                post_wire.append(QUANTUM_WIRE)

    def multi_qubit_gate(self, circuit_op, wires):
        matrix = self._circuit_matrix

        if self.on_empty_column():
            self.add_column()

        for wire in wires:
            wire_ops = matrix[wire]
            wire_ops[-1] = circuit_op

    def controlled_qubit_gate(self, source_wire, target_wire, circuit_op):
        matrix = self._circuit_matrix
        source_ops = matrix[source_wire]
        target_ops = matrix[target_wire]
        distance = target_wire - source_wire

        if Circuit.is_empty(source_ops[-1]) and Circuit.is_empty(target_ops[-1]):
            source_ops[-1] = CONTROL.format(distance)
            target_ops[-1] = circuit_op
        else:
            for wire in range(len(matrix)):
                if wire == source_wire:
                    matrix[wire].append(CONTROL.format(distance))
                elif wire == target_wire:
                    matrix[wire].append(circuit_op)
                else:
                    matrix[wire].append(QUANTUM_WIRE)

    # helpers

    def on_empty_column(self):
        matrix = self._circuit_matrix

        empty_column = True
        for wire in matrix:
            wire_ops = matrix[wire]
            if not Circuit.is_empty(wire_ops[-1]):
                empty_column = False
                break

        return empty_column

    def add_column(self):
        self._circuit_matrix = [wire.append(QUANTUM_WIRE) for wire in self._circuit_matrix]

    @staticmethod
    def is_empty(op):
        return op == QUANTUM_WIRE

    # cosmetic

    def set_column_spacing(self, spacing):
        self._column_spacing = spacing

    def set_row_spacing(self, spacing):
        self._row_spacing = spacing

    @staticmethod
    def pad_with_spaces(string):
        return ' ' + string + ' '

    # latex translation

    def dump_to_document(self):
        self.init_document()
        self.apply_spacing()
        self.begin_circuit()

        for wire in range(len(self._circuit_matrix)):
            for wire_op in self._circuit_matrix[wire]:
                self.write_operation_to_document(wire_op)
            self.end_wire()

        self.end_circuit()
        self.end_document()

    def compile_document(self, tex_dir='circuit_tex', pdf_dir='circuit_pdfs'):
        file_name = "output_{0}".format(datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p"))
        output_file = open(f'{tex_dir}/{file_name}.tex', "w")
        output_file.write(self._document)
        try:
            subprocess.call([f'pdflatex -output-directory {pdf_dir} {tex_dir}/{file_name}.tex'])
            return f'{pdf_dir}/{file_name}.pdf'
        except OSError:
            print('pdflatex not configured!')
            return -1

    def init_document(self):
        self._document = INIT_DOCUMENT

    def end_document(self):
        self._document += DOCUMENT_END

    def begin_circuit(self):
        self._document += CIRCUIT_BODY_START

    def end_circuit(self):
        self._document += CIRCUIT_BODY_TERMINATOR

    def end_wire(self):
        self._document += WIRE_TERMINATOR

    def apply_spacing(self):
        if self._column_spacing is not None:
            self._document += Circuit.pad_with_spaces(COLUMN_SPACING.format(self._column_spacing))
        if self._row_spacing is not None:
            self._document += Circuit.pad_with_spaces(ROW_SPACING.format(self._row_spacing))

    def write_operation_to_document(self, operation):
        self._document += Circuit.pad_with_spaces(WIRE_OPERATION.format(operation))

    def __str__(self):
        return self._document

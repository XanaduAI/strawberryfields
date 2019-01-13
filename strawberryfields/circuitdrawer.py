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
Circuit drawer
==============

A Strawberry Fields module that provides an object-oriented interface for building
quantum circuit representations of continuous-variable circuits using the
:math:`\LaTeX` `Qcircuit package <https://ctan.org/pkg/qcircuit>`_.

The following features of Qcircuit are currently supported:

Loading Q-circuit
-----------------
\input{Qcircuit}

Making Circuits
---------------
\Qcircuit

Spacing
-------
@C=#1
@R=#1

Wires
-----
\qw[#1]

Gates
-----
\gate {#1}
\targ
\qswap

Control
-------
\ctrl{#1}

The drawing of the following Xanadu supported operations will be implemented:

single-mode gates
-----------------
Dgate
Xgate
Zgate
Sgate
Rgate
Pgate
Vgate
Kgate
Fouriergate

Two-mode gates
--------------
BSgate
S2gate
CXgate
CZgate
CKgate
"""

from .qcircuit_strings import QUANTUM_WIRE, PAULI_X_COMP, PAULI_Z_COMP, CONTROL, \
    TARGET, COLUMN_SPACING, ROW_SPACING, DOCUMENT_END, WIRE_OPERATION, WIRE_TERMINATOR, CIRCUIT_BODY_TERMINATOR, \
    CIRCUIT_BODY_START, INIT_DOCUMENT, PIPE, S_COMP, D_COMP, R_COMP, P_COMP, V_COMP, FOURIER_COMP, BS_COMP, S_COMP, \
    K_COMP
import datetime
import subprocess


class ModeMismatchException(Exception):
    pass


class UnsupportedGateException(Exception):
    pass


class LatexConfigException(Exception):
    pass


class Circuit:

    _circuit_matrix = []

    def __init__(self, wires):
        self._document = ''
        self._circuit_matrix = [[QUANTUM_WIRE.format(1)] for wire in range(wires)]
        self._column_spacing = None
        self._row_spacing = None

        self.single_mode_gates = {
            'Xgate': self.x,
            'Zgate': self.z,
            'Dgate': self.d,
            'Sgate': self.s,
            'Rgate': self.r,
            'Pgate': self.p,
            'Vgate': self.v,
            'Kgate': self.k,
            'FourierGate': self.fourier
        }

        self.two_mode_gates = {
             'CXgate': self.cx,
             'CZgate': self.cz,
             'CKgate': self.ck,
             'BSgate': self.bs,
             'S2gate': self.s2
        }

    # operations

    def gate_from_operator(self, op):
        operator = str(op).split(PIPE)[0]
        method = None
        mode = None

        for two_mode_gate in self.two_mode_gates.keys():
            if two_mode_gate in operator:
                method = self.two_mode_gates[two_mode_gate]
                mode = 2

        if method is None:
            for single_mode_gate in self.single_mode_gates.keys():
                if single_mode_gate in operator:
                    method = self.single_mode_gates[single_mode_gate]
                    mode = 1

        return method, mode

    def parse_op(self, op):
        method, mode = self.gate_from_operator(op)
        wires = list(map(lambda register: register.ind, op.reg))

        if method is None:
            raise UnsupportedGateException('Unsupported operation {0} not printable by circuit builder!'.format(str(op)))
        elif mode == len(wires):
            method(*wires)
        elif mode != len(wires):
            raise ModeMismatchException('{0} mode gate applied to {1} wires!'.format(mode, len(wires)))

    def x(self, wire):
        self.single_mode_gate(wire, PAULI_X_COMP)

    def z(self, wire):
        self.single_mode_gate(wire, PAULI_Z_COMP)

    def s(self, wire):
        self.single_mode_gate(wire, S_COMP)

    def d(self, wire):
        self.single_mode_gate(wire, D_COMP)

    def r(self, wire):
        self.single_mode_gate(wire, R_COMP)

    def p(self, wire):
        self.single_mode_gate(wire, P_COMP)

    def v(self, wire):
        self.single_mode_gate(wire, V_COMP)

    def k(self, wire):
        self.single_mode_gate(wire, K_COMP)

    def fourier(self, wire):
        self.single_mode_gate(wire, FOURIER_COMP)

    def cx(self, source_wire, target_wire):
        self.controlled_mode_gate(source_wire, target_wire, TARGET)

    def cz(self, source_wire, target_wire):
        self.controlled_mode_gate(source_wire, target_wire, PAULI_Z_COMP)

    def ck(self, source_wire, target_wire):
        self.controlled_mode_gate(source_wire, target_wire, K_COMP)

    def bs(self, first_wire, second_wire):
        self.multi_mode_gate(BS_COMP, [first_wire, second_wire])

    def s2(self, first_wire, second_wire):
        self.multi_mode_gate(S_COMP, [first_wire, second_wire])

    # operation types

    def single_mode_gate(self, wire, circuit_op):
        matrix = self._circuit_matrix
        wire_ops = matrix[wire]

        if Circuit.is_empty(wire_ops[-1]):
            wire_ops[-1] = circuit_op
        else:
            wire_ops.append(circuit_op)
            for prev_wire in matrix[:wire]:
                prev_wire.append(QUANTUM_WIRE.format(1))
            for post_wire in matrix[wire + 1:]:
                post_wire.append(QUANTUM_WIRE.format(1))

    def multi_mode_gate(self, circuit_op, wires):
        matrix = self._circuit_matrix

        if not self.on_empty_column():
            self.add_column()

        for wire in wires:
            wire_ops = matrix[wire]
            wire_ops[-1] = circuit_op
            matrix[wire] = wire_ops

        self._circuit_matrix = matrix

    def controlled_mode_gate(self, source_wire, target_wire, circuit_op):
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
                    matrix[wire].append(QUANTUM_WIRE.format(1))

    # helpers

    def on_empty_column(self):
        matrix = self._circuit_matrix

        empty_column = True
        for wire in range(len(matrix)):
            wire_ops = matrix[wire]
            if not Circuit.is_empty(wire_ops[-1]):
                empty_column = False
                break

        return empty_column

    def add_column(self):
        self._circuit_matrix = [wire.append(QUANTUM_WIRE.format(1)) for wire in self._circuit_matrix]

    @staticmethod
    def is_empty(op):
        return op == QUANTUM_WIRE.format(1)

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

        return self._document

    def compile_document(self, tex_dir='circuit_tex', pdf_dir='circuit_pdfs'):
        file_name = "output_{0}".format(datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p"))
        output_file = open('{0}/{1}.tex'.format(tex_dir, file_name), "w")
        output_file.write(self._document)
        try:
            subprocess.call(['pdflatex -output-directory {0} {1}/{2}.tex'.format(pdf_dir, tex_dir, file_name)])
            return '{0}/{1}.pdf'.format(pdf_dir, file_name)
        except OSError:
            raise LatexConfigException('pdflatex not configured!')

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

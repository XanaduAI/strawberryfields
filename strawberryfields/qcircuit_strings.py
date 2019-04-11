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
Qcircuit strings
==============

A file containing the string constants used by :class:`~strawberryfields.circuitdrawer.Circuit` to transform Strawberry
Fields operators to latex code.
"""

DOCUMENT_CLASS = r'\documentclass{article}'
QCIRCUIT_PACKAGE = r'\usepackage{qcircuit}'
BEGIN_DOCUMENT = r'\begin{document}'
DOCUMENT_END = r'\end{document}'
CIRCUIT_START = r'\Qcircuit'
COLUMN_SPACING = '@C={0}' #spacing i.e. "1em"
ROW_SPACING = '@R={0}' #spacing
UNIFORM_ROW_SPACING = '@!R'
UNIFORM_COLUMN_SPACING = '@!C'
UNIFORM_ELEMENT_SPACING = '@!'
QUANTUM_WIRE = r'\qw'
MULTI_QUANTUM_WIRE = r'\qw[{0}]' #length vector i.e. "-1"
VERTICAL_QUANTUM_WIRE = r'\qwx[{0}]' #length vector
WIRE_END = r'\qwa[{0}]' #length vector
CLASSICAL_WIRE = r'\cw[{0}]' #length vector
CLASSICAL_WIRE_END = r'\cwa[{0}]' #length vector
VERTICAL_CLASSICAL_WIRE = r'\cwx[{0}]' #length vector
LABELLED_GATE = r'\gate{{{0}}}' #label i.e. "X"
TARGET = r'\targ'
SWAP = r'\qswap'
MULTIGATE = r'\multigate{{{0}}}{{{1}}}' #depth i.e. "2", label
NON_ADJACENT_MULTIGATE = r'\sgate{{{0}}}{{{1}}}' #gate i.e. "X", second wire vector i.e. "1"
GHOST = r'\ghost{{{0}}}' #gate
CLASSICAL_GHOST = r'\cghost{{{0}}}' #gate
NO_GHOST = r'\nghost{{{0}}}' #gate
CONTROL = r'\ctrl{{{0}}}' #length vector
CONTROL_ON_ZERO = r'\ctrlo{{{0}}}' #length vector
CLASSICAL_CONTROL = r'\cctrl{{{0}}}' #length vector
CLASSICAL_CONTROL_ON_ZERO = r'\cctrlo{{{0}}}' #length vector
ISOLATED_CONTROL = r'\control'
ISOLATED_CONTROL_ON_ZERO = r'\controlo'
METER = r'\meter'
BASIS_METER = r'\meterB{{{0}}}' #basis i.e. \ket{\xi_\pm}
SPLIT_BASIS_METER = r'\smeterB{{{0}}}{{{1}}}' #basis, second wire vector
MEASURE = r'\measuretab{{{0}}}' #label
MULTIMEASURE = r'\multimeasure{{{0}}}{{{1}}}' #depth, label
LEFT_WIRE_LABEL = r'\lstick{{{0}}}' #label
RIGHT_WIRE_LABEL = r'\rstick{{{0}}}' #label
BRA = r'\bra{{{0}}}' #state i.e. "\psi"
KET = r'\ket{{{0}}}' #state

HADAMARD_COMP = LABELLED_GATE.format('H')
PAULI_X_COMP = LABELLED_GATE.format('X')
PAULI_Y_COMP = LABELLED_GATE.format('Y')
PAULI_Z_COMP = LABELLED_GATE.format('Z')
D_COMP = LABELLED_GATE.format('D')
S_COMP = LABELLED_GATE.format('S')
R_COMP = LABELLED_GATE.format('R')
P_COMP = LABELLED_GATE.format('P')
V_COMP = LABELLED_GATE.format('V')
K_COMP = LABELLED_GATE.format('K')
FOURIER_COMP = LABELLED_GATE.format('F')

BS_MULTI_COMP = 'BS'
S_MULTI_COMP = 'S'

WIRE_OPERATION = '& {0}'
WIRE_TERMINATOR = r'\\' + '\n'
CIRCUIT_BODY_TERMINATOR = '}\n'
CIRCUIT_BODY_START = ' {' + '\n'
INIT_DOCUMENT = DOCUMENT_CLASS + '\n' + QCIRCUIT_PACKAGE + '\n' + BEGIN_DOCUMENT + '\n' + CIRCUIT_START

PIPE = '|'
LINE_RETURN = "\n"

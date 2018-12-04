DOCUMENT_CLASS = r'\documentclass{article}'
QCIRCUIT_PACKAGE = r'\usepackage{qcircuit}'
DOCUMENT_END = r'\end{document}'
CIRCUIT_START = r'\Qcircuit'
COLUMN_SPACING = '@C={0}' #spacing i.e. "1em"
ROW_SPACING = '@R={0}' #spacing
UNIFORM_ROW_SPACING = '@!R'
UNIFORM_COLUMN_SPACING = '@!C'
UNIFORM_ELEMENT_SPACING = '@!'
QUANTUM_WIRE = r'\qw[{0}]' #length vector i.e. "-1"
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

WIRE_OPERATION = '& {0}'
WIRE_TERMINATOR = r'\\' + '\n'
CIRCUIT_BODY_TERMINATOR = '\n}\n'
CIRCUIT_BODY_START = ' {' + '\n'
INIT_DOCUMENT = DOCUMENT_CLASS + '\n' + QCIRCUIT_PACKAGE + '\n' + CIRCUIT_START

PIPE = '|'
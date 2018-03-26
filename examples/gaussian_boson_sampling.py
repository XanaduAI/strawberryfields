#!/usr/bin/env python3
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *

# initialise the engine and register
eng, q = sf.Engine(4)

with eng:
    # prepare the input squeezed states
    S = Sgate(1)
    S | q[0]
    S | q[1]
    S | q[2]
    S | q[3]

    # rotation gates
    Rgate(0.5719)  | q[0]
    Rgate(-1.9782) | q[1]
    Rgate(2.0603)  | q[2]
    Rgate(0.0644)  | q[3]

    # beamsplitter array
    BSgate(0.7804, 0.8578)  | (q[0], q[1])
    BSgate(0.06406, 0.5165) | (q[2], q[3])
    BSgate(0.473, 0.1176)   | (q[1], q[2])
    BSgate(0.563, 0.1517)   | (q[0], q[1])
    BSgate(0.1323, 0.9946)  | (q[2], q[3])
    BSgate(0.311, 0.3231)   | (q[1], q[2])
    BSgate(0.4348, 0.0798)  | (q[0], q[1])
    BSgate(0.4368, 0.6157)  | (q[2], q[3])
    # end circuit

# run the engine
state = eng.run('gaussian')

# Fock states to measure at output
measure_states = [[0,0,0,0], [1,1,0,0], [0,1,0,1], [1,1,1,1], [2,0,0,0]]

# extract the probabilities of calculating several
# different Fock states at the output, and print them to the terminal
for i in measure_states:
    prob = state.fock_prob(i)
    print("|{}>: {}".format("".join(str(j) for j in i), prob))

#!/usr/bin/env python3
import strawberryfields as sf
from strawberryfields.ops import *

# initialize engine and program objects
eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 7})
boson_sampling = sf.Program(4)

import numpy as np
U = np.array(
    [[ 0.2195-0.2565j,  0.6111+0.5242j, -0.1027+0.4745j, -0.0273+0.0373j],
     [ 0.4513+0.6026j,  0.457 +0.0123j,  0.1316-0.4504j,  0.0353-0.0532j],
     [ 0.0387+0.4927j, -0.0192-0.3218j, -0.2408+0.5244j, -0.4584+0.3296j],
     [-0.1566+0.2246j,  0.11  -0.1638j, -0.4212+0.1836j,  0.8188+0.068j ]])

with boson_sampling.context as q:
    # prepare the input fock states
    Fock(1) | q[0]
    Fock(1) | q[1]
    Vac     | q[2]
    Fock(1) | q[3]

    Interferometer(U) | q

    # rotation gates
    Rgate(0.5719) | q[0]
    Rgate(-1.9782) | q[1]
    Rgate(2.0603) | q[2]
    Rgate(0.0644) | q[3]

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
results = eng.run(boson_sampling)

# extract the joint Fock probabilities
probs = results.state.all_fock_probs()

# print the joint Fock state probabilities
print(probs[1, 1, 0, 1])
print(probs[2, 0, 0, 1])

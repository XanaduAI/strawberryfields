#!/usr/bin/env python3
import strawberryfields as sf
from strawberryfields.ops import *

# initialize engine and program objects
eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 7})
boson_sampling = sf.Program(4)

with boson_sampling.context as q:
    # prepare the input fock states
    Fock(1) | q[0]
    Fock(1) | q[1]
    Vac     | q[2]
    Fock(1) | q[3]

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

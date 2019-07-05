#!/usr/bin/env python3
import strawberryfields as sf
from strawberryfields.ops import *

# initialize engine and program objects
eng = sf.Engine(backend="gaussian")
gate_teleportation = sf.Program(4)

with gate_teleportation.context as q:
    # create initial states
    Squeezed(0.1) | q[0]
    Squeezed(-2)  | q[1]
    Squeezed(-2)  | q[2]

    # apply the gate to be teleported
    Pgate(0.5) | q[1]

    # conditional phase entanglement
    CZgate(1) | (q[0], q[1])
    CZgate(1) | (q[1], q[2])

    # projective measurement onto
    # the position quadrature
    Fourier.H | q[0]
    MeasureX | q[0]
    Fourier.H | q[1]
    MeasureX | q[1]
    # compare against the expected output
    # X(q1/sqrt(2)).F.P(0.5).X(q0/sqrt(0.5)).F.|z>
    # not including the corrections
    Squeezed(0.1) | q[3]
    Fourier       | q[3]
    Xgate(q[0])   | q[3]
    Pgate(0.5)    | q[3]
    Fourier       | q[3]
    Xgate(q[1])   | q[3]
    # end circuit

results = eng.run(gate_teleportation)
print(results.state.reduced_gaussian([2]))
print(results.state.reduced_gaussian([3]))

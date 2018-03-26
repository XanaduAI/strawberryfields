#!/usr/bin/env python3
import strawberryfields as sf
from strawberryfields.ops import *

# initialise the engine and register
eng, q = sf.Engine(4)

with eng:
    # prepare the input squeezed states
    S = Sgate(-1)
    S | q[0]
    S | q[1]
    S | q[2]
    S | q[3]

    # gates diagonal in the x quadrature
    CZgate(0.5719) | (q[0], q[1])
    Vgate(-1.9782) | q[2]
    Zgate(2.0603)  | q[3]
    Zgate(-0.7804) | q[0]
    Zgate(0.8578)  | q[2]
    CZgate(1.321)  | (q[1], q[2])
    Zgate(0.473)   | q[3]
    CZgate(0.9946) | (q[0], q[2])
    Zgate(0.1223)  | q[3]
    Vgate(0.6157)  | q[0]
    Vgate(0.3110)  | q[1]
    Zgate(-1.243)  | q[2]
    # end circuit

# run the engine
eng.run("fock", cutoff_dim=5)
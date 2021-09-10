#!/usr/bin/env python3
import strawberryfields as sf
from strawberryfields.ops import (
    Coherent,
    Squeezed,
    BSgate,
    Xgate,
    Zgate,
    MeasureX,
    MeasureP,
)

import numpy as np
from numpy import pi, sqrt

# set the random seed
np.random.seed(42)

prog = sf.Program(3)

alpha = 1 + 0.5j
r = np.abs(alpha)
phi = np.angle(alpha)

with prog.context as q:
    # prepare initial states
    Coherent(r, phi) | q[0]
    Squeezed(-2) | q[1]
    Squeezed(2) | q[2]

    # apply gates
    BS = BSgate(pi / 4, pi)
    BS | (q[1], q[2])
    BS | (q[0], q[1])

    # Perform homodyne measurements
    MeasureX | q[0]
    MeasureP | q[1]

    # Displacement gates conditioned on
    # the measurements
    Xgate(sqrt(2) * q[0].par) | q[2]
    Zgate(sqrt(2) * q[1].par) | q[2]

eng = sf.Engine("fock", backend_options={"cutoff_dim": 15})
result = eng.run(prog)

# view output state and fidelity
print(result.samples)
print(result.state)
print(result.state.fidelity_coherent([0, 0, alpha]))

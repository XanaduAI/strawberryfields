#!/usr/bin/env python3
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import scale
from numpy import pi, sqrt
import numpy as np

# initialise the engine and register
eng, q = sf.Engine(4)

with eng:
    # state to be cloned
    Coherent(0.7+1.2j) | q[0]

    # 50-50 beamsplitter
    BS = BSgate(pi/4, 0)

    # symmetric Gaussian cloning scheme
    BS | (q[0], q[1])
    BS | (q[1], q[2])
    MeasureX | q[1]
    MeasureP | q[2]
    Xgate(scale(q[1], sqrt(2))) | q[0]
    Zgate(scale(q[2], sqrt(2))) | q[0]

    # after the final beamsplitter, modes q[0] and q[3]
    # will contain identical approximate clones of the
    # initial state Coherent(0.1+0j)
    BS | (q[0], q[3])
    # end circuit

# run the engine
state = eng.run('gaussian', modes=[0,3])

# return the cloning fidelity
fidelity = sqrt(state.fidelity_coherent([0.7+1.2j, 0.7+1.2j]))
# return the cloned displacement
alpha = state.displacement()

# run the engine over an ensemble
shots = 1000
f = np.empty([shots])
a = np.empty([shots], dtype=np.complex128)

for i in range(shots):
    state = eng.run('gaussian', reset_backend=True, modes=[0])
    f[i] = state.fidelity_coherent([0.7+1.2j])
    a[i] = state.displacement()

print(np.mean(f))
print(np.mean(a))
print(np.cov([a.real, a.imag]))
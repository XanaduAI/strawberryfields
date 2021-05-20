import numpy as np
from numpy import pi, sqrt

import strawberryfields as sf
from strawberryfields.ops import *

# initialize engine and program objects
eng = sf.Engine(backend="gaussian")
gaussian_cloning = sf.Program(4)

with gaussian_cloning.context as q:
    # state to be cloned
    Coherent(0.7+1.2j) | q[0]

    # 50-50 beamsplitter
    BS = BSgate(pi/4, 0)

    # symmetric Gaussian cloning scheme
    BS | (q[0], q[1])
    BS | (q[1], q[2])
    MeasureX | q[1]
    MeasureP | q[2]
    Xgate(q[1].par * sqrt(2)) | q[0]
    Zgate(q[2].par * sqrt(2)) | q[0]

    # after the final beamsplitter, modes q[0] and q[3]
    # will contain identical approximate clones of the
    # initial state Coherent(0.1+0j)
    BS | (q[0], q[3])
    # end circuit

# run the engine
results = eng.run(gaussian_cloning, modes=[0, 3])

# return the cloning fidelity
fidelity = sqrt(results.state.fidelity_coherent([0.7+1.2j, 0.7+1.2j]))
# return the cloned displacement
alpha = results.state.displacement()

# run the engine over an ensemble
reps = 1000
f = np.empty([reps])
a = np.empty([reps], dtype=np.complex128)

for i in range(reps):
    eng.reset()
    results = eng.run(gaussian_cloning, modes=[0])
    f[i] = results.state.fidelity_coherent([0.7+1.2j])
    a[i] = results.state.displacement()

print("Fidelity of cloned state:", np.mean(f))
print("Mean displacement of cloned state:", np.mean(a))
print("Mean covariance matrix of cloned state:", np.cov([a.real, a.imag]))

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np

"""
Test to check that the triangular and rectangular meshes give the same results in the
interferometer opreation.
The tests creates a random M by M unitary matrix and applies it to a M-mode interferometer
with the rectangular and triangular meshes. The test checks that the output probabilities.
"""

M = 5
# Create a random complex matrix
random_matrix = np.random.rand(M, M) + 1j * np.random.rand(M, M)
# Perform QR decomposition to get a unitary matrix
q, r = np.linalg.qr(random_matrix)
# Normalize to ensure unitary property (q should already be unitary, but we'll double-check)
unitary_matrix = q / np.linalg.norm(q, axis=0)

boson_sampling_triangular = sf.Program(M)
with boson_sampling_triangular.context as q:
    # prepare the input fock states
    for i in range(M):
        if i % 2 == 0:
            Fock(1) | q[i]
        else:
            Vac | q[i]
    Interferometer(unitary_matrix, mesh = 'triangular', tol = 1e-10) | q

boson_sampling_triangular.compile(compiler="fock")
eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
results = eng.run(boson_sampling_triangular)
# extract the joint Fock probabilities
probs_triangular = results.state.all_fock_probs()

#now try with mesh = 'rectangular'
boson_sampling_rectangular = sf.Program(M)

with boson_sampling_rectangular.context as q:
    for i in range(M):
        if i % 2 == 0:
            Fock(1) | q[i]
        else:
            Vac | q[i]
    Interferometer(unitary_matrix, mesh = 'rectangular', tol = 1e-10) | q
    
boson_sampling_rectangular.compile(compiler="fock")
eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
results = eng.run(boson_sampling_rectangular)
# extract the joint Fock probabilities
probs_rect = results.state.all_fock_probs()
assert(np.allclose(probs_triangular, probs_rect))
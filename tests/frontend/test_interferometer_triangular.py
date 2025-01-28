import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np

M = 4
# Create a random complex matrix
random_matrix = np.random.rand(M, M) + 1j * np.random.rand(M, M)

# Perform QR decomposition to get a unitary matrix
q, r = np.linalg.qr(random_matrix)

# Normalize to ensure unitary property (q should already be unitary, but we'll double-check)
unitary_matrix = q / np.linalg.norm(q, axis=0)


boson_sampling_triangular = sf.Program(4)
#initiate a starting state |1101>
with boson_sampling_triangular.context as q:
    # prepare the input fock states
    Fock(1) | q[0]
    Fock(1) | q[1]
    Vac     | q[2]
    Fock(1) | q[3]

    # apply the triangular interferometer
    Interferometer(unitary_matrix, mesh = 'triangular', tol = 1e-10) | q

#run the simulation
boson_sampling_triangular.compile(compiler="fock")
eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
results = eng.run(boson_sampling_triangular)
# extract the joint Fock probabilities
probs_triangular = results.state.all_fock_probs()

#now try with mesh = 'rectangular'
boson_sampling_rectangular = sf.Program(4)

with boson_sampling_rectangular.context as q:
    # prepare the input fock states
    Fock(1) | q[0]
    Fock(1) | q[1]
    Vac     | q[2]
    Fock(1) | q[3]

    # apply the rectangular interferometer
    Interferometer(unitary_matrix, mesh = 'rectangular', tol = 1e-10) | q
    
boson_sampling_rectangular.compile(compiler="fock")
eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
results = eng.run(boson_sampling_rectangular)
# extract the joint Fock probabilities
probs_rect = results.state.all_fock_probs()
#all probabilities should be the same here! Sp the output should be True
print(f"{np.allclose(probs_triangular, probs_rect)}")
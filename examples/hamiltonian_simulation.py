#!/usr/bin/env python3
import strawberryfields as sf
from strawberryfields.ops import *
from numpy import pi

# initialize engine and program objects
eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
ham_simulation = sf.Program(2)

# set the Hamiltonian parameters
J = 1           # hopping transition
U = 1.5         # on-site interaction
k = 20          # Lie product decomposition terms
t = 1.086       # timestep
theta = -J*t/k  
r = -U*t/(2*k)

with ham_simulation.context as q:
    # prepare the initial state
    Fock(2) | q[0]

    # Two node tight-binding
    # Hamiltonian simulation

    for i in range(k):
        BSgate(theta, pi/2) | (q[0], q[1])
        Kgate(r)  | q[0]
        Rgate(-r) | q[0]
        Kgate(r)  | q[1]
        Rgate(-r) | q[1]
    # end circuit

# run the engine
results = eng.run(ham_simulation)
# the output state probabilities
print(results.state.fock_prob([0,2]))
print(results.state.fock_prob([1,1]))
print(results.state.fock_prob([2,0]))

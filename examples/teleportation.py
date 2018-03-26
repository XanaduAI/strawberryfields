import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import scale
from numpy import pi, sqrt

eng,q = sf.Engine(3)

with eng:
    psi, alice, bob = q[0], q[1], q[2]

    # state to be teleported:
    Coherent(1+0.5j) | psi

    # 50-50 beamsplitter
    BS = BSgate(pi/4, 0)

    # maximally entangled states
    Squeezed(-2) | alice
    Squeezed(2) | bob
    BS | (alice, bob)

    # Alice performs the joint measurement
    # in the maximally entangled basis
    BS | (psi, alice)
    MeasureX | psi
    MeasureP | alice

    # Bob conditionally displaces his mode
    # based on Alice's measurement result
    Xgate(scale(psi, sqrt(2))) | bob
    Zgate(scale(alice, sqrt(2))) | bob
    # end circuit

state = eng.run('gaussian')
# view Bob's output state and fidelity
print(q[0].val,q[1].val)
print(state.displacement([2]))
print(state.fidelity_coherent([0,0,1+0.5j]))
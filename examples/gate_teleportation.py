import strawberryfields as sf
from strawberryfields.ops import *

eng, q = sf.Engine(4)

with eng:
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

state = eng.run('gaussian')
print(state.reduced_gaussian([2]))
print(state.reduced_gaussian([3]))
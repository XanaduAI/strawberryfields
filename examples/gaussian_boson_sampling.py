#!/usr/bin/env python3
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *

# initialize engine and program objects
eng = sf.Engine(backend="gaussian")
gbs = sf.Program(4)

# define the linear interferometer
U = np.array([
 [ 0.219546940711-0.256534554457j, 0.611076853957+0.524178937791j,
    -0.102700187435+0.474478834685j,-0.027250232925+0.03729094623j],
 [ 0.451281863394+0.602582912475j, 0.456952590016+0.01230749109j,
    0.131625867435-0.450417744715j, 0.035283194078-0.053244267184j],
 [ 0.038710094355+0.492715562066j,-0.019212744068-0.321842852355j,
    -0.240776471286+0.524432833034j,-0.458388143039+0.329633367819j],
 [-0.156619083736+0.224568570065j, 0.109992223305-0.163750223027j,
    -0.421179844245+0.183644837982j, 0.818769184612+0.068015658737j]
])


with gbs.context as q:
    # prepare the input squeezed states
    S = Sgate(1)
    S | q[0]
    S | q[1]
    S | q[2]
    S | q[3]

    # linear interferometer
    Interferometer(U) | q
    # end circuit

# run the engine
results = eng.run(gbs)

# Fock states to measure at output
measure_states = [[0,0,0,0], [1,1,0,0], [0,1,0,1], [1,1,1,1], [2,0,0,0]]

# extract the probabilities of calculating several
# different Fock states at the output, and print them to the terminal
for i in measure_states:
    prob = results.state.fock_prob(i)
    print("|{}>: {}".format("".join(str(j) for j in i), prob))

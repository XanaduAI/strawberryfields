#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *

# define width and depth of CV quantum neural network
modes = 1
layers = 8
cutoff_dim = 6

# defining desired state (single photon state)
target_state = np.zeros(cutoff_dim)
target_state[1] = 1
target_state = tf.constant(target_state, dtype=tf.complex64)


# define interferometer
def interferometer(theta, phi, rphi, q):
    """Parameterised interferometer acting on N qumodes

    Args:
        theta (list): list of length N(N-1)/2 real parameters
        phi (list): list of length N(N-1)/2 real parameters
        rphi (list): list of length N-1 real parameters
        q (list): list of qumodes the interferometer is to be applied to
    """
    N = len(q)

    if N == 1:
        # the interferometer is a single rotation
        Rgate(rphi[0]) | q[0]
        return

    n = 0  # keep track of free parameters

    # Apply the rectangular beamsplitter array
    # The array depth is N
    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            # skip even or odd pairs depending on layer
            if (l + k) % 2 != 1:
                BSgate(theta[n], phi[n]) | (q1, q2)
                n += 1

    # apply the final local phase shifts to all modes except the last one
    for i in range(max(1, N - 1)):
        Rgate(rphi[i]) | q[i]
    # Rgate only applied to first N - 1 modes


# define layer
def layer(q):
    """CV quantum neural network layer acting on N modes

    Args:
        q (list): list of qumodes the layer is to be applied to
    """
    N = len(q)
    BS_variable_number = int(N * (N - 1) / 2)
    R_variable_number = max(1, N - 1)

    theta_variables_1 = tf.Variable(tf.random_normal(shape=[BS_variable_number]))
    phi_variables_1 = tf.Variable(tf.random_normal(shape=[BS_variable_number]))
    rphi_variables_1 = tf.Variable(tf.random_normal(shape=[R_variable_number]))

    theta_variables_2 = tf.Variable(tf.random_normal(shape=[BS_variable_number]))
    phi_variables_2 = tf.Variable(tf.random_normal(shape=[BS_variable_number]))
    rphi_variables_2 = tf.Variable(tf.random_normal(shape=[R_variable_number]))

    s_variables = tf.Variable(tf.random_normal(shape=[N], stddev=0.0001))
    d_variables_r = tf.Variable(tf.random_normal(shape=[N], stddev=0.0001))
    d_variables_phi = tf.Variable(tf.random_normal(shape=[N]))
    k_variables = tf.Variable(tf.random_normal(shape=[N], stddev=0.0001))

    # begin layer
    interferometer(theta_variables_1, phi_variables_1, rphi_variables_1, q)

    for i in range(N):
        Sgate(s_variables[i]) | q[i]

    interferometer(theta_variables_2, phi_variables_2, rphi_variables_2, q)

    for i in range(N):
        Dgate(d_variables_r[i], d_variables_phi[i]) | q[i]
        Kgate(k_variables[i]) | q[i]
    # end layer


# initialize engine and program objects
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
qnn = sf.Program(modes)

with qnn.context as q:
    for _ in range(layers):
        layer(q)

# starting the engine
results = eng.run(qnn, eval=False)
ket = results.state.ket()

# defining cost function
difference = tf.reduce_sum(tf.abs(ket - target_state))
fidelity = tf.abs(tf.reduce_sum(tf.conj(ket) * target_state)) ** 2
cost = difference

# setting up optimizer
optimiser = tf.train.AdamOptimizer()
minimize_cost = optimiser.minimize(cost)

print("Beginning optimization")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

fidelity_before = sess.run(fidelity)

for i in range(5000):
    sess.run(minimize_cost)

fidelity_after = sess.run(fidelity)

print("Fidelity before optimization: " + str(fidelity_before))
print("Fidelity after optimization: " + str(fidelity_after))
print("\nTarget state: " + str(sess.run(target_state)))
print("Output state: " + str(np.round(sess.run(ket), decimals=3)))

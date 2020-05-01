#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops


# =========================================================================
# Utility functions
# =========================================================================


# define interferometer
def interferometer(params, q):
    """Parameterised interferometer acting on ``N`` modes.

    Args:
        params (list[float]): list of length ``max(1, N-1) + (N-1)*N`` parameters.

            * The first ``N(N-1)/2`` parameters correspond to the beamsplitter angles
            * The second ``N(N-1)/2`` parameters correspond to the beamsplitter phases
            * The final ``N-1`` parameters correspond to local rotation on the first N-1 modes

        q (list[RegRef]): list of Strawberry Fields quantum registers the interferometer
            is to be applied to
    """
    N = len(q)
    theta = params[:N*(N-1)//2]
    phi = params[N*(N-1)//2:N*(N-1)]
    rphi = params[-N+1:]

    if N == 1:
        # the interferometer is a single rotation
        ops.Rgate(rphi[0]) | q[0]
        return

    n = 0  # keep track of free parameters

    # Apply the rectangular beamsplitter array
    # The array depth is N
    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            # skip even or odd pairs depending on layer
            if (l + k) % 2 != 1:
                ops.BSgate(theta[n], phi[n]) | (q1, q2)
                n += 1

    # apply the final local phase shifts to all modes except the last one
    for i in range(max(1, N - 1)):
        ops.Rgate(rphi[i]) | q[i]
    # Rgate only applied to first N - 1 modes


# define layer
def layer(params, q):
    """CV quantum neural network layer acting on ``N`` modes.

    Args:
        params (list[float]): list of length ``2*(max(1, N-1) + N**2 + n)`` containing
            the number of parameters for the layer
        q (list[RegRef]): list of Strawberry Fields quantum registers the layer
            is to be applied to
    """
    N = len(q)
    M = int(N * (N - 1)) + max(1, N - 1)

    int1 = params[:M]
    s = params[M:M+N]
    int2 = params[M+N:2*M+N]
    dr = params[2*M+N:2*M+2*N]
    dp = params[2*M+2*N:2*M+3*N]
    k = params[2*M+3*N:2*M+4*N]

    # begin layer
    interferometer(int1, q)

    for i in range(N):
        ops.Sgate(s[i]) | q[i]

    interferometer(int2, q)

    for i in range(N):
        ops.Dgate(dr[i], dp[i]) | q[i]
        ops.Kgate(k[i]) | q[i]
    # end layer


def init_weights(modes, layers, active_sd=0.0001, passive_sd=0.1):
    """Initialize a 2D TensorFlow Variable containing normally-distributed
    random weights for an ``N`` mode quantum neural network with ``L`` layers.

    Args:
        modes (int): the number of modes in the quantum neural network
        layers (int): the number of layers in the quantum neural network
        active_sd (float): the standard deviation used when initializing
            the normally-distributed weights for the active parameters
            (displacement, squeezing, and Kerr magnitude)
        passive_sd (float): the standard deviation used when initializing
            the normally-distributed weights for the passive parameters
            (beamsplitter angles and all gate phases)

    Returns:
        tf.Variable[tf.float32]: A TensorFlow Variable of shape
        ``[layers, 2*(max(1, modes-1) + modes**2 + modes)]``, where the Lth
        row represents the layer parameters for the Lth layer.
    """
    # Number of interferometer parameters:
    M = int(modes * (modes - 1)) + max(1, modes - 1)

    # Create the TensorFlow variables
    int1_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    s_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    int2_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    dr_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    dp_weights = tf.random.normal(shape=[layers, modes], stddev=passive_sd)
    k_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)

    weights = tf.concat([int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights], axis=1)
    weights = tf.Variable(weights)

    return weights


# =========================================================================
# Define the optimization problem
# =========================================================================


# set the random seed
tf.random.set_seed(137)
np.random.seed(137)


# define width and depth of CV quantum neural network
modes = 1
layers = 8
cutoff_dim = 6


# defining desired state (single photon state)
target_state = np.zeros(cutoff_dim)
target_state[1] = 1
target_state = tf.constant(target_state, dtype=tf.complex64)


# initialize engine and program
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
qnn = sf.Program(modes)


# initialize QNN weights
weights = init_weights(modes, layers)
num_params = np.prod(weights.shape)


# Create array of Strawberry Fields symbolic gate arguments, matching
# the size of the weights Variable.
sf_params = np.arange(num_params).reshape(weights.shape).astype(np.str)
sf_params = np.array([qnn.params(*i) for i in sf_params])


# Construct the symbolic Strawberry Fields program by
# looping and applying layers to the program.
with qnn.context as q:
    for k in range(layers):
        layer(sf_params[k], q)


def cost(weights):
    # Create a dictionary mapping from the names of the Strawberry Fields
    # symbolic gate parameters to the TensorFlow weight values.
    mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))}

    # run the engine
    state = eng.run(qnn, args=mapping).state
    ket = state.ket()

    difference = tf.reduce_sum(tf.abs(ket - target_state))
    fidelity = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target_state)) ** 2
    return difference, fidelity, ket, tf.math.real(state.trace())


# set up optimizer
opt = tf.keras.optimizers.Adam()
cost_before, fidelity_before, _, _ = cost(weights)

print("Beginning optimization")


# Perform the optimization
for i in range(1000):
    # reset the engine if it has already been executed
    if eng.run_progs:
        eng.reset()

    with tf.GradientTape() as tape:
        loss, fid, _, trace = cost(weights)

    # one repetition of the optimization
    gradients = tape.gradient(loss, weights)
    opt.apply_gradients(zip([gradients], [weights]))

    # Prints progress at every rep
    if i % 1 == 0:
        print("Rep: {} Cost: {:.4f} Fidelity: {:.4f} Trace: {:.4f}".format(i, loss, fid, trace))


cost_after, fidelity_after, ket_after, _ = cost(weights)


print("\nFidelity before optimization: ", fidelity_before.numpy())
print("Fidelity after optimization: ", fidelity_after.numpy())
print("\nTarget state: ", target_state.numpy())
print("Output state: ", np.round(ket_after.numpy(), decimals=3))

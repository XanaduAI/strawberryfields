r"""
Quantum gate synthesis
======================

This demonstration works through the process used to produce the gate
synthesis results presented in `"Machine learning method for state
preparation and gate synthesis on photonic quantum
computers" <https://arxiv.org/abs/1807.10781>`__.

This tutorial uses the TensorFlow backend of Strawberry Fields, giving us
access to a number of additional functionalities including: GPU integration,
automatic gradient computation, built-in optimization algorithms, and other
machine learning tools.


Variational quantum circuits
----------------------------

A key element of machine learning is optimization. We can use
TensorFlow's automatic differentiation tools to optimize the parameters
of variational quantum circuits constructed using Strawberry Fields. In
this approach, we fix a circuit architecture where the states, gates,
and/or measurements may have learnable parameters :math:`\vec{\theta}`
associated with them. We then define a loss function based on the output
state of this circuit. In this case, we define a loss function such that the
action of the variational quantum circuit is close to some specified target
unitary.

.. note::

    For more details on the TensorFlow backend in Strawberry Fields, please see
    :ref:`machine_learning_tutorial`.

For arbitrary gate synthesis using optimization, we need to make use of
a quantum circuit with a layer structure that is **universal** - that
is, by 'stacking' the layers, we can guarantee that we can produce *any*
CV state with at-most polynomial overhead. Therefore, the architecture
we choose must consist of layers with each layer containing
parameterized Gaussian *and* non-Gaussian gates. **The non-Gaussian
gates provide both the nonlinearity and the universality of the model.**
To this end, we employ the CV quantum neural network architecture
described below:

.. figure:: https://i.imgur.com/NEsaVIX.png
   :alt: layer

Here,

-  :math:`\mathcal{U}_i(\theta_i,\phi_i)` is an N-mode linear optical
   interferometer composed of two-mode beamsplitters
   :math:`BS(\theta,\phi)` and single-mode rotation gates
   :math:`R(\phi)=e^{i\phi\hat{n}}`,

-  :math:`\mathcal{D}(\alpha_i)` are single mode displacements in the
   phase space by complex value :math:`\alpha_i`,

-  :math:`\mathcal{S}(r_i, \phi_i)` are single mode squeezing operations
   of magnitude :math:`r_i` and phase :math:`\phi_i`, and

-  :math:`\Phi(\lambda_i)` is a single mode non-Gaussian operation, in
   this case chosen to be the Kerr interaction
   :math:`\mathcal{K}(\kappa_i)=e^{i\kappa_i\hat{n}^2}` of strength
   :math:`\kappa_i`.

Hyperparameters
---------------

First, we must define the **hyperparameters** of our layer structure:

-  ``cutoff``: the simulation Fock space truncation we will use in the
   optimization. The TensorFlow backend will perform numerical
   operations in this truncated Fock space when performing the
   optimization.

-  ``depth``: The number of layers in our variational quantum
   circuit. As a general rule, increasing the number of layers (and
   thus, the number of parameters we are optimizing over) increases the
   optimizer's chance of finding a reasonable local minimum in the
   optimization landscape.

-  ``reps``: the number of steps in the optimization routine performing
   gradient descent

Some other optional hyperparameters include:

-  The standard deviation of initial parameters. Note that we make a
   distinction between the standard deviation of *passive* parameters
   (those that preserve photon number when changed, such as phase
   parameters), and *active* parameters (those that introduce or remove
   energy from the system when changed).
"""
import numpy as np

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import operation

# Cutoff dimension
cutoff = 10

# gate cutoff
gate_cutoff = 4

# Number of layers
depth = 15

# Number of steps in optimization routine performing gradient descent
reps = 200

# Learning rate
lr = 0.025

# Standard deviation of initial parameters
passive_sd = 0.1
active_sd = 0.001


######################################################################
# Note that, unlike in state learning, we must also specify a *gate
# cutoff* :math:`d`. This restricts the target unitary to its action on a
# :math:`d`-dimensional subspace of the truncated Fock space, where
# :math:`d\leq D`, where :math:`D` is the overall simulation Fock basis
# cutoff. As a result, we restrict the gate synthesis optimization to only
# :math:`d` input-output relations.
#
# The layer parameters :math:`\vec{\theta}`
# -----------------------------------------
#
# We use TensorFlow to create the variables corresponding to the gate
# parameters. Note that each variable has shape ``(depth,)``, with each
# individual element representing the gate parameter in layer :math:`i`.
#

import tensorflow as tf

# set the random seed
tf.random.set_seed(42)
np.random.seed(42)

# squeeze gate
sq_r = tf.random.normal(shape=[depth], stddev=active_sd)
sq_phi = tf.random.normal(shape=[depth], stddev=passive_sd)

# displacement gate
d_r = tf.random.normal(shape=[depth], stddev=active_sd)
d_phi = tf.random.normal(shape=[depth], stddev=passive_sd)

# rotation gates
r1 = tf.random.normal(shape=[depth], stddev=passive_sd)
r2 = tf.random.normal(shape=[depth], stddev=passive_sd)

# kerr gate
kappa = tf.random.normal(shape=[depth], stddev=active_sd)


######################################################################
# For convenience, we store the TensorFlow variables representing the weights as
# a tensor:

weights = tf.convert_to_tensor([r1, sq_r, sq_phi, r2, d_r, d_phi, kappa])
weights = tf.Variable(tf.transpose(weights))

######################################################################
# Since we have a depth of 15 (so 15 layers), and each layer takes
# 7 different types of parameters, the final shape of our weights
# array should be :math:`\text{depth}\times 7` or ``(15, 7)``:

print(weights.shape)


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     (15, 7)


######################################################################
# Constructing the circuit
# ------------------------
# We can now construct the corresponding single-mode Strawberry Fields program:
#

# Single-mode Strawberry Fields program
prog = sf.Program(1)

# Create the 7 Strawberry Fields free parameters for each layer
sf_params = []
names = ["r1", "sq_r", "sq_phi", "r2", "d_r", "d_phi", "kappa"]

for i in range(depth):
    # For the ith layer, generate parameter names "r1_i", "sq_r_i", etc.
    sf_params_names = ["{}_{}".format(n, i) for n in names]
    # Create the parameters, and append them to our list ``sf_params``.
    sf_params.append(prog.params(*sf_params_names))

######################################################################
# ``sf_params`` is now a nested list of shape ``(depth, 7)``, matching
# the shape of ``weights``.

sf_params = np.array(sf_params)
print(sf_params.shape)


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     (15, 7)

######################################################################
# Now, we can create a function to define the :math:`i`\ th layer, acting
# on qumode ``q``. We add the :class:`~.utils.operation` decorator so that the layer can be used
# as a single operation when constructing our circuit within the usual
# Strawberry Fields Program context.

# layer architecture
@operation(1)
def layer(i, q):
    Rgate(sf_params[i][0]) | q
    Sgate(sf_params[i][1], sf_params[i][2]) | q
    Rgate(sf_params[i][3]) | q
    Dgate(sf_params[i][4], sf_params[i][5]) | q
    Kgate(sf_params[i][6]) | q
    return q

######################################################################
# We must also specify the input states to the variational quantum circuit
# - these are the Fock state :math:`\ket{i}`, :math:`i=0,\dots,d`,
# allowing us to optimize the circuit parameters to learn the target
# unitary acting on all input Fock states within the :math:`d`-dimensional
# subspace.
#

in_ket = np.zeros([gate_cutoff, cutoff])
np.fill_diagonal(in_ket, 1)

# Apply circuit of layers with corresponding depth
with prog.context as q:
    Ket(in_ket) | q
    for k in range(depth):
        layer(k) | q[0]

######################################################################
# Performing the optimization
# ---------------------------
#
# :math:`\newcommand{ket}[1]{\left|#1\right\rangle}`
# With the Strawberry Fields TensorFlow backend calculating the
# resulting state of the circuit symbolically, we can use TensorFlow to
# optimize the gate parameters to minimize the cost function we specify.
# With gate synthesis, we minimize the overlaps in the Fock basis
# between the target and learnt unitaries via the following cost function:
#
# .. math::
#
#     C(\vec{\theta}) = \frac{1}{d}\sum_{i=0}^{d-1} \left| \langle i \mid
#     V^\dagger U(\vec{\theta})\mid i\rangle - 1\right|
#
# where :math:`V` is the target unitary, :math:`U(\vec{\theta})` is the
# learnt unitary, and :math:`d` is the gate cutoff. Note that this is a
# generalization of state preparation to more than one input-output
# relation.
#
# For our target unitary, let's use Strawberry Fields to generate a 4x4
# random unitary:
#

from strawberryfields.utils import random_interferometer

# define unitary up to gate_cutoff
random_unitary = random_interferometer(gate_cutoff)
print(random_unitary)

# extend unitary up to cutoff
target_unitary = np.identity(cutoff, dtype=np.complex128)
target_unitary[:gate_cutoff, :gate_cutoff] = random_unitary


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     [[ 0.23648826-0.48221431j  0.06829648+0.04447898j  0.51150074-0.09529866j
#        0.55205719-0.35974699j]
#      [-0.11148167+0.69780321j -0.24943828+0.08410701j  0.46705929-0.43192981j
#        0.16220654-0.01817602j]
#      [-0.22351926-0.25918352j  0.24364996-0.05375623j -0.09259829-0.53810588j
#        0.27267708+0.66941977j]
#      [ 0.11519953-0.28596729j -0.90164923-0.22099186j -0.09627758-0.13105595j
#       -0.0200152 +0.12766128j]]

######################################################################
# This matches the gate cutoff of :math:`d=4` that we chose above when
# defining our hyperparameters.
#
# Now, we instantiate the Strawberry Fields TensorFlow backend:

eng = sf.Engine('tf', backend_options={"cutoff_dim": cutoff, "batch_size": gate_cutoff})

######################################################################
# Here, we use the ``batch_size`` argument to perform the optimization in
# parallel - each batch calculates the variational quantum circuit acting
# on a different input Fock state: :math:`U(\vec{\theta}) | n\rangle`.
#

in_state = np.arange(gate_cutoff)

# extract action of the target unitary acting on
# the allowed input fock states.
target_kets = np.array([target_unitary[:, i] for i in in_state])
target_kets = tf.constant(target_kets, dtype=tf.complex64)

######################################################################
# Using this target unitary, we define the cost function we would like to
# minimize. We must use TensorFlow functions to manipulate this data, as well
# as a ``GradientTape`` to keep track of the corresponding gradients!
#
def cost(weights):
    # Create a dictionary mapping from the names of the Strawberry Fields
    # free parameters to the TensorFlow weight values.
    mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))}

    # Run engine
    state = eng.run(prog, args=mapping).state

    # Extract the statevector
    ket = state.ket()

    # overlaps
    overlaps = tf.math.real(tf.einsum('bi,bi->b', tf.math.conj(target_kets), ket))
    mean_overlap = tf.reduce_mean(overlaps)

    # Objective function to minimize
    cost = tf.abs(tf.reduce_sum(overlaps - 1))
    return cost, overlaps, ket



######################################################################
# Now that the cost function is defined, we can define and run the
# optimization. Below, we choose the Adam optimizer that is built into
# TensorFlow.
#

# Using Adam algorithm for optimization
opt = tf.keras.optimizers.Adam(learning_rate=lr)


######################################################################
# We then loop over all repetitions, storing the best predicted overlap
# value.
#

overlap_progress = []
cost_progress = []

# Run optimization
for i in range(reps):

    # reset the engine if it has already been executed
    if eng.run_progs:
        eng.reset()

    # one repetition of the optimization
    with tf.GradientTape() as tape:
        loss, overlaps_val, ket_val = cost(weights)

    # calculate the mean overlap
    # This gives us an idea of how the optimization is progressing
    mean_overlap_val = np.mean(overlaps_val)

    # store cost at each step
    cost_progress.append(loss)
    overlap_progress.append(overlaps_val)

    # one repetition of the optimization
    gradients = tape.gradient(loss, weights)
    opt.apply_gradients(zip([gradients], [weights]))

    # Prints progress at every rep
    if i % 1 == 0:
        # print progress
        print("Rep: {} Cost: {:.4f} Mean overlap: {:.4f}".format(i, loss, mean_overlap_val))

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Rep: 0 Cost: 4.2947 Mean overlap: -0.0737
#     Rep: 1 Cost: 3.2125 Mean overlap: 0.1969
#     Rep: 2 Cost: 4.3774 Mean overlap: -0.0944
#     Rep: 3 Cost: 3.6844 Mean overlap: 0.0789
#     Rep: 4 Cost: 3.7196 Mean overlap: 0.0701
#     Rep: 5 Cost: 3.2360 Mean overlap: 0.1910
#     Rep: 6 Cost: 3.1782 Mean overlap: 0.2055
#     Rep: 7 Cost: 3.1090 Mean overlap: 0.2228
#     Rep: 8 Cost: 3.0612 Mean overlap: 0.2347
#     Rep: 9 Cost: 3.0602 Mean overlap: 0.2349
#     Rep: 10 Cost: 2.7062 Mean overlap: 0.3234
#     Rep: 11 Cost: 2.7678 Mean overlap: 0.3081
#     Rep: 12 Cost: 2.2194 Mean overlap: 0.4451
#     Rep: 13 Cost: 2.0853 Mean overlap: 0.4787
#     Rep: 14 Cost: 2.2453 Mean overlap: 0.4387
#     Rep: 15 Cost: 1.6812 Mean overlap: 0.5797
#     Rep: 16 Cost: 1.9051 Mean overlap: 0.5237
#     Rep: 17 Cost: 1.4323 Mean overlap: 0.6419
#     Rep: 18 Cost: 1.4057 Mean overlap: 0.6486
#     Rep: 19 Cost: 1.2089 Mean overlap: 0.6978
#     Rep: 20 Cost: 1.1891 Mean overlap: 0.7027
#     Rep: 21 Cost: 1.2000 Mean overlap: 0.7000
#     Rep: 22 Cost: 1.1956 Mean overlap: 0.7011
#     Rep: 23 Cost: 1.1900 Mean overlap: 0.7025
#     Rep: 24 Cost: 1.1602 Mean overlap: 0.7100
#     Rep: 25 Cost: 1.0940 Mean overlap: 0.7265
#     Rep: 26 Cost: 1.0667 Mean overlap: 0.7333
#     Rep: 27 Cost: 1.0020 Mean overlap: 0.7495
#     Rep: 28 Cost: 0.9791 Mean overlap: 0.7552
#     Rep: 29 Cost: 0.9648 Mean overlap: 0.7588
#     Rep: 30 Cost: 0.9356 Mean overlap: 0.7661
#     Rep: 31 Cost: 0.9526 Mean overlap: 0.7618
#     Rep: 32 Cost: 0.8805 Mean overlap: 0.7799
#     Rep: 33 Cost: 0.9031 Mean overlap: 0.7742
#     Rep: 34 Cost: 0.8137 Mean overlap: 0.7966
#     Rep: 35 Cost: 0.8456 Mean overlap: 0.7886
#     Rep: 36 Cost: 0.8192 Mean overlap: 0.7952
#     Rep: 37 Cost: 0.8201 Mean overlap: 0.7950
#     Rep: 38 Cost: 0.8197 Mean overlap: 0.7951
#     Rep: 39 Cost: 0.7660 Mean overlap: 0.8085
#     Rep: 40 Cost: 0.7770 Mean overlap: 0.8058
#     Rep: 41 Cost: 0.7360 Mean overlap: 0.8160
#     Rep: 42 Cost: 0.7346 Mean overlap: 0.8163
#     Rep: 43 Cost: 0.7328 Mean overlap: 0.8168
#     Rep: 44 Cost: 0.7027 Mean overlap: 0.8243
#     Rep: 45 Cost: 0.7108 Mean overlap: 0.8223
#     Rep: 46 Cost: 0.6900 Mean overlap: 0.8275
#     Rep: 47 Cost: 0.6831 Mean overlap: 0.8292
#     Rep: 48 Cost: 0.6847 Mean overlap: 0.8288
#     Rep: 49 Cost: 0.6627 Mean overlap: 0.8343
#     Rep: 50 Cost: 0.6624 Mean overlap: 0.8344
#     Rep: 51 Cost: 0.6518 Mean overlap: 0.8370
#     Rep: 52 Cost: 0.6354 Mean overlap: 0.8412
#     Rep: 53 Cost: 0.6388 Mean overlap: 0.8403
#     Rep: 54 Cost: 0.6310 Mean overlap: 0.8422
#     Rep: 55 Cost: 0.6186 Mean overlap: 0.8453
#     Rep: 56 Cost: 0.6168 Mean overlap: 0.8458
#     Rep: 57 Cost: 0.6052 Mean overlap: 0.8487
#     Rep: 58 Cost: 0.5878 Mean overlap: 0.8531
#     Rep: 59 Cost: 0.5823 Mean overlap: 0.8544
#     Rep: 60 Cost: 0.5790 Mean overlap: 0.8552
#     Rep: 61 Cost: 0.5666 Mean overlap: 0.8583
#     Rep: 62 Cost: 0.5546 Mean overlap: 0.8614
#     Rep: 63 Cost: 0.5487 Mean overlap: 0.8628
#     Rep: 64 Cost: 0.5416 Mean overlap: 0.8646
#     Rep: 65 Cost: 0.5304 Mean overlap: 0.8674
#     Rep: 66 Cost: 0.5214 Mean overlap: 0.8696
#     Rep: 67 Cost: 0.5165 Mean overlap: 0.8709
#     Rep: 68 Cost: 0.5098 Mean overlap: 0.8726
#     Rep: 69 Cost: 0.4999 Mean overlap: 0.8750
#     Rep: 70 Cost: 0.4911 Mean overlap: 0.8772
#     Rep: 71 Cost: 0.4850 Mean overlap: 0.8788
#     Rep: 72 Cost: 0.4789 Mean overlap: 0.8803
#     Rep: 73 Cost: 0.4711 Mean overlap: 0.8822
#     Rep: 74 Cost: 0.4627 Mean overlap: 0.8843
#     Rep: 75 Cost: 0.4551 Mean overlap: 0.8862
#     Rep: 76 Cost: 0.4484 Mean overlap: 0.8879
#     Rep: 77 Cost: 0.4420 Mean overlap: 0.8895
#     Rep: 78 Cost: 0.4360 Mean overlap: 0.8910
#     Rep: 79 Cost: 0.4307 Mean overlap: 0.8923
#     Rep: 80 Cost: 0.4261 Mean overlap: 0.8935
#     Rep: 81 Cost: 0.4217 Mean overlap: 0.8946
#     Rep: 82 Cost: 0.4175 Mean overlap: 0.8956
#     Rep: 83 Cost: 0.4135 Mean overlap: 0.8966
#     Rep: 84 Cost: 0.4106 Mean overlap: 0.8974
#     Rep: 85 Cost: 0.4089 Mean overlap: 0.8978
#     Rep: 86 Cost: 0.4093 Mean overlap: 0.8977
#     Rep: 87 Cost: 0.4116 Mean overlap: 0.8971
#     Rep: 88 Cost: 0.4185 Mean overlap: 0.8954
#     Rep: 89 Cost: 0.4284 Mean overlap: 0.8929
#     Rep: 90 Cost: 0.4437 Mean overlap: 0.8891
#     Rep: 91 Cost: 0.4490 Mean overlap: 0.8877
#     Rep: 92 Cost: 0.4441 Mean overlap: 0.8890
#     Rep: 93 Cost: 0.4154 Mean overlap: 0.8962
#     Rep: 94 Cost: 0.3849 Mean overlap: 0.9038
#     Rep: 95 Cost: 0.3642 Mean overlap: 0.9089
#     Rep: 96 Cost: 0.3601 Mean overlap: 0.9100
#     Rep: 97 Cost: 0.3686 Mean overlap: 0.9079
#     Rep: 98 Cost: 0.3814 Mean overlap: 0.9047
#     Rep: 99 Cost: 0.3918 Mean overlap: 0.9020
#     Rep: 100 Cost: 0.3890 Mean overlap: 0.9027
#     Rep: 101 Cost: 0.3765 Mean overlap: 0.9059
#     Rep: 102 Cost: 0.3562 Mean overlap: 0.9110
#     Rep: 103 Cost: 0.3395 Mean overlap: 0.9151
#     Rep: 104 Cost: 0.3304 Mean overlap: 0.9174
#     Rep: 105 Cost: 0.3291 Mean overlap: 0.9177
#     Rep: 106 Cost: 0.3333 Mean overlap: 0.9167
#     Rep: 107 Cost: 0.3396 Mean overlap: 0.9151
#     Rep: 108 Cost: 0.3465 Mean overlap: 0.9134
#     Rep: 109 Cost: 0.3496 Mean overlap: 0.9126
#     Rep: 110 Cost: 0.3499 Mean overlap: 0.9125
#     Rep: 111 Cost: 0.3426 Mean overlap: 0.9144
#     Rep: 112 Cost: 0.3324 Mean overlap: 0.9169
#     Rep: 113 Cost: 0.3190 Mean overlap: 0.9203
#     Rep: 114 Cost: 0.3071 Mean overlap: 0.9232
#     Rep: 115 Cost: 0.2975 Mean overlap: 0.9256
#     Rep: 116 Cost: 0.2909 Mean overlap: 0.9273
#     Rep: 117 Cost: 0.2868 Mean overlap: 0.9283
#     Rep: 118 Cost: 0.2845 Mean overlap: 0.9289
#     Rep: 119 Cost: 0.2838 Mean overlap: 0.9290
#     Rep: 120 Cost: 0.2848 Mean overlap: 0.9288
#     Rep: 121 Cost: 0.2888 Mean overlap: 0.9278
#     Rep: 122 Cost: 0.2966 Mean overlap: 0.9258
#     Rep: 123 Cost: 0.3116 Mean overlap: 0.9221
#     Rep: 124 Cost: 0.3308 Mean overlap: 0.9173
#     Rep: 125 Cost: 0.3530 Mean overlap: 0.9118
#     Rep: 126 Cost: 0.3528 Mean overlap: 0.9118
#     Rep: 127 Cost: 0.3306 Mean overlap: 0.9173
#     Rep: 128 Cost: 0.2866 Mean overlap: 0.9283
#     Rep: 129 Cost: 0.2533 Mean overlap: 0.9367
#     Rep: 130 Cost: 0.2439 Mean overlap: 0.9390
#     Rep: 131 Cost: 0.2555 Mean overlap: 0.9361
#     Rep: 132 Cost: 0.2737 Mean overlap: 0.9316
#     Rep: 133 Cost: 0.2785 Mean overlap: 0.9304
#     Rep: 134 Cost: 0.2655 Mean overlap: 0.9336
#     Rep: 135 Cost: 0.2411 Mean overlap: 0.9397
#     Rep: 136 Cost: 0.2240 Mean overlap: 0.9440
#     Rep: 137 Cost: 0.2216 Mean overlap: 0.9446
#     Rep: 138 Cost: 0.2291 Mean overlap: 0.9427
#     Rep: 139 Cost: 0.2360 Mean overlap: 0.9410
#     Rep: 140 Cost: 0.2332 Mean overlap: 0.9417
#     Rep: 141 Cost: 0.2215 Mean overlap: 0.9446
#     Rep: 142 Cost: 0.2068 Mean overlap: 0.9483
#     Rep: 143 Cost: 0.1970 Mean overlap: 0.9507
#     Rep: 144 Cost: 0.1943 Mean overlap: 0.9514
#     Rep: 145 Cost: 0.1964 Mean overlap: 0.9509
#     Rep: 146 Cost: 0.1992 Mean overlap: 0.9502
#     Rep: 147 Cost: 0.1992 Mean overlap: 0.9502
#     Rep: 148 Cost: 0.1956 Mean overlap: 0.9511
#     Rep: 149 Cost: 0.1886 Mean overlap: 0.9529
#     Rep: 150 Cost: 0.1805 Mean overlap: 0.9549
#     Rep: 151 Cost: 0.1729 Mean overlap: 0.9568
#     Rep: 152 Cost: 0.1670 Mean overlap: 0.9582
#     Rep: 153 Cost: 0.1629 Mean overlap: 0.9593
#     Rep: 154 Cost: 0.1602 Mean overlap: 0.9599
#     Rep: 155 Cost: 0.1586 Mean overlap: 0.9603
#     Rep: 156 Cost: 0.1581 Mean overlap: 0.9605
#     Rep: 157 Cost: 0.1588 Mean overlap: 0.9603
#     Rep: 158 Cost: 0.1613 Mean overlap: 0.9597
#     Rep: 159 Cost: 0.1671 Mean overlap: 0.9582
#     Rep: 160 Cost: 0.1782 Mean overlap: 0.9554
#     Rep: 161 Cost: 0.1975 Mean overlap: 0.9506
#     Rep: 162 Cost: 0.2244 Mean overlap: 0.9439
#     Rep: 163 Cost: 0.2522 Mean overlap: 0.9369
#     Rep: 164 Cost: 0.2572 Mean overlap: 0.9357
#     Rep: 165 Cost: 0.2291 Mean overlap: 0.9427
#     Rep: 166 Cost: 0.1776 Mean overlap: 0.9556
#     Rep: 167 Cost: 0.1401 Mean overlap: 0.9650
#     Rep: 168 Cost: 0.1342 Mean overlap: 0.9664
#     Rep: 169 Cost: 0.1540 Mean overlap: 0.9615
#     Rep: 170 Cost: 0.1779 Mean overlap: 0.9555
#     Rep: 171 Cost: 0.1819 Mean overlap: 0.9545
#     Rep: 172 Cost: 0.1622 Mean overlap: 0.9595
#     Rep: 173 Cost: 0.1354 Mean overlap: 0.9662
#     Rep: 174 Cost: 0.1232 Mean overlap: 0.9692
#     Rep: 175 Cost: 0.1301 Mean overlap: 0.9675
#     Rep: 176 Cost: 0.1448 Mean overlap: 0.9638
#     Rep: 177 Cost: 0.1523 Mean overlap: 0.9619
#     Rep: 178 Cost: 0.1451 Mean overlap: 0.9637
#     Rep: 179 Cost: 0.1301 Mean overlap: 0.9675
#     Rep: 180 Cost: 0.1182 Mean overlap: 0.9704
#     Rep: 181 Cost: 0.1162 Mean overlap: 0.9709
#     Rep: 182 Cost: 0.1223 Mean overlap: 0.9694
#     Rep: 183 Cost: 0.1296 Mean overlap: 0.9676
#     Rep: 184 Cost: 0.1324 Mean overlap: 0.9669
#     Rep: 185 Cost: 0.1285 Mean overlap: 0.9679
#     Rep: 186 Cost: 0.1208 Mean overlap: 0.9698
#     Rep: 187 Cost: 0.1133 Mean overlap: 0.9717
#     Rep: 188 Cost: 0.1091 Mean overlap: 0.9727
#     Rep: 189 Cost: 0.1089 Mean overlap: 0.9728
#     Rep: 190 Cost: 0.1114 Mean overlap: 0.9721
#     Rep: 191 Cost: 0.1149 Mean overlap: 0.9713
#     Rep: 192 Cost: 0.1178 Mean overlap: 0.9706
#     Rep: 193 Cost: 0.1194 Mean overlap: 0.9702
#     Rep: 194 Cost: 0.1191 Mean overlap: 0.9702
#     Rep: 195 Cost: 0.1176 Mean overlap: 0.9706
#     Rep: 196 Cost: 0.1148 Mean overlap: 0.9713
#     Rep: 197 Cost: 0.1119 Mean overlap: 0.9720
#     Rep: 198 Cost: 0.1090 Mean overlap: 0.9728
#     Rep: 199 Cost: 0.1064 Mean overlap: 0.9734


######################################################################
# Results and visualisation
# -------------------------
#
# Plotting the cost vs. optimization step:
#

from matplotlib import pyplot as plt
# %matplotlib inline
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
plt.style.use('default')

plt.plot(cost_progress)
plt.ylabel('Cost')
plt.xlabel('Step')
plt.show()


######################################################################
# .. image:: /_static/images/sphx_glr_run_gate_synthesis_001.png
#     :class: sphx-glr-single-img


######################################################################
# We can use matrix plots to plot the real and imaginary components of the
# target unitary :math:`V` and learnt unitary :math:`U`.
#

learnt_unitary = ket_val.numpy().T[:gate_cutoff, :gate_cutoff]
target_unitary = target_unitary[:gate_cutoff, :gate_cutoff]

fig, ax = plt.subplots(1, 4, figsize=(7, 4))
ax[0].matshow(target_unitary.real, cmap=plt.get_cmap('Reds'))
ax[1].matshow(target_unitary.imag, cmap=plt.get_cmap('Greens'))
ax[2].matshow(learnt_unitary.real, cmap=plt.get_cmap('Reds'))
ax[3].matshow(learnt_unitary.imag, cmap=plt.get_cmap('Greens'))

ax[0].set_xlabel(r'$\mathrm{Re}(V)$')
ax[1].set_xlabel(r'$\mathrm{Im}(V)$')
ax[2].set_xlabel(r'$\mathrm{Re}(U)$')
ax[3].set_xlabel(r'$\mathrm{Im}(U)$')
fig.show()

######################################################################
# .. image:: /_static/images/sphx_glr_run_gate_synthesis_002.png
#     :class: sphx-glr-single-img

######################################################################
# Process fidelity
# ----------------
#
# The process fidelity between the two unitaries is defined by
#
# .. math:: F_e  = \left| \left\langle \Psi(V) \mid \Psi(U)\right\rangle\right|^2
#
# where:
#
# -  :math:`\left|\Psi(V)\right\rangle` is the action of :math:`V` on one
#    half of a maximally entangled state :math:`\left|\phi\right\rangle`:
#
# .. math:: \left|\Psi(V)\right\rangle = (I\otimes V)\left|\phi\right\rangle,
#
# -  :math:`V` is the target unitary,
# -  :math:`U` the learnt unitary.
#

I = np.identity(gate_cutoff)
phi = I.flatten()/np.sqrt(gate_cutoff)
psiV = np.kron(I, target_unitary) @ phi
psiU = np.kron(I, learnt_unitary) @ phi



######################################################################
# Therefore, after 200 repetitions, the learnt unitary synthesized via a
# variational quantum circuit has the following process fidelity to the target
# unitary:
#

print(np.abs(np.vdot(psiV, psiU))**2)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     0.9481729779485119

######################################################################
# Circuit parameters
# ------------------
#
# We can also query the optimal variational circuit parameters
# :math:`\vec{\theta}` that resulted in the learnt unitary. For example,
# to determine the maximum squeezing magnitude in the variational quantum
# circuit:
#

print(np.max(np.abs(weights[:, 0])))


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     0.39022577

######################################################################
# Further results
# ---------------
#
# After downloading the tutorial, even more refined results can be obtained by
# increasing the number of repetitions (``reps``), changing the depth of the
# circuit or altering the gate cutoff!
#

######################################################################
# References
# ----------
#
# 1. Juan Miguel Arrazola, Thomas R. Bromley, Josh Izaac, Casey R. Myers,
#    Kamil Brádler, and Nathan Killoran. Machine learning method for state
#    preparation and gate synthesis on photonic quantum computers. `Quantum
#    Science and Technology, 4
#    024004 <https://iopscience.iop.org/article/10.1088/2058-9565/aaf59e>`__,
#    (2019).
#
# 2. Nathan Killoran, Thomas R. Bromley, Juan Miguel Arrazola, Maria Schuld,
#    Nicolas Quesada, and Seth Lloyd. Continuous-variable quantum neural networks.
#    `Physical Review Research, 1(3), 033063.
#    <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.1.033063>`__,
#    (2019).

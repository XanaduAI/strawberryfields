r"""
Quantum gate synthesis
======================

This demonstration works through the process used to produce the gate
synthesis results presented in `“Machine learning method for state
preparation and gate synthesis on photonic quantum
computers” <https://arxiv.org/abs/1807.10781>`__.

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
is, by ‘stacking’ the layers, we can guarantee that we can produce *any*
CV state with at-most polynomial overhead. Therefore, the architecture
we choose must consist of layers with each layer containing
parameterized Gaussian *and* non-Gaussian gates. **The non-Gaussian
gates provide both the nonlinearity and the universality of the model.**
To this end, we employ the CV quantum neural network architecture
described below:

.. figure:: https://i.imgur.com/NEsaVIX.png
   :alt: layer

   layer

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
# Now, we can create a function to define the :math:`i`\ th layer, acting
# on qumode ``q``. We add the :class:`~.utils.operation` decorator so that the layer can be used
# as a single operation when constructing our circuit within the usual
# Strawberry Fields Program context

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
target_unitary = np.identity(cutoff, dtype=np.complex128)
target_unitary[:gate_cutoff, :gate_cutoff] = random_interferometer(4)
print(target_unitary)


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
# We can use matrix plots to plot the real and imaginary components of the
# target and learnt unitary.
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
# Further results
# ---------------
#
# Even more refined results can be obtained by increasing the number of
# repetitions (``reps``) after downloading the tutorial!
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

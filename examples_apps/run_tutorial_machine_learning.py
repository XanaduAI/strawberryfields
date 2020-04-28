r"""
.. _machine_learning_tutorial:

Optimization & machine learning
===============================

.. note::

	The content in this page is suited to more advanced users who already have an understanding
	of Strawberry Fields, e.g., those who have completed the :ref:`teleportation tutorial <tutorial>`.
	Some basic knowledge of `TensorFlow <https://www.tensorflow.org/>`_ is also helpful.


In this demonstration, we show how the user can carry out optimization and machine learning on quantum
circuits in Strawberry Fields. This functionality is provided via the TensorFlow simulator
backend. By leveraging TensorFlow, we have access to a number of additional functionalities,
including GPU integration, automatic gradient computation, built-in optimization algorithms,
and other machine learning tools.

Basic functionality
-------------------

As usual, we can initialize a Strawberry Fields program:
"""
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *

prog = sf.Program(2)


##############################################################################
# Replacing numbers with Tensors
# ------------------------------
#
# When a circuit contains only numerical parameters, the TensorFlow backend works the same
# as the other backends. However, with the TensorFlow backend, we have the additional option
# to use TensorFlow ``tf.Variable`` and ``tf.tensor`` objects for the parameters of Blackbird
# states, gates, and measurements.
#
# To construct the circuit to allow for TensorFlow objects, we **must** use **symbolic
# parameters** as the gate arguments. These can be created via the ``Program.params()``
# method:

import tensorflow as tf

# we can create symbolic parameters one by one
alpha = prog.params("alpha")

# or create multiple at the same time
theta_bs, phi_bs = prog.params("theta_bs", "phi_bs")

with prog.context as q:
    # States
    Coherent(alpha) | q[0]
    # Gates
    BSgate(theta_bs, phi_bs) | (q[0], q[1])
    # Measurements
    MeasureHomodyne(0.0) | q[0]


##############################################################################
# To run a Strawberry Fields simulation with the TensorFlow backend, we need to specify
# ``'tf'`` as the backend argument when initializing the engine:

eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})

##############################################################################
# We can now run our program using :meth:`eng.run() <.Engine.run>`. However, directly
# evaluating a circuit which contains symbolic parameters using :meth:`eng.run() <.Engine.run>`
# will produce errors. The reason for this is that :meth:`eng.run() <.Engine.run>` tries,
# by default, to numerically evaluate any measurement result. But we have not provided
# numerical values for the symbolic arguments yet!
#
# .. code-block:: python3
#
#     eng.run(prog)
#
# .. rst-class:: sphx-glr-script-out
#
#     Out:
#
#     .. code-block:: none
#
#         ParameterError: {alpha}: unbound parameter with no default value.
#
# To bind a numerical value to the free parameters, we pass a mapping from parameter
# name to value using the ``args`` keyword argument when running the engine:

mapping = {"alpha": tf.Variable(0.5), "theta_bs": tf.constant(0.4), "phi_bs": tf.constant(0.0)}

result = eng.run(prog, args=mapping)


##############################################################################
# This code will execute without error, and both the output results and the
# output samples will contain numeric values based on the given value for the angle ``phi``:

print(result.samples)

##############################################################################
# We can select measurement results at other angles by supplying different values for
# ``phi`` in ``mapping``. We can also return and perform processing on the underlying state:

state = result.state
print("Density matrix element [0,0,1,2]:", state.dm()[0, 0, 1, 2])

##############################################################################
# Computing gradients
# -------------------
#
# When used within a gradient tape context, any output generated from the state
# class can be differentiated natively using ``GradientTape().gradient()``.
# For example, lets displace a one-mode program, and then compute the gradient
# of the mean photon number:

eng.reset()
prog = sf.Program(1)

alpha = prog.params("alpha")

with prog.context as q:
    Dgate(alpha) | q

# Assign our TensorFlow variables, so that we can
# refer to them later when differentiating/training.
a = tf.Variable(0.43)

with tf.GradientTape() as tape:
    # Here, we map our quantum free parameter `alpha`

    # to our TensorFlow variable `a` and pass it to the engine.

    result = eng.run(prog, args={"alpha": a})
    state = result.state

    # Note that all processing, including state-based post-processing,

    # must be done within the gradient tape context!
    mean, var = state.mean_photon(0)

# test that the gradient of the mean photon number is correct

grad = tape.gradient(mean, [a])
print("Gradient:", grad)

##############################################################################
# The mean photon number of a displaced state :math:`|\alpha\rangle` is given by
# :math:`\langle \hat{n} \rangle(\alpha) = |\alpha|^2`, and thus, for real-valued
# :math:`\alpha`, the gradient is given by :math:`\langle \hat{n}\rangle'(\alpha) = 2\alpha`:

print("Exact gradient:", 2 * a)
print("Exact and TensorFlow gradient agree:", np.allclose(grad, 2 * a))

##############################################################################
# Processing data
# ---------------
#
# The parameters for Blackbird states, gates, and measurements may be more complex
# than just raw data or machine learning weights. These can themselves be the outputs
# from some learnable function like a neural network [#]_.
#
# We can also use the ``sf.math`` module to allow arbitrary processing of measurement
# results within the TensorFlow backend, by manipulating the ``.par`` attribute of
# the measured register. Note that the math functions in ``sf.math`` will be automatically
# converted to the equivalent TensorFlow ``tf.math`` function:

eng.reset()
prog = sf.Program(2)

with prog.context as q:
    MeasureX | q[0]
    Dgate(sf.math.sin(q[0].par)) | q[1]

result = eng.run(prog)
print("Measured Homodyne sample from mode 0:", result.samples[0][0])

mean, var = result.state.mean_photon(0)
print("Mean photon number of mode 0:", mean)

mean, var = result.state.mean_photon(1)
print("Mean photon number of mode 1:", mean)


##############################################################################
# The mean photon number of mode 1 should be given by :math:`\sin(q[0])^2`, where :math:`q[0]`
# is the measured Homodyne sample value:

q0 = result.samples[0][0]
print(np.sin(q0) ** 2)

##############################################################################
# Working with batches
# --------------------
#
# It is common in machine learning to process data in *batches*. Strawberry Fields supports both
# unbatched and batched data when using the TensorFlow backend. Unbatched operation is the default
# behaviour (shown above). To enable batched operation, you should provide an extra
# ``batch_size`` argument within the ``backend_options`` dictionary [#]_, e.g.,


# run simulation in batched-processing mode
batch_size = 3
prog = sf.Program(1)
eng = sf.Engine("tf", backend_options={"cutoff_dim": 15, "batch_size": batch_size})

x = prog.params("x")

with prog.context as q:
    Thermal(x) | q[0]

x_val = tf.Variable([0.1, 0.2, 0.3])
result = eng.run(prog, args={"x": x_val})
print("Mean photon number of mode 0 (batched):", result.state.mean_photon(0)[0])

##############################################################################
# .. note:: The batch size should be static, i.e., not changing over the course of a computation.
#
# Parameters supplied to a circuit in batch-mode operation can either be scalars or vectors
# (of length ``batch_size``). Scalars are automatically broadcast over the batch dimension.

alpha = tf.Variable([0.5] * batch_size)
theta = tf.constant(0.0)
phi = tf.Variable([0.1, 0.33, 0.5])

##############################################################################
# Measurement results will be returned as Tensors with shape ``(batch_size,)``.
# We can picture batch-mode operation as simulating multiple circuit configurations at the
# same time. Combined with appropriate parallelized hardware like GPUs, this can result in
# significant speedups compared to serial evaluation.


##############################################################################
# Example: variational quantum circuit optimization
# -------------------------------------------------
#
# A key element of machine learning is optimization. We can use TensorFlow's automatic differentiation
# tools to optimize the parameters of *variational quantum circuits*. In this approach, we fix a
# circuit architecture where the states, gates, and/or measurements may have learnable parameters
# associated with them. We then define a loss function based on the output state of this circuit.
#
# .. warning::
#
#     The state representation in the simulator can change from a ket (pure) to a density
#     matrix (mixed) if we use certain operations (e.g., state preparations). We can check ``state.is_pure``
#     to determine which representation is being used.
#
# In the example below, we optimize a :class:`~.Dgate` to produce an output with the largest overlap
# with the Fock state :math:`n=1`. In this example, we compute the loss function imperatively
# within a gradient tape, and then apply the gradients using the chosen optimizer:

# initialize engine and program objects
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})
circuit = sf.Program(1)

tf_alpha = tf.Variable(0.1)
tf_phi = tf.Variable(0.1)

alpha, phi = circuit.params("alpha", "phi")

with circuit.context as q:
    Dgate(alpha, phi) | q[0]

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
steps = 50

for step in range(steps):

    # reset the engine if it has already been executed
    if eng.run_progs:
        eng.reset()

    with tf.GradientTape() as tape:
        # execute the engine
        results = eng.run(circuit, args={"alpha": tf_alpha, "phi": tf_phi})
        # get the probability of fock state |1>
        prob = results.state.fock_prob([1])
        # negative sign to maximize prob
        loss = -prob

    gradients = tape.gradient(loss, [tf_alpha, tf_phi])
    opt.apply_gradients(zip(gradients, [tf_alpha, tf_phi]))
    print("Probability at step {}: {}".format(step, prob))

##############################################################################
# After 50 or so iterations, the optimization should converge to the optimal probability value of
# :math:`e^{-1}\approx 0.3678` at a parameter of :math:`\alpha=1` [#]_.
#
# In cases where we want to take advantage of TensorFlow's graph mode,
# or do not need to extract/accumulate intermediate values used to calculate
# the cost function, we could also make use of TensorFlow's ``Optimizer().minimize``
# method, which requires the loss be defined as a function with no arguments:


def loss():
    # reset the engine if it has already been executed
    if eng.run_progs:
        eng.reset()

    # execute the engine
    results = eng.run(circuit, args={"alpha": tf_alpha, "phi": tf_phi})
    # get the probability of fock state |1>
    prob = results.state.fock_prob([1])
    # negative sign to maximize prob
    return -prob


tf_alpha = tf.Variable(0.1)
tf_phi = tf.Variable(0.1)


for step in range(steps):
    # In eager mode, calling Optimizer.minimize performs a single optimization step,
    # and automatically updates the parameter values.
    _ = opt.minimize(loss, [tf_alpha, tf_phi])
    parameter_vals = [tf_alpha.numpy(), tf_phi.numpy()]
    print("Parameter values at step {}: {}".format(step, parameter_vals))


###############################################################################
# Other high-level optimization interfaces, such as Keras, may also
# be used to train models built using the Strawberry Fields TensorFlow backend.
#
# .. warning::
#
#     When optimizing circuits which contains energy-changing operations (displacement, squeezing, etc.),
#     one should be careful to monitor that the state does not leak out of the given truncation level. This can
#     be accomplished by regularizing or using ``tf.clip_by_value`` on the relevant parameters.
#
# TensorFlow supports a large set of mathematical and machine learning operations which can be applied to
# a circuit's output state to enable further processing. Examples include ``tf.norm``,
# ``tf.linalg.eigh``, ``tf.linalg.svd``, and ``tf.linalg.inv`` [#]_.
#
# Exercise: Hong-Ou-Mandel Effect
# -------------------------------
#
# Use the optimization methods outlined above to find the famous
# `Hong-Ou-Mandel effect <https://en.wikipedia.org/wiki/Hong%E2%80%93Ou%E2%80%93Mandel_effect>`_,
# where photons bunch together in the same mode. Your circuit should contain two modes, each with a
# single-photon input state, and a beamsplitter with variable parameters. By optimizing the beamsplitter
# parameters, minimize the probability of the :math:`|1,1\rangle` Fock-basis element of the output state.
#
# .. rubric:: Footnotes
#
# .. [#] Note that certain operations---in particular, measurements---may not have gradients defined within
#        TensorFlow. When optimizing via gradient descent, we must be careful to define a circuit which is end-to-end differentiable.
#
#
# .. [#] Note that ``batch_size`` should not be set to 1. Instead, use ``batch_size=None``, or just
#        omit the ``batch_size`` argument.
#
# .. [#] In this tutorial, we have applied classical machine learning tools to learn a quantum optical circuit.
#        Of course, there are many other possibilities for combining machine learning and quantum computing,
#        e.g., using quantum algorithms to speed up machine learning subroutines, or fully quantum learning on
#        unprocessed quantum data.
#
# .. [#] Remember that it might be necessary to reshape a ket or density matrix before using some of these functions.

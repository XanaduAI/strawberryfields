r"""
.. _machine_learning_tutorial:

Optimization & machine learning
===============================

.. note::

	The content in this page is suited to more advanced users who already have an understanding
	of Strawberry Fields, e.g., those who have completed the :ref:`teleportation tutorial <tutorial>`.
	Some basic knowledge of `Tensorflow <https://www.tensorflow.org/>`_ is also helpful.

In this page, we show how the user can carry out optimization and machine learning on quantum
circuits in Strawberry Fields. This functionality is provided via the Tensorflow simulator
backend. By leveraging Tensorflow, we have access to a number of additional funtionalities,
including GPU integration, automatic gradient computation, built-in optimization algorithms,
and other machine learning tools.

Basic functionality
===================

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
# as the other backends. However, with the Tensorflow backend, we have the additional option
# to use Tensorflow ``tf.Variable`` and ``tf.tensor`` objects for the parameters of Blackbird
# states, gates, and measurements.
#
# To construct the circuit to allow for TensorFlow objects, we **must** use **symbolic
# parameters** as the gate arguments. These can be created via the ``Program.params()``
# method:

import tensorflow as tf

alpha = prog.params("alpha")
theta_bs = prog.params("theta_bs")
phi_bs = prog.params("phi_bs")

with prog.context as q:
    # States
    Coherent(alpha)            | q[0]
    # Gates
    BSgate(theta_bs, phi_bs)   | (q[0], q[1])
    # Measurements
    MeasureHomodyne(0.0)       | q[0]


##############################################################################
# To run a Strawberry Fields simulation with the Tensorflow backend, we need to specify
# ``'tf'`` as the backend argument when initializing the engine:

eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})

##############################################################################
# We can now run our program using :meth:`eng.run() <.Engine.run>`. However, directly
# evaluating a circuit which contains symbolic parameters using :meth:`eng.run() <.Engine.run>`
# will produce errors. The reason for this is that :meth:`eng.run() <.Engine.run>` tries,
# by default, to numerically evaluate any measurement result. But we have not provided
# numerical values for the symbolic arguments yet!
#
# >>> eng.run(prog)
#
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#    strawberryfields.parameters.ParameterError: {alpha}: unbound parameter with no default value.
#
# To bind a numerical value to the free parameters, we pass a mapping from parameter
# name to value using the ``args`` keyword argument when running the engine:

mapping = {
    "alpha": tf.Variable(0.5),
    "theta_bs": tf.constant(0.4),
    "phi_bs": tf.constant(0.0)
}

result = eng.run(prog, args=mapping)


##############################################################################
# This code will execute without error, and both the output results and the
# output samples will contain numeric values based on the given value for the angle ``phi``:

print(result.samples)

##############################################################################
# We can select measurement results at other angles by supplying different values for
# ``phi`` in ``mapping``. We can also return and perform processing on the underlying state:

state = result.state
print("Density matrix element [0, 0, 1, 2]:", state.dm()[0, 0, 1, 2])

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
    # Here, we pass map our quantum free parameter `alpha`
    # to our TensorFlow variable `a`.
    result = eng.run(prog, args={"alpha": a})
    state = result.state

    # Note that all processing, even state-based post-processing,
    # must be done within the gradient tape!
    mean, var = state.mean_photon(0)

# test the gradient of the variance is correct
grad = tape.gradient(mean, [a])
print(grad)

##############################################################################
# The mean photon number of a displaced state :math:|\alpha\rangle` is given by
#
# .. math:: \langle \hat{n} \rangle(\alpha) = |\alpha|^2
#
# and thus, for real-valued :math:`\alpha`, the gradient is given by
# :math:`\langle \hat{n}\rangle'(\alpha) = 2\alpha`:

print(2*a)
print(np.allclose(grad, 2*a))

##############################################################################
# Processing data
# ---------------
#
# The parameters for Blackbird states, gates, and measurements may be more complex
# than just raw data or machine learning weights. These can themselves be the outputs
# from some learnable function like a neural network.
#
# We can also use the ``sf.math`` module to allow arbitrary processing of measurement
# results within the Tensorflow backend, by manipulating the ``.par`` attribute of
# the measured register. Note that the math functions in ``sf.math`` will be automatically
# converted to the equivalent TensorFlow ``tf.math`` function:

eng.reset()
prog = sf.Program(2)

with prog.context as q:
    MeasureX | q[0]
    Dgate(sf.math.sin(q[0].par)) | q[1]

result = eng.run(prog)
print("Measured Homodyne sample from mode 0:", result.samples[0])

mean, var = result.state.mean_photon(0)
print("Mean photon number of mode 0:", mean)

mean, var = result.state.mean_photon(1)
print("Mean photon number of mode 1:", mean)

##############################################################################
# Working with batches
# --------------------
#


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

import strawberryfields as sf
from strawberryfields.ops import *

prog = sf.Program(2)

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

    alpha = tf.Variable(0.5)
    theta_bs = tf.constant(0.0)
    phi_bs = tf.constant(0.0)
    phi = tf.placeholder(tf.float32)

    with prog.context as q:
        # States
        Coherent(alpha)            | q[0]

        # Gates
        BSgate(theta_bs, phi_bs)   | (q[0], q[1])

        # Measurements
        MeasureHomodyne(phi)       | q[0]

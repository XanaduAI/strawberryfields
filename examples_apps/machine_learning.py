r"""
.. _machine_learning_tutorial:

Optimization & machine learning
===============================

.. note::

	The content in this page is suited to more advanced users who already have an understanding
	of Strawberry Fields, e.g., those who have completed the :ref:`teleportation tutorial <tutorial>`.
	Some basic knowledge of `TensorFlow <https://www.tensorflow.org/>`_ is also helpful.

.. note::

    This tutorial requires TensorFlow 2.0 and above. TensorFlow can be installed via ``pip``:

    .. code-block:: console

        pip install tensorflow

    For more installation details and instructions, please refer to the
    `TensorFlow documentation <https://www.tensorflow.org/install>`_.

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
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     [[<tf.Tensor: id=794, shape=(), dtype=complex64, numpy=(2.6271262+0j)>
#       None]]

##############################################################################
# We can select measurement results at other angles by supplying different values for
# ``phi`` in ``mapping``. We can also return and perform processing on the underlying state:

state = result.state
print("Density matrix element [0,0,1,2]:", state.dm()[0, 0, 1, 2])

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Density matrix element [0,0,1,2]: tf.Tensor((0.0050359764+0j), shape=(), dtype=complex64)

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
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Gradient: [<tf.Tensor: id=1343, shape=(), dtype=float32, numpy=0.85999966>]

##############################################################################
# The mean photon number of a displaced state :math:`|\alpha\rangle` is given by
# :math:`\langle \hat{n} \rangle(\alpha) = |\alpha|^2`, and thus, for real-valued
# :math:`\alpha`, the gradient is given by :math:`\langle \hat{n}\rangle'(\alpha) = 2\alpha`:

print("Exact gradient:", 2 * a)
print("Exact and TensorFlow gradient agree:", np.allclose(grad, 2 * a))

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Exact gradient: tf.Tensor(0.86, shape=(), dtype=float32)
#     Exact and TensorFlow gradient agree: True

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
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Measured Homodyne sample from mode 0: tf.Tensor((1.099311+0j), shape=(), dtype=complex64)
#     Mean photon number of mode 0: tf.Tensor(0.0, shape=(), dtype=float32)
#     Mean photon number of mode 1: tf.Tensor(0.793553, shape=(), dtype=float32)

##############################################################################
# The mean photon number of mode 1 should be given by :math:`\sin(q[0])^2`, where :math:`q[0]`
# is the measured Homodyne sample value:

q0 = result.samples[0][0]
print(np.sin(q0) ** 2)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     (0.7936931737133115+0j)

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
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Mean photon number of mode 0 (batched): tf.Tensor([0.09999999 0.19999996 0.30000004], shape=(3,), dtype=float32)


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
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Probability at step 0: 0.009900497272610664
#     Probability at step 1: 0.03843098133802414
#     Probability at step 2: 0.08083382993936539
#     Probability at step 3: 0.1331305205821991
#     Probability at step 4: 0.19035020470619202
#     Probability at step 5: 0.24677760899066925
#     Probability at step 6: 0.29659587144851685
#     Probability at step 7: 0.3348417580127716
#     Probability at step 8: 0.35853680968284607
#     Probability at step 9: 0.3676064908504486
#     Probability at step 10: 0.3649454414844513
#     Probability at step 11: 0.355256587266922
#     Probability at step 12: 0.3432542383670807
#     Probability at step 13: 0.3323868215084076
#     Probability at step 14: 0.32456544041633606
#     Probability at step 15: 0.32050856947898865
#     Probability at step 16: 0.3201928436756134
#     Probability at step 17: 0.3231735825538635
#     Probability at step 18: 0.3287637233734131
#     Probability at step 19: 0.3361213207244873
#     Probability at step 20: 0.3443049192428589
#     Probability at step 21: 0.3523356020450592
#     Probability at step 22: 0.3592878580093384
#     Probability at step 23: 0.364411860704422
#     Probability at step 24: 0.36726585030555725
#     Probability at step 25: 0.3678152561187744
#     Probability at step 26: 0.36645036935806274
#     Probability at step 27: 0.36389175057411194
#     Probability at step 28: 0.36100488901138306
#     Probability at step 29: 0.35858556628227234
#     Probability at step 30: 0.35718992352485657
#     Probability at step 31: 0.35705265402793884
#     Probability at step 32: 0.35809454321861267
#     Probability at step 33: 0.3599957227706909
#     Probability at step 34: 0.36230403184890747
#     Probability at step 35: 0.3645508885383606
#     Probability at step 36: 0.3663528263568878
#     Probability at step 37: 0.36747899651527405
#     Probability at step 38: 0.36787670850753784
#     Probability at step 39: 0.36765244603157043
#     Probability at step 40: 0.3670196831226349
#     Probability at step 41: 0.366233766078949
#     Probability at step 42: 0.3655299246311188
#     Probability at step 43: 0.3650795519351959
#     Probability at step 44: 0.3649670481681824
#     Probability at step 45: 0.365190327167511
#     Probability at step 46: 0.3656746745109558
#     Probability at step 47: 0.36629951000213623
#     Probability at step 48: 0.36692991852760315
#     Probability at step 49: 0.3674468994140625

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

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Parameter values at step 0: [0.17085811, 0.10000666]
#     Parameter values at step 1: [0.2653996, 0.10000824]
#     Parameter values at step 2: [0.3756343, 0.10003564]
#     Parameter values at step 3: [0.498407, 0.10004087]
#     Parameter values at step 4: [0.6317963, 0.10005679]
#     Parameter values at step 5: [0.7733106, 0.10010292]
#     Parameter values at step 6: [0.91795504, 0.100087106]
#     Parameter values at step 7: [1.056465, 0.09979849]
#     Parameter values at step 8: [1.1767575, 0.0995149]
#     Parameter values at step 9: [1.2695967, 0.099197276]
#     Parameter values at step 10: [1.3319764, 0.09904715]
#     Parameter values at step 11: [1.3654184, 0.098881036]
#     Parameter values at step 12: [1.3732629, 0.098441504]
#     Parameter values at step 13: [1.3591939, 0.09790572]
#     Parameter values at step 14: [1.3267639, 0.09771576]
#     Parameter values at step 15: [1.2793746, 0.09764076]
#     Parameter values at step 16: [1.2204303, 0.097491466]
#     Parameter values at step 17: [1.1535465, 0.09737021]
#     Parameter values at step 18: [1.0827549, 0.09747333]
#     Parameter values at step 19: [1.0126097, 0.097629994]
#     Parameter values at step 20: [0.9480358, 0.09750217]
#     Parameter values at step 21: [0.89378, 0.09736596]
#     Parameter values at step 22: [0.85358965, 0.09750438]
#     Parameter values at step 23: [0.8295505, 0.09758021]
#     Parameter values at step 24: [0.82193875, 0.097489305]
#     Parameter values at step 25: [0.82952106, 0.09749036]
#     Parameter values at step 26: [0.85000145, 0.09736506]
#     Parameter values at step 27: [0.8804064, 0.09721753]
#     Parameter values at step 28: [0.9173693, 0.09727109]
#     Parameter values at step 29: [0.957366, 0.09753705]
#     Parameter values at step 30: [0.99695796, 0.0978495]
#     Parameter values at step 31: [1.0330576, 0.09814763]
#     Parameter values at step 32: [1.0631745, 0.09897915]
#     Parameter values at step 33: [1.0855812, 0.09966516]
#     Parameter values at step 34: [1.0993629, 0.10014479]
#     Parameter values at step 35: [1.1043644, 0.10067672]
#     Parameter values at step 36: [1.1010801, 0.10097134]
#     Parameter values at step 37: [1.090526, 0.10137434]
#     Parameter values at step 38: [1.0741205, 0.102029756]
#     Parameter values at step 39: [1.0535735, 0.10259121]
#     Parameter values at step 40: [1.0307757, 0.102993794]
#     Parameter values at step 41: [1.0076759, 0.103460334]
#     Parameter values at step 42: [0.9861408, 0.10372111]
#     Parameter values at step 43: [0.967804, 0.10424348]
#     Parameter values at step 44: [0.9539274, 0.10505462]
#     Parameter values at step 45: [0.9452985, 0.105316624]
#     Parameter values at step 46: [0.9421848, 0.10547027]
#     Parameter values at step 47: [0.94434804, 0.105495274]
#     Parameter values at step 48: [0.9511101, 0.10539721]
#     Parameter values at step 49: [0.9614545, 0.1051061]


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

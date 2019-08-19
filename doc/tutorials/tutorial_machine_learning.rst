.. _machine_learning_tutorial:

Optimization & machine learning tutorial
########################################

.. sectionauthor:: Nathan Killoran <nathan@xanadu.ai>

.. note:: This is a more advanced tutorial for users who already have an understanding of Strawberry Fields, e.g., those who have completed the initial :ref:`teleportation tutorial <tutorial>`. Some basic knowledge of `Tensorflow <https://www.tensorflow.org/>`_ is also helpful.

In this tutorial, we show how the user can carry out optimization and machine learning on quantum circuits in Strawberry Fields. This functionality is provided via the Tensorflow simulator backend. By leveraging Tensorflow, we have
access to a number of additional funtionalities, including GPU integration, automatic gradient computation, built-in optimization algorithms, and other machine learning tools.

Basic functionality
===================

As usual, we can initialize a Strawberry Fields program:

.. code-block:: python

    import strawberryfields as sf
    from strawberryfields.ops import *

    prog = sf.Program(2)


Replacing numbers with Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a circuit contains only numerical parameters, the :ref:`tensorflow_backend` works the same as the other backends. However, with the Tensorflow backend, we have the additional option to use Tensorflow objects (e.g., :code:`tf.Variable`, :code:`tf.constant`, :code:`tf.placeholder`, or :code:`tf.Tensor`) for the parameters of Blackbird states, gates, and measurements.

.. code-block:: python

    import tensorflow as tf
    alpha = tf.Variable(0.5)
    theta_bs = tf.constant(0.0)
    phi_bs = tf.sigmoid(0.0) # this will be a tf.Tensor object
    phi = tf.placeholder(tf.float32)

    with prog.context as q:
        # States
        Coherent(alpha)            | q[0]

        # Gates
        BSgate(theta_bs, phi_bs)   | (q[0], q[1])

        # Measurements
        MeasureHomodyne(phi)       | q[0]


To run a Strawberry Fields simulation with the Tensorflow backend, we need to specify :code:`'tf'` as the backend argument when initializing the engine:

.. code-block:: python

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})

We can now run our program using :meth:`eng.run() <.Engine.run>`. However, directly evaluating a circuit which contains Tensorflow objects using :meth:`eng.run() <.Engine.run>` will produce errors. The reason for this is that :meth:`eng.run() <.Engine.run>` tries, by default, to numerically evaluate any measurement result. But Tensorflow requires several extra ingredients to do this:

1. Numerical computations must be carried out using a :code:`tf.Session`.

2. All :code:`tf.Variable` objects must be initialized within this :code:`tf.Session` (the initial values are supplied when creating the variables).

3. Numerical values must be provided for any :code:`tf.placeholder` objects using a feed dictionary (:code:`feed_dict`).

To properly evaluate measurement results, we must therefore do a little more work:

.. code-block:: python

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = {phi: 0.0}

    results = eng.run(prog, run_options={"session": sess, "feed_dict": feed_dict})

This code will execute without error, and both the output results and the register :code:`q` will contain numeric values based on the given value for the angle phi. We can select measurement results at other angles by supplying different values for :code:`phi` in :code:`feed_dict`.

.. note:: When being used as a numerical simulator (similar to the other backends), the Tensorflow backend creates temporary sessions in order to evaluate measurement results numerically.

Symbolic computation
~~~~~~~~~~~~~~~~~~~~

Supplying a :code:`Session` and :code:`feed_dict` to :code:`eng.run()` is okay for checking one or two numerical values.
However, each call of :code:`eng.run()` will create additional redundant nodes in the underlying Tensorflow computational graph.
A better method is to make the single call :code:`eng.run(prog, run_options={"eval": False})`. This will carry out the computation symbolically but not numerically.
The results returned by the engine will instead contain *unevaluted Tensors*. These Tensors can be evaluated numerically by running the :code:`tf.Session` and supplying the desired values for any placeholders:

.. code-block:: python

    prog = sf.Program(2)

    with prog.context as q:
        Dgate(alpha)         | q[0]
        MeasureHomodyne(phi) | q[0]

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})
    results = eng.run(prog, run_options={"eval": False})

    state_density_matrix = results.state.dm()
    homodyne_meas = results.samples[0]
    dm_x, meas_x = sess.run([state_density_matrix, homodyne_meas], feed_dict={phi: 0.0})

Processing data
~~~~~~~~~~~~~~~

The parameters for Blackbird states, gates, and measurements may be more complex than just raw data or machine learning weights. These can themselves be the outputs from some learnable function like a neural network:

.. code-block:: python

    input_ = tf.placeholder(tf.float32, shape=(2,1))
    weights = tf.Variable([[0.1, 0.1]])
    bias = tf.Variable(0.0)
    NN = tf.sigmoid(tf.matmul(weights, input_) + bias)
    NNDgate = Dgate(NN)

We can also use the :func:`strawberryfields.convert` decorator to allow arbitrary processing of measurement results with the Tensorflow backend [#]_.

.. code-block:: python

    @sf.convert
    def sigmoid(x):
        return tf.sigmoid(x)

    prog = sf.Program(2)

    with prog.context as q:
        MeasureX             | q[0]
        Dgate(sigmoid(q[0])) | q[1]

Working with batches
~~~~~~~~~~~~~~~~~~~~

It is common in machine learning to process data in *batches*. Strawberry Fields supports both unbatched and batched data when using the Tensorflow backend. Unbatched operation is the default behaviour (shown above). To enable batched operation, you should provide an extra :code:`batch_size` argument [#]_ within the :code:`backend_options` dictionary, e.g.,

.. code-block:: python

    # run simulation in batched-processing mode
    batch_size = 3
    prog = sf.Program(2)
    eng = sf.Engine('tf', backend_options={"cutoff_dim": 7, "batch_size": batch_size})

    with prog.context as q:
        Dgate(tf.Variable([0.1] * batch_size)) | q[0]

    result = eng.run(prog, run_options={"eval": False})

.. note:: The batch size should be static, i.e., not changing over the course of a computation.

Parameters supplied to a circuit in batch-mode operation can either be scalars or vectors (of length :code:`batch_size`). Scalars are automatically broadcast over the batch dimension.

.. code-block:: python

    alpha = tf.Variable([0.5] * batch_size)
    theta = tf.constant(0.0)
    phi = tf.Variable([0.1, 0.33, 0.5])

Measurement results will be returned as Tensors with shape :code:`(batch_size,)`. We can picture batch-mode operation as simulating multiple circuit configurations at the same time. Combined with appropriate parallelized hardware like GPUs, this can result in significant speedups compared to serial evaluation.

Example: variational quantum circuit optimization
=================================================

A key element of machine learning is optimization. We can use Tensorflow's automatic differentiation tools to optimize the parameters of *variational quantum circuits*. In this approach, we fix a circuit architecture where the states, gates, and/or measurements may have learnable parameters associated with them. We then define a loss function based on the output state of this circuit.

.. warning:: The state representation in the simulator can change from a ket (pure) to a density matrix (mixed) if we use certain operations (e.g., state preparations). We can check :code:`state.is_pure` to determine which representation is being used.

In the example below, we optimize a :class:`~.Dgate` to produce an output with the largest overlap with the Fock state :math:`n=1`.

.. literalinclude:: ../../examples/optimization.py
    :language: python

.. note:: This example program is included with Strawberry Fields as :download:`examples/optimization.py <../../examples/optimization.py>`.

After 50 or so iterations, the optimization should converge to the optimal probability value of :math:`e^{-1}\approx 0.3678` at a parameter of :math:`\alpha=1` [#]_.

.. warning:: When optimizing circuits which contains energy-changing operations (displacement, squeezing, etc.), one should be careful to monitor that the state does not leak out of the given truncation level. This can be accomplished by regularizing or using :code:`tf.clip_by_value` on the relevant parameters.

Tensorflow supports a large set of mathematical and machine learning operations which can be applied to a circuit's output state to enable further processing. Examples include :code:`tf.norm`, :code:`tf.self_adjoint_eig`, :code:`tf.svd`, and :code:`tf.inv` [#]_.

Exercise: Hong-Ou-Mandel Effect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the optimization methods outlined above to find the famous `Hong-Ou-Mandel effect <https://en.wikipedia.org/wiki/Hong%E2%80%93Ou%E2%80%93Mandel_effect>`_, where photons bunch together in the same mode. Your circuit should contain two modes, each with a single-photon input state, and a beamsplitter with variable parameters. By optimizing the beamsplitter parameters, minimize the probability of the :math:`|1,1\rangle` Fock-basis element of the output state.

.. rubric:: Footnotes

.. [#] Note that certain operations---in particular, measurements---may not have gradients defined within Tensorflow. When optimizing via gradient descent, we must be careful to define a circuit which is end-to-end differentiable.

.. [#] Note that :code:`batch_size` should not be set to 1. Instead, use ``batch_size=None``, or just omit the ``batch_size`` argument.

.. [#] In this tutorial, we have applied classical machine learning tools to learn a quantum optical circuit. Of course, there are many other possibilities for combining machine learning and quantum computing, e.g., using quantum algorithms to speed up machine learning subroutines, or fully quantum learning on unprocessed quantum data.

.. [#] Remember that it might be necessary to reshape a ket or density matrix before using some of these functions.

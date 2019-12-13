Circuits
========

.. raw:: html

    <style>
      .sphx-glr-thumbcontainer {
        float: center;
        margin-top: 20px;
        margin-right: 50px;
              margin-bottom: 20px;
      }
          #right-column.card {
              box-shadow: none!important;
          }
          #right-column.card:hover {
              box-shadow: none!important;
          }
    </style>


.. toctree::
    :maxdepth: 1
    :hidden:

    circuits/glossary

In Strawberry Fields, photonic quantum circuits are represented as :class:`~.Program`
objects. By creating a program, quantum operations can be applied, measurements performed,
and the program can then be simulated using the various Strawberry Fields backends.

.. seealso::

    New to photonic quantum computing? See the :doc:`circuits/glossary` for a glossary
    of some of the terms used in Strawberry Fields.

Creating a quantum program
--------------------------

To construct a photonic quantum circuit in Strawberry Fields, a :class:`~.Program` object
must be created, and operations applied.

.. code-block:: python3

    import strawberryfields as sf
    from strawberryfields import ops

    # create a 3 mode quantum program
    prog = sf.Program(3)

    with prog.context as q:
        ops.Sgate(0.54) | q[0]
        ops.Sgate(0.54) | q[1]
        ops.Sgate(0.54) | q[2]
        ops.BSgate(0.43, 0.1) | (q[0], q[2])
        ops.BSgate(0.43, 0.1) | (q[1], q[2])
        ops.MeasureFock() | q

Here, we are creating a 3-mode program, and applying a :class:`~.Sgate` and
a :class:`~.BSgate` to various modes. Constructing quantum circuits always follows
the same structure as the above example; in particular,

* A ``with`` statement is used to populate the program with quantum operations.

* ``Program.context`` is used within the ``with`` statement to return ``q``,
  a representation of the quantum registers (qumodes or modes).

* Quantum operations are applied to the modes of the program using the syntax

  .. code-block:: python3

      ops.GateName(arg1, arg2) | (q[i], q[j], ...)

.. note::

    The contents of a program can be viewed by calling :meth:`.Program.print` method,
    or output as a qcircuit :math:`\LaTeX{}` document by using the :meth:`.Program.draw_circuit`
    method.

.. seealso::

    Visit the :doc:`ops` page to see an overview of available quantum operations.


Simulating your program
-----------------------

Strawberry Fields provides several backend simulators for simulating
your quantum program. To access the simulators, an **engine** must be initialized,
which is responsible for executing the program on a specified backend
(which can be either a local simulator, or a remote simulator/hardware device).

.. code-block:: python3

    # intialize the fock backend with a
    # Fock cutoff dimension (truncation) of 5
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})

:class:`~.Engine` accepts two arguments, the backend name, and a dictionary of
backend options. Available backends include:

* The ``'fock'`` backend, written in NumPy.

  This backend represents the quantum state
  and operations via the Fock basis, so can represent all possible CV states and
  operations. However numerical error is also introduced due to truncation of the Fock
  space—increasing the cutoff results in higher accuracy at a cost of increased memory consumption.

* The ``'gaussian'`` backend, written in NumPy.

  This backend represents the quantum
  state as a Gaussian, and operations as quantum operations. It is numerically exact,
  and consumes less memory and is less computationally intensive then the Fock backends.
  However, it cannot represent non-Gaussian operations and states, with the exception of
  terminal Fock measurements.

* The ``'tf'`` backend, written in TensorFlow 1.3.

  This backend represents the quantum
  state and operations via the Fock basis, but allows for backpropagation and optimization
  using TensorFlow.

Once the engine has been initialized, the quantum program can be executed on the
selected backend via :meth:`~.Engine.run`:

.. code-block:: python3

    result = eng.run(prog)

Execution results
-----------------

The returned :class:`~Result` object provides several useful properties
for accessing the results of your program execution:

* ``results.state``: The quantum state object contains details and methods
  for manipulation of the final circuit state. Not available for remote
  backends.

  .. code-block:: python

      >>> print(result.state)
      <FockState: num_modes=3, cutoff=15, pure=False, hbar=2.0>
      >>> state = result.state
      >>> state.trace()    # trace of the quantum state
      0.999998
      >>> state.dm().shape # density matrix
      [5, 5, 5]

* ``results.samples``: Measurement samples from any measurements performed.

  .. code-block:: pycon

      >>> results.samples
      [0, 0, 2]

.. seealso::

    Visit the :doc:`states` page to see an overview of available quantum state methods.


.. note::

    Measured modes will always be returned to the vacuum state.

.. warning::

    To avoid significant numerical error when working with Fock backends, ensure that
    the trace of your program after simulation remains reasonably close to 1,
    by calling :meth:`state.trace() <.BaseFockState.trace>`. If it is :math:`\leq1`, you will need
    to increase the cutoff dimension.


Compilation
-----------

The :class:`~.Program` object also provides *compilation* methods, that
automatically transform your circuit into an *equivalent* circuit with
a particular layout or topology. For example, the ``gbs`` compile target will
compile a circuit consisting of Gaussian operations and Fock measurements
into canonical Gaussian boson sampling form.

.. code-block:: python3

    prog2 = prog.compile('gbs')


Related tutorials
-----------------

For more details and guides on creating and simulating photonic quantum
circuits, see the following tutorials.

.. customgalleryitem::
    :tooltip: Building photonic quantum circuits
    :description: :doc:`/tutorials/blackbird`

.. customgalleryitem::
    :tooltip: Quantum teleportation
    :description: :doc:`Basic tutorial: teleportation </tutorials/tutorial_teleportation>`
    :figure: /_static/teleport.png

.. customgalleryitem::
    :tooltip: Making photonic measurements
    :description: :doc:`Measurements and post-selection </tutorials/tutorial_post_selection>`
    :figure: /_static/bs_measure.png

.. raw:: html

        <div style='clear:both'></div>
        <br>

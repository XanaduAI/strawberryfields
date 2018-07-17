.. _tutorial:

Quantum teleportation tutorial
##############################

.. sectionauthor:: Josh Izaac <josh@xanadu.ai>

.. note:: This tutorial is also available in the form of an interactive Jupter Notebook :download:`QuantumTeleportation.ipynb <../../examples/QuantumTeleportation.ipynb>`

To see how to construct and simulate a simple continuous-variable (CV) quantum circuit in Strawberry Fields, let's consider the case of **state teleportation** (see the respective section on the :ref:`quantum algorithms <state_teleportation>` page for a more technical discussion).

Importing Strawberry Fields
============================

To start with, create a new text file with name :file:`teleport.py`, and open it in a text editor. The first thing we need to do is import Strawberry Fields; we do this with the following import statements:

.. code-block:: python

    #!/usr/bin/env python3
    import strawberryfields as sf
    from strawberryfields.ops import *
    from strawberryfields.utils import scale
    from numpy import pi, sqrt

The first import statement imports Strawberry Fields as ``sf``, allowing us to access the engine and backends. The second import statement imports all available CV gates into the global namespace, while the third import imports a prebuilt classical processing function from :mod:`strawberryfields.utils`, a module containing Strawberry Fields utilities and extensions. Finally, we import :math:`\pi` and the square root from ``NumPy`` so that we can pass angle parameters to gates such as beamsplitters, and perform some custom classical processing.

.. Finally, the third import imports various utilities we will need for teleportaton, in this case the :func:`~.convert` function.

Engine initialization
======================

We can now initialize our chosen backend, engine, and quantum register, using the 
:func:`strawberryfields.Engine()` function, which has the following syntax:

``sf.Engine(num_subsystems, hbar=2)``

where

* ``num_subsystems`` (*int*) is the number of modes we want to initialize in our quantum register
* ``hbar`` (*float*) is the value of :math:`\hbar` we choose for the commutation relation :math:`[\x,\p]=i\hbar`. This is a matter of convention, and by default, if not provided, is assumed to be 2. See :ref:`conventions` for more details.

.. warning:: The value of :math:`\hbar` chosen modifies the application of the :class:`~.Xgate` and :class:`~.Zgate`, as well as the measurements returned by Homodyne measurement :class:`~.MeasureHomodyne`, so this must be taken into account if the value of :math:`\hbar` is modified. All other gates are unaffected.

Therefore, to initialize our engine:

.. code-block:: python

    eng, q = sf.Engine(3)

The ``Engine`` function returns two objects - the quantum engine itself, ``eng``, and our 3 mode quantum register, ``q``. Let's get to building our teleportation circuit.

Circuit construction
=====================

To prepare states and apply gates, we must be inside the context of the engine we initialized, using the ``with`` statement. Everything within the engine context is written using the :ref:`Blackbird quantum programming language <blackbird>`. For example, to construct the following state teleportation circuit

.. raw:: html

    <br>

.. image:: ../_static/teleport.svg
   :width: 60%
   :align: center
   :target: javascript:void(0);

.. raw:: html

    <br>
    
to teleport the coherent state :math:`\ket{\alpha}` where :math:`\alpha=1+0.5i`:

.. code-block:: python

    @sf.convert
    def custom(x):
        return -x*sqrt(2)

    with eng:
        # prepare initial states
        Coherent(1+0.5j) | q[0]
        Squeezed(-2) | q[1]
        Squeezed(2) | q[2]

        # apply gates
        BS = BSgate(pi/4, pi)
        BS | (q[1], q[2])
        BS | (q[0], q[1])

        # Perform homodyne measurements
        MeasureX | q[0]
        MeasureP | q[1]

        # Displacement gates conditioned on
        # the measurements
        Xgate(scale(q[0], sqrt(2))) | q[2]
        Zgate(custom(q[1])) | q[2]

A couple of things to note here:

* **The quantum register returned from the** :func:`strawberryfields.Engine` **function is a sequence**. Individual modes can be accessed via standard Python indexing and slicing techniques.

.. 
    * **Preparing initial states, measurements, and gate operations all make use of the following syntax:**

..      ``Operation([arg1, arg2, ...]) | reg``

..      where the number of arguments depends on the specific operation, and ``reg`` is either a single mode or a sequence of modes, depending on how many modes the operation acts on. For a full list of operations and gates available, see the :ref:`quantum gates <gates>` documentation.

* **Every time a operation is applied it is added to the command queue**, ready to be simulated by the backend.

.. 

* **Operations must be applied in temporal order**. Different operation orderings can result in the same quantum circuit, providing the operations do not apply sequentially to the same mode. For example, we can permute the line containing ``MeasureX`` and ``MeasureP`` without changing the result.

.. 

* **Gates are standard Python objects, and can be treated as such**. In this case, since both beamsplitters use the same parameters, a single instance is being instantiated and stored under variable ``BS``.

.. 

* **The results of measured modes are passed to gates simply by passing the measured mode as an argument.** In order to perform additional classical processing to the measured mode, we can use the basic classical processing functions available in :mod:`strawberryfields.utils`; here we used the :func:`~.scale` function. In addition, we use the :func:`~strawberryfields.convert` decorator that we imported earlier to do more complicated classical processing, by converting our user-defined function, ``custom(x)``, to one that accepts quantum registers as arguments.

.. note:: By choosing a different phase for the 50-50 beamsplitter, that is, ``BSgate(pi/4,0)``, we can avoid having to negate the :class:`Zgate` correction! However, for the purposes of this tutorial, we will continue to use the currently defined beamsplitter so as to show how the :func:`~.convert` decorator works.

Running the engine
==================

Once the circuit is constructed, you can run the engine via the :func:`strawberryfields.engine.Engine.run` method:

``eng.run(backend=None, return_state=True, modes=None, apply_history=False, **kwargs)``

The :meth:`eng.run <.Engine.run>` method accepts the arguments

* ``backend``: a string or :class:`~.BaseBackend` object representing the Strawberry Fields backend we wish to use; we have the choice of two Fock backends [#]_, the NumPy based (``'fock'``) and Tensorflow (``'tf'``), and one Gaussian backend [#]_ (``'gaussian'``).

  This is *required* the first time running the engine, but optional for subsequent runs - if not provided, the previously used backend will continue to be used.

  Note that if the backend string is altered in a later call to :meth:`eng.run <.Engine.run>`, for example by switching from ``'fock'`` to ``'gaussian'``, this is treated as a new backend, initialised in the vacuum state.

.. 

* ``return_state``: if true, returns an object representing the quantum state after the circuit simulation.

  Depending on backend used, the state returned might be a :class:`~.BaseFockState`, which represents the state using the Fock/number basis, or might be a :class:`~.BaseGaussianState`, which represents the state using Gaussian representation, as a vector of means and a covariance matrix. Many methods are provided for state manipulation, see :ref:`state_class` for more details.

.. 

* ``modes``: a list of integers, that specifies which modes we wish to return in the state object. If the state is a mixed state represented by a density matrix, then the engine will automatically perform a partial trace to return only the modes specified. Note that this only affects the returned state object - all modes remain in the backend circuit.

.. 

* ``apply_history``: a boolean that instructs the engine to begin the simulation by reapplying all previously applied circuit operations, before applying any newly queued operations. This is useful if you are changing or resetting the backend, and want to quickly bring the new circuit back to the same state, without re-queueing the same operations.

For more details on the technical differences between the backends, see :ref:`backends`.

Let's choose the Fock backend for this particular example. Since we are working in the Fock basis, we must also specify the Fock basis *cutoff dimension*; let's choose ``cutoff_dim=15``, such that a state :math:`\ket{\psi}` has approximation

.. math::

    \ket{\psi} = \sum_{n=0}^\infty c_n\ket{n} \approx \sum_{n=0}^{\texttt{cutoff_dim}-1} c_n\ket{n}

in our truncated Fock basis. We now have all the parameters ready to run the simulation:

.. code-block:: python

    state = eng.run('fock', cutoff_dim=15)

.. warning::

    To avoid significant numerical error when working with the Fock backend, we need to make sure from now on that all initial states and gates we apply result in negligible amplitude in the Fock basis for Fock states :math:`\ket{n}, ~~n\geq \texttt{cutoff_dim}`.

    For example, to prepare a squeezed vacuum state in the :math:`x` quadrature with ``cutoff_dim=10``, a squeezing factor of :math:`r=1` provides an acceptable approximation, since :math:`|\braketD{n}{z}|^2<0.02` for :math:`n\geq 10`.

Note that :meth:`eng.run <strawberryfields.engine.Engine.run>` can either go inside or outside the engine context used earlier, and will execute all operations within the command queue. You can continue to queue operations after running the engine, allowing you to break your circuit execution into multiple command queues and engine runs.

Other useful engine methods that can be called at any time include:

* :func:`eng.print_queue() <strawberryfields.engine.Engine.print_queue>`: print the command queue (the operations to be applied on the next call to :meth:`eng.run <strawberryfields.engine.Engine.run>`)

* :func:`eng.print_applied() <strawberryfields.engine.Engine.print_applied>`: prints all commands applied using :meth:`eng.run <strawberryfields.engine.Engine.run>` since the last backend reset/initialisation.

  - Unlike ``print_queue()``, this shows all applied gate decompositions, which may differ depending on the backend.

* :func:`eng.reset() <strawberryfields.engine.Engine.reset>`: clear all operations from the command queue, and resets the backend circuit to the vacuum state.

  - Optionally, if the keyword argument ``keep_history=True`` is provided, the entire circuit history up to that point, as well as the current command queue, are retained and will be applied on the next call to ``eng.run()``.

* :func:`eng.reset_queue() <strawberryfields.engine.Engine.reset_queue>`: clear all operations from the command queue, leaving the backend circuit unchanged.

Results and visualization
==========================

To analyze these results, it is convenient to now move to a Python console or interactive environment, such as `iPython <https://ipython.org/>`_ or `Jupyter Notebook <http://jupyter.org/>`_. In the following, Python input will be specified with the prompt ``>>>``, and output will follow.

Once the engine has been run, we can extract results of measurements and the quantum state from the circuit. Any measurements performed on a mode are stored in the mode attribute :func:`RegRef.val <strawberryfields.engine.RegRef.val>`:

.. code-block:: pycon
    
    >>> q[0].val
    2.9645296452964534
    >>> q[1].val
    -2.9465294652946525

If a mode has not been measured, this attribute simply returns ``None``:

.. code-block:: pycon

    >>> print(q[2].val)
    None

In this particular example, we are using the Fock backend, and so the state that was returned by the ``eng.run`` is in the Fock basis. To double check this, we can inspect it with the ``print`` function:

.. code-block:: python

    >>> print(state)
    <FockState: num_modes=3, cutoff=15, pure=False, hbar=2.0>

In addition to the parameters we have already configured when creating and running the engine, the line ``pure=False``, indicates that this is a mixed state represented as a density matrix, and not a state vector.

To return the density matrix representing the Fock state, we can use the method :meth:`state.dm <.BaseFockState.dm>` [#]_. In this case, the density matrix has dimension

.. code-block:: pycon
    
    >>> state.dm().shape
    (15, 15, 15, 15, 15, 15)

Here, we use the convention that every pair of consecutive dimensions corresponds to a subsystem; i.e.,

.. math::

    \rho_{\underbrace{ij}_{q[0]}~\underbrace{kl}_{q[1]}~\underbrace{mn}_{q[2]}}

Thus we can calculate the reduced density matrix for mode ``q[2]``, :math:`\rho_2`:

.. code-block:: pycon

    >>> import numpy as np
    >>> rho2 = np.einsum('kkllij->ij', state.dm())
    >>> rho2.shape
    (15, 15)

.. note:: The Fock state also provides the method :meth:`~.BaseFockState.reduced_dm` for extracting the reduced density matrix automatically.

The diagonal values of the reduced density matrix contain the marginal Fock state probabilities :math:`|\braketD{i}{\rho_2}|^2,~~ 0\leq i\leq 14`:

.. code-block:: pycon

    >>> probs = np.real_if_close(np.diagonal(rho2))
    >>> print(probs)
    array([  2.61948280e-01,   3.07005910e-01,   2.44374603e-01,
         1.22884591e-01,   3.79861250e-02,   1.27283154e-02,
         2.40961681e-03,   1.79702250e-04,   1.10907533e-05,
         2.54431653e-05,   3.30439758e-05,   1.38338559e-05,
         4.72489428e-05,   2.11951333e-05,   9.01969688e-06])

We can then use a package such as matplotlib to plot the marginal Fock state probability distributions for the first 6 Fock states, for the teleported mode ``q[2]``:

.. code-block:: pycon

    >>> from matplotlib import pyplot as plt
    >>> plt.bar(range(7), probs[:7])
    >>> plt.xlabel('Fock state')
    >>> plt.ylabel('Marginal probability')
    >>> plt.title('Mode 2')
    >>> plt.show()

.. raw:: html

    <br>

.. image:: ../_static/fock_teleport.svg
    :align: center
    :width: 50%
    :target: javascript:void(0);

.. raw:: html

    <br>

.. _fock_prob_tutorial:

Note that this information can also be extracted automatically via the Fock state method :meth:`~.BaseFockState.all_fock_probs`:

.. code-block:: pycon
    
    >>> fock_probs = state.all_fock_probs()
    >>> fock_probs.shape
    (15,15,15)
    >>> np.sum(fock_probs, axis=(0,1))
    array([  2.61948280e-01,   3.07005910e-01,   2.44374603e-01,
         1.22884591e-01,   3.79861250e-02,   1.27283154e-02,
         2.40961681e-03,   1.79702250e-04,   1.10907533e-05,
         2.54431653e-05,   3.30439758e-05,   1.38338559e-05,
         4.72489428e-05,   2.11951333e-05,   9.01969688e-06])


Full program
============

:file:`teleport.py`:

.. code-block:: python

    #!/usr/bin/env python3
    import strawberryfields as sf
    from strawberryfields.ops import *
    from strawberryfields.utils import scale
    from numpy import pi, sqrt

    eng, q = sf.Engine(3)

    @sf.convert
    def custom(x):
        return -x*sqrt(2)

    with eng:
        # prepare initial states
        Coherent(1+0.5j) | q[0]
        Squeezed(-2) | q[1]
        Squeezed(2) | q[2]

        # apply gates
        BS = BSgate(pi/4, pi)
        BS | (q[1], q[2])
        BS | (q[0], q[1])

        # Perform homodyne measurements
        MeasureX | q[0]
        MeasureP | q[1]

        # Displacement gates conditioned on
        # the measurements
        Xgate(scale(q[0], sqrt(2))) | q[2]
        Zgate(custom(q[1])) | q[2]

    state = eng.run('fock', cutoff_dim=15)


.. rubric:: Footnotes

.. [#] Fock backends are backends which represent the quantum state and operations via the Fock basis. These can represent *all* possible CV states and operations, but also introduce numerical error due to truncation of the Fock space, and consume more memory.
.. [#] The Gaussian backend, due to its ability to represent states and operations as Gaussian objects/transforms in the phase space, consumes less memory and is less computationally intensive then the Fock backends. However, it cannot represent non-Gaussian operations and states (such as the cubic phase gate, Fock measurements, and Fock states, amongst others).
.. [#] If using the Gaussian backend, state methods and attributes available for extracting the state information include:
    
    * :meth:`~.BaseGaussianState.means` and :meth:`~.BaseGaussianState.cov` for returning the vector of means and the covariance matrix of the specified modes
    * :meth:`~.BaseState.fock_prob` for returning the probability that the photon counting pattern specified by ``n`` occurs
    * :meth:`~.BaseState.reduced_dm` for returning the reduced density matrix in the fock basis of mode ``n``

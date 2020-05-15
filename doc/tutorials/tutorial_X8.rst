.. _starship:

Executing programs on X8 devices
================================

In this tutorial, we demonstrate how Strawberry Fields can execute
quantum programs on the X8 family of remote photonic devices, via the Xanadu cloud platform. Topics
covered include:

* Configuring Strawberry Fields to access the cloud platform,
* Using the :class:`~.RemoteEngine` to execute jobs remotely on ``X8``,
* Compiling quantum programs for specific quantum chip architectures,
* Saving and running Blackbird scripts, an assembly language for quantum photonic hardware, and
* Embedding and sampling from bipartite graphs on ``X8``.

.. warning::

    An authentication token is required to access the Xanadu cloud platform. In the
    following, replace ``AUTHENTICATION_TOKEN`` with your personal API token.

    If you do not have an authentication token, you can request hardware access via email
    at sf@xanadu.ai.

.. note::

    **What's in a name?**

    ``X8``, rather than being a single hardware chip, corresponds to a family
    of physical devices, with individual names including ``X8_01``.

    While ``X8`` is used here to specify *any* chip in the ``X8`` device class (including ``X8_01``),
    you may also specify a specific chip ID (such as ``X8_01``) during program compilation
    and execution.


Configuring your credentials
----------------------------

Before using the :class:`~.RemoteEngine` for the first time, we need to configure
Strawberry Fields with an authentication token that provides you access to the Xanadu
cloud platform. We can do this by using the :func:`~.store_account` function from within Python:

>>> import strawberryfields as sf
>>> sf.store_account("AUTHENTICATION_TOKEN")

This only needs to be executed the first time configuring Strawberry Fields;
your account details will be stored to disk, and automatically loaded from now
on.

To test that your account credentials correctly authenticate against the cloud platform,
you can use the :func:`~.ping` command,

>>> sf.ping()
You have successfully authenticated to the platform!

.. note::

    Strawberry Fields also provides a command line interface for configuring
    access to the cloud platform and for submitting jobs:

    .. code-block:: console

        $ sf configure --token AUTHENTICATION_TOKEN
        $ sf --ping
        You have successfully authenticated to the platform!

For more details on configuring Strawberry Fields for cloud access, including
using the command line interface, see the :doc:`/introduction/photonic_hardware`
quickstart guide.

Device details
--------------

Below, we will be submitting a job to run on ``X8``.

``X8`` is an 8-mode chip, with the following restrictions:


* The initial states are two-mode squeezed states (:class:`~.S2gate`). We call modes 0 to 3 the
  *signal modes* and modes 4 to 7 the *idler modes*. Two-mode squeezing is between the pairs
  of modes: (0, 4), (1, 5), (2, 6), (3, 7).

* Any arbitrary :math:`4\times 4` unitary (consisting of :class:`~.BSgate`, :class:`~.MZgate`,
  :class:`~.Rgate`, and :class:`~.Interferometer` operations) can be applied identically
  on both the signal and idler modes.

* Finally, the chip terminates with photon-number resolving measurements (:class:`~.MeasureFock`).

.. figure:: /_static/X8.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

At this point, only the parameters :code:`r=1`, :code:`phi=0` are allowed for the two-mode
squeezing gates between any pair of signal and idler modes. Eventually, a range of
squeezing amplitudes :code:`r` will be supported.

Executing jobs
--------------

In this section, we will use Strawberry Fields to submit a simple
circuit to the chip.

First, we import NumPy and Strawberry Fields, including the remote engine.

.. code-block:: python3

    import numpy as np

    import strawberryfields as sf
    from strawberryfields import ops
    from strawberryfields import RemoteEngine

Lets use the :func:`~.random_interferometer` function to generate a random :math:`4\times 4`
unitary:

>>> from strawberryfields.utils import random_interferometer
>>> U = random_interferometer(4)
>>> print(U)
array([[-0.13879438-0.47517904j,-0.29303954-0.47264099j,-0.43951987+0.12977568j, -0.03496718-0.48418713j],
[ 0.06065372-0.11292765j, 0.54733962+0.1215551j, -0.50721513+0.56195975j, -0.15923161+0.26606674j],
[ 0.42212573-0.53182417j, -0.2642572 +0.50625182j, 0.19448705+0.28321781j,  0.30281396-0.05582391j],
[ 0.43097587-0.30288974j, 0.07419772-0.21155126j, 0.28335618-0.13633175j, -0.75113453+0.09580304j]])

Next we create an 8-mode quantum program:

.. code-block:: python3

    prog = sf.Program(8, name="remote_job1")

    with prog.context as q:
        # Initial squeezed states
        # Allowed values are r=1.0 or r=0.0
        ops.S2gate(1.0) | (q[0], q[4])
        ops.S2gate(1.0) | (q[1], q[5])
        ops.S2gate(1.0) | (q[3], q[7])

        # Interferometer on the signal modes (0-3)
        ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
        ops.BSgate(0.543, 0.123) | (q[2], q[0])
        ops.Rgate(0.453) | q[1]
        ops.MZgate(0.65, -0.54) | (q[2], q[3])

        # *Same* interferometer on the idler modes (4-7)
        ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
        ops.BSgate(0.543, 0.123) | (q[6], q[4])
        ops.Rgate(0.453) | q[5]
        ops.MZgate(0.65, -0.54) | (q[6], q[7])

        ops.MeasureFock() | q

Finally, we create the engine. Similarly to the :class:`~.LocalEngine`, the :class:`~.RemoteEngine`
is in charge of compiling and executing programs. However, it differs in that the program will be
executed on *remote* devices, rather than on local simulators.

Below, we create a remote engine to submit and execute quantum programs
on the ``X8`` photonic device.

>>> eng = RemoteEngine("X8")

We can now run the program by calling ``eng.run``, and passing the program to be executed
as well as additional runtime options.

>>> results = eng.run(prog, shots=20)
>>> results.samples
array([[0, 0, 1, 0, 1, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 2],
       [0, 0, 0, 0, 0, 1, 0, 0],
       [1, 0, 0, 0, 0, 0, 3, 0],
       [3, 0, 0, 0, 2, 0, 1, 0],
       [0, 1, 0, 0, 0, 1, 1, 0],
       [0, 1, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0],
       [1, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 1, 1, 0, 2, 1, 2],
       [2, 0, 1, 0, 1, 0, 0, 0]])

The samples returned correspond to 20 measurements (or shots) of the 8 mode quantum program
above. Some modes have measured zero photons, and others have
detected single photons, with a few even detecting 2 or 3.

By taking the average of the returned array along the shots axis, we can estimate the
mean photon number of each mode:

>>> np.mean(results.samples, axis=0)
array([0.4, 0.1, 0.15, 0.05, 0.3, 0.3, 0.45, 0.35])

We can also use the Python collections module to convert the samples into
counts:

>>> from collections import Counter
>>> bitstrings = [tuple(i) for i in results.samples]
>>> counts = {k:v for k, v in Counter(bitstrings).items()}
>>> counts[(0, 0, 0, 0, 0, 0, 0)]
2

.. note::

    The :class:`~.operation` decorator allows you to create your own Strawberry Fields
    operation. This can make it easier to ensure that the same unitary is always
    applied to the signal and idler modes.

    .. code-block:: python3

        from strawberryfields.utils import operation

        @operation(4)
        def unitary(q):
            ops.Interferometer(U) | q
            ops.BSgate(0.543, 0.123) | (q[2], q[0])
            ops.Rgate(0.453) | q[1]
            ops.MZgate(0.65, -0.54) | (q[2], q[3])

        prog = sf.Program(8)

        with prog.context as q:
            ops.S2gate(1.0) | (q[0], q[4])
            ops.S2gate(1.0) | (q[1], q[5])
            ops.S2gate(1.0) | (q[3], q[7])

            unitary() | q[:4]
            unitary() | q[4:]

            ops.MeasureFock() | q

    Refer to the :class:`~.operation` documentation for more details.

Job management
~~~~~~~~~~~~~~

Above, when we called ``eng.run()``, we had to wait for the
remote device to execute the program and the result to be returned before we could
continue executing code. That is, ``eng.run()`` is a **blocking** method.

Sometimes, however, it is useful to submit the program
and continue performing computation locally, every now and again checking to see
if the job is complete and the results are ready. This is possible with
the **non-blocking** :meth:`eng.run_async() <.RemoteEngine.run_async>` method:

>>> job = engine.run_async(program, shots=100)

Unlike ``eng.run()``, it returns a :class:`~.Job` instance, which allows us to
check the status of our submitted job:

>>> job.id
"e6ead866-04c9-4d48-ba28-680e8639fc41"
>>> job.status
"queued"

When the job result is ready, it is available via the ``result`` property.
If the job result is not yet available, however, an ``InvalidJobOperationError``
will be raised:

>>> job.result
InvalidJobOperationError

To check when the results are ready, the job can be refreshed, and the status
checked:

>>> job.refresh()
>>> job.status
"complete"
>>> result = job.result
>>> result.samples.shape
(100, 8)

Finally, an incomplete job can be *cancelled* by calling :meth:`job.cancel() <.Job.cancel>`.

Hardware compilation
--------------------

When creating a quantum program to run on hardware, Strawberry Fields can compile
any collection of the following gates into a multi-mode unitary:

* `General beamsplitters <https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.BSgate.html>`_ (:class:`~.ops.BSgate`),

* `Mach-Zehnder interferometers <https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.MZgate.html>`_ (:class:`~.ops.MZgate`), or

* `rotations/phase shifts <https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.Rgate.html>`_ (:class:`~.ops.Rgate`).

Furthermore, several automatic decompositions are supported:

* You can use the :class:`~.ops.Interferometer` command to directly pass a
  unitary matrix to be decomposed and compiled to match the device architecture.
  This performs a rectangular decomposition using Mach-Zehnder interferometers.

* You can use :class:`~.ops.BipartiteGraphEmbed` to embed a bipartite graph on
  the photonic device.

  .. warning::

      Decomposed squeezing values depend on the graph
      structure, so only bipartite graphs that result in equal squeezing on all
      modes can be executed on ``X8`` chips.

Before sending the program to the cloud platform to be executed, however, Strawberry Fields
must **compile** the program to match the physical architecture or layout of the photonic chip, in this case ``X8``.
This happens implicitly when using the remote engine, however we can use the :meth:`~.Program.compile`
method to explicitly compile the program for a specific chip.

For example, lets compile the program we created in the previous section:

>>> prog_compiled = prog.compile("X8")
>>> prog_compiled.print()
S2gate(1, 0) | (q[0], q[4])
S2gate(1, 0) | (q[3], q[7])
S2gate(1, 0) | (q[2], q[6])
MZgate(1.573, 4.368) | (q[2], q[3])
MZgate(1.573, 4.368) | (q[6], q[7])
S2gate(1, 0) | (q[1], q[5])
MZgate(1.228, 5.006) | (q[0], q[1])
MZgate(4.414, 3.859) | (q[1], q[2])
MZgate(2.98, 3.316) | (q[2], q[3])
Rgate(-0.7501) | (q[3])
MZgate(5.397, 5.494) | (q[0], q[1])
MZgate(5.152, 4.891) | (q[1], q[2])
Rgate(2.544) | (q[2])
MZgate(1.228, 5.006) | (q[4], q[5])
MZgate(4.414, 3.859) | (q[5], q[6])
MZgate(2.98, 3.316) | (q[6], q[7])
Rgate(-0.7501) | (q[7])
MZgate(5.397, 5.494) | (q[4], q[5])
MZgate(5.152, 4.891) | (q[5], q[6])
Rgate(2.544) | (q[6])
Rgate(-1.173) | (q[1])
Rgate(1.902) | (q[4])
Rgate(1.902) | (q[0])
Rgate(-1.173) | (q[5])
MeasureFock | (q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])

While equivalent to the uncompiled program, we can now see the low-level hardware
operations that are applied on the physical chip.


Working with Blackbird scripts
------------------------------

When submitting quantum programs to be executed remotely, they are communicated to
the cloud platform using Blackbird---a quantum photonic assembly language.
Strawberry Fields also supports exporting programs directly as Blackbird scripts
(an ``xbb`` file); Blackbird scripts can then be submitted to be executed via the
Strawberry Fields :doc:`command line interface </code/sf_cli>`.

For example, lets consider a Blackbird script
:download:`examples/example_job_X8.xbb <../../examples/example_job_X8.xbb>`
representing the same quantum program we constructed above:

.. code-block:: python3

    name remote_job1
    version 1.0
    target X8 (shots = 20)

    complex array U[4, 4] =
        -0.13879438-0.47517904j, -0.29303954-0.47264099j, -0.43951987+0.12977568j, -0.03496718-0.48418713j
        0.06065372-0.11292765j, 0.54733962+0.1215551j, -0.50721513+0.56195975j, -0.15923161+0.26606674j
        0.42212573-0.53182417j, -0.2642572+0.50625182j, 0.19448705+0.28321781j, 0.30281396-0.05582391j
        0.43097587-0.30288974j, 0.07419772-0.21155126j, 0.28335618-0.13633175j, -0.75113453+0.09580304j

    # Initial states are two-mode squeezed states
    S2gate(1.0, 0.0) | [0, 4]
    S2gate(1.0, 0.0) | [1, 5]
    S2gate(1.0, 0.0) | [3, 7]

    # Apply the unitary matrix above to
    # the first pair of modes, as well
    # as a beamsplitter
    Interferometer(U) | [0, 1, 2, 3]
    BSgate(0.543, 0.123) | [2, 0]
    Rgate(0.453) | 1
    MZgate(0.65, -0.54) | [2, 3]

    # Duplicate the above unitary for
    # the second pair of modes
    Interferometer(U) | [4, 5, 6, 7]
    BSgate(0.543, 0.123) | [6, 4]
    Rgate(0.453) | 5
    MZgate(0.65, -0.54) | [6, 7]

    # Perform a PNR measurement in the Fock basis
    MeasureFock() | [0, 1, 2, 3, 4, 5, 6, 7]

The above Blackbird script can be remotely executed using the command line,

.. code-block:: console

    $ sf run program1.xbb --output out.txt

After executing the above command, the result will be stored in ``out.txt`` in the
current working directory. You can also omit the ``--output`` parameter to print the
result to the screen.

.. note::

    Saved Blackbird scripts can be imported as Strawberry Fields programs
    using the :func:`~.load` function:

    >>> prog = load("test.xbb")

    Strawberry Fields programs can also be exported as Blackbird scripts
    using :func:`~.save`:

    >>> sf.save("program1.xbb", prog)


Embedding bipartite graphs
--------------------------

The X8 device class supports embedding bipartite graphs,
i.e., those with adjacency matrices

.. math:: A = \begin{bmatrix}0 & B\\ B^T & 0\end{bmatrix}

where :math:`B` represents the edges between the two sets of
vertices in the graph. However, the devices are currently restricted
to bipartite graphs with equally sized partitions, such that the singular values form the set :math:`\{0, d\}`
for some real value :math:`d`.

Here, we will
consider a `complete bipartite graph <https://en.wikipedia.org/wiki/Complete_bipartite_graph>`_,
since the singular values are of the form :math:`\{0, d\}`.

.. code-block:: python3

    B = np.ones([4, 4])
    A = np.block([[0*B, B], [B.T, 0*B]])

    prog = sf.Program(8)

    # the following mean photon number per mode
    # quantity is set to ensure that the singular values
    # are scaled such that all Sgates have squeezing value r=1
    m = 0.345274461385554870545

    with prog.context as q:
        ops.BipartiteGraphEmbed(A, mean_photon_per_mode=m) | q
        ops.MeasureFock() | q


>>> prog.compile("X8").print()
S2gate(1, 0) | (q[0], q[4])
S2gate(0, 0) | (q[3], q[7])
S2gate(0, 0) | (q[2], q[6])
MZgate(3.598, 5.444) | (q[2], q[3])
MZgate(3.598, 5.444) | (q[6], q[7])
S2gate(0, 0) | (q[1], q[5])
MZgate(0, 5.236) | (q[0], q[1])
MZgate(4.886, 5.496) | (q[1], q[2])
MZgate(0.7106, 4.492) | (q[2], q[3])
Rgate(0.9284) | (q[3])
MZgate(2.922, 3.142) | (q[0], q[1])
MZgate(4.528, 3.734) | (q[1], q[2])
Rgate(-2.51) | (q[2])
MZgate(0, 5.236) | (q[4], q[5])
MZgate(4.886, 5.496) | (q[5], q[6])
MZgate(0.7106, 4.492) | (q[6], q[7])
Rgate(0.9284) | (q[7])
MZgate(2.922, 3.142) | (q[4], q[5])
MZgate(4.528, 3.734) | (q[5], q[6])
Rgate(-2.51) | (q[6])
Rgate(-2.51) | (q[1])
Rgate(-0.8273) | (q[4])
Rgate(-0.8273) | (q[0])
Rgate(-2.51) | (q[5])
MeasureFock | (q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])

If the bipartite graph to be embedded does not satisfy the aforementioned
restriction on the singular values, an error message will be raised on
compilation:

>>> B = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 1], [1, 0, 1, 0]])
>>> A = np.block([[np.zeros_like(B), B], [B.T, np.zeros_like(B)]])
>>> prog = sf.Program(8)
>>> with prog.context as q:
...     ops.BipartiteGraphEmbed(A, mean_photon_per_mode=1) | q
...     ops.MeasureFock() | q
CircuitError: Incorrect squeezing value(s) (r, phi)={(1.336, 0.0), (0.177, 0.0), (0.818, 0.0)}.
Allowed squeezing value(s) are (r, phi)={(1.0, 0.0), (0.0, 0.0)}.

.. _starship:

Xanadu cloud platform
#####################

In this section, we provide a tutorial of the **RemoteEngine**, an engine used to connect to the Xanadu
cloud platform and execute jobs remotely (e.g., on a quantum chip).

Configuring the RemoteEngine
----------------------------

Before using the RemoteEngine, you need to configure the hostname and authentication token that will provide
you access to the API. This can be done in one of two ways:

1. By using the :func:`~.store_account` function to store your account credentials:

   >>> sf.store_account("my_api_token")

2. By using the ``sf`` command line interface to configure Strawberry Fields with your
   account credentials:

   >>> sf configure --token my_api_token

In both of the above code snippets, ``my_api_token`` should be replaced with your personal
API token! For more details on configuring Strawberry Fields for cloud access, including
creating local per-project configuration files, see the :doc:`/introduction/configuration`
quickstart guide.

To test that your account credentials correctly authenticate against the cloud platform,
you can use the ``ping`` command, from within Strawberry Fields,

>>> sf.cli.ping()
You have successfully authenticated to the platform!

or via the command line:

>>> sf --ping
You have successfully authenticated to the platform!

.. _first_program:

Submitting a Blackbird script
-----------------------------

The easiest way to execute a program using the RemoteEngine is to create a Blackbird script (an ``xbb`` file)
and place it in your current working directory.

For this example, consider the following Blackbird script, which represents a quantum program that matches
exactly the gate layout of the ``chip2`` photonic hardware device. We will save the following
file as ``test.xbb`` in our current working directory:

.. code-block:: python

    name template_4x2_chip0     # Name of the program
    version 1.0                 # Blackbird version number
    target chip2 (shots = 100)   # This program will run on chip2 for 50 shots

    # define the squeezing amplitude
    float r = 1.0

    # Define the interferometer phase values
    float phi0 = 0.574
    float phi1 = 1.33
    float phi2 = 0.654
    float phi3 = -2.3
    float phi4 = 0.065
    float phi5 = 0.654
    float phi6 = 1.23
    float phi7 = -1.63
    float phi8 = 0.065
    float phi9 = 0.654
    float phi10 = 1.23
    float phi11 = -1.63

    # Initial states are two-mode squeezed states
    S2gate(r, 0.0) | [0, 4]
    S2gate(r, 0.0) | [1, 5]
    S2gate(r, 0.0) | [2, 6]
    S2gate(r, 0.0) | [3, 7]

    # A standard four-mode interferometer is applied
    # to the signal modes (the ones lower in frequency)
    MZgate(phi0, phi1) | [0, 1]
    MZgate(phi2, phi3) | [2, 3]
    MZgate(phi4, phi5) | [1, 2]
    MZgate(phi6, phi7) | [0, 1]
    MZgate(phi8, phi9) | [2, 3]
    MZgate(phi10, phi11) | [1, 2]

    # final local phases
    Rgate(0.765)  | [0]
    Rgate(-0.123) | [1]
    Rgate(0.654)  | [2]
    Rgate(-0.651) | [3]

    # The 4x4 interferometer above is duplicated
    # for the idler modes (the ones higher in frequency)
    MZgate(phi0, phi1) | [4, 5]
    MZgate(phi2, phi3) | [6, 7]
    MZgate(phi4, phi5) | [5, 6]
    MZgate(phi6, phi7) | [4, 5]
    MZgate(phi8, phi9) | [6, 7]
    MZgate(phi10, phi11) | [5, 6]

    # final local phases
    Rgate(0.765)  | [4]
    Rgate(-0.123) | [5]
    Rgate(0.654)  | [6]
    Rgate(-0.651) | [7]

    # Perform a photon number counting measurement
    MeasureFock() | [0, 1, 2, 3, 4, 5, 6, 7]

After you have created your Blackbird script, you can execute it using the command line, or using a Python shell.


Executing your Blackbird script using Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To execute this file using Python, you can use a code block like this:

.. code-block:: python3

    from strawberryfields import RemoteEngine
    from strawberryfields.io import load

    eng = RemoteEngine("chip2")
    prog = load("test.xbb")
    result = eng.run(prog)
    print(result.samples)


Executing your Blackbird script from the command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To execute this file from the command line, use the ``sf`` command as follows:

.. code-block:: console

    sf run test.xbb --output out.txt

After executing the above command, the result will be stored in ``out.txt`` in the current working directory.
You can also omit the ``--output`` parameter to print the result to the screen.


Submitting via Strawberry Fields
--------------------------------

In this section, we will use Strawberry Fields to submit a simple
circuit to the chip.

.. code-block:: python3

    import numpy as np

    import strawberryfields as sf
    from strawberryfields import ops
    from strawberryfields import RemoteEngine
    from strawberryfields.utils import random_interferometer

We choose a random 4x4 interferometer

>>> U = random_interferometer(4)
>>> print(U)
array([[-0.13879438-0.47517904j,-0.29303954-0.47264099j,-0.43951987+0.12977568j, -0.03496718-0.48418713j],
[ 0.06065372-0.11292765j, 0.54733962+0.1215551j, -0.50721513+0.56195975j, -0.15923161+0.26606674j],
[ 0.42212573-0.53182417j, -0.2642572 +0.50625182j, 0.19448705+0.28321781j,  0.30281396-0.05582391j],
[ 0.43097587-0.30288974j, 0.07419772-0.21155126j, 0.28335618-0.13633175j, -0.75113453+0.09580304j]])

Next we create the program

.. code-block:: python3

    prog = sf.Program(8)

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

We create the engine. The engine is in charge of compiling and executing
programs on the remote device.

>>> eng = RemoteEngine("chip2")

We run the engine by calling ``eng.run``, and pass it the program we
want to run.

>>> results = eng.run(prog, shots=20)
Job e6ead866-04c9-4d48-ba28-680e8639fc41 is sent to server.
>>> results.samples.T
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
>>> np.mean(results.samples.T, axis=0)
array([0.4 , 0.1 , 0.15, 0.05, 0.3 , 0.3 , 0.45, 0.35])


We can convert the samples into counts using the following function:

.. code-block:: python3

     from collections import Counter

     def count(samples):
          bitstrings = [tuple(i) for i in samples]
          return {k:v for k, v in Counter(bitstrings).items()}

>>> samples = np.array([[0, 2],[1, 0],[0, 1],[0, 0],[0, 0],[2, 0],[0, 1],[0, 1]])
>>> counts = count(samples)
>>> print(counts)
{(0, 2): 1, (1, 0): 1, (0, 1): 3, (0, 0): 2, (2, 0): 1}
>>> counts[(0, 0)]
2

.. _compilation:

Program compilation
-------------------

In addition to using the program template above, which directly matches the physical
layout of the hardware device, you can apply any four-mode interferometer to the pairs of modes.

Primitive gates supported by chip2 include any combination of:

* `General beamsplitters <https://strawberryfields.readthedocs.io/en/stable/code/ops.html#strawberryfields.ops.BSgate>`_ (``ops.BSgate``),

* `Mach-Zehnder interfomerters <https://strawberryfields.readthedocs.io/en/stable/code/ops.html#strawberryfields.ops.MZgate>`_ (``ops.MZgate``), or

* `rotations/phase shifts <https://strawberryfields.readthedocs.io/en/stable/code/ops.html#strawberryfields.ops.Rgate>`_ (``ops.Rgate``).

Furthermore, several automatic decompositions are supported:

* You can use the :class:`~.ops.Interferometer` command to directly pass a
  unitary matrix to be decomposed and compiled to match the device architecture.
  This performs a rectangular decomposition using Mach-Zehnder interferometers.

* You can use :class:`~.ops.BipartiteGraphEmbed` to embed a bipartite graph on
  the GBS chip. Note, however, that the decomposed squeezing values depends on the graph
  structure, so only bipartite graphs that result in equal squeezing on all
  modes can currently be executed on chip2.

For example, consider the following Blackbird script:

.. code-block:: python

    name compilation_example  # Name of the program
    version 1.0               # Blackbird version number
    target chip2 (shots=100)   # This program will run on chip0 for 50 shots

    # Define a unitary matrix
    complex array U[4, 4] =
         0.09980516-0.78971535j,  0.53374613+0.07984545j, -0.21161788+0.10047649j, -0.01337026-0.14167555j
         -0.12759979-0.00425289j,  0.14089156+0.40091225j, 0.31942372-0.21453252j, -0.79775306+0.13657774j
         -0.18224807+0.30281836j,  0.26930442-0.04644871j, -0.46045639-0.55359506j, -0.0737605-0.52580999j
         0.19903677-0.43076659j, -0.50320649-0.44750373j, -0.01617065-0.52755812j, -0.19729219+0.06200712j

    # Initial states are two-mode squeezed states
    S2gate(1.0, 0.0) | [0, 4]
    S2gate(1.0, 0.0) | [1, 5]
    S2gate(1.0, 0.0) | [2, 6]
    S2gate(1.0, 0.0) | [3, 7]

    # Apply the unitary matrix above to
    # the first pair of modes, as well
    # as a beamsplitter
    Interferometer(U) | [0, 1, 2, 3]
    BSgate(0.543, -0.123) | [0, 1]

    # Duplicate the above unitary for
    # the second pair of modes
    Interferometer(U) | [4, 5, 6, 7]
    BSgate(0.543, -0.123) | [4, 5]

    # Perform a PNR measurement in the Fock basis
    MeasureFock() | [0, 1, 2, 3, 4, 5, 6, 7]


**Note:** You may use ``random_interferometer`` to generate arbitrary random unitaries.

This program will execute following the same steps as above; ``RemoteEngine`` will automatically
compile the program to match the layout of the chip.

You may wish to view the compiled program; this can be easily done in Python using
the ``Program.compile`` method:


>>> from strawberryfields import RemoteEngine
>>> from strawberryfields.io import load
>>> prog = load("test.xbb")
>>> prog = prog.compile("chip2")
>>> prog.print()
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

and even saved as a new Blackbird script using the ``io.save`` function:

>>> from strawberryfields.io import save
>>> save("test_compiled.xbb", prog)


Tips and tricks
---------------

.. code-block:: python3

    from strawberryfields.utils import operation

We can define an operation to make it easier to apply the same unitary
to both signal and idler modes.

.. code-block:: python3

    @operation(4)
    def unitary(q):
        ops.Interferometer(U) | q
        ops.BSgate(0.543, 0.123) | (q[2], q[0])

    prog = sf.Program(8)

    with prog.context as q:
        ops.S2gate(1.0) | (q[0], q[4])
        ops.S2gate(1.0) | (q[1], q[5])
        ops.S2gate(1.0) | (q[2], q[6])
        ops.S2gate(1.0) | (q[3], q[7])

        unitary() | q[:4]
        unitary() | q[4:]


Embedding bipartite graphs
--------------------------

We can embed bipartite graphs, with the restriction that the singular
values form the set :math:`\{0, d\}` for some real value :math:`d`.

The matrix :math:`B` represents the edges between the two sets of
vertices in the graph, and :math:`A` is the full adjacency matrix
:math:`A = \begin{bmatrix}0 & B\\ B^T & 0\end{bmatrix}`. Here, we will
consider a complete bipartite graph, since we know that the singular
values are of the form :math:`\{d, 0\}`.

.. code-block:: python3

    B = np.ones([4, 4])
    A = np.block([[0*B, B], [B.T, 0*B]])

    prog = sf.Program(8)

    # the following mean photon number per mode
    # quantity is set to ensure that the singular values
    # are scaled such that all squeezers have value 1
    m = 0.345274461385554870545

    with prog.context as q:
        ops.BipartiteGraphEmbed(A, mean_photon_per_mode=m) | q
        ops.MeasureFock() | q


>>> prog.compile("chip2").print()
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

The squeezing values required to embed this bipartite graph are given by
the following relation:

>>> from thewalrus.quantum import find_scaling_adjacency_matrix
>>> c = find_scaling_adjacency_matrix(A, 2*4*0.345274461385554870545)
>>> set(np.arctanh(np.linalg.svd(c*A)[1]))
{0.0, 1.0000000000000002}

Note that the above squeezing values must be of the form :math:`\{0,1\}`
to be embedded on the chip. Consider a bipartite graph where this is not
the case:

>>> B = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 1], [1, 0, 1, 0]])
>>> A = np.block([[0*B, B], [B.T, 0*B]])
>>> c = find_scaling_adjacency_matrix(A, 2*4*1)
>>> set(np.arctanh(np.linalg.svd(c*A)[1]))
{0.0,
3.2937343775007984e-32,
0.17674864137317442,
0.17674864137317453,
0.8180954232791708,
0.8180954232791715,
1.3361892276414615}

The program will fail to compile for chip2:

.. code-block:: python3

    prog = sf.Program(8)

    with prog.context as q:
        ops.BipartiteGraphEmbed(A, mean_photon_per_mode=1) | q
        ops.MeasureFock() | q

    prog.compile("chip2").print()

.. code-block:: bash

    ---------------------------------------------------------------------------

    CircuitError                              Traceback (most recent call last)

    <ipython-input-23-f713320d8c3b> in <module>
          5     ops.MeasureFock() | q
          6
    ----> 7 prog.compile("chip2").print()


    ~/Dropbox/Work/Xanadu/sf_cloud/strawberryfields/program.py in compile(self, target, **kwargs)
        522         # does the circuit spec  have its own compilation method?
        523         if db.compile is not None:
    --> 524             seq = db.compile(seq, self.register)
        525
        526         # create the compiled Program


    ~/Dropbox/Work/Xanadu/sf_cloud/strawberryfields/circuitspecs/chip2.py in compile(self, seq, registers)
        137             raise CircuitError(
        138                 "Incorrect squeezing value(s) (r, phi)={}. Allowed squeezing "
    --> 139                 "value(s) are (r, phi)={}.".format(wrong_params, allowed_sq_value)
        140             )
        141


    CircuitError: Incorrect squeezing value(s) (r, phi)={(1.336, 0.0), (0.177, 0.0), (0.818, 0.0)}. Allowed squeezing value(s) are (r, phi)={(1, 0.0), (0.0, 0.0)}.

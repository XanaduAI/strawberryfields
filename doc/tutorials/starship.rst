.. _starship:

Running Jobs with Starship
################################

.. sectionauthor:: Zeid Zabaneh <zeid@xanadu.ai>

In this section, we provide a tutorial of the **Starship**, an engine used to connect to the Xanadu
cloud platform and execute jobs remotely (e.g., on a quantum chip).

Configuring Starship
==========================

Before using Starship, you need to configure the hostname and authentication token that will provide
you access to the API. The easiest way is to create a configuration file named ``config.toml`` in your
working directory. A typical file looks like this:

.. code-block:: console

   [api]
   hostname = "platform.example.com"
   authentication_token = "ElUFm3O6m6q1DXPmpi5g4hWEhYHXFxBc"

You can generate this file interactively by using the ``starship`` command as follows, answering the questions in the prompts.

.. code-block:: console

   starship --reconfigure
   Please enter the hostname of the server to connect to: [platform.example.com]
   Please enter the authentication token to use when connecting: [] ElUFm3O6m6q1DXPmpi5g4hWEhYHXFxBc
   Would you like to save these settings to a local cofiguration file in the current directory? [Y/n] y
   Writing configuration file to current working directory...


To test connectivity, you can use the following command:

.. code-block:: console

   starship --hello
   You have successfully authenticated to the platform!


.. _first_program:

Executing your first program
============================
The easiest way to execute a program using Starship is to create a Blackbird script (an ``xbb`` file)
and place it in your current working directory. Check the :ref:`Blackbird tutorial <blackbird>` for how to create this file.

For this example, consider the following Blackbird script, which represents a quantum program that matches
exactly the gate layout of the `chip0` photonic hardware device. We will save the following file as ``test.xbb``
in our current working directory:

.. code-block:: python

   name template_2x2_chip0     # Name of the program
   version 1.0                 # Blackbird version number
   target chip0 (shots = 50)   # This program will run on chip0 for 50 shots

   # Define the interferometer phase values
   float phi0 = 0.574
   float phi1 = 1.33

   # final local phases
   float local_phase_0 = -0.543
   float local_phase_1 = 2.43
   float local_phase_2 = 0.11
   float local_phase_3 = -3.21

   # Initial states are two-mode squeezed states
   S2gate(1.0, 0.0) | [0, 2]
   S2gate(1.0, 0.0) | [1, 3]

   # A standard two-mode interferometer is applied
   # to the first pair of modes
   Rgate(phi0) | [0]
   BSgate(pi/4, pi/2) | [0, 1]
   Rgate(phi1) | [0]
   BSgate(pi/4, pi/2) | [0, 1]

   # The 2x2 interferometer above is duplicated
   # for the second pair of modes
   Rgate(phi0) | [2]
   BSgate(pi/4, pi/2) | [2, 3]
   Rgate(phi1) | [2]
   BSgate(pi/4, pi/2) | [2, 3]

   # final local phases
   Rgate(local_phase_0) | 0
   Rgate(local_phase_1) | 1
   Rgate(local_phase_2) | 2
   Rgate(local_phase_3) | 3

   # Perform a photon number counting measurement
   MeasureFock() | [0, 1, 2, 3]

After you have created your Blackbird script, you can execute it using the command line, or using a Python shell.


Executing your Blackbird script using Python
--------------------------------------------

To execute this file using Python, you can use a code block like this:

.. code-block:: python3

   from strawberryfields import Starship
   from strawberryfields.io import load

   eng = Starship()
   prog = load("test.xbb")
   result = eng.run(prog)
   print(result.samples)


Executing your Blackbird script from the command line
-----------------------------------------------------

To execute this file from the command line, use the ``starship`` command as follows:

.. code-block:: console

   starship --input test.xbb --output out.txt

After executing the above command, the result will be stored in ``out.txt`` in the current working directory.
You can also omit the ``--output`` parameter to print the result to the screen.


Program compilation
===================

In addition to using the program template above, which directly matches the physical
layout of the hardware device, you can apply any two-mode interferometer to the pairs of modes.
The interferometer can be composed of any combination
of beamsplitters (:class:`~.ops.BSgate`), rotations/phase shifts (:class:`~.ops.Rgate`).
Furthermore, you can use the :class:`~.ops.Interferometer` command to directly pass a
unitary matrix to be decomposed and compiled to match the device architecture.

For example, consider the following Blackbird script:


.. code-block:: python

   name compilation_example  # Name of the program
   version 1.0               # Blackbird version number
   target chip0 (shots=50)   # This program will run on chip0 for 50 shots

   # Define a unitary matrix
   complex array U[2, 2] =
      -0.1955885-0.16833594j, 0.77074506+0.58254631j
      -0.03596574+0.96546083j, 0.00676031+0.2579654j

   # Initial states are two-mode squeezed states,
   # applied to alternating pairs of modes.
   S2gate(1.0, 0.0) | [0, 2]
   S2gate(1.0, 0.0) | [1, 3]

   # Apply the unitary matrix above to
   # the first pair of modes, as well
   # as a beamsplitter
   Interferometer(U) | [0, 1]
   BSgate(0.543, -0.123) | [0, 1]

   # Duplicate the above unitary for
   # the second pair of modes
   Interferometer(U) | [2, 3]
   BSgate(0.543, -0.123) | [2, 3]

   # Perform a PNR measurement in the Fock basis
   MeasureFock() | [0, 1, 2, 3]


.. note:: You may use :func:`~.random_interferometer` to generate arbitrary random unitaries.

This program will execute following the same steps as above; ``Starship`` will automatically
compile the program to match the layout of the chip described in :ref:`first_program`.

You may wish to view the compiled program; this can be easily done in Python using
the :meth:`~.Program.compile` method:


>>> from strawberryfields import Starship
>>> from strawberryfields.io import load
>>> prog = load("test.xbb")
>>> prog = prog.compile("chip0")
>>> prog.print()
S2gate(1, 0) | (q[0], q[2])
S2gate(1, 0) | (q[1], q[3])
Rgate(0.9355) | (q[0])
BSgate(0.7854, 1.571) | (q[0], q[1])
Rgate(4.886) | (q[0])
BSgate(0.7854, 1.571) | (q[0], q[1])
Rgate(-0.3742) | (q[0])
Rgate(-0.05099) | (q[1])
Rgate(0.9355) | (q[2])
BSgate(0.7854, 1.571) | (q[2], q[3])
Rgate(4.886) | (q[2])
BSgate(0.7854, 1.571) | (q[2], q[3])
Rgate(-0.3742) | (q[2])
Rgate(-0.05099) | (q[3])
MeasureFock | (q[0], q[1], q[2], q[3])

and even saved as a new Blackbird script using the :func:`~io.save` function:

>>> from strawberryfields.io import save
>>> save("test_compiled.xbb", prog)

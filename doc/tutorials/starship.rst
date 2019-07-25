.. _starship:

Running Jobs with StarshipEngine
################################

.. sectionauthor:: Zeid Zabaneh <zeid@xanadu.ai>

In this section, we provide a tutorial of the **StarshipEngine**, an engine used to connect to the Xanadu cloud platform and execute jobs remotely (e.g., on a quantum chip).

Configuring StarshipEngine
==========================

Before using StarshipEngine, you need to configure the hostname and authentication token that will provide you access to the API. The easiest way is to create a configuration file named ``config.toml`` in your working directory. A typical file looks like this:

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


Executing your first program
============================
The easiest way to execute a program using StarshipEngine is to create a blackbird script (an ``xbb`` file) and place it in your current working directory. Check the :ref:`blackbird tutorial <tutorial>` for how to create this file.

For this example, we will use the following file and save it to ``test.xbb`` in our current working directory:

.. code-block:: python

   name template_2x2_chip0
   version 1.0
   target chip0 (shots = 50)

   float sqz0 = 1.0
   float sqz1 = 1.0
   float phi0 = 0.574
   float phi1 = 1.33

   # for n spatial degrees, first n signal modes, then n idler modes, phase zero
   S2gate(sqz0, 0.0) | [0, 2]
   S2gate(sqz1, 0.0) | [1, 3]

   # standard 2x2 interferometer for the signal modes (the lower ones in frequency)
   Rgate(phi0) | [0]
   BSgate(pi/4, pi/2) | [0, 1]
   Rgate(phi1) | [0]
   BSgate(pi/4, pi/2) | [0, 1]

   #duplicate the interferometer for the idler modes (the higher ones in frequency)
   Rgate(phi0) | [2]
   BSgate(pi/4, pi/2) | [2, 3]
   Rgate(phi1) | [2]
   BSgate(pi/4, pi/2) | [2, 3]

   # Measurement in Fock basis
   MeasureFock() | [0, 1, 2, 3]

After you have created your ``xbb`` file, you can execute it using the command line, or using a python shell.

Executing your xbb file using Python
====================================
To execute this file using Python, you can use a code block like this:

.. code-block:: python3

   from strawberryfields import StarshipEngine
   from strawberryfields.io import load

   eng = StarshipEngine()
   prog = load("test.xbb")
   result = eng.run(prog)
   print(result.samples)

Executing your xbb file from the command line
=============================================
To execute this file from the command line, use the ``starship`` command as follows:

.. code-block:: console

   starship --input test.xbb --output out.txt

After executing the above command, the result will be stored in ``out.txt`` in the current working directory. You can also omit the ``--output`` parameter to print the result to the screen.

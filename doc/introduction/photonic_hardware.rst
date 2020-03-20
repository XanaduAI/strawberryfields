Photonic hardware
=================

Using Strawberry Fields, you can submit quantum programs
to be executed on photonic hardware using the Xanadu cloud platform.


Configuring your account
------------------------

Before using the Xanadu cloud platform, you will need to configure your credentials. This
can be done using several methods:

* **Configuration file (recommended)**. Use the Strawberry Fields :func:`~.store_account` function from within Python or a Jupyter notebook,

  >>> sf.store_account("AUTHENTICATION_TOKEN")

  or the ``sf configure`` :doc:`command line interface </code/sf_cli>`,

  .. code-block:: bash

      $ sf configure --token AUTHENTICATION_TOKEN

  to store a configuration file containing your credentials. Replace
  ``AUTHENTICATION_TOKEN`` above with your Xanadu cloud access token.

  .. note::

      The above configuration file need only be created once. Once created,
      Strawberry Fields will automatically load the stored authentication
      credentials.

* **Environment variables**. Set the ``SF_API_AUTHENTICATION_TOKEN`` environment variable
  with your API token.

* **Keyword argument**. Pass the keyword argument ``token`` when initializing the
  :class:`~.RemoteEngine`.

.. seealso::

    For more details on configuring Strawberry Fields for cloud access, including
    creating local per-project configuration files, see the :doc:`configuration </code/sf_configuration>` module and :func:`~.store_account` function.


Verifying your connection
-------------------------

To verify that your account credentials correctly authenticate against the cloud
platform, the :func:`~.ping` function can be used from within Python,

>>> sf.cli.ping()
You have successfully authenticated to the platform!

or the ``ping`` flag can be used on the command line:

.. code-block:: bash

    $ sf --ping
    You have successfully authenticated to the platform!

.. seealso::

   :doc:`command line interface </code/sf_cli>`

Submitting jobs
---------------

Once connected to the Xanadu cloud platform, the Strawberry Fields
:class:`~.RemoteEngine` can be used to submit quantum programs to available
photonic hardware:

>>> eng = sf.RemoteEngine("chip_name")
>>> result = eng.run(prog)
>>> print(result.samples)

Alternatively, if you have your quantum program available as a Blackbird script,
you can submit the Blackbird script file to be executed remotely using
the Strawberry Fields command line interface:

.. code-block:: console

    $ sf run blackbird_script.xbb --output out.txt

After executing the above command, the result will be stored in ``out.txt`` in the current working directory.
You can also omit the ``--output`` parameter to print the result to the screen.

Tutorials
---------

For more details on submitting jobs to photonic hardware, check out the following
tutorials.

.. customgalleryitem::
    :tooltip: RemoteEngine
    :description: :doc:`/introduction/remote`
    :figure: /gallery/gate_visualisation/GateVisualisation.gif

.. raw:: html

        <div style='clear:both'></div>
        <br>

Hardware
========

Using Strawberry Fields, you can submit quantum programs
to be executed on photonic hardware using the Xanadu cloud platform.

.. warning::

    An authentication token is required to access the Xanadu cloud platform. In the
    following, replace ``AUTHENTICATION_TOKEN`` with your personal API token.

    If you do not have an authentication token, you can request hardware access `via the sign up
    form on the Xanadu website <https://xanadu.ai/access>`__.

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

>>> sf.ping()
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

In the above example, :meth:`eng.run() <.RemoteEngine.run>` is a **blocking** method;
Python will wait until the job result has been returned from the cloud before further lines
will execute.

Jobs can also be submitted using the **non-blocking** :meth:`eng.run_async() <.RemoteEngine.run_async>`
method. Unlike :meth:`eng.run() <.RemoteEngine.run>`, which returns a :class:`~.Results` object once the computation is
complete, this method instead returns a :class:`~.Job` object directly that can be queried
to get the job's status.

>>> job = engine.run_async(prog, shots=3)
>>> job.status
"queued"
>>> job.refresh()
>>> job.status
"complete"
>>> result = job.result
>>> result.samples
array([[0, 0, 1, 0, 1, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 2]])

Job objects have the following properties and methods:

.. raw:: html

    <div class="summary-table">

.. currentmodule:: strawberryfields.api.job.Job

.. autosummary::
    id
    status
    result
    refresh
    cancel

.. raw:: html

    </div>

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
    :tooltip: Submit quantum jobs to the X8 photonic chip
    :description: :doc:`demos/tutorial_X8`
    :figure: /_static/chip.png

.. customgalleryitem::
    :tooltip: Characterize the squeezing of the X8 chip
    :description: :doc:`demos/squeezer_tests`

.. raw:: html

        <div style='clear:both'></div>
        <br>

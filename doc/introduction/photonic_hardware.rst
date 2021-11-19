Hardware and cloud
==================

Using Strawberry Fields, you can submit quantum programs to be executed on
photonic hardware or to be run on a cloud simulator via the Xanadu Cloud.

.. warning::

    An API key is required to access the Xanadu Cloud platform. If you do not
    have a Xanadu Cloud API key, you can request hardware access `via the sign
    up form on the Xanadu website <https://xanadu.ai/access>`__.

Configuring your account
------------------------

Before using the Xanadu Cloud platform, you will need to configure your
credentials using the `Xanadu Cloud Client
<https://github.com/XanaduAI/xanadu-cloud-client#setup>`__. The simplest way
to do this is through the ``xcc config`` command line interface:

.. code-block:: console

    $ xcc config set REFRESH_TOKEN "Xanadu Cloud API key goes here"

You can verify that your account credentials successfully authenticate against
the Xanadu Cloud platform by running

.. code-block:: console

    $ xcc ping
    Successfully connected to the Xanadu Cloud.

Submitting jobs on hardware
---------------------------

Once connected to the Xanadu cloud platform, the Strawberry Fields
:class:`~.RemoteEngine` can be used to submit quantum programs to available
photonic hardware:

>>> eng = sf.RemoteEngine("chip_name")
>>> result = eng.run(prog)

In the above example, :meth:`eng.run() <.RemoteEngine.run>` is a **blocking** method;
Python will wait until the job result has been returned from the cloud before further lines
will execute.

Jobs can also be submitted using the **non-blocking** :meth:`eng.run_async() <.RemoteEngine.run_async>`
method. Unlike :meth:`eng.run() <.RemoteEngine.run>`, which returns a :class:`~.Result` object once the computation is
complete, this method instead returns a :class:`xcc.Job` object directly that can be queried
to get the job's status.

>>> job = engine.run_async(prog, shots=3)
>>> job.status
"queued"
>>> job.wait()
>>> job.status
"complete"
>>> result = sf.Result(job.result)
>>> result.samples
array([[0, 0, 1, 0, 1, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 2]])

Further details on the :class:`xcc.Job` class can be found in the
`Xanadu Cloud Client documentation <https://xanadu-cloud-client.readthedocs.io/en/stable/api/xcc.Job.html>`_.

Alternatively, if you have your quantum program available as a Blackbird script,
you can submit the Blackbird script file to be executed remotely using
the Xanadu Cloud Client command line interface:

.. code-block:: console

    $ xcc job submit \
        --name "example" \
        --target "X8_01" \
        --language "blackbird:1.0" \
        --circuit "name example\nversion 1.0\ntarget X8_01 (shots=3)\n ..."

Cloud simulator
---------------

In addition to submitting jobs to be run on quantum hardware, it is also
possible to run jobs on cloud simulators (which we refer to as "simulons") via
the Xanadu Quantum Cloud. The process is very similar to running jobs on
hardware. You will need to configure your account, as described above, and
submit a job via the ``RemoteEngine``, using a simulator as the target instead
of a specific chip:

>>> eng = sf.RemoteEngine("simulon_gaussian")
>>> result = eng.run(prog)

Simulator jobs can also be submitted asynchronously using ``eng.run_async``, or
by submitting a Blackbird script with the ``target`` set to a simulator target
in the Blackbird header.

See the `Submitting jobs on hardware`_ section above for more details.

.. note::

    The ``simulon_gaussian`` simulator runs on the ``gaussian`` backend (see
    :ref:`simulating_your_program`) and thus only supports Gaussian operations,
    including homodyne and heterodyne measurements, as well terminal Fock
    measurements. Note that there are limits to how many measurements a circuit
    can have depending on the type of measurement. These can be retrieved by
    calling ``engine.device_spec.modes`` with ``engine =
    sf.RemoteEngine("simulon_gaussian")``.

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

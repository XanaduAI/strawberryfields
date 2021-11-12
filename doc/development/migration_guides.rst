Migration guides
================

The migration guides below offer guidance for dealing with major breaking
changes between Strawberry Fields releases.

Version 0.20.0
--------------

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^

The following table shows the equivalent ``xcc`` (v0.20.0) command for
each ``sf`` (v0.19.0) command:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - **Strawberry Fields v0.19.0**
     - **Strawberry Fields v0.20.0**
   * - ``sf configure --token "foo"``
     - ``xcc config set REFRESH_TOKEN "foo"``
   * - ``sf --ping``
     - ``xcc ping``
   * - ``sf run "foo.xbb"``
     - ``xcc job submit --name "bar" --target "X8_01" --circuit "$(cat foo.xbb)"``

.. note::

  To submit a job using Windows PowerShell, simply replace ``cat foo.xbb`` with ``Get-Content foo.xbb -Raw``.

Connection
^^^^^^^^^^

All ``strawberryfields.api.Connection`` instances must be replaced by their
equivalent `XCC Connection <https://xanadu-cloud-client.readthedocs.io/en/stable/api/xcc.Connection.html>`_
counterparts. For example, consider the following instantiation of a Xanadu
Cloud connection in Strawberry Fields v0.19.0:

.. code-block:: Python

    from strawberryfields.api import Connection

    connection = Connection(
        token="Xanadu Cloud API key goes here",
        host="platform.strawberryfields.ai",
        port=443,
        use_ssl=True,
    )

The (semantically) equivalent code in Strawberry Fields v0.20.0 is

.. code-block:: Python

    import xcc

    connection = xcc.Connection(
        refresh_token="Xanadu Cloud API key goes here",  # See "token" argument above.
        host="platform.strawberryfields.ai",
        port=443,
        tls=True,                                        # See "token" argument above.
    )

Job
^^^

``strawberryfields.api.Job`` has been replaced with an equivalent
`XCC Job <https://xanadu-cloud-client.readthedocs.io/en/stable/api/xcc.Job.html>`_
counterpart. This will affect the object returned when running an asynchronous job on the
``RemoteEngine``. Any code that uses the ``Job`` object returned by an asynchronous run will need to
be adapted to work with the new :class:`xcc.Job`.

In Strawberry Fields v0.19.0 this could look as follows:

.. code-block:: pycon

    >>> job = engine.run_async(program, shots=1)
    >>> job.status
    'queued'
    >>> job.result
    InvalidJobOperationError
    >>> job.refresh()
    >>> job.status
    'complete'
    >>> job.result
    [[0 1 0 2 1 0 0 0]]

  In Strawberry Fields v0.20.0, the (semantically) equivalent `Job` object would work slightly
  differently:

.. code-block:: pycon

    >>> job = engine.run_async(program, shots=1)
    >>> job.status
    'queued'
    >>> job.wait()
    >>> job.status
    'complete'
    >>> job.result
    {'output': [array([[0 1 0 2 1 0 0 0]])]}

  The ``job.wait()`` method is a blocking method that will wait for the job to finish. Alternatively,
  ``job.clear()`` can be called to clear the cache, allowing ``job.status`` to re-fetch the job status.

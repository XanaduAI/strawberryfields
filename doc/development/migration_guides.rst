Migration guides
================

The migration guides below offer guidance for dealing with major breaking
changes between Strawberry Fields releases.

.. raw:: html

    <h2 style="font-size:15px;color:white;background-color:#c30010;" >&nbsp;&nbsp;&nbsp;Xanadu's Quantum Cloud is no longer available. This material is maintained for reference purposes only.</h2>


Version 0.20.0
--------------

Configuration
^^^^^^^^^^^^^

The replacement of the ``strawberryfields.configuration`` module with the `XCC
Settings <https://xanadu-cloud-client.readthedocs.io/en/stable/api/xcc.Settings.html>`_
class has several consequences. Specifically, Xanadu Cloud credentials are now
stored in exactly one location, the path to which depends on your operating
system:

#. Windows: ``C:\Users\%USERNAME%\AppData\Local\Xanadu\xanadu-cloud\.env``

#. MacOS: ``/home/$USER/Library/Application\ Support/xanadu-cloud/.env``

#. Linux: ``/home/$USER/.config/xanadu-cloud/.env``

The format of the configuration file has also changed to `.env
<https://saurabh-kumar.com/python-dotenv/>`_ and the names of some options have
been updated. For example, the Strawberry Fields v0.19.0 configuration

.. code-block:: toml

  # config.toml
  [api]
  authentication_token = "Xanadu Cloud API key goes here"
  hostname = "platform.strawberryfields.ai"
  port = 443
  use_ssl = true

is equivalent to the Strawberry Fields v0.20.0 configuration

.. code-block:: python

  # .env
  XANADU_CLOUD_REFRESH_TOKEN='Xanadu Cloud API key goes here'
  XANADU_CLOUD_HOST='platform.strawberryfields.ai'
  XANADU_CLOUD_PORT=443
  XANADU_CLOUD_TLS=True

Similarly, the names of the configuration environment variables have changed from

.. code-block:: bash

  # Strawberry Fields v0.19.0
  export SF_API_AUTHENTICATION_TOKEN="Xanadu Cloud API key goes here"
  export SF_API_HOSTNAME="platform.strawberryfields.ai"
  export SF_API_PORT=443
  export SF_API_USE_SSL=true

to

.. code-block:: bash

  # Strawberry Fields v0.20.0
  export XANADU_CLOUD_REFRESH_TOKEN="Xanadu Cloud API key goes here"
  export XANADU_CLOUD_HOST="platform.strawberryfields.ai"
  export XANADU_CLOUD_PORT=443
  export XANADU_CLOUD_TLS=true

Finally, ``strawberryfields.store_account()`` has been replaced such that

.. code-block:: python

  # Strawberry Fields v0.19.0
  import strawberryfields as sf
  sf.store_account("Xanadu Cloud API key goes here")

becomes

.. code-block:: python

  # Strawberry Fields v0.20.0
  import xcc
  xcc.Settings(REFRESH_TOKEN="Xanadu Cloud API key goes here").save()

.. note::

  In most cases, running the following command is sufficient to configure
  Strawberry Fields v0.20.0:

  .. code-block:: console

      $ xcc config set REFRESH_TOKEN "Xanadu Cloud API key goes here"


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

.. warning::

  Windows PowerShell users should write ``Get-Content foo.xbb -Raw`` instead of ``cat foo.xbb``.

Connection
^^^^^^^^^^

All ``strawberryfields.api.Connection`` instances must be replaced by their
equivalent `XCC Connection <xcc.Connection>`_
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
        tls=True,                                        # See "use_ssl" argument above.
    )

Job
^^^

``strawberryfields.api.Job`` has been replaced with an equivalent
`XCC Job <https://xanadu-cloud-client.readthedocs.io/en/stable/api/xcc.Job.html>`_
class. This will affect the object returned when running an asynchronous job on the
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

  In Strawberry Fields v0.20.0, the (semantically) equivalent ``Job`` object would work slightly
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

Result
^^^^^^

When running local or remote jobs, a ``strawberryfields.Result`` object will be returned. This
object works slightly differently in Strawberry Fields v0.20.0, compared to the
``strawberryfields.api.Result`` object in Strawberry Fields v0.19.0.

While ``Result.samples`` should return the same type and shape as before, the ``Result.all_samples``
property has been renamed to ``Result.samples_dict``. This property returns the samples as a
dictionary with corresponding measured modes as keys.

.. code-block:: pycon

    >>> res = eng.run(prog, shots=3)
    >>> res.samples
    array([[1, 0], [0, 1], [1, 1]])
    >>> res.samples_dict
    {0: [np.array([1, 0, 1])], 1: [np.array([0, 1, 1])]}

All instances of ``Result.all_samples`` must be replaced with ``Result.samples_dict`` in Strawberry Fields v0.20.0.

Device specification
^^^^^^^^^^^^^^^^^^^^

The ``DeviceSpec`` no longer has a connection object, and is simply a container for a remote device
specification. The target argument has been removed in favour of using the target entry in the
device specification dictionary.

Creating a ``DeviceSpec`` object in Strawberry Fields v0.19.0:

.. code-block:: python

    connection = sf.api.Connection()
    spec = {"target": "X8", "layout": "", "modes": 8, "gate_parameters": {}}
    device_spec = sf.api.DeviceSpec(target="X8", spec=spec, connection=connection)

The (semantically) equivalent code in Strawberry Fields v0.20.0 is

.. code-block:: python

    spec = {"target": "X8", "layout": "", "modes": 8, "gate_parameters": {}}
    device_spec = sf.DeviceSpec(spec=spec)

.. note::

    The remote specification dictionary keys "target", "layout", "modes" and "gate_parameters" are
    mandatory in Strawberry Fields v0.20.0. If one or more are missing, a ``DeviceSpec`` object
    cannot be created.

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

Result
^^^^^^

When running local or remote jobs, a ``strawberryfields.api.Result`` object will be returned. This
object will function slightly differently in Strawberry Fields v0.20.0.

While ``Result.samples`` should return the same type and shape as before, the `Result.all_samples`
property has been renamed to `Result.samples_dict`. This property returns the samples as a
dictionary with corresponding measured modes as keys.

.. code-block:: pycon

    >>> res = eng.run(prog, shots=3)
    >>> res.samples
    array([[1, 0], [0, 1], [1, 1]])
    >>> res.samples_dict
    {0: [np.array([1, 0, 1])], 1: [np.array([0, 1, 1])]}

All instances of ``Result.all_samples`` must be replaced with ``Result.samples_dict`` in Strawberry Fields v0.20.0.

Device Specification
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
    device_spec = sf.api.DeviceSpec(spec=spec)

.. note::

    The remote specification dictionary keys "target", "layout", "modes" and "gate_parameters" are
    mandatory in Strawberry Fields v0.20.0. If one or more are missing, a ``DeviceSpec`` object
    cannot be created.

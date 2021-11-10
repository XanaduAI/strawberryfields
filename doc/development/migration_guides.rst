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

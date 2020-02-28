Configuration
=============

On first import, Strawberry Fields attempts to load the configuration file ``config.toml``, by
scanning the following three directories in order of preference:

1. The current directory
2. The path stored in the environment variable ``SF_CONF``
3. The default user configuration directory:

   * On Linux: ``~/.config/strawberryfields``
   * On Windows: ``~C:\Users\USERNAME\AppData\Local\Xanadu\strawberryfields``
   * On MacOS: ``~/Library/Application\ Support/strawberryfields``

If no configuration file is found, a warning message will be displayed in the logs,
and all device parameters will need to be passed as keyword arguments when
loading the device.

The user can access the initialized configuration via `strawberryfields.config`, view the
loaded configuration filepath, print the configurations options, access and modify
them via keys, and save/load new configuration files.

Configuration files
-------------------

The configuration file ``config.toml`` uses the `TOML standard <https://github.com/toml-lang/toml>`_,
and has the following format:

.. code-block:: toml

    [api]
    # Options for the Strawberry Fields Cloud API
    authentication_token = "071cdcce-9241-4965-93af-4a4dbc739135"
    hostname = "localhost"
    use_ssl = true
    port = 443

Configuration options
---------------------

**authentication_token (str)** (*required*)
    API token for authentication to the Xanadu Cloud platform. This is required
    for submitting remote jobs using :class:`~.StarshipEngine`. Corresponding
    environment variable: ``SF_API_AUTHENTICATION_TOKEN``

**hostname (str)** (*optional*)
    The hostname of the server to connect to. Defaults to ``localhost``. Must
    be one of the allowed hosts. Corresponding environment variable:
    ``SF_API_HOSTNAME``

**use_ssl (bool)** (*optional*)
    Whether to use SSL or not when connecting to the API. True or False.
    Corresponding environment variable: ``SF_API_USE_SSL``

**port (int)** (*optional*)
    The port to be used when connecting to the remote service.
    Corresponding environment variable: ``SF_API_PORT``

Store your account
------------------

Using the :func:`~.store_account` function, a configuration file containing your Xanadu Cloud credentials
will be created. By default, this configuration file is saved *globally*, and will be used every time
a remote job is submitted.

In these examples ``"MyToken"`` should be replaced with a valid authentication token.

.. code-block:: python

    import strawberryfields as sf
    sf.store_account("MyToken")

.. note::

    By default, the account is stored in a ``config.toml`` file located in the
    *Strawberry Fields configuration directory*.

.. warning::

    The ``store_account`` function only needs to be executed once, when
    initially configuring your system to connect to the Xanadu cloud platform.
    
    Take care not to share or publicly commit your authentication token, as it provides
    full access to your account.

The following code snippet can be run to create a configuration file locally in
the *same directory* of a Python script or Jupyter Notebook that uses
Strawberry Fields:

.. code-block:: python

    import strawberryfields as sf
    sf.store_account("MyToken", location="local")

For more detailed examples, visit the :func:`~.store_account`
documentation.

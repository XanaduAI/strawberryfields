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
    The authentication token to use when connecting to the API. Will be sent with every request in
    the header. Corresponding environment variable: ``SF_API_AUTHENTICATION_TOKEN``

**hostname (str)** (*optional*)
    The hostname of the server to connect to. Defaults to ``localhost``. Must be one of the allowed
    hosts. Corresponding environment variable: ``SF_API_HOSTNAME``

**use_ssl (bool)** (*optional*)
    Whether to use SSL or not when connecting to the API. True or False.
    Corresponding environment variable: ``SF_API_USE_SSL``

**port (int)** (*optional*)
    The port to be used when connecting to the remote service.
    Corresponding environment variable: ``SF_API_PORT``

Store your account
------------------

Using the :func:`~.store_account` function, a configuration file can be created easily. It only requires specifying the authentication token. Apart from that, optional configuration options can be passed as keyword arguments.

Configure for the current SF project
************************************

The following is an example for using ``store_account`` with defaults:

.. code-block:: python

    import strawberryfields as sf
    sf.store_account("MyToken")

where ``"MyToken"`` contains the user specific authentication token.

It is advised to execute this code snippet **only once** per configuration in the same directory where the SF project can be found. It should also be separated from any other Strawberry Fields scripts. Using the default options it will store the account in the *current working directory* by creating a ``config.toml`` file.

Configure for every SF project
******************************

The following code snippet can be used to create a configuration file for *every Strawberry Fields project*.

.. code-block:: python

    import strawberryfields as sf
    sf.store_account("MyToken", location="user_config")

where ``"MyToken"`` is the user specific authentication token.
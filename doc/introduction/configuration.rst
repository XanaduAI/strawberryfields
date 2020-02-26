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

Summary of options
------------------

Keyword arguments
*****************

    **authentication_token (str)** (*required*)
        The authentication token to use when connecting to the API. Will be sent with every request in
        the header.
    **hostname (str)** (*optional*)
        The hostname of the server to connect to. Defaults to localhost. Must be one of the allowed
        hosts.
    **use_ssl (bool)** (*optional*)
        Whether to use SSL or not when connecting to the API. True or False.
    **port (int)** (*optional*)
        The port to be used when connecting to the remote service.

Environment variables
*********************

* SF_API_AUTHENTICATION_TOKEN
* SF_API_HOSTNAME
* SF_API_USE_SSL
* SF_API_PORT
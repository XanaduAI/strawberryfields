sf.configuration
================

.. currentmodule:: strawberryfields.configuration

This module contains functionality for configuring Strawberry Fields.


Configuration files
-------------------

When connecting to the Xanadu cloud platform, Strawberry Fields will attempt to load
the configuration file ``config.toml``, by
scanning the following three directories in order of preference:

1. The current directory
2. The path stored in the environment variable ``SF_CONF`` (if specified)
3. The default user configuration directory:

   * On Linux: ``~/.config/strawberryfields``
   * On Windows: ``C:\Users\USERNAME\AppData\Local\Xanadu\strawberryfields``
   * On MacOS: ``~/Library/Application\ Support/strawberryfields``

The configuration file ``config.toml`` uses the `TOML standard <https://github.com/toml-lang/toml>`_,
and has the following format:

.. code-block:: toml

    [api]
    # Options for the Strawberry Fields cloud API
    authentication_token = "AUTHENTICATION_TOKEN"
    hostname = "platform.strawberryfields.ai"
    use_ssl = true
    port = 443

Configuration options
---------------------

``[api]``
^^^^^^^^^

Settings for the Xanadu cloud platform.


**authentication_token (str)** (*required*)
    API token for authentication to the Xanadu cloud platform. This is required
    for submitting remote jobs using :class:`~.RemoteEngine`.

    *Corresponding environment variable:* ``SF_API_AUTHENTICATION_TOKEN``

**hostname (str)** (*optional*)
    The hostname of the server to connect to. Defaults to ``platform.strawberryfields.ai``. Must be one of the allowed hosts.

    *Corresponding environment variable:* ``SF_API_HOSTNAME``

**use_ssl (bool)** (*optional*)
    Whether to use SSL or not when connecting to the API. ``true`` or ``false``. Defaults to ``true``.

    *Corresponding environment variable:* ``SF_API_USE_SSL``

**port (int)** (*optional*)
    The port to be used when connecting to the remote service. Defaults to 443.

    *Corresponding environment variable:* ``SF_API_PORT``

Functions
---------

.. warning::

    Unless you are a Strawberry Fields developer, you likely do not need
    to use the functions in this module directly.

.. automodapi:: strawberryfields.configuration
    :no-heading:
    :skip: user_config_dir, store_account, active_configs, reset_config, create_logger
    :no-inheritance-diagram:

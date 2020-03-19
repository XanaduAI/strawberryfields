Configuration
=============

Configuring your account
------------------------

Before using the Xanadu cloud platform, you will need to configure your credentials. This
can be done using several methods:

* **Configuration file**. Use the Strawberry Fields :ref:`command
  line interface <cli>`,

  .. code-block:: bash

      $ sf configure --token MYAUTH

  or the :func:`~.store_account` function,

  >>> sf.store_account("MYAUTH")

  to create a configuration file containing your credentials. This is the recommended approach.

  By default, the account is stored in a ``config.toml`` file located in the
  *Strawberry Fields configuration directory*. For more detailed examples, visit the
  :func:`~.store_account` documentation.

* **Environment variables**. Set the ``SF_API_AUTHENTICATION_TOKEN`` environment variable
  with your API token.

* **Keyword argument**. Pass the keyword argument ``token`` when initializing the
  :class:`~.RemoteEngine`.


.. _cli:

Command line interface
----------------------

The Strawberry Fields command line program ``sf`` provides several utilities
including:

* ``sf configure [--token] [--local]``: configure the connection to the cloud platform

* ``sf run INFILE [--output OUTFILE]``: submit and execute quantum programs from the command line

* ``sf --ping``: verify your connection to the Xanadu cloud platform.

Using the ``sf configure`` command to create a configuration file on Linux:

.. code-block:: bash

    $ sf configure --token MYAUTH
    $ cat  ~/.config/strawberryfields/config.toml
    [api]
    authentication_token = "MYAUTH"
    hostname = "platform.strawberryfields.ai"
    use_ssl = true
    port = 443

Using the ``sf configure`` command to create a configuration file on Windows:

.. code-block:: ps1con

    C:\> sf configure --token MYAUTH
    C:\> cat C:\Users\USERNAME\AppData\Local\Xanadu\strawberryfields\config.toml
    [api]
    authentication_token = "MYAUTH"
    hostname = "platform.strawberryfields.ai"
    use_ssl = true
    port = 443

For more advanced configuration options see the :func:`~.configuration_wizard` page.

**Flags:**

* ``--token``: the authentication token to use
* ``--local``: whether or not to create the configuration file locally in the current directory

If not provided, the configuration file will be saved in the user's configuration directory.


Configuration files
-------------------

When connecting to the Xanadu cloud platform, Strawberry Fields will attempt to load
the configuration file ``config.toml``, by
scanning the following three directories in order of preference:

1. The current directory
2. The path stored in the environment variable ``SF_CONF``
3. The default user configuration directory:

   * On Linux: ``~/.config/strawberryfields``
   * On Windows: ``C:\Users\USERNAME\AppData\Local\Xanadu\strawberryfields``
   * On MacOS: ``~/Library/Application\ Support/strawberryfields``

The configuration file ``config.toml`` uses the `TOML standard <https://github.com/toml-lang/toml>`_,
and has the following format:

.. code-block:: toml

    [api]
    # Options for the Strawberry Fields cloud API
    authentication_token = "071cdcce-9241-4965-93af-4a4dbc739135"
    hostname = "platform.strawberryfields.ai"
    use_ssl = true
    port = 443

Configuration options
---------------------

**authentication_token (str)** (*required*)
    API token for authentication to the Xanadu cloud platform. This is required
    for submitting remote jobs using :class:`~.RemoteEngine`. Corresponding
    environment variable: ``SF_API_AUTHENTICATION_TOKEN``

**hostname (str)** (*optional*)
    The hostname of the server to connect to. Defaults to ``platform.strawberryfields.ai``. Must
    be one of the allowed hosts. Corresponding environment variable:
    ``SF_API_HOSTNAME``

**use_ssl (bool)** (*optional*)
    Whether to use SSL or not when connecting to the API. True or False.
    Corresponding environment variable: ``SF_API_USE_SSL``

**port (int)** (*optional*)
    The port to be used when connecting to the remote service.
    Corresponding environment variable: ``SF_API_PORT``

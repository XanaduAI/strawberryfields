sf.cli
======

.. currentmodule:: strawberryfields.cli

The Strawberry Fields command line program ``sf`` provides several utilities
including:

* ``sf configure [--token] [--local]``: configure the connection to the cloud platform

* ``sf run input [--output FILE]``: submit and execute quantum programs from the command line

* ``sf --ping``: verify your connection to the Xanadu cloud platform



Configure
---------

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

**Flags:**

* ``--token``: the authentication token to use

* ``--local``: whether or not to create the configuration file locally in the current directory


If not provided, the configuration file will be saved in the user's configuration directory.

Configuration wizard
~~~~~~~~~~~~~~~~~~~~

For more advanced configuration options, the configuration wizard can be used.

Here we show how the interactive wizard can be used to configure each API
option. Once done, we have a look at the newly created configuration file.

Example run in a Linux-based operating system:

.. code-block:: bash

    $ sf configure
    Please enter the authentication token to use when connecting: [] MYAUTH
    Please enter the hostname of the server to connect to: [platform.strawberryfields.ai] MYHOST
    Should the client attempt to connect over SSL? [y] N
    Please enter the port number to connect with: [443] 12345

    $cat  ~/.config/strawberryfields/config.toml
    [api]
    authentication_token = "MYAUTH"
    hostname = "MYHOST"
    use_ssl = false
    port = "12345"

The same example run on a Windows operating system:

.. code-block::  ps1con


    C:\>  sf configure

    Please enter the authentication token to use when connecting: [] MYAUTH
    Please enter the hostname of the server to connect to: [platform.strawberryfields.ai] MYHOST
    Should the client attempt to connect over SSL? [y] N
    Please enter the port number to connect with: [443] 12345

    C:\>  cat C:\Users\USERNAME\AppData\Local\Xanadu\strawberryfields\config.toml

    [api]
    authentication_token = "MYAUTH"
    hostname = "MYHOST"
    use_ssl = false
    port = "12345"

Testing your connection
-----------------------

To test that your account credentials correctly authenticate against the cloud platform, the ``ping`` flag can be used:

.. code-block:: bash

    $ sf --ping
    You have successfully authenticated to the platform!

Submitting jobs
---------------

To execute a Blackbird ``.xbb`` file from the command line, the ``sf run`` command
line program can be used:

.. code-block:: console

    sf run test.xbb --output out.txt

After executing the above command, the result will be stored in ``out.txt``
in the current working directory.
You can also omit the ``--output`` parameter to print the result to the screen.

Functions
---------

.. warning::

    Unless you are a Strawberry Fields developer, you likely do not need
    to access the functions in this module directly.

.. automodapi:: strawberryfields.cli
    :no-heading:
    :no-inheritance-diagram:
    :skip: store_account, create_config, load, RemoteEngine, Connection, ConfigurationError

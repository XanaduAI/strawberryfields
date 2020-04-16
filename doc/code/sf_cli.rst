sf.cli
======

.. currentmodule:: strawberryfields.cli

The Strawberry Fields command line program ``sf`` provides several utilities
including:

* ``sf configure [--token] [--local]``: configure the connection to the cloud platform

* ``sf run INFILE [--output FILE]``: submit and execute Blackbird scripts containing quantum programs from the command line

* ``sf --ping``: verify your connection to the Xanadu cloud platform


Configure
---------

Using the ``sf configure`` command to create a configuration file on Linux:

.. code-block:: console

    $ sf configure --token AUTHENTICATION_TOKEN
    $ cat  ~/.config/strawberryfields/config.toml
    [api]
    authentication_token = "AUTHENTICATION_TOKEN"
    hostname = "platform.strawberryfields.ai"
    use_ssl = true
    port = 443

Using the ``sf configure`` command to create a configuration file on Windows:

.. code-block:: doscon

    C:\> sf configure --token AUTHENTICATION_TOKEN
    C:\> type C:\Users\USERNAME\AppData\Local\Xanadu\strawberryfields\config.toml
    [api]
    authentication_token = "AUTHENTICATION_TOKEN"
    hostname = "platform.strawberryfields.ai"
    use_ssl = true
    port = 443

In both the above examples, replace ``AUTHENTICATION_TOKEN`` above with your
Xanadu cloud access token.

**Flags:**

* ``--token``: the authentication token to use

* ``--local``: causes the configuration file to be created locally in the current directory

If ``--local`` is not provided, the configuration file will be saved in the user's configuration directory.

.. seealso::

    :doc:`Configuration options </code/sf_configuration>`


Configuration wizard
^^^^^^^^^^^^^^^^^^^^

For more advanced configuration options, the configuration wizard can be used.

Here we show how the interactive wizard can be used to configure each API
option. Once done, we have a look at the newly created configuration file.

Example run in a Linux-based operating system:

.. code-block:: console

    $ sf configure
    Please enter the authentication token to use when connecting: [] AUTHENTICATION_TOKEN
    Please enter the hostname of the server to connect to: [platform.strawberryfields.ai] MYHOST
    Should the client attempt to connect over SSL? [y] N
    Please enter the port number to connect with: [443] 12345

    $ cat  ~/.config/strawberryfields/config.toml
    [api]
    authentication_token = "AUTHENTICATION_TOKEN"
    hostname = "MYHOST"
    use_ssl = false
    port = "12345"

The same example run on a Windows operating system:

.. code-block::  doscon

    C:\> sf configure
    Please enter the authentication token to use when connecting: [] AUTHENTICATION_TOKEN
    Please enter the hostname of the server to connect to: [platform.strawberryfields.ai] MYHOST
    Should the client attempt to connect over SSL? [y] N
    Please enter the port number to connect with: [443] 12345

    C:\> type C:\Users\USERNAME\AppData\Local\Xanadu\strawberryfields\config.toml
    [api]
    authentication_token = "AUTHENTICATION_TOKEN"
    hostname = "MYHOST"
    use_ssl = false
    port = "12345"


Testing your connection
-----------------------

To test that your account credentials correctly authenticate against the cloud platform, the ``ping`` flag can be used:

.. code-block:: console

    $ sf --ping
    You have successfully authenticated to the platform!


Submitting jobs
---------------

To execute a Blackbird ``.xbb`` file from the command line, the ``sf run`` command
line program can be used:

.. code-block:: console

    $ sf run test.xbb --output out.txt

After executing the above command, the result will be stored in ``out.txt``
in the current working directory.
You can also omit the ``--output`` parameter to print the result to the screen.


Code details
------------

.. warning::

    Unless you are a Strawberry Fields developer, you likely do not need
    to access the functions in this module directly.

.. automodapi:: strawberryfields.cli
    :no-heading:
    :no-inheritance-diagram:
    :skip: store_account, create_config, load, RemoteEngine, Connection, ConfigurationError, ping

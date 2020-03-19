The cloud platform
==================

Using Strawberry Fields, you can submit quantum programs
to be executed on photonic hardware using the Xanadu cloud platform.


Configuring your account
------------------------

Before using the Xanadu cloud platform, you will need to configure your credentials. This
can be done using several methods:

* **Configuration file (recommended)**. Use the Strawberry Fields :func:`~.store_account` function,

  >>> sf.store_account("AUTHENTICATION_TOKEN")

  or the ``sf configure`` :doc:`command line interface </code/sf_cli>`,

  .. code-block:: bash

      $ sf configure --token AUTHENTICATION_TOKEN

  to create a configuration file containing your credentials. Replace
  ``AUTHENTICATION_TOKEN`` above with your Xanadu cloud access token.

  By default, the account is stored in a ``config.toml`` file located in the
  *Strawberry Fields configuration directory*.

* **Environment variables**. Set the ``SF_API_AUTHENTICATION_TOKEN`` environment variable
  with your API token.

* **Keyword argument**. Pass the keyword argument ``token`` when initializing the
  :class:`~.RemoteEngine`.

.. seealso::

    :doc:`Configuration options </code/sf_configuration>`_, :doc:`command line interface </code/sf_cli>`_


Verifying your connection
~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that your account credentials correctly authenticate against the cloud
platform, the :func:`~.ping` function can be used from within Python,

>>> sf.cli.ping()
You have successfully authenticated to the platform!

or the ``ping`` flag can be used on the command line:

.. code-block:: bash

    $ sf --ping
    You have successfully authenticated to the platform!


Submitting jobs
---------------



Software tests
==============

The Strawberry Fields test suite requires `pytest <https://docs.pytest.org/en/latest/>`_ and `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ for coverage reports. These can both be installed via ``pip``:
::

    $ pip install pytest pytest-cov


To ensure that Strawberry Fields is working correctly after installation, the test suite can be run by navigating to the source code folder and running
::

    $ make test

Note that this runs *all* of the tests, using *all* available backends, in both pure *and* mixed modes, so can be quite slow (it should take around 1 hour to complete). Alternatively, you can run the full test suite for a particular component by running
::

    $ make test-[component]

where ``[component]`` should be replaced with either ``frontend`` for the Strawberry Fields frontend UI, or one of the :ref:`backend <backends>` you would like to test (``fock``, ``tf``, or ``gaussian``).

Pytest can accept a boolean logic string specifying exactly which tests to run, if finer control is needed. For example, to run all tests for the frontend and the Gaussian backend, as well as the Fock backend (but only for pure states), you can run:
::

    $ make test-"gaussian or frontend or (fock and pure)"

The above syntax also works for the ``make coverage`` command, as well as ``make batch-test`` command for running the tests in batched mode.


Individual test modules are run by invoking pytest directly from the command line:
::

    $ pytest tests/test_gate.py


.. note:: **Adding tests to Strawberry Fields**

    The ``tests`` folder is organised into three subfolders: ``backend`` for tests that
    only import a Strawberry Fields backend, ``frontend`` for tests that import the Strawberry
    Fields UI but do not make use of a backend, and ``integration`` for tests that test
    integration of the frontend and backends.

    When writing new tests, make sure to mark what components it tests. For a backend test,
    you can use the ``backends`` mark, which accepts the names of the backends:

    .. code-block:: python

        pytest.mark.backends("fock", "gaussian")

    For a frontend-only test, you can use the frontend mark:

    .. code-block:: python

        pytest.mark.frontend

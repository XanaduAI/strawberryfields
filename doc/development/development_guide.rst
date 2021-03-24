Development guide
=================

Dependencies
------------

Strawberry Fields requires the following libraries be installed:

* `Python <http://python.org/>`_ >= 3.6

as well as the following Python packages:

* `NumPy <http://numpy.org/>`_ >= 1.17.4
* `SciPy <http://scipy.org/>`_ >= 1.0.0
* `NetworkX <http://networkx.github.io/>`_ >= 2.0
* `Blackbird <https://quantum-blackbird.readthedocs.io>`_ >= 0.2.0
* `The Walrus <https://the-walrus.readthedocs.io>`_ >= 0.7.0
* `toml <https://pypi.org/project/toml/>`_
* `appdirs <https://pypi.org/project/appdirs/>`_


If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version
of Python packaged for scientific computation.

Installation
------------

For development purposes, it is recommended to install the Strawberry Fields source code
using development mode:

.. code-block:: bash

    git clone https://github.com/XanaduAI/strawberryfields
    cd strawberryfields
    pip install -e .

The ``-e`` flag ensures that edits to the source code will be reflected when
importing StrawberryFields in Python.


TensorFlow support
------------------

To use Strawberry Fields with TensorFlow, version 2.0 of
TensorFlow (or higher) is required. This can be installed alongside Strawberry Fields
as follows:

.. code-block:: console

    pip install strawberryfields tensorflow

Or, to install Strawberry Fields and TensorFlow with GPU and CUDA support:

.. code-block:: console

    pip install strawberryfields tensorflow-gpu

Development environment
-----------------------

Strawberry fields uses a ``pytest`` suite for testing and ``black`` for formatting. These
dependencies can be installed via ``pip``:

.. code-block:: bash

    pip install -r dev_requirements.txt

Software tests
--------------

The Strawberry Fields test suite includes `pytest <https://docs.pytest.org/en/latest/>`_,
`pytest-mocks <https://github.com/pytest-dev/pytest-mock/>`_, 
`pytest-randomly <https://github.com/pytest-dev/pytest-randomly>`_,
and `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ for coverage reports.

To ensure that Strawberry Fields is working correctly after installation, the test suite
can be run by navigating to the source code folder and running

.. code-block:: bash

    make test

Note that this runs *all* of the tests, using *all* available backends, so can be quite
slow (it should take around 40 minutes to complete). Alternatively, you can run the full
test suite for a particular component by running

.. code-block:: bash

    make test-[component]

where ``[component]`` should be replaced with either ``frontend`` for the Strawberry Fields
frontend UI, ``apps`` for the applications layer, or one of the :doc:`backends </code/sf_backends>`
you would like to test (``fock``, ``tf``, or ``gaussian``).

Pytest can accept a boolean logic string specifying exactly which tests to run,
if finer control is needed. For example, to run all tests for the frontend and the
Gaussian backend, as well as the Fock backend (but only for pure states), you can run:

.. code-block:: bash

    make test-"gaussian or frontend or (fock and pure)"

The above syntax also works for the ``make coverage`` command, as well as
``make batch-test`` command for running the tests in batched mode.

Individual test modules are run by invoking pytest directly from the command line:

.. code-block:: bash

    pytest tests/test_gate.py


.. note:: **Adding tests to Strawberry Fields**

    The ``tests`` folder is organised into several subfolders:

    - ``backend`` for tests that only import a Strawberry Fields backend,
    - ``frontend`` for tests that import the Strawberry Fields UI but do not make use of a backend,
    - ``integration`` for tests that verify integration of the frontend and backends,
    - ``apps`` for tests of the applications layer
    - ``api`` for tests that only import and use the strawberryfields.api package

    When writing new tests, make sure to mark what components they test.

    Certain tests that are related to a specific backend, e.g. test cases for
    its operations or the states returned by a backend. For a backend test, you
    can use the ``backends`` mark, which accepts the names of the backends:

    .. code-block:: python

        pytest.mark.backends("fock", "gaussian")

    For specific test cases, the decorator can be used to mark only classes and
    test functions:

    .. code-block:: python

        @pytest.mark.backends("fock", "gaussian")
        def test_fock_and_gaussian_feature():

    Adding tests for an engine, operations, parameters and other parts of the
    user interface can be added as part of the frontend tests. For a
    frontend-only test, you can use the frontend mark:

    .. code-block:: python

        pytest.mark.frontend

    This could then be used on the module level to mark not just a single test
    case, but the entire test file as a frontend test:

    .. code-block:: python

        mark = pytest.mark.frontend


.. note:: **Run options for Strawberry Fields tests**

    Several run options can be helpful for testing Strawberry Fields.

    Marks mentioned in the previous section are useful also when running tests
    and selecting only certain tests to be run. They can be specified by using
    the ``-m`` option for ``pytest``.

    The following command can be used for example, to run tests related to the
    ``"Fock"`` backend:

    .. code-block:: console

        pytest -m fock

    When running tests, it can also be useful to examine a single failing test.
    The following command stops at the first failing test:

    .. code-block:: console

        pytest -x

    For further useful options (e.g. ``-k``, ``-s``, ``--tb=short``, etc.)
    refer to the ``pytest --help`` command line usage description or the
    ``pytest`` online documentation.


Test coverage
^^^^^^^^^^^^^

Test coverage can be checked by running

.. code-block:: bash

    make coverage

The output of the above command will show the coverage percentage of each
file, as well as the line numbers of any lines missing test coverage.

To obtain coverage, the ``pytest-cov`` plugin is needed.

The coverage of a specific file can also be checked by generating a report:

.. code-block:: console

    pytest tests/backend/test_states.py --cov=strawberryfields/location/to/module --cov-report=term-missing

Here the coverage report will be created relative to the module specified by
the path passed to the ``--cov=`` option.

The previously mentioned ``pytest`` options can be combined with the coverage
options. As an example, the ``-k`` option allows you to pass a boolean string
using file names, test class/test function names, and marks. Using ``-k`` in
the following command we can get the report of a specific file while also
filtering out certain tests:

.. code-block:: console

    pytest tests/backend/test_states.py --cov --cov-report=term-missing -k 'not TestBaseGaussianMethods'

Passing the ``--cov`` option without any modules specified will generate a
coverage report for all modules of Strawberry Fields.

Format
------

Contributions are checked for format alignment in the pipeline. With ``black``
installed, changes can be formatted locally using:

.. code-block:: bash

    make format

Contributors without ``make`` installed can run ``black`` directly using:

.. code-block:: bash

    black -l 100 strawberryfields

Documentation
-------------

Additional packages are required to build the documentation, as specified in
``doc/requirements.txt``. These packages can be installed using:

.. code-block:: bash

    pip install -r doc/requirements.txt

from within the top-level directory. To then build the HTML documentation, run

.. code-block:: bash

    make docs

The documentation can be found in the :file:`doc/_build/html/` directory.


Adding a new module to the docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several steps to adding a new module to the documentation:

1. Make sure your module has a one-to-two line module docstring, that summarizes
   what the module purpose is, and what it contains.

2. Add a file ``doc/code/sf_module_name.rst``, that contains the following:

   .. literalinclude:: example_module_rst.txt
       :language: rest

3. Add ``code/sf_module_name`` to the table of contents at the bottom of ``doc/index.rst``.


Adding a new package to the docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adding a new subpackage to the documentation requires a slightly different process than
a module:

1. Make sure your package ``__init__.py`` file has a one-to-two line module docstring,
   that summarizes what the package purpose is, and what it contains.

2. At the bottom of the ``__init__.py`` docstring, add an autosummary table that contains
   all modules in your package:

   .. literalinclude:: example_module_autosummary.txt
       :language: rest

   All modules should also contain a module docstring that summarizes the module.

3. Add a file ``doc/code/sf_package_name.rst``, that contains the following:

   .. literalinclude:: example_package_rst.txt
       :language: rest

4. Add ``code/sf_package_name`` to the table of contents at the bottom of ``doc/index.rst``.


Submitting a pull request
-------------------------

Before submitting a pull request, please make sure the following is done:

* **All new features must include a unit test.** If you've fixed a bug or added
  code that should be tested, add a test to the ``tests`` directory.

  Strawberry Fields uses pytest for testing; common fixtures can be found in the ``tests/conftest.py``
  file.

* **All new functions and code must be clearly commented and documented.**

  Have a look through the source code at some of the existing function docstrings---
  the easiest approach is to simply copy an existing docstring and modify it as appropriate.

  If you do make documentation changes, make sure that the docs build and render correctly by
  running ``make docs``.

* **Ensure that the test suite passes**, by running ``make test``.

* **Make sure the modified code in the pull request conforms to the PEP8 coding standard.**

  The Strawberry Fields source code conforms to `PEP8 standards <https://www.python.org/dev/peps/pep-0008/>`_.
  Before submitting the PR, you can autoformat your code changes using the
  `Black <https://github.com/psf/black>`_ Python autoformatter, with max-line length set to 100:

  .. code-block:: bash

      black -l 100 strawberryfields/path/to/modified/file.py

  We check all of our code against `Pylint <https://www.pylint.org/>`_ for errors.
  To lint modified files, simply ``pip install pylint``, and then from the source code
  directory, run

  .. code-block:: bash

      pylint strawberryfields/path/to/modified/file.py


When ready, submit your fork as a `pull request <https://help.github.com/articles/about-pull-requests>`_
to the Strawberry Fields repository, filling out the pull request template. This template is added
automatically to the comment box when you create a new issue.

* When describing the pull request, please include as much detail as possible
  regarding the changes made/new features added/performance improvements. If including any
  bug fixes, mention the issue numbers associated with the bugs.

* Once you have submitted the pull request, three things will automatically occur:

  - The **test suite** will automatically run on `GitHub Actions
    <https://github.com/XanaduAI/strawberryfields/actions?query=workflow%3ATests>`_
    to ensure that all tests continue to pass.

  - Once the test suite is finished, a **code coverage report** will be generated on
    `Codecov <https://codecov.io/gh/XanaduAI/strawberryfields>`_. This will calculate the percentage
    of Strawberry Fields covered by the test suite, to ensure that all new code additions
    are adequately tested.

  - Finally, the **code quality** is calculated by
    `Codefactor <https://app.codacy.com/app/XanaduAI/strawberryfields/dashboard>`_,
    to ensure all new code additions adhere to our code quality standards.

Based on these reports, we may ask you to make small changes to your branch before
merging the pull request into the master branch. Alternatively, you can also
`grant us permission to make changes to your pull request branch
<https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/>`_.

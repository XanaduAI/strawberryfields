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

To use Strawberry Fields with TensorFlow, version 1.3 of
TensorFlow is required. This can be installed alongside Strawberry Fields
as follows:

.. code-block:: console

    pip install strawberryfields tensorflow==1.3

Or, to install Strawberry Fields and TensorFlow with GPU and CUDA support:

.. code-block:: console

    pip install strawberryfields tensorflow-gpu==1.3

Note that TensorFlow version 1.3 is only supported on Python versions
less than 3.7. You can use the following command to check your
Python version:

.. code-block:: console

    python --version

If the above prints out 3.7, and you still want to use TensorFlow, you will need
to install Python 3.6. The recommended method is to install
`Anaconda for Python 3 <https://www.anaconda.com/download/>`_.

Once installed, you can then create a Python 3.6 Conda environment:

.. code-block:: console

    conda create --name sf_tensorflow_env python=3.6
    conda activate sf_tensorflow_env
    pip install strawberryfields tensorflow==1.3

Software tests
--------------

The Strawberry Fields test suite requires `pytest <https://docs.pytest.org/en/latest/>`_
and `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ for coverage reports.
These can both be installed via ``pip``:

.. code-block:: bash

    pip install pytest pytest-cov


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

    The ``tests`` folder is organised into four subfolders: ``backend`` for tests that
    only import a Strawberry Fields backend, ``frontend`` for tests that import the Strawberry
    Fields UI but do not make use of a backend, ``integration`` for tests that test
    integration of the frontend and backends, and ``apps`` for tests of the applications layer.

    When writing new tests, make sure to mark what components it tests. For a backend test,
    you can use the ``backends`` mark, which accepts the names of the backends:

    .. code-block:: python

        pytest.mark.backends("fock", "gaussian")

    For a frontend-only test, you can use the frontend mark:

    .. code-block:: python

        pytest.mark.frontend

Test coverage
^^^^^^^^^^^^^

Test coverage can be checked by running

.. code-block:: bash

    make coverage

The output of the above command will show the coverage percentage of each
file, as well as the line numbers of any lines missing test coverage.


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

  - The **test suite** will automatically run on `Travis CI <https://travis-ci.org/XanaduAI/strawberryfields>`_
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

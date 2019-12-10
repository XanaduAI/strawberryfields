Development guide
=================

Dependencies
------------

Strawberry Fields requires the following libraries be installed:

* `Python <http://python.org/>`_ >=3.5,<3.7

as well as the following Python packages:

* `NumPy <http://numpy.org/>`_  >=1.17.4
* `SciPy <http://scipy.org/>`_  >=1.0.0
* `NetworkX <http://networkx.github.io/>`_ >=2.0
* `Blackbird <https://quantum-blackbird.readthedocs.io>`_ >=0.2.0
* `The Walrus <https://the-walrus.readthedocs.io>`_ >=0.7.0
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
frontend UI, ``apps`` for the applications layer, or one of the :ref:`backend <backends>`
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

Test coverage
^^^^^^^^^^^^^

Test coverage can be checked by running

.. code-block:: bash

    make coverage

The output of the above command will show the coverage percentage of each
file, as well as the line numbers of any lines missing test coverage.


Documentation
-------------

To build the documentation, the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ >=2.2.2
* `graphviz <http://graphviz.org/>`_ >=2.38
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ == 0.4.2
* `sphinx-automodapi <https://github.com/astropy/sphinx-automodapi>`_

These can all be installed via ``pip``.

To build the HTML documentation, go to the top-level directory and run

.. code-block:: bash

    make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.

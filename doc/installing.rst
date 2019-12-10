.. _installation:

Installation and downloads
#################################

.. .. include:: ../README.rst
   :start-line: 6

Dependencies
============

.. highlight:: bash

Strawberry Fields requires the following libraries be installed:

* `Python <http://python.org/>`_ >=3.5,<3.7

as well as the following Python packages:

* `NumPy <http://numpy.org/>`_  >=1.16.3
* `SciPy <http://scipy.org/>`_  >=1.0.0
* `NetworkX <http://networkx.github.io/>`_ >=2.0
* `Blackbird <https://quantum-blackbird.readthedocs.io>`_ >=0.2.0
* `The Walrus <https://the-walrus.readthedocs.io>`_ >=0.7.0
* `toml <https://pypi.org/project/toml/>`_
* `appdirs <https://pypi.org/project/appdirs/>`_


If you currently do not have Python 3 installed, we recommend `Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version of Python packaged for scientific computation.


Installation
============

Installation of Strawberry Fields, as well as all required Python packages mentioned above, can be installed via ``pip``:

.. code-block:: console

    pip install strawberryfields


TensorFlow support
==================

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

If the above prints out 3.7, and you still want to use TensorFlow, you will need to install Python 3.6.
The recommended method is to install `Anaconda for Python 3 <https://www.anaconda.com/download/>`_.

Once installed, you can then create a Python 3.6 Conda environment:

.. code-block:: console

    conda create --name sf_tensorflow_env python=3.6
    conda activate sf_tensorflow_env
    pip install strawberryfields tensorflow==1.3


Notebook downloads
===================

Two of the tutorials provided in the documentation, quantum teleporation and Gaussian boson sampling, are also provided in the form of interactive Jupyter notebooks:

1. :download:`QuantumTeleportation.ipynb <../examples/QuantumTeleportation.ipynb>`

2. :download:`GaussianBosonSampling.ipynb <../examples/GaussianBosonSampling.ipynb>`

To open them, launch the Jupyter notebook environment by clicking on the 'Jupyter notebook' shortcut in the start menu (Windows), or by running the following in the Anaconda Prompt/Command Prompt/Terminal:
::

    jupyter notebook

Your web browser should open with the Jupyter notebook home page; simply click the 'Upload' button, browse to the tutorial file you downloaded above, and upload the file. You will now be able to open it and work through the tutorial.



Software tests
==============

The Strawberry Fields test suite requires `pytest <https://docs.pytest.org/en/latest/>`_ and `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ for coverage reports. These can both be installed via ``pip``:
::

    $ pip install pytest pytest-cov


To ensure that Strawberry Fields is working correctly after installation, the test suite can be run by navigating to the source code folder and running
::

    $ make test

Note that this runs *all* of the tests, using *all* available backends, so can be quite slow (it should take around 40 minutes to complete). Alternatively, you can run the full test suite for a particular component by running
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



Documentation
=============

To build the documentation, the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `graphviz <http://graphviz.org/>`_ >=2.38
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6

If using Ubuntu, they can be installed via a combination of ``apt`` and ``pip``:
::

    $ sudo apt install graphviz
    $ pip install sphinx --user
    $ pip install sphinxcontrib-bibtex --user

To build the HTML documentation, go to the top-level directory and run
::

  $ make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.

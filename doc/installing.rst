.. _installation:

Installation and downloads
#################################

.. .. include:: ../README.rst
   :start-line: 6

Dependencies
============

.. highlight:: bash

Strawberry Fields requires the following libraries be installed:

* `Python <http://python.org/>`_ >=3.5

as well as the following Python packages:

* `NumPy <http://numpy.org/>`_  >=1.13.3
* `SciPy <http://scipy.org/>`_  >=1.0.0
* `NetworkX <http://networkx.github.io/>`_ >=2.0
* `Tensorflow <https://www.tensorflow.org/>`_ >=1.3,<1.7


If you currently do not have Python 3 installed, we recommend `Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version of Python packaged for scientific computation.


Installation
============

Installation of Strawberry Fields, as well as all required Python packages mentioned above, can be installed via ``pip``:
::

   	python -m pip install strawberryfields


Make sure you are using the Python 3 version of pip.

Alternatively, you can install Strawberry Fields from the source code by navigating to the top directory and running
::

	python setup.py install


Notebook downloads
===================

Two of the tutorials provided in the documentation, boson sampling and Gaussian boson sampling, are also provided in the form of interactive Jupyter notebooks:

1. :download:`BosonSampling.ipynb <../examples/BosonSampling.ipynb>`

2. :download:`GaussianBosonSampling.ipynb <../examples/GaussianBosonSampling.ipynb>`

To open them, launch the Jupyter notebook environment by clicking on the 'Jupyter notebook' shortcut in the start menu (Windows), or by running the following in the Anaconda Prompt/Command Prompt/Terminal:
::

	jupyter notebook

Your web browser should open with the Jupyter notebook home page; simply click the 'Upload' button, browse to the tutorial file you downloaded above, and upload the file. You will now be able to open it and work through the tutorial.



Software tests
==============

To ensure that Strawberry Fields is working correctly after installation, the test suite can be run by navigating to the source code folder and running
::

	make test

Note that this runs *all* of the tests, using *all* available backends, so can be quite slow (it should take around 40 minutes to complete). Alternatively, you can run the full test suite for a particular backend by running
::

	make test-[backend]

where ``[backend]`` should be replaced with the :ref:`backend <backends>` you would like to test (``fock``, ``tf``, or ``gaussian``).

Individual test modules are run using

::

	python tests/test_gate.py --backend=[backend]


Documentation
=============

To build the documentation, the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `graphviz <http://graphviz.org/>`_ >=2.38
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6

If using Ubuntu, they can be installed via a combination of ``apt`` and ``pip``:
::

	sudo apt install graphviz
	pip install sphinx --user
	pip install sphinxcontrib-bibtex --user

To build the HTML documentation, go to the top-level directory and run
::

  make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.

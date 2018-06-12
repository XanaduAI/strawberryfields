Software tests
==============

To ensure that Strawberry Fields is working correctly after installation, the test suite can be run by navigating to the source code folder and running
::

  make test

Note that this runs *all* of the tests, using *all* available backends, so can be quite slow (it should take around 40 minutes to complete). Alternatively, you can run the full test suite for a particular backend by running
::

  make test-[backend]

where ``[backend]`` should be replaced with the backend you would like to test (``fock``, ``tf``, or ``gaussian``).

Individual test modules are run using
::

  python tests/test_gate.py --backend=[backend]

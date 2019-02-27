"""
Default parameters, commandline arguments and common routines for the unit tests.
"""

import argparse
import unittest
import os
import sys

import logging

import numpy as np

# Make sure strawberryfields is always imported from the same source distribution where the tests reside, not e.g. from site-packages.
# See https://docs.python-guide.org/en/latest/writing/structure/#test-suite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import strawberryfields
from strawberryfields import backends


BACKEND = "fock"
TOLERANCE = 1e-3
CUTOFF_N = 5
ALPHA = 0.1
BATCH_SIZE = 2
HBAR = 2
BATCHED = False
MIXED = False


if "BACKEND" in os.environ:
    BACKEND = os.environ["BACKEND"]
    print('Backend:', BACKEND)


if "BATCHED" in os.environ:
    BATCHED = bool(int(os.environ["BATCHED"]))
    print('Batched:', BATCHED)


if "MIXED" in os.environ:
    MIXED = bool(int(os.environ["MIXED"]))
    print('Mixed:', MIXED)


if "LOGGING" in os.environ:
    logLevel = os.environ["LOGGING"]
    print('Logging:', logLevel)
    numeric_level = getattr(logging, logLevel.upper(), 10)
else:
    numeric_level = 100

logging.basicConfig(level=numeric_level, format='\n%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logging.captureWarnings(True)

def get_commandline_args():
    """Parse the commandline arguments for the unit tests.
    If none are given (e.g. when the test is run as a module instead of a script), the defaults are used.

    Returns:
      argparse.Namespace: parsed arguments in a namespace container
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backend',   type=str,   default=BACKEND,   help='Backend to use for tests.')
    parser.add_argument('-t', '--tolerance', type=float, default=TOLERANCE, help='Numerical tolerance for equality tests.')
    parser.add_argument('-N', '--cutoffN',   type=int,   default=CUTOFF_N,  help='Maximum number state in the Fock basis.')
    parser.add_argument('-a', '--alpha',     type=float, default=ALPHA,     help='Maximum magnitude of coherent states used in tests.')
    parser.add_argument('-H', '--hbar',      type=float, default=HBAR,         help='The value of hbar.')
    parser.add_argument('--batch_size',      type=int,   default=BATCH_SIZE,         help='Batch size.')

    batch_parser = parser.add_mutually_exclusive_group(required=False)
    batch_parser.add_argument('--batched', dest='batched', action='store_true')
    batch_parser.add_argument('--no-batched', dest='batched', action='store_false')
    parser.set_defaults(batched=BATCHED)

    mixed_parser = parser.add_mutually_exclusive_group(required=False)
    mixed_parser.add_argument('--mixed', dest='mixed', action='store_true')
    mixed_parser.add_argument('--pure', dest='mixed', action='store_false')
    parser.set_defaults(mixed=MIXED)

    # HACK: We only parse known args to enable unittest test discovery without parsing errors.
    args, _ = parser.parse_known_args()
    setup_backend(args)
    return args

def setup_backend(args):
    """Loads the chosen backend, checks that it supports the requested properties.
    """
    args.backend_name = args.backend
    backend = backends.load_backend(args.backend)
    args.backend = backend

    args.fock_support     = backend.supports("fock_basis")
    args.gaussian_support = backend.supports("gaussian")
    args.symbolic_support = backend.supports("symbolic")

    if args.mixed and not backend.supports("mixed_states"):
        raise RuntimeError("Current backend does not support mixed states.")
    if args.batched and not backend.supports("batched"):
        raise RuntimeError("Current backend does not support batched operation.")


# parse any possible commandline arguments
args = get_commandline_args()


class BaseTest(unittest.TestCase):
    """ABC for tests.
    Encapsulates the user-given commandline parameters for the test run.
    """
    num_subsystems = None  #: int: number of modes for the backend, must be overridden by child classes

    def setUp(self):
        self.args = args
        self.backend = args.backend
        self.circuit = args.backend  # alias for backend
        self.backend_name = args.backend_name
        self.tol = args.tolerance
        self.D = args.cutoffN + 1  # NOTE the +1
        self.hbar = args.hbar
        self.batched = args.batched

        # keyword arguments for the backend
        self.kwargs = dict(cutoff_dim=self.D, pure=not args.mixed)
        if args.batched:
            self.kwargs["batch_size"] = args.batch_size
            self.bsize = args.batch_size
        else:
            self.bsize = 1

        self.backend.begin_circuit(num_subsystems=self.num_subsystems, hbar=self.hbar, **self.kwargs)

    def logTestName(self):
        logging.info('{}'.format(self.id()))

    def assertAllAlmostEqual(self, first, second, delta, msg=None):
        """
        Like assertAlmostEqual, but works with arrays. All the corresponding elements have to be almost equal.
        """
        if isinstance(first, tuple):
            # check each element of the tuple separately (needed for when the tuple elements are themselves batches)
            if np.all([np.all(first[idx] == second[idx]) for idx, _ in enumerate(first)]):
                return
            if np.all([np.all(np.abs(first[idx] - second[idx])) <= delta for idx, _ in enumerate(first)]):
                return
        else:
            if np.all(first == second):
                return
            if np.all(np.abs(first - second) <= delta):
                return
        standardMsg = '{} != {} within {} delta'.format(first, second, delta)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertAllEqual(self, first, second, msg=None):
        """
        Like assertEqual, but works with arrays. All the corresponding elements have to be equal.
        """
        return self.assertAllAlmostEqual(first, second, delta=0.0, msg=msg)

    def assertAllTrue(self, value, msg=None):
        """
        Like assertTrue, but works with arrays. All the corresponding elements have to be True.
        """
        return self.assertTrue(np.all(value))


class FockBaseTest(BaseTest):
    """ABC for Fock-basis dependent tests."""
    def setUp(self):
        super().setUp()
        if not self.args.fock_support:
            raise unittest.SkipTest('Backend does not support Fock basis operations.')


class GaussianBaseTest(BaseTest):
    """ABC for Gaussian-dependent tests."""
    def setUp(self):
        super().setUp()
        if not self.args.gaussian_support:
            raise unittest.SkipTest('Backend does not support gaussian operations.')


class SymbolicBaseTest(BaseTest):
    """ABC for symbolic tests."""
    def setUp(self):
        super().setUp()
        if not self.args.symbolic_support:
            raise unittest.SkipTest('Backend does not support symbolic operations.')


class FockBackendTest(FockBaseTest):
    """ABC for Fock-basis dependent tests."""
    def setUp(self):
        super().setUp()
        if not isinstance(self.backend, backends.FockBackend):
            raise unittest.SkipTest('Test is only relevant for the Fock backend.')


class TFBackendTest(FockBaseTest):
    """ABC for Fock-basis dependent tests."""
    def setUp(self):
        super().setUp()
        if not isinstance(self.backend, backends.TFBackend):
            raise unittest.SkipTest('Test is only relevant for the Tensorflow backend.')

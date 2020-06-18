# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Default parameters, environment variables, fixtures, and common routines for the unit tests.
"""
# pylint: disable=redefined-outer-name
import os
import pytest

import numpy as np

np.random.seed(42)

import strawberryfields as sf
from strawberryfields.engine import LocalEngine
from strawberryfields.program import Program
from strawberryfields.backends.base import BaseBackend
from strawberryfields.backends.fockbackend import FockBackend
from strawberryfields.backends.gaussianbackend import GaussianBackend


try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError) as e:
    tf_available = False
    tf_version = False
else:
    tf_available = True
    tf_version = tf.__version__


backend_params = [
    pytest.param(FockBackend, marks=pytest.mark.fock),
    pytest.param(GaussianBackend, marks=pytest.mark.gaussian),
]


eng_backend_params = [
    pytest.param("fock", marks=pytest.mark.fock),
    pytest.param("gaussian", marks=pytest.mark.gaussian),
]


if tf_available and tf.__version__[:2] == "2.":
    from strawberryfields.backends.tfbackend import TFBackend

    backend_params.append(pytest.param(TFBackend, marks=pytest.mark.tf))
    eng_backend_params.append(pytest.param("tf", marks=pytest.mark.tf))
else:
    tf_available = False


# defaults
TOL = 1e-3
CUTOFF = 6
ALPHA = 0.1
HBAR = 1.7
BATCHED = False
BATCHSIZE = 2


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope="session")
def cutoff():
    """Fock state cutoff"""
    return int(os.environ.get("CUTOFF", CUTOFF))


@pytest.fixture(scope="session")
def alpha():
    """Maximum magnitude of coherent states used in tests"""
    return float(os.environ.get("ALPHA", ALPHA))


@pytest.fixture(scope="session")
def hbar():
    """The value of hbar"""
    sf.hbar = float(os.environ.get("HBAR", HBAR))
    return sf.hbar


@pytest.fixture(
    params=[
        pytest.param(True, marks=pytest.mark.pure),
        pytest.param(False, marks=pytest.mark.mixed),
    ]
)
def pure(request):
    """Whether to run the backend in pure or mixed state mode"""
    return request.param  # bool(int(os.environ.get("PURE", PURE)))


@pytest.fixture(scope="session")
def batch_size():
    """Whether to run the backend in batched mode"""
    if "BATCHSIZE" in os.environ:
        # if user-specified BATCHSIZE provided, then batching is assumed (even if BATCHED=0)
        return int(os.environ["BATCHSIZE"])

    # check if batching is turned on
    batched = bool(int(os.environ.get("BATCHED", BATCHED)))

    if batched:
        # use the default batch size
        return BATCHSIZE

    return None  # no batching


@pytest.fixture
def backend(monkeypatch):
    """Create a mocked out backend fixture for front-end only tests"""
    dummy_backend = BaseBackend()
    with monkeypatch.context() as m:
        # mock out the base backend
        m.setattr(dummy_backend, "add_mode", lambda n: None)
        m.setattr(dummy_backend, "del_mode", lambda n: None)
        m.setattr(dummy_backend, "displacement", lambda r, phi, modes: None)
        m.setattr(dummy_backend, "prepare_coherent_state", lambda r, phi, modes: None)
        m.setattr(dummy_backend, "squeeze", lambda r, phi, modes: None)
        m.setattr(dummy_backend, "rotation", lambda r, modes: None)
        m.setattr(dummy_backend, "beamsplitter", lambda t, r, m1, m2: None)
        m.setattr(dummy_backend, "measure_homodyne", lambda phi, modes, select, shots: np.array([[5]]))
        m.setattr(dummy_backend, "state", lambda modes, shots: None)
        m.setattr(dummy_backend, "reset", lambda: None)
        dummy_backend.two_mode_squeeze = lambda r, phi, mode1, mode2: None
        dummy_backend.get_cutoff_dim = lambda: 6
        yield dummy_backend


@pytest.fixture(scope="session")
def print_fixtures(cutoff, hbar, batch_size):
    """Print the test configuration at the beginning of the session"""
    print(
        "FIXTURES: cutoff = {}, pure = {}, hbar = {}, batch_size = {}".format(
            cutoff, hbar, pure, batch_size
        )
    )


@pytest.fixture(params=backend_params)
def setup_backend(
    request, cutoff, pure, batch_size
):  # pylint: disable=redefined-outer-name
    """Parameterized fixture, used to automatically create a backend of certain number of modes.

    This fixture should only be used in backend tests, as it bypasses Engine and
    initializes the backend directly.

    Every test that uses this fixture, or a fixture that depends on it
    will be run three times, one for each backend.

    To explicitly mark a test to only work on certain backends, you may
    use the ``@pytest.mark.backends()`` fixture. For example, for a test that
    only works on the TF and Fock backends, ``@pytest.mark.backends('tf', 'fock').
    """

    def _setup_backend(num_subsystems):
        """Factory function"""
        backend = request.param()
        backend.begin_circuit(
            num_subsystems=num_subsystems,
            cutoff_dim=cutoff,
            pure=pure,
            batch_size=batch_size,
        )
        return backend

    return _setup_backend


@pytest.fixture(params=eng_backend_params)
def setup_backend_pars(
    request, cutoff, pure, batch_size
):  # pylint: disable=redefined-outer-name
    """Parameterized fixture, a container for the backend parameters.

    Every test that uses this fixture, or a fixture that depends on it
    will be run three times, one for each backend.

    To explicitly mark a test to only work on certain backends, you may
    use the ``@pytest.mark.backends()`` fixture. For example, for a test that
    only works on the TF and Fock backends, ``@pytest.mark.backends('tf', 'fock').
    """
    return request.param, {"cutoff_dim": cutoff, "pure": pure, "batch_size": batch_size}


@pytest.fixture
def setup_eng(setup_backend_pars):  # pylint: disable=redefined-outer-name
    """Parameterized fixture, used to automatically create an Engine and a Program with a certain number of modes."""

    def _setup_eng(num_subsystems, **kwargs):
        """Factory function"""
        prog = Program(num_subsystems)
        backend, backend_options = setup_backend_pars
        backend_options.update(kwargs)  # override defaults with kwargs
        eng = LocalEngine(backend=backend, backend_options=backend_options)
        return eng, prog

    return _setup_eng

def pytest_runtest_setup(item):
    """Automatically skip tests if they are marked for only certain backends"""
    if tf_available:
        allowed_backends = {"gaussian", "tf", "fock"}
    else:
        allowed_backends = {"gaussian", "fock"}

    # load the marker specifying what the backend is
    marks = {mark.name for mark in item.iter_markers() if mark.name in allowed_backends}

    # load a marker specifying whether the test only works with certain backends
    test_backends = [mark.args for mark in item.iter_markers(name="backends")]

    if not test_backends:
        # if the test hasn't specified that it runs on particular backends,
        # assume it will work with all backends
        test_backends = allowed_backends
    else:
        # otherwise, extract the set of backend strings the test works with
        test_backends = test_backends[0]

    for b in marks:
        if b not in test_backends:
            pytest.skip(
                "\nTest {} only runs with {} backend(s), "
                "but {} backend provided".format(item.nodeid, test_backends, b)
            )

    # skip broken tests
    for mark in item.iter_markers(name="broken"):
        if mark.args:
            pytest.skip("Broken test skipped: {}".format(*mark.args))
        else:
            pytest.skip("Test skipped as corresponding code base is currently broken!")

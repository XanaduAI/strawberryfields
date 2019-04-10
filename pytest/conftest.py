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

from strawberryfields import Engine
from strawberryfields.backends.base import BaseBackend
from strawberryfields.backends.fockbackend import FockBackend
from strawberryfields.backends.gaussianbackend import GaussianBackend
from strawberryfields.backends.tfbackend import TFBackend


# defaults
TOL = 1e-3
CUTOFF = 6
ALPHA = 0.1
HBAR = 2
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
    return float(os.environ.get("HBAR", HBAR))


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
        m.setattr(dummy_backend, "displacement", lambda alpha, modes: None)
        m.setattr(dummy_backend, "squeeze", lambda r, modes: None)
        m.setattr(dummy_backend, "rotation", lambda r, modes: None)
        m.setattr(dummy_backend, "beamsplitter", lambda t, r, m1, m2: None)
        m.setattr(dummy_backend, "measure_homodyne", lambda phi, modes, select: 5)
        m.setattr(dummy_backend, "state", lambda modes: None)
        m.setattr(dummy_backend, "reset", lambda: None)
        yield dummy_backend


@pytest.fixture(scope="session")
def print_fixtures(cutoff, hbar, batch_size):
    """Print the test configuration at the beginning of the session"""
    print(
        "FIXTURES: cutoff = {}, hbar = {}, batch_size = {}".format(
            cutoff, hbar, pure, batch_size
        )
    )


@pytest.fixture(
    params=[
        pytest.param(FockBackend, marks=pytest.mark.fock),
        pytest.param(GaussianBackend, marks=pytest.mark.gaussian),
        pytest.param(TFBackend, marks=pytest.mark.tf),
    ]
)
def setup_backend(
    request, cutoff, hbar, pure, batch_size
):  # pylint: disable=redefined-outer-name
    """Parameterized fixture, used to automatically create a backend of certain number of modes.

    Every test that uses this fixture, or a fixture that depends on it (such as ``setup_eng``)
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
            hbar=hbar,
            pure=pure,
            batch_size=batch_size,
        )
        return backend

    return _setup_backend


@pytest.fixture
def setup_eng(setup_backend):  # pylint: disable=redefined-outer-name
    """Parameterized fixture, used to automatically create an engine with certain number of modes"""

    def _setup_eng(num_subsystems):
        """Factory function"""
        eng, q = Engine(num_subsystems)
        eng.backend = setup_backend(num_subsystems)
        return eng, q

    return _setup_eng


def pytest_runtest_setup(item):
    """Automatically skip tests if they are marked for only certain backends"""
    allowed_backends = {"gaussian", "tf", "fock"}

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

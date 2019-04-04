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
Default parameters, commandline arguments and common routines for the TensorFlow backend unit tests.
"""
import os
import pytest

from strawberryfields import Engine
from tfbackend import TFBackend


# defaults
TOL = 1e-3
CUTOFF = 6
ALPHA = 0.1
HBAR = 2
PURE = True
BATCHED = False
BATCHSIZE = 2


@pytest.fixture
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture
def cutoff():
    """Fock state cutoff"""
    return int(os.environ.get("CUTOFF", CUTOFF))


@pytest.fixture
def alpha():
    """Maximum magnitude of coherent states used in tests"""
    return float(os.environ.get("ALPHA", ALPHA))


@pytest.fixture
def hbar():
    """The value of hbar"""
    return float(os.environ.get("HBAR", HBAR))


@pytest.fixture
def pure():
    """Whether to run the backend in pure or mixed state mode"""
    return bool(int(os.environ.get("PURE", PURE)))


# if BATCHSIZE is specified, then batching is assumed (even if BATCHED=0)
@pytest.fixture
def batched_and_size():
    """Whether to run the backend in batched mode"""
    if "BATCHSIZE" in os.environ:
        batched = True
        batch_size =  int(os.environ.get("BATCHSIZE", BATCHSIZE)) # user-specified batch_size
    else:
        batched = bool(int(os.environ.get("BATCHED", BATCHED)))
        if batched:
            batch_size = BATCHSIZE # use default
        else:
            batch_size = None # no batching
    return batched, batch_size


@pytest.fixture
def setup_backend(cutoff, hbar, pure, batched_and_size): #pylint: disable=redefined-outer-name
    """Parameterized fixture, used to automatically create a backend of certain number of modes"""
    batched, batch_size = batched_and_size
    def _setup_backend(num_subsystems):
        """Factory function"""
        backend = TFBackend()
        backend.begin_circuit(num_subsystems=num_subsystems, cutoff_dim=cutoff, hbar=hbar, pure=pure, batch_size=batch_size)
        return backend
    return _setup_backend


@pytest.fixture
def setup_eng(setup_backend): #pylint: disable=redefined-outer-name
    """Parameterized fixture, used to automatically create an engine with certain number of modes"""
    def _setup_eng(num_subsystems):
        """Factory function"""
        eng, q = Engine(num_subsystems)
        eng.backend = setup_backend(num_subsystems)
        return eng, q
    return _setup_eng

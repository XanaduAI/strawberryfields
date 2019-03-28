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
Default parameters, commandline arguments and common routines for the Fock backend unit tests.
"""
import os
import pytest

from fockbackend import FockBackend


# defaults
TOL = 1e-3
CUTOFF = 6
ALPHA = 0.1
HBAR = 2
PURE = True
BATCHED = False


@pytest.fixture
def tol():
    """Numerical tolerance for equality tests."""
    return os.environ.get("TOL", TOL)


@pytest.fixture
def cutoff():
    """Fock state cutoff"""
    return os.environ.get("CUTOFF", CUTOFF)


@pytest.fixture
def alpha():
    """Maximum magnitude of coherent states used in tests"""
    return os.environ.get("ALPHA", ALPHA)


@pytest.fixture
def hbar():
    """The value of hbar"""
    return os.environ.get("HBAR", HBAR)


@pytest.fixture
def pure():
    """Whether to run the backend in pure or mixed state mode"""
    return os.environ.get("PURE", PURE)


@pytest.fixture
def batched():
    """Whether to run the backend in batched mode"""
    return os.environ.get("BATCHED", BATCHED)


@pytest.fixture
def begin_circuit(cutoff, hbar, pure): #pylint: disable=redefined-outer-name
    """Parameterized fixture, used to automatically create a backend of certain number of modes"""
    def create_backend(num_subsystems):
        """Factory function"""
        backend = FockBackend()
        backend.begin_circuit(num_subsystems=num_subsystems, cutoff_dim=cutoff, hbar=hbar, pure=pure)
        return backend
    return create_backend

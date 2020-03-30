# Copyright 2020 Xanadu Quantum Technologies Inc.

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
Test fixtures and shared functions for strawberryfields.api tests
"""
import pytest

from strawberryfields import Program, ops
from strawberryfields.api import Connection

# pylint: disable=expression-not-assigned


@pytest.fixture
def prog():
    """Program fixture."""
    program = Program(8)
    with program.context as q:
        ops.Rgate(0.5) | q[0]
        ops.Rgate(0.5) | q[4]
        ops.MeasureFock() | q
    return program


@pytest.fixture
def connection():
    """A mock connection object."""
    return Connection(token="token", host="host", port=123, use_ssl=True)


def mock_return(return_value):
    """A helper function for defining a mock function that returns the given value for
    any arguments.
    """
    return lambda *args, **kwargs: return_value

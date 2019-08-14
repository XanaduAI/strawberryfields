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
r"""Unit tests for the parameters.py module."""
import pytest

#pytestmark = pytest.mark.frontend
pytestmark = pytest.mark.skip('Unused for now, add some Sympy tests later.')

import numpy as np

import strawberryfields as sf
from strawberryfields import ops

# make test deterministic
np.random.seed(32)


TEST_VALUES = [3, 0.14, 4.2 + 0.5j, np.random.random(3)]
TEST_PARAMETERS = [i for i in TEST_VALUES]



@pytest.mark.parametrize("p", TEST_PARAMETERS)
@pytest.mark.parametrize("q", TEST_PARAMETERS)
def test_parameter_arithmetic(p, q):
    """Test parameter addition works as expected"""
    # TODO: make sure values are actually adding/etc together!
    assert isinstance(p + q, Parameter)
    assert isinstance(p - q, Parameter)
    assert isinstance(p * q, Parameter)
    assert isinstance(p / q, Parameter)
    assert isinstance(p ** q, Parameter)


@pytest.mark.parametrize("p", TEST_VALUES)
@pytest.mark.parametrize("q", TEST_PARAMETERS)
def test_parameter_left_literal_arithmetic(p, q):
    """Test parameter addition works as expected"""
    # TODO: make sure values are actually adding/etc together!
    assert isinstance(p + q, Parameter)
    assert isinstance(p - q, Parameter)
    assert isinstance(p * q, Parameter)
    assert isinstance(p / q, Parameter)
    assert isinstance(p ** q, Parameter)


@pytest.mark.parametrize("p", TEST_PARAMETERS)
@pytest.mark.parametrize("q", TEST_VALUES)
def test_parameter_right_literal_arithmetic(p, q):
    """Test parameter addition works as expected"""
    # TODO: make sure values are actually adding/etc together!
    assert isinstance(p + q, Parameter)
    assert isinstance(p - q, Parameter)
    assert isinstance(p * q, Parameter)
    assert isinstance(p / q, Parameter)
    assert isinstance(p ** q, Parameter)


@pytest.mark.parametrize("p", TEST_PARAMETERS)
def test_parameter_unary_negation(p):
    """Test unary negation addition works as expected"""
    assert isinstance(-p, Parameter)


# TODO: This barely scratches the surface of the parameters.py module,
# a lot more unit tests are needed.

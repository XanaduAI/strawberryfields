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

pytestmark = pytest.mark.frontend

import numpy as np

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.parameters import Parameter

# make test deterministic
np.random.random(32)


TEST_VALUES = [3, 0.14, 4.2 + 0.5j, np.random.random(3)]


try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError) as e:
    tf_available = False
else:
    tf_available = True
    TEST_VALUES.extend([tf.Variable(2), tf.Variable(0.4), tf.Variable(0.8 + 1.1j)])


TEST_PARAMETERS = [Parameter(i) for i in TEST_VALUES]


def test_parameter_wrapping_integer():
    """Tests that ensure wrapping works with integers"""
    var = 5
    res = Parameter._wrap(var)
    assert isinstance(res, Parameter)
    assert res.x == var


def test_parameter_wrapping_parameters():
    """Tests that ensure wrapping works with other parameters"""
    var = 5
    var = Parameter(var)
    res = Parameter._wrap(var)
    assert isinstance(res, Parameter)
    assert res.x == var


def test_parameter_no_shape():
    """Tests that ensure wrapping occurs as expected"""
    var = Parameter(5)
    res = var.shape
    assert res is None


def test_parameter_shape():
    """Tests that ensure wrapping occurs as expected"""
    var = np.array([[1, 2, 3], [4, 5, 6]])
    p = Parameter(var)
    res = p.shape
    assert res == (2, 3)


@pytest.mark.skipif(not tf_available, reason="Test only works with TensorFlow installed")
def test_parameter_shape_tf():
    """Tests that ensure wrapping occurs as expected"""
    var = np.array([[1, 2, 3], [4, 5, 6]])
    p = Parameter(tf.convert_to_tensor(var))
    res = p.shape
    assert res == (2, 3)


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

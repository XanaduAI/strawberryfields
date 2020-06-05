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
r"""
Tests for strawberryfields.apps.qchem.dynamics
"""
import numpy as np
import pytest

from types import FunctionType

import strawberryfields as sf

from strawberryfields.apps.qchem import dynamics

pytestmark = pytest.mark.apps


t1 = 0.0
t2 = 100.0
U1 = np.array([[0.707106781, -0.707106781], [0.707106781, 0.707106781]])
U2 = np.array(
    [
        [0.31364917, -0.11345076, 0.9051936],
        [0.05043594, -0.01799135, -0.0036580],
        [0.60597119, -0.40844145, -0.1135415],
    ]
)
w1 = np.array([3914.92, 3787.59])
w2 = np.array([3914.92, 3787.59, 1627.01])


@pytest.mark.parametrize("time, unitary, frequency", [(t1, U1, w1), (t2, U2, w2)])
def test_type(time, unitary, frequency):
    """Test if the function ``strawberryfields.apps.qchem.dynamics.dynamics_observable`` outputs
    the correct type"""
    isinstance(type(dynamics.dynamics_observable(time, unitary, frequency)), FunctionType)


@pytest.mark.parametrize("time, unitary, frequency", [(t1, U1, w1), (t2, U2, w2)])
def test_type(time, unitary, frequency):
    """Test if the function ``strawberryfields.apps.qchem.dynamics.dynamics_observable`` outputs
    the correct number of operations"""
    obs = dynamics.dynamics_observable(time, unitary, frequency)
    gbs = sf.Program(len(unitary))
    with gbs.context as q:
        obs() | q
    assert len(gbs.circuit) == (len(unitary) + 2)

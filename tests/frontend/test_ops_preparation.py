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
r"""Unit tests for Preparation classes in ops.py"""
import pytest

pytestmark = pytest.mark.frontend

import numpy as np

from strawberryfields import ops
from strawberryfields.program import Program
from strawberryfields.program_utils import MergeFailure, RegRefError
from strawberryfields import utils
from strawberryfields.parameters import Parameter


@pytest.mark.parametrize("state", ops.simple_state_preparations)  # these have __init__ methods with default arguments
class TestStatePreparationBasics:
    """Basic properties of state preparation operations."""

    def test_merge(self, state):
        """Test that merging states simply returns the second state"""
        # state to test against
        V = ops.Vacuum()
        G = state()

        # merging with another state returns the last one
        assert np.all(G.merge(V) == V)
        assert np.all(V.merge(G) == G)

    def test_exceptions(self, state):
        """Test exceptions raised if state prep used on invalid modes"""
        G = state()
        prog = Program(2)

        with prog.context:
            # all states act on a single mode
            with pytest.raises(ValueError):
                G.__or__([0, 1])

            # can't repeat the same index
            with pytest.raises(RegRefError):
                ops.All(G).__or__([0, 0])

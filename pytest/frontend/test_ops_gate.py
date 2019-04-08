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
r"""Unit tests for Gate classes in ops.py"""
import pytest

pytestmark = pytest.mark.frontend

import numpy as np

import strawberryfields as sf

from strawberryfields import ops
from strawberryfields.engine import Engine, MergeFailure, RegRefError
from strawberryfields import utils
from strawberryfields.parameters import Parameter

# make test deterministic
np.random.random(42)
a = np.random.random()
b = np.random.random()
c = np.random.random()


@pytest.mark.parametrize("gate", ops.gates)
class TestGateBasics:
    """Test the basic properties of gates"""

    @pytest.fixture(autouse=True)
    def eng(self):
        """Dummy engine context for each test"""
        eng, _ = sf.Engine(2)
        Engine._current_context = eng
        yield eng
        Engine._current_context = None

    @pytest.fixture
    def Q(self):
        """The common test gate"""
        return ops.Xgate(0.5)

    @pytest.fixture
    def G(self, gate):
        """Initialize each gate"""
        if gate in ops.zero_args_gates:
            return gate

        if gate in ops.one_args_gates:
            return gate(a)

        if gate in ops.two_args_gates:
            return gate(a, b)

    @pytest.fixture
    def H(self, gate):
        """Second gate fixture of the same class, with same phase as G"""
        if gate in ops.zero_args_gates:
            return gate

        if gate in ops.one_args_gates:
            return gate(c)

        if gate in ops.two_args_gates:
            return gate(c, b)

    def test_merge_inverse(self, G):
        """gate merged with its inverse is the identity"""
        assert G.merge(G.H) is None

    def test_merge_different_gate(self, G, Q):
        """gates cannot be merged with a different type of gate"""
        if isinstance(G, Q.__class__):
            pytest.skip("Gates are the same type.")

        with pytest.raises(MergeFailure):
            Q.merge(G)

        with pytest.raises(MergeFailure):
            G.merge(Q)

    def test_wrong_number_subsystems(self, G):
        """wrong number of subsystems"""
        if G.ns == 1:
            with pytest.raises(ValueError):
                G.__or__([0, 1])
        else:
            with pytest.raises(ValueError):
                G.__or__(0)

    def test_repeated_index(self, G):
        """multimode gates: can't repeat the same index"""
        if G.ns == 2:
            with pytest.raises(RegRefError):
                G.__or__([0, 0])

    def test_non_trivial_merging(self, G, H):
        """test the merging of two gates (with default values
        for optional parameters)"""
        if G in ops.zero_args_gates:
            pytest.skip("Gates with no arguments are not merged")

        a, b = np.random.random([2])
        merged = G.merge(H)

        # should not be the identity
        assert merged is not None

        # first parameters should be added
        assert merged.p[0] == G.p[0] + H.p[0]

        # merge G with conjugate transpose of H
        merged = G.merge(H.H)

        # should not be the identity
        assert merged is not None

        # first parameters should be subtracted
        assert merged.p[0] == G.p[0] - H.p[0]

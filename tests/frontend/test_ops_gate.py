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
from strawberryfields.program import (Program, MergeFailure, RegRefError)
from strawberryfields import utils
from strawberryfields.parameters import Parameter

# make test deterministic
np.random.random(42)
A = np.random.random()
B = np.random.random()
C = np.random.random()


@pytest.mark.parametrize("gate", ops.gates)
class TestGateBasics:
    """Test the basic properties of gates"""

    @pytest.fixture(autouse=True)
    def prog(self):
        """Dummy program context for each test"""
        prog = sf.Program(2)
        Program._current_context = prog
        yield prog
        Program._current_context = None

    @pytest.fixture
    def Q(self):
        """The common test gate"""
        return ops.Xgate(0.5)

    @pytest.fixture
    def G(self, gate):
        """Initialize each gate"""
        if gate in ops.zero_args_gates:
            return gate()

        if gate in ops.one_args_gates:
            return gate(A)

        if gate in ops.two_args_gates:
            return gate(A, B)

    @pytest.fixture
    def H(self, gate):
        """Second gate fixture of the same class, with same phase as G"""
        if gate in ops.zero_args_gates:
            return gate()

        if gate in ops.one_args_gates:
            return gate(C)

        if gate in ops.two_args_gates:
            return gate(C, B)

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
        if G.__class__ in ops.zero_args_gates:
            pytest.skip("Gates with no arguments are not merged")

        A, B = np.random.random([2])
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

    def test_gate_dagger(self, G, monkeypatch):
        """Test the dagger functionality of the gates"""
        G2 = G.H
        assert not G.dagger
        assert G2.dagger

        def dummy_apply(self, reg, backend, eval_params=False):
            """Dummy apply function, used to store the evaluated params"""
            self.res = [x.evaluate() for x in self.p]

        with monkeypatch.context() as m:
            # patch the standard Operation class apply method
            # with our dummy method, that stores the applied parameter
            # in the attribute res. This allows us to extract
            # and verify the parameter was properly negated.
            m.setattr(ops.Operation, "apply", dummy_apply)
            G2.apply(None, None)

        orig_params = [x.evaluate() for x in G2.p]
        applied_params = G2.res
        # dagger should negate the first param
        assert applied_params == [-orig_params[0]] + orig_params[1:]


def test_merge_regrefs():
    """Test merging two gates with regref parameters, that are
    the inverse of each other."""
    prog = sf.Program(2)

    with prog.context as q:
        ops.MeasureX | q[0]
        D = ops.Dgate(q[0])

    merged = D.merge(D.H)
    assert merged is None

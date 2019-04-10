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
r"""Unit tests for engine optimizer engine.py"""
import pytest

pytestmark = pytest.mark.frontend

import numpy as np

import strawberryfields as sf

from strawberryfields import ops

# make test deterministic
np.random.random(42)
A = np.random.random()


# all single-mode gates with at least one parameter
single_mode_gates = [x for x in ops.one_args_gates + ops.two_args_gates if x.ns == 1]


@pytest.fixture
def permute_gates():
    """Returns a list containing an instance of each
    single mode gate, with random parameters"""
    params = np.random.random(len(single_mode_gates))
    gates = [G(p) for G, p in zip(single_mode_gates, params)]

    # random permutation
    return np.random.permutation(gates)


@pytest.mark.parametrize("G", single_mode_gates)
def test_merge_dagger(G):
    """Optimizer merging single-mode gates with their daggered versions."""
    eng, _ = sf.Engine(1)
    G = G(A)

    with eng:
        G | 0
        G.H | 0

    eng.optimize()
    assert not eng.cmd_queue


@pytest.mark.parametrize("G", single_mode_gates)
def test_merge_negated(G):
    """Optimizer merging single-mode gates with their negated versions."""
    eng, _ = sf.Engine(1)
    G1 = G(A)
    G2 = G(-A)

    with eng:
        G1 | 0
        G2 | 0

    eng.optimize()
    assert not eng.cmd_queue


def test_merge_palindromic_cancelling(permute_gates):
    """Optimizer merging chains cancel out with palindromic cancelling"""
    eng, _ = sf.Engine(3)

    with eng:
        for G in permute_gates:
            G | 0
        for G in reversed(permute_gates):
            G.H | 0

    eng.optimize()
    assert not eng.cmd_queue


def test_merge_pairwise_cancelling(permute_gates):
    """Optimizer merging chains cancel out with pairwise cancelling"""
    eng, _ = sf.Engine(3)

    with eng:
        for G in permute_gates:
            G | 0
            G.H | 0

    eng.optimize()
    assert not eng.cmd_queue


def test_merge_interleaved_chains_two_modes(permute_gates):
    """Optimizer merging chains cancel out with palindromic
    cancelling with interleaved chains on two modes"""
    eng, _ = sf.Engine(3)

    # create a random vector of 0s and 1s, corresponding
    # to the applied mode of each gates
    modes = np.random.randint(2, size=len(permute_gates))

    with eng:
        for G, m in zip(permute_gates, modes):
            G | m
        for G, m in zip(reversed(permute_gates), reversed(modes)):
            G.H | m

    eng.optimize()
    assert not eng.cmd_queue


def test_merge_incompatible():
    """Test merging of incompatible gates does nothing"""
    eng, _ = sf.Engine(3)

    with eng:
        ops.Xgate(0.6) | 0
        ops.Zgate(0.2) | 0

    eng.optimize()
    assert len(eng.cmd_queue) == 2

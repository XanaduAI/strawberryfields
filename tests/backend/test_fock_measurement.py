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

r"""Unit tests for measurements in the Fock basis"""
import pytest

# fock measurements only supported by fock backends
pytestmark = pytest.mark.backends("fock", "tf", "gaussian")

import numpy as np


NUM_REPEATS = 50


class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    def test_vacuum_measurements(self, setup_backend, pure):
        """Tests Fock measurement on the vacuum state."""
        backend = setup_backend(3)

        for _ in range(NUM_REPEATS):
            backend.reset(pure=pure)

            meas = backend.measure_fock([0, 1, 2])[0]
            assert np.all(np.array(meas) == 0)

    def test_normalized_conditional_states(self, setup_backend, cutoff, pure, tol):
        """Tests if the conditional states resulting from Fock measurements in a subset of modes are normalized."""
        state_preps = [n for n in range(cutoff)] + [
            cutoff - n for n in range(cutoff)
        ]  # [0, 1, 2, ..., cutoff-1, cutoff, cutoff-1, ..., 2, 1]
        backend = setup_backend(3)

        for idx in range(NUM_REPEATS):
            backend.reset(pure=pure)

            # cycles through consecutive triples in `state_preps`
            backend.prepare_fock_state(state_preps[idx % cutoff], 0)
            backend.prepare_fock_state(state_preps[(idx + 1) % cutoff], 1)
            backend.prepare_fock_state(state_preps[(idx + 2) % cutoff], 2)

            for mode in range(3):
                backend.measure_fock([mode])
                state = backend.state()
                tr = state.trace()
                assert np.allclose(tr, 1, atol=tol, rtol=0)

    def test_fock_measurements(self, setup_backend, cutoff, batch_size, pure, tol):
        """Tests if Fock measurements results on a variety of multi-mode Fock states are correct."""
        state_preps = [n for n in range(cutoff)] + [
            cutoff - n for n in range(cutoff)
        ]  # [0, 1, 2, ..., cutoff-1, cutoff, cutoff-1, ..., 2, 1]

        singletons = [(0,), (1,), (2,)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        triples = [(0, 1, 2)]
        mode_choices = singletons + pairs + triples

        backend = setup_backend(3)

        for idx in range(NUM_REPEATS):
            backend.reset(pure=pure)

            n = [
                state_preps[idx % cutoff],
                state_preps[(idx + 1) % cutoff],
                state_preps[(idx + 2) % cutoff],
            ]
            n = np.array(n)
            meas_modes = np.array(
                mode_choices[idx % len(mode_choices)]
            )  # cycle through mode choices

            backend.prepare_fock_state(n[0], 0)
            backend.prepare_fock_state(n[1], 1)
            backend.prepare_fock_state(n[2], 2)

            meas_result = backend.measure_fock(meas_modes)
            ref_result = n[meas_modes]

            if batch_size is not None:
                ref_result = tuple(np.array([i] * batch_size) for i in ref_result)

            assert np.allclose(meas_result, ref_result, atol=tol, rtol=0)



class TestRepresentationIndependent:
    """Basic implementation-independent tests."""
    def test_vacuum_measurements(self, setup_backend, pure):
        """Tests Fock measurement on the vacuum state."""
        backend = setup_backend(3)

        for _ in range(NUM_REPEATS):
            backend.reset(pure=pure)

            meas = backend.measure_fock([0, 1, 2])[0]
            assert np.all(np.array(meas) == 0)
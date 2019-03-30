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
from itertools import combinations
import numpy as np

NUM_REPEATS = 50

class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    def test_vacuum_measurements(self, setup_backend):
        """Tests Fock measurement on the vacuum state."""

        for _ in range(NUM_REPEATS):
            backend = setup_backend(3)

            meas = backend.measure_fock([0,1,2])[0]
            assert np.all(np.array(meas)==0)

    def test_normalized_conditional_states(self, setup_backend, cutoff, tol):
        """Tests if the conditional states resulting from Fock measurements in a subset of modes are normalized."""

        state_preps = [n for n in range(cutoff)] + [cutoff - n for n in range(cutoff)]
        for idx in range(NUM_REPEATS):
            backend = setup_backend(3)

            backend.prepare_fock_state(state_preps[idx % cutoff], 0)
            backend.prepare_fock_state(state_preps[(idx + 1) % cutoff], 1)
            backend.prepare_fock_state(state_preps[(idx + 2) % cutoff], 2)
            state = backend.state()
            tr = state.trace()
            assert np.allclose(tr, 1, atol=tol, rtol=0)

    def _test_fock_measurements(self, cutoff, tol):
        """Tests if Fock measurements results on a variety of multi-mode Fock states are correct."""

        state_preps = [n for n in range(cutoff)] + [cutoff - n for n in range(cutoff)]
        mode_choices = [p for p in combinations(range(3), 1)] + \
                       [p for p in combinations(range(3), 2)] + \
                       [p for p in combinations(range(3), 3)]
        for idx in range(NUM_REPEATS):
            n = [state_preps[idx % cutoff],
                   state_preps[(idx + 1) % cutoff],
                   state_preps[(idx + 2) % cutoff]]
            n = np.array(n)
            meas_modes = np.array(mode_choices[idx % len(mode_choices)])
            backend = setup_backend(3)

            backend.prepare_fock_state(n[0], 0)
            backend.prepare_fock_state(n[1], 1)
            backend.prepare_fock_state(n[2], 2)
            meas_result = backend.measure_fock(meas_modes)
            ref_result = n[meas_modes]
            assert np.all(meas_result, ref_result)


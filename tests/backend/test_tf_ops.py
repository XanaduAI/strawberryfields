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
r""" Tests for the file tfbackend/ops.py"""

import pytest

import numpy as np
import tensorflow as tf

from strawberryfields.backends.tfbackend.ops import reduced_density_matrix

@pytest.mark.backends("tf", "fock")
class TestTFOps:
    """Testing for tfbackend/ops.py"""

    @pytest.mark.parametrize("num_modes", [2, 3, 4])
    def test_reduced_density_matrix_fock_states(self, num_modes, setup_backend, cutoff, tol):
        """Test the reduced_density_matrix returns the correct reduced density matrices
        when prepared with initial fock states"""

        zero_photon_state = np.zeros([cutoff])
        zero_photon_state[0] = 1.
        one_photon_state = np.zeros([cutoff])
        one_photon_state[1] = 1.

        # create a single-photon state in the second mode
        state = np.outer(zero_photon_state, one_photon_state)
        for _ in range(2, num_modes):
            state = np.multiply.outer(state, zero_photon_state)
        state = tf.constant(state)

        # get reduced density matrix of last subsystem
        mode = num_modes - 1
        reduced_dm = reduced_density_matrix(state, mode, state_is_pure=True)

        if num_modes == 2:
            expected = np.multiply.outer(one_photon_state, one_photon_state)
        else:
            expected = np.multiply.outer(zero_photon_state, zero_photon_state)
        expected = tf.constant(expected)

        assert np.allclose(reduced_dm, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("modes", [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]])
    def test_reduced_density_matrix_multiple_modes(self, setup_backend, cutoff, modes, tol):
        """Test the reduced_density_matrix returns the correct reduced density matrices
        when tracing out more than one mode."""
        state = np.zeros([cutoff, cutoff, cutoff])
        state[0, 0, 0] = 1 / np.sqrt(2)
        state[1, 1, 1] = 1 / np.sqrt(2)
        state = np.multiply.outer(state, state.conj())

        # transpose the state due to different conventions
        state = state.transpose([0, 3, 1, 4, 2, 5])

        reduced_dm = reduced_density_matrix(state, modes, state_is_pure=False)

        alpha_str = np.array(["ab", "cd", "ef"])
        betha_str = np.array(["aa", "bb", "cc"])
        betha_str[modes] = alpha_str[modes]
        expected = np.einsum(f"{''.join(betha_str)}->{''.join(alpha_str[modes])}", state)

        assert np.allclose(reduced_dm, expected, atol=tol, rtol=0)

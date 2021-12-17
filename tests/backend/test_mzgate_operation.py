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
Unit tests for Mach-Zehnder Interferometer operations
"""

import pytest

import numpy as np

PHI_IN = np.linspace(0, 2 * np.pi, 4, endpoint=False, dtype=np.float64)
PHI_EX = np.linspace(0, 2 * np.pi, 4, endpoint=False, dtype=np.float64)


@pytest.mark.backends("fock", "tf")
class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    @pytest.mark.parametrize("phi_in", PHI_IN)
    @pytest.mark.parametrize("phi_ex", PHI_EX)
    def test_normalized_mzgate_output(self, setup_backend, phi_in, phi_ex, tol):
        """Tests if a range of MZ gate outputs states are normalized."""

        cutoff = 6
        photon_11 = np.zeros((cutoff, cutoff), dtype=np.complex64)
        photon_11[1, 1] = 1.0 + 0.0j

        backend = setup_backend(2)
        backend.prepare_ket_state(photon_11, [0, 1])
        backend.mzgate(phi_in, phi_ex, 0, 1)
        state = backend.state()
        tr = state.trace()
        assert np.allclose(tr, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("phi_in", PHI_IN)
    @pytest.mark.parametrize("phi_ex", PHI_EX)
    def test_mzgate_two_photon_input(self, setup_backend, phi_in, phi_ex, batch_size, tol):
        """Tests if a range of MZ gate outputs states are equal to the gate decomposition."""

        if batch_size is not None:
            pytest.skip("Does not support batch mode")

        cutoff = 6
        photon_11 = np.zeros((cutoff, cutoff), dtype=np.complex64)
        photon_11[1, 1] = 1.0 + 0.0j

        # decomposition
        backend = setup_backend(2)
        backend.prepare_ket_state(photon_11, [0, 1])
        backend.rotation(phi_ex, 0)
        backend.beamsplitter(np.pi / 4, np.pi / 2, 0, 1)
        backend.rotation(phi_in, 0)
        backend.beamsplitter(np.pi / 4, np.pi / 2, 0, 1)
        state = backend.state()
        amp_11 = state.all_fock_probs()[1, 1]
        amp_02 = state.all_fock_probs()[0, 2]

        assert np.allclose(amp_11, np.cos(phi_in) ** 2, atol=tol, rtol=0)
        assert np.allclose(amp_02, (0.5) * np.sin(phi_in) ** 2, atol=tol, rtol=0)

    @pytest.mark.parametrize("phi_in", PHI_IN)
    @pytest.mark.parametrize("phi_ex", PHI_EX)
    def test_mzgate_operation_equals_decomposition(
        self, setup_backend, phi_in, phi_ex, batch_size, tol
    ):
        """Tests if a range of MZ gate outputs states are equal to the gate decomposition."""

        if batch_size is not None:
            pytest.skip("Does not support batch mode")

        cutoff = 6
        photon_11 = np.zeros((cutoff, cutoff), dtype=np.complex64)
        photon_11[1, 1] = 1.0 + 0.0j

        # decomposition
        backend1 = setup_backend(2)
        backend1.prepare_ket_state(photon_11, [0, 1])
        backend1.rotation(phi_ex, 0)
        backend1.beamsplitter(np.pi / 4, np.pi / 2, 0, 1)
        backend1.rotation(phi_in, 0)
        backend1.beamsplitter(np.pi / 4, np.pi / 2, 0, 1)
        state1 = backend1.state()
        amp_11_decomp = state1.all_fock_probs()[1, 1]
        amp_02_decomp = state1.all_fock_probs()[0, 2]

        # gate
        backend2 = setup_backend(2)
        backend2.prepare_ket_state(photon_11, [0, 1])
        backend2.mzgate(phi_in, phi_ex, 0, 1)
        state2 = backend2.state()
        amp_11_gate = state2.all_fock_probs()[1, 1]
        amp_02_gate = state2.all_fock_probs()[0, 2]

        assert np.allclose(amp_11_gate, amp_11_decomp, atol=tol, rtol=0)
        assert np.allclose(amp_02_gate, amp_02_decomp, atol=tol, rtol=0)

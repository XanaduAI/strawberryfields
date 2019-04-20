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

r"""Unit tests for the methods which extract Fock state probabilities from a state"""

import pytest

import numpy as np
from scipy.special import factorial


MAG_ALPHAS = np.linspace(0, .8, 4)
PHASE_ALPHAS = np.linspace(0, 2 * np.pi, 7, endpoint=False)


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    def test_prob_fock_of_coherent_state(self, setup_backend, mag_alpha, batch_size, phase_alpha, cutoff, tol):
        """Tests that probabilities of particular Fock states |n> are correct for a coherent state."""
        backend = setup_backend(1)

        alpha = mag_alpha * np.exp(1j * phase_alpha)
        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n))
        ref_probs = np.abs(ref_state) ** 2

        if batch_size is not None:
            # note that we 'vertically stack' the reference probabilities,
            # to match the output of state.fock_prob during batching
            ref_probs = np.tile(ref_probs, (batch_size, 1)).T

        backend.prepare_coherent_state(alpha, 0)
        state = backend.state()
        prob_n = np.array([state.fock_prob([n]) for n in range(cutoff)])
        assert np.allclose(prob_n, ref_probs, atol=tol, rtol=0)


class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    def test_all_fock_probs_of_coherent_state(self, setup_backend, mag_alpha, phase_alpha, batch_size, cutoff, tol):
        """Tests that the numeric probabilities in the full Fock basis are correct for a coherent state."""
        backend = setup_backend(1)

        alpha = mag_alpha * np.exp(1j * phase_alpha)
        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n))
        ref_probs = np.abs(ref_state) ** 2

        if batch_size is not None:
            ref_probs = np.tile(ref_probs, batch_size)

        backend.prepare_coherent_state(alpha, 0)

        state = backend.state()
        probs = state.all_fock_probs().flatten()
        assert np.allclose(probs, ref_probs, atol=tol, rtol=0)

    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    def test_prob_fock_of_nongaussian_state(self, setup_backend, mag_alpha, phase_alpha, batch_size, cutoff, tol):
        """Tests that Fock probabilities are correct for a specific family of nongaussian states
        (tensor product of coherent state and fock state)."""
        backend = setup_backend(2)

        alpha = mag_alpha * np.exp(1j * phase_alpha)
        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n))
        ref_probs = np.abs(ref_state) ** 2

        if batch_size is not None:
            # note that we 'vstack' the reference probabilities,
            # to match the output of state.fock_prob during batching
            ref_probs = np.tile(ref_probs, (batch_size, 1)).T

        backend.prepare_coherent_state(alpha, 0)
        backend.prepare_fock_state(cutoff // 2, 1)

        state = backend.state()
        prob_n = [state.fock_prob([n, cutoff // 2]) for n in range(cutoff)]
        assert np.allclose(prob_n, ref_probs, atol=tol, rtol=0)

    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    def test_all_fock_state_probs_of_product_coherent_state(self, setup_backend, mag_alpha, phase_alpha, cutoff, batch_size, tol):
        """Tests that the numeric probabilities in the full Fock basis are correct for a two-mode product coherent state."""
        backend = setup_backend(2)

        alpha = mag_alpha * np.exp(1j * phase_alpha)
        n = np.arange(cutoff)
        ref_state1 = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n))
        ref_state2 = np.exp(-0.5 * np.abs(-alpha) ** 2) * (-alpha) ** n / np.sqrt(factorial(n))
        ref_state = np.outer(ref_state1, ref_state2)
        ref_probs = np.abs(np.reshape(ref_state ** 2, -1))

        if batch_size is not None:
            ref_probs = np.tile(ref_probs, batch_size)

        backend.prepare_coherent_state(alpha, 0)
        backend.prepare_coherent_state(-alpha, 1)

        state = backend.state()
        probs = state.all_fock_probs().flatten()
        assert np.allclose(probs, ref_probs, atol=tol, rtol=0)

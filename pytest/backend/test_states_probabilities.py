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
r"""Unit tests for the states.py fock probabilities methods"""
import pytest

import numpy as np
from scipy.special import factorial as fac

from strawberryfields import backends
from strawberryfields import utils


MAG_ALPHAS = np.linspace(0, 0.8, 3)
PHASE_ALPHAS = np.linspace(0, 2 * np.pi, 3, endpoint=False)


@pytest.mark.parametrize("a", MAG_ALPHAS)
@pytest.mark.parametrize("phi", PHASE_ALPHAS)
class TestFockProbabilities:
    """Tests for the fock_prob state method"""

    def test_gaussian(self, a, phi, setup_backend, cutoff, tol):
        """Tests that probabilities of particular Fock states
        |n> are correct for a gaussian state."""
        backend = setup_backend(1)

        alpha = a * np.exp(1j * phi)
        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(fac(n))
        ref_probs = np.abs(ref_state) ** 2

        backend.prepare_coherent_state(alpha, 0)
        state = backend.state()

        for n in range(cutoff):
            prob_n = state.fock_prob([n])
            assert np.allclose(prob_n, ref_probs[n], atol=tol, rtol=0)

    @pytest.mark.backends("fock", "tf")
    def test_nongaussian(self, a, phi, setup_backend, cutoff, tol):
        """Tests that probabilities of particular Fock states |n> are
        correct for a nongaussian state."""
        backend = setup_backend(2)

        alpha = a * np.exp(1j * phi)
        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(fac(n))
        ref_probs = np.abs(ref_state) ** 2

        backend.prepare_coherent_state(alpha, 0)
        backend.prepare_fock_state(cutoff // 2, 1)
        state = backend.state()

        for n in range(cutoff):
            prob_n = state.fock_prob([n, cutoff // 2])
            assert np.allclose(prob_n, ref_probs[n], atol=tol, rtol=0)


@pytest.mark.parametrize("a", MAG_ALPHAS)
@pytest.mark.parametrize("phi", PHASE_ALPHAS)
@pytest.mark.backends("fock", "tf")
class TestAllFockProbs:
    """Tests for the all_fock_probs state method"""

    def test_pure(self, a, phi, setup_backend, cutoff, batch_size, tol):
        """Tests that the numeric probabilities in the full Fock basis are
        correct for a one-mode pure state."""
        backend = setup_backend(1)

        alpha = a * np.exp(1j * phi)
        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(fac(n))
        ref_probs = np.abs(ref_state) ** 2

        backend.prepare_coherent_state(alpha, 0)
        state = backend.state()

        probs = state.all_fock_probs().flatten()

        if batch_size is not None:
            ref_probs = np.tile(ref_probs, batch_size)

        assert np.allclose(probs, ref_probs, atol=tol, rtol=0)

    def test_two_mode_gaussian(self, a, phi, setup_backend, batch_size, cutoff, tol):
        """Tests that the numeric probabilities in the full Fock basis are
        correct for a two-mode gaussian state."""
        if a == 0.0:
            pytest.skip("Test only runs for states with non-zero displacement")

        backend = setup_backend(2)

        alpha = a * np.exp(1j * phi)

        n = np.arange(cutoff)
        ref_state1 = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(fac(n))
        ref_state2 = (
            np.exp(-0.5 * np.abs(-alpha) ** 2) * (-alpha) ** n / np.sqrt(fac(n))
        )

        ref_state = np.outer(ref_state1, ref_state2)
        ref_probs = np.abs(np.reshape(ref_state ** 2, -1))

        if batch_size is not None:
            ref_probs = np.tile(ref_probs, batch_size)

        backend.prepare_coherent_state(alpha, 0)
        backend.prepare_coherent_state(-alpha, 1)
        state = backend.state()

        for n in range(cutoff):
            for m in range(cutoff):
                probs = state.all_fock_probs().flatten()
                assert np.allclose(probs, ref_probs, atol=tol, rtol=0)

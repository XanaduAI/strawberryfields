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

r"""Unit tests for the loss channel
Convention: The loss channel N(T) has the action
N(T){|n><m|} = \sum_{l=0}^{min(m,n)} ((1-T)/T) ^ l * T^((n+m)/2) / l! * \sqrt(n!m!/((n-l)!(m-l)!))n-l><m-l|
"""


import pytest

import numpy as np
from scipy.special import factorial

LOSS_TS = np.linspace(0., 1., 3, endpoint=True)
MAG_ALPHAS = np.linspace(0, .75, 3)
PHASE_ALPHAS = np.linspace(0, 2 * np.pi, 3, endpoint=False)
MAX_FOCK = 5

###################################################################


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    @pytest.mark.parametrize("T", LOSS_TS)
    def test_loss_channel_on_vacuum(self, setup_backend, T, tol):
        """Tests loss channels on vacuum (result should be vacuum)."""

        backend = setup_backend(1)

        backend.loss(T, 0)
        assert np.all(backend.is_vacuum(tol))

    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    def test_full_loss_channel_on_coherent_states(self, setup_backend, mag_alpha, phase_alpha, tol):
        """Tests the full-loss channel on various states (result should be vacuum)."""

        T = 0.0
        alpha = mag_alpha * np.exp(1j * phase_alpha)

        backend = setup_backend(1)
        backend.displacement(alpha, 0)
        backend.loss(T, 0)
        assert np.all(backend.is_vacuum(tol))

class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    @pytest.mark.parametrize("T", LOSS_TS)
    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    def test_normalized_after_loss_channel_on_coherent_state(self, setup_backend, T, mag_alpha, phase_alpha, tol):
        """Tests if a range of loss states are normalized."""
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        backend = setup_backend(1)

        backend.prepare_coherent_state(alpha, 0)
        backend.loss(T, 0)
        state = backend.state()
        tr = state.trace()
        assert np.allclose(tr, 1., atol=tol, rtol=0.)

    @pytest.mark.parametrize("T", LOSS_TS)
    @pytest.mark.parametrize("n", range(MAX_FOCK))
    def test_normalized_after_loss_channel_on_fock_state(self, setup_backend, T, n, tol):
        """Tests if a range of loss states are normalized."""

        backend = setup_backend(1)

        backend.prepare_fock_state(n, 0)
        backend.loss(T, 0)
        state = backend.state()
        tr = state.trace()
        assert np.allclose(tr, 1., atol=tol, rtol=0.)

    @pytest.mark.parametrize("n", range(MAX_FOCK))
    def test_full_loss_channel_on_fock_states(self, setup_backend, n, tol):
        """Tests the full-loss channel on various states (result should be vacuum)."""

        T = 0.0
        backend = setup_backend(1)

        backend.prepare_fock_state(n, 0)
        backend.loss(T, 0)
        assert np.all(backend.is_vacuum(tol))

    @pytest.mark.parametrize("T", LOSS_TS)
    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    def test_loss_channel_on_coherent_states(self, setup_backend, T, mag_alpha, phase_alpha, cutoff, tol):
        """Tests various loss channels on coherent states (result should be coherent state with amplitude weighted by sqrt(T)."""

        alpha = mag_alpha * np.exp(1j * phase_alpha)

        rootT_alpha = np.sqrt(T) * alpha

        backend = setup_backend(1)

        backend.prepare_coherent_state(alpha, 0)
        backend.loss(T, 0)
        s = backend.state()
        if s.is_pure:
            numer_state = s.ket()
        else:
            numer_state = s.dm()
        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(rootT_alpha) ** 2) * rootT_alpha ** n / np.sqrt(factorial(n))
        ref_state = np.outer(ref_state, np.conj(ref_state))
        assert np.allclose(numer_state, ref_state, atol=tol, rtol=0.)

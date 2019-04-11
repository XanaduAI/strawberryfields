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

r"""Unit tests for operations that prepare squeezed coherent states"""

import pytest

import numpy as np
from scipy.special import factorial

MAG_ALPHAS = np.linspace(0.1, 0.5, 2)
PHASE_ALPHAS = np.linspace(np.pi / 6, 2 * np.pi, 2, endpoint=False)
SQZ_R = np.linspace(0.01, 0.1, 2)
SQZ_PHI = np.linspace(np.pi / 3, 2 * np.pi, 2, endpoint=False)


def sech(x):
    """Hyperbolic secant"""
    return 1 / np.cosh(x)


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    @pytest.mark.parametrize("phi", SQZ_PHI)
    def test_no_squeezing_no_displacement(self, setup_backend, phi, tol):
        """Tests squeezing operation in the limiting case where the result should be a vacuum state."""
        alpha = 0
        r = 0

        backend = setup_backend(1)
        backend.prepare_displaced_squeezed_state(alpha, r, phi, 0)
        assert np.all(backend.is_vacuum(tol))

    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    def test_displaced_squeezed_with_no_squeezing(
        self, setup_backend, mag_alpha, phase_alpha, tol
    ):
        """Tests if a squeezed coherent state with no squeezing is equal to a coherent state."""
        r = phi = 0
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        backend = setup_backend(1)

        backend.prepare_displaced_squeezed_state(alpha, r, phi, 0)
        state = backend.state()
        fidel = state.fidelity_coherent([alpha])
        assert np.allclose(fidel, 1, atol=tol, rtol=0)


@pytest.mark.backends("fock", "tf")
class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    @pytest.mark.parametrize("r", SQZ_R)
    @pytest.mark.parametrize("phi", SQZ_PHI)
    def test_normalized_displaced_squeezed_state(
        self, setup_backend, mag_alpha, phase_alpha, r, phi, tol
    ):
        """Tests if a range of squeezed vacuum states are normalized."""
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        backend = setup_backend(1)

        backend.prepare_displaced_squeezed_state(alpha, r, phi, 0)
        state = backend.state()
        tr = state.trace()
        assert np.allclose(tr, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("r", SQZ_R)
    @pytest.mark.parametrize("phi", SQZ_PHI)
    def test_displaced_squeezed_with_no_displacement(
        self, setup_backend, r, phi, cutoff, batch_size, pure, tol
    ):
        """Tests if a squeezed coherent state with no displacement is equal to a squeezed state (Eq. (5.5.6) in Loudon)."""
        alpha = 0
        backend = setup_backend(1)

        backend.prepare_displaced_squeezed_state(alpha, r, phi, 0)
        state = backend.state()

        if state.is_pure:
            num_state = state.ket()
        else:
            num_state = state.dm()

        n = np.arange(0, cutoff, 2)
        even_refs = (
            np.sqrt(sech(r))
            * np.sqrt(factorial(n))
            / factorial(n / 2)
            * (-0.5 * np.exp(1j * phi) * np.tanh(r)) ** (n / 2)
        )

        if batch_size is not None:
            if pure:
                even_entries = num_state[:, ::2]
            else:
                even_entries = num_state[:, ::2, ::2]
                even_refs = np.outer(even_refs, np.conj(even_refs))
        else:
            if pure:
                even_entries = num_state[::2]
            else:
                even_entries = num_state[::2, ::2]
                even_refs = np.outer(even_refs, np.conj(even_refs))

        assert np.allclose(even_entries, even_refs, atol=tol, rtol=0)

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
Unit tests for beamsplitter operations
Convention: The beamsplitter operation transforms
\hat{a} -> t \hat{a} + r \hat{b}
\hat{b} -> - r^* \hat{a} + t^* \hat{b}
where \hat{a}, \hat{b} are the photon creation operators of the two modes
Equivalently, we have t:=\cos(\theta) (t assumed real) and r:=\exp{i\phi}\sin(\theta)
"""

import pytest

import numpy as np
from scipy.special import factorial


T_VALUES = np.linspace(-0.2, 1.0, 3)
PHASE_R = np.linspace(0, 2 * np.pi, 3, endpoint=False)
ALPHA = 0.1
MAG_ALPHAS = np.linspace(0.0, ALPHA, 3)


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    def test_complex_t(self, setup_backend):
        """Test exception raised if t is complex"""
        t = 0.1 + 0.5j
        r = np.exp(1j * 0.2) * np.sqrt(1.0 - np.abs(t) ** 2)
        backend = setup_backend(2)

        with pytest.raises(ValueError, match="must be a float"):
            backend.beamsplitter(t, r, 0, 1)

    @pytest.mark.parametrize("t", T_VALUES)
    @pytest.mark.parametrize("r_phi", PHASE_R)
    def test_vacuum_beamsplitter(self, setup_backend, t, r_phi, tol):
        """Tests beamsplitter operation in some limiting cases where the output
           should be the vacuum in both modes."""
        r = np.exp(1j * r_phi) * np.sqrt(1.0 - np.abs(t) ** 2)
        backend = setup_backend(2)

        backend.beamsplitter(t, r, 0, 1)
        assert np.all(backend.is_vacuum(tol))

    @pytest.mark.parametrize("t", T_VALUES)
    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS[1:])
    @pytest.mark.parametrize("r_phi", PHASE_R)
    def test_coherent_vacuum_interfered(self, setup_backend, t, mag_alpha, r_phi, tol):
        r"""Tests if a range of beamsplitter output states (formed from a coherent state interfering with vacuum)
            have the correct fidelity with the expected coherent states outputs.
            |\psi_in> = |\alpha>|0> --> |t \alpha>|r \alpha> = |\psi_out>
            and for each output mode,
            |\gamma> = exp(-0.5 |\gamma|^2) \sum_n \gamma^n / \sqrt{n!} |n>"""
        phase_alpha = np.pi / 5
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        r = np.exp(1j * r_phi) * np.sqrt(1.0 - np.abs(t) ** 2)
        backend = setup_backend(2)

        backend.displacement(alpha, 0)
        backend.beamsplitter(t, r, 0, 1)
        alpha_outA = t * alpha
        alpha_outB = r * alpha
        state = backend.state()
        fidel = state.fidelity_coherent([alpha_outA, alpha_outB])
        assert np.allclose(fidel, 1, atol=tol, rtol=0)


@pytest.mark.backends("tf", "fock")
class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    @pytest.mark.parametrize("t", T_VALUES)
    @pytest.mark.parametrize("r_phi", PHASE_R)
    def test_normalized_beamsplitter_output(self, setup_backend, t, r_phi, tol):
        """Tests if a range of beamsplitter outputs states are normalized."""

        alpha = ALPHA * np.exp(1j * np.pi / 3)
        r = np.exp(1j * r_phi) * np.sqrt(1.0 - np.abs(t) ** 2)
        backend = setup_backend(2)

        backend.displacement(alpha, 1)
        backend.beamsplitter(t, r, 0, 1)
        state = backend.state()
        tr = state.trace()
        assert np.allclose(tr, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("t", T_VALUES)
    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS[1:])
    @pytest.mark.parametrize("r_phi", PHASE_R)
    def test_coherent_vacuum_interfered_fock_elements(
        self, setup_backend, mag_alpha, t, r_phi, cutoff, pure, tol
    ):
        r"""Tests if a range of beamsplitter output states (formed from a coherent state interfering with vacuum)
            have the correct Fock basis elements.
            |\psi_in> = |\alpha>|0> --> |t \alpha>|r \alpha> = |\psi_out>
            and for each output mode,
            |\gamma> = exp(-0.5 |\gamma|^2) \sum_n \gamma^n / \sqrt{n!} |n>"""

        phase_alpha = np.pi / 5
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        r = np.exp(1j * r_phi) * np.sqrt(1.0 - np.abs(t) ** 2)
        backend = setup_backend(2)

        backend.displacement(alpha, 0)
        backend.beamsplitter(t, r, 0, 1)
        state = backend.state()

        if state.is_pure:
            numer_state = state.ket()
        else:
            numer_state = state.dm()

        alpha_outA = t * alpha
        alpha_outB = r * alpha

        n = np.arange(cutoff)
        ref_stateA = (
            np.exp(-0.5 * np.abs(alpha_outA) ** 2)
            * alpha_outA ** n
            / np.sqrt(factorial(n))
        )
        ref_stateB = (
            np.exp(-0.5 * np.abs(alpha_outB) ** 2)
            * alpha_outB ** n
            / np.sqrt(factorial(n))
        )

        ref_state = np.einsum("i,j->ij", ref_stateA, ref_stateB)

        if not pure:
            ref_state = np.einsum(
                "i,j,k,l->ijkl",
                ref_stateA,
                np.conj(ref_stateA),
                ref_stateB,
                np.conj(ref_stateB),
            )

        assert np.allclose(numer_state, ref_state, atol=tol, rtol=0)

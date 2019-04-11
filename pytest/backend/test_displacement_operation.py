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
Unit tests for displacement operations
Convention: The displacement unitary is fixed to be
U(\alpha) = \exp(alpha \hat{a}^\dagger - \alpha^* \hat{a})
where \hat{a}^\dagger is the photon creation operator.
"""
# pylint: disable=too-many-arguments
import pytest

import numpy as np
from scipy.special import factorial as fac


MAG_ALPHAS = np.linspace(0, 0.8, 4)
PHASE_ALPHAS = np.linspace(0, 2 * np.pi, 7, endpoint=False)


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    def test_no_displacement(self, setup_backend, tol):
        """Tests displacement operation where the result should be a vacuum state."""
        backend = setup_backend(1)
        backend.displacement(0, 0)
        assert np.all(backend.is_vacuum(tol))

    @pytest.mark.parametrize("r", MAG_ALPHAS)
    @pytest.mark.parametrize("p", PHASE_ALPHAS)
    def test_fidelity_coherent(self, setup_backend, r, p, tol):
        """Tests if a range of alpha-displaced states have the correct fidelity
        with the corresponding coherent state.
        """
        alpha = r * np.exp(1j * p)

        backend = setup_backend(1)
        backend.displacement(alpha, 0)
        state = backend.state()

        fid = state.fidelity_coherent([alpha])
        assert np.allclose(fid, 1, atol=tol, rtol=0)


@pytest.mark.backends("fock", "tf")
class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    @pytest.mark.parametrize("r", MAG_ALPHAS)
    @pytest.mark.parametrize("p", PHASE_ALPHAS)
    def test_normalized_displaced_state(self, setup_backend, r, p, tol):
        """Tests if a range of displaced states are normalized."""
        alpha = r * np.exp(1j * p)

        backend = setup_backend(1)
        backend.displacement(alpha, 0)
        state = backend.state()

        assert np.allclose(state.trace(), 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("r", MAG_ALPHAS)
    @pytest.mark.parametrize("p", PHASE_ALPHAS)
    def test_coherent_state_fock_elements(self, setup_backend, r, p, cutoff, pure, tol):
        r"""Tests if a range of alpha-displaced states have the correct Fock basis elements:
           |\alpha> = exp(-0.5 |\alpha|^2) \sum_n \alpha^n / \sqrt{n!} |n>
        """
        alpha = r * np.exp(1j * p)

        backend = setup_backend(1)
        backend.displacement(alpha, 0)
        state = backend.state()

        if state.is_pure:
            numer_state = state.ket()
        else:
            numer_state = state.dm()

        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(fac(n))

        if not pure:
            ref_state = np.outer(ref_state, np.conj(ref_state))

        assert np.allclose(numer_state, ref_state, atol=tol, rtol=0)

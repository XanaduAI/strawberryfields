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

r"""Unit tests for phase-shifter operations
Convention: The phase-shift unitary is fixed to be
U(\theta) = \exp(i * \theta \hat{n})
where \hat{n} is the number operator. For positive \theta, this corresponds
to a clockwise rotation of the state in phase space."""

import pytest

import numpy as np

SHIFT_THETAS = np.linspace(0, 2 * np.pi, 7, endpoint=False)


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    @pytest.mark.parametrize("theta", SHIFT_THETAS)
    def test_rotated_vacuum(self, setup_backend, theta, tol):
        """Tests phase shift operation in some limiting cases where the result should be a vacuum state."""
        backend = setup_backend(1)
        backend.rotation(theta, 0)
        assert np.all(backend.is_vacuum(tol))


@pytest.mark.backends("fock", "tf")
class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    @pytest.mark.parametrize("theta", SHIFT_THETAS)
    def test_normalized_rotated_coherent_states(self, setup_backend, theta, tol):
        """Tests if a range of phase-shifted coherent states are normalized."""
        alpha = 1.0
        backend = setup_backend(1)

        backend.prepare_coherent_state(alpha, 0)
        backend.rotation(theta, 0)

        state = backend.state()
        tr = state.trace()
        assert np.allclose(tr, 1.0, atol=tol, rtol=0.0)

    @pytest.mark.parametrize("theta", SHIFT_THETAS)
    def test_rotated_fock_states(self, setup_backend, theta, pure, cutoff, tol):
        """Tests if a range of phase-shifted fock states |n> are equal to the form of
        exp(i * theta * n)|n>"""
        backend = setup_backend(1)

        for n in range(cutoff):
            backend.reset(pure=pure)

            backend.prepare_fock_state(n, 0)
            backend.rotation(theta, 0)
            s = backend.state()

            if s.is_pure:
                numer_state = s.ket()
            else:
                numer_state = s.dm()

            k = np.arange(cutoff)
            ref_state = np.where(k == n, np.exp(1j * theta * k), 0)

            if not pure:
                ref_state = np.outer(ref_state, np.conj(ref_state))

            assert np.allclose(numer_state, ref_state, atol=tol, rtol=0.0)

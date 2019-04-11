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

r"""Unit tests for operations that prepare squeezed states
Convention: The squeezing unitary is fixed to be
U(z) = \exp(0.5 (z^* \hat{a}^2 - z (\hat{a^\dagger}^2)))
where \hat{a} is the photon annihilation operator."""

import pytest

import numpy as np
from scipy.special import factorial


SQZ_R = np.linspace(0.0, 0.1, 5)
SQZ_THETA = np.linspace(0, 2 * np.pi, 3, endpoint=False)


def sech(x):
    """Hyberbolic secant"""
    return 1 / np.cosh(x)


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    @pytest.mark.parametrize("theta", SQZ_THETA)
    def test_no_squeezing(self, setup_backend, theta, tol):
        """Tests squeezing operation in some limiting cases where the result should be a vacuum state."""
        backend = setup_backend(1)
        backend.prepare_squeezed_state(0, theta, 0)
        assert np.all(backend.is_vacuum(tol))


@pytest.mark.backends("fock", "tf")
class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    @pytest.mark.parametrize("r", SQZ_R)
    @pytest.mark.parametrize("theta", SQZ_THETA)
    def test_normalized_squeezed_state(self, setup_backend, r, theta, tol):
        """Tests if a range of squeezed vacuum states are normalized."""
        backend = setup_backend(1)

        backend.prepare_squeezed_state(r, theta, 0)
        state = backend.state()
        tr = state.trace()
        assert np.allclose(tr, 1.0, atol=tol, rtol=0.0)

    @pytest.mark.parametrize("r", SQZ_R)
    @pytest.mark.parametrize("theta", SQZ_THETA)
    def test_no_odd_fock(self, setup_backend, r, theta, batch_size):
        """Tests if a range of squeezed vacuum states have
        only nonzero entries for even Fock states."""
        backend = setup_backend(1)

        backend.prepare_squeezed_state(r, theta, 0)
        s = backend.state()

        if s.is_pure:
            num_state = s.ket()
        else:
            num_state = s.dm()

        if batch_size is not None:
            odd_entries = num_state[:, 1::2]
        else:
            odd_entries = num_state[1::2]

        assert np.all(odd_entries == 0)

    @pytest.mark.parametrize("r", SQZ_R)
    @pytest.mark.parametrize("theta", SQZ_THETA)
    def test_reference_squeezed_states(
        self, setup_backend, r, theta, batch_size, pure, cutoff, tol
    ):
        """Tests if a range of squeezed vacuum states are equal to the form of Eq. (5.5.6) in Loudon."""
        backend = setup_backend(1)

        backend.prepare_squeezed_state(r, theta, 0)
        s = backend.state()

        if s.is_pure:
            num_state = s.ket()
        else:
            num_state = s.dm()

        n = np.arange(0, cutoff, 2)
        even_refs = (
            np.sqrt(sech(r))
            * np.sqrt(factorial(n))
            / factorial(n / 2)
            * (-0.5 * np.exp(1j * theta) * np.tanh(r)) ** (n / 2)
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

        assert np.allclose(even_entries, even_refs, atol=tol, rtol=0.0)

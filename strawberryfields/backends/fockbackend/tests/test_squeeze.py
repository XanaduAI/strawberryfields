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
Unit tests for squeezing operation
Convention: The squeezing unitary is fixed to be
U(z) = \exp(0.5 (z^* \hat{a}^2 - z (\hat{a^\dagger}^2)))
where \hat{a} is the photon annihilation operator.
"""
#pylint: disable=too-many-arguments
import pytest

import numpy as np
from scipy.special import factorial as fac, gammaln as lg


PHASE = np.linspace(0, 2 * np.pi, 3, endpoint=False)
MAG = np.linspace(0.0, 0.2, 5, endpoint=False)


def matrix_elem(n, r, m):
    """Matrix element corresponding to squeezed density matrix[n, m]"""
    eps = 1e-10

    if n % 2 != m % 2:
        return 0.0

    if r == 0.:
        return np.complex(n == m) # delta function

    k = np.arange(m % 2, min([m, n]) + 1, 2)
    res = np.sum(
        (-1)**((n-k)/2)
        * np.exp((lg(m+1) + lg(n+1))/2 - lg(k+1) - lg((m-k)/2+1) - lg((n-k)/2+1))
        * (np.sinh(r)/2+eps)**((n+m-2*k)/2) / (np.cosh(r)**((n+m+1)/2))
        )
    return res


def test_no_squeezing(begin_circuit, tol):
    """Tests displacement operation where the result should be a vacuum state."""
    backend = begin_circuit(1)
    backend.squeeze(0, 0)
    assert backend.is_vacuum(tol)


@pytest.mark.parametrize("r", MAG)
@pytest.mark.parametrize("p", PHASE)
def test_normalized_squeezed_state(begin_circuit, r, p, tol):
    """Tests if a range of alpha-displaced states have the correct fidelity
    with the corresponding coherent state.
    """
    z = r*np.exp(1j*p)

    backend = begin_circuit(1)
    backend.squeeze(z, 0)
    state = backend.state()

    assert np.abs(state.trace() - 1) < tol


@pytest.mark.parametrize("r", MAG)
@pytest.mark.parametrize("p", PHASE)
def test_no_odd_fock(begin_circuit, r, p, batched):
    """Tests if a range of squeezed vacuum states have
    only nonzero entries for even Fock states."""
    z = r * np.exp(1j * p)

    backend = begin_circuit(1)
    backend.squeeze(z, 0)
    state = backend.state()

    if state.is_pure:
        num_state = state.ket()
    else:
        num_state = state.dm()

    if batched:
        odd_entries = num_state[:, 1::2]
    else:
        odd_entries = num_state[1::2]

    assert np.all(odd_entries == 0)


@pytest.mark.parametrize("r", MAG)
@pytest.mark.parametrize("p", PHASE)
def test_reference_squeezed_vacuum(begin_circuit, r, p, cutoff, batched, pure, tol):
    """Tests if a range of squeezed vacuum states are equal to the form of Eq. (5.5.6) in Loudon."""
    z = r * np.exp(1j * p)

    backend = begin_circuit(1)
    backend.squeeze(z, 0)
    state = backend.state()

    if state.is_pure:
        num_state = state.ket()
    else:
        num_state = state.dm()

    k = np.arange(0, cutoff, 2)
    even_refs = np.sqrt(1/np.cosh(r))*np.sqrt(fac(k))/fac(k / 2)*(-0.5*np.exp(1j*p)*np.tanh(r))**(k/2)

    if pure:
        if batched:
            even_entries = num_state[:, ::2]
        else:
            even_entries = num_state[::2]
    else:
        even_refs = np.outer(even_refs, even_refs.conj())
        if batched:
            even_entries = num_state[:, ::2, ::2]
        else:
            even_entries = num_state[::2, ::2]

    assert np.allclose(even_entries, even_refs, atol=tol, rtol=0)


@pytest.mark.parametrize("r", MAG)
def test_reference_squeezed_fock(begin_circuit, r, cutoff, pure, tol):
    """Tests if a range of squeezed fock states are equal to the form of Eq. (20)
    in 'On the Squeezed Number States and their Phase Space Representations'
    (https://arxiv.org/abs/quant-ph/0108024).
    """
    backend = begin_circuit(1)

    for m in range(cutoff):
        backend.reset(pure=pure)
        backend.prepare_fock_state(m, 0)
        backend.squeeze(r, 0)
        state = backend.state()

        if state.is_pure:
            num_state = state.ket()
        else:
            num_state = state.dm()

        ref_state = np.array([matrix_elem(n, r, m) for n in range(cutoff)])

        if not pure:
            ref_state = np.outer(ref_state, ref_state.conj())

        print(num_state, ref_state)

        assert np.allclose(num_state, ref_state, atol=tol, rtol=0)

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

r"""Unit tests for probabilities of Fock outcomes of Gaussian states"""
import numpy as np
from scipy.special import factorial as fac

NUM_REPEATS = 50


def squeezed_matelem(r, n):
    r"""Calculate a squeezed state vector element.

    Args:
        r (complex): squeezing parameter
        n (int): Fock state

    Returns:
        complex: the element :math:`\langle n \mid S(r) \mid 0\rangle`
    """
    if n%2 == 0:
        return (1/np.cosh(r))*(np.tanh(r)**(n))*fac(n)/(2**(n/2)*fac(n/2))**2

    return 0.0


def test_coherent_state(setup_backend, cutoff, pure, batch_size, tol):
    """Tests Fock probabilities on a coherent state agree with fock_prob and the state vector"""
    alpha = 1

    backend = setup_backend(1)
    backend.prepare_coherent_state(alpha, 0)
    state = backend.state()

    if state.is_pure:
        bsd = state.ket()
    else:
        bsd = state.dm()

    if pure:
        if batch_size is not None:
            bsd = np.array([np.outer(b, b.conj()) for b in bsd])
        else:
            bsd = np.outer(bsd, bsd.conj())

    if batch_size is not None:
        bsd_diag = np.array([np.diag(b) for b in bsd])
        gbs = np.empty((batch_size, cutoff))
    else:
        bsd_diag = np.diag(bsd)
        gbs = np.empty((1, cutoff))

    for i in range(cutoff):
        gbs[:, i] = state.fock_prob([i])

    n = np.arange(cutoff)
    exact = np.exp(-np.abs(alpha)**2)*(np.abs(alpha)**2**n)/fac(n)

    if batch_size is not None:
        exact = np.tile(exact, batch_size)

    assert np.allclose(gbs.flatten(), bsd_diag.real.flatten(), atol=tol, rtol=0)
    assert np.allclose(exact.flatten(), bsd_diag.real.flatten(), atol=tol, rtol=0)


def test_squeezed_state(setup_backend, cutoff, pure, batch_size, tol):
    """Tests Fock probabilities on a squeezed state agree with  fock_prob and the state vector."""
    r = 1.0

    backend = setup_backend(1)
    backend.prepare_squeezed_state(r, 0, 0)
    state = backend.state()

    if state.is_pure:
        bsd = state.ket()
    else:
        bsd = state.dm()

    if pure:
        if batch_size is not None:
            bsd = np.array([np.outer(b, b.conj()) for b in bsd])
        else:
            bsd = np.outer(bsd, bsd.conj())

    if batch_size is not None:
        bsd_diag = np.array([np.diag(b) for b in bsd])
        gbs = np.empty((batch_size, cutoff))
    else:
        bsd_diag = np.diag(bsd)
        gbs = np.empty((1, cutoff))

    for i in range(cutoff):
        gbs[:, i] = state.fock_prob([i])

    n = np.arange(cutoff)
    exact = np.where(n%2 == 0, (1/np.cosh(r))*(np.tanh(r)**(n))*fac(n)/(2**(n/2)*fac(n/2))**2, 0)

    if batch_size is not None:
        exact = np.tile(exact, batch_size)

    assert np.allclose(gbs.flatten(), bsd_diag.real.flatten(), atol=tol, rtol=0)
    assert np.allclose(exact.flatten(), bsd_diag.real.flatten(), atol=tol, rtol=0)


def test_displaced_squeezed_state(setup_backend, cutoff, pure, batch_size, tol):
    """Tests Fock probabilities on a dispalced squeezed state agree with  fock_prob and the state vector."""
    r = 1.0
    alpha = 3+4*1j

    backend = setup_backend(1)
    backend.prepare_squeezed_state(r, 0, 0)
    backend.displacement(alpha, 0)
    state = backend.state()

    if state.is_pure:
        bsd = state.ket()
    else:
        bsd = state.dm()

    if pure:
        if batch_size is not None:
            bsd = np.array([np.outer(b, b.conj()) for b in bsd])
        else:
            bsd = np.outer(bsd, bsd.conj())

    if batch_size is not None:
        bsd_diag = np.array([np.diag(b) for b in bsd])
        gbs = np.empty((batch_size, cutoff))
    else:
        bsd_diag = np.diag(bsd)
        gbs = np.empty((1, cutoff))

    for i in range(cutoff):
        gbs[:, i] = state.fock_prob([i])

    assert np.allclose(gbs.flatten(), bsd_diag.real.flatten(), atol=tol, rtol=0)


def test_two_mode_squeezed(setup_backend, batch_size, tol):
    """Tests Fock probabilities on a two mode squeezed state agree with  fock_prob and the state vector."""
    r = np.arcsinh(np.sqrt(1)) #pylint: disable=assignment-from-no-return
    nmax = 3
    theta = np.sqrt(0.5)
    phi = np.sqrt(0.5)

    backend = setup_backend(2)

    backend.prepare_squeezed_state(r, 0, 0)
    backend.prepare_squeezed_state(-r, 0, 1)
    backend.beamsplitter(theta, phi, 0, 1)

    if batch_size is None:
        bsize = 1
    else:
        bsize = batch_size

    gbs = np.empty((bsize, nmax, nmax))
    state = backend.state()

    for i in range(nmax):
        for j in range(nmax):
            gbs[:, i, j] = state.fock_prob(np.array([i, j]))

    n = np.arange(nmax)
    exact = np.diag((1/np.cosh(r)**2) * np.tanh(r)**(2*n))
    exact = np.tile(exact.flatten(), bsize)

    assert np.allclose(gbs.flatten(), exact.flatten(), atol=tol, rtol=0)


def test_two_mode_squeezed_complex_phase(setup_backend, batch_size, tol):
    """Tests Fock probabilities on a two mode squeezed state with complex phase
    agree with fock_prob and the state vector."""
    r = np.arcsinh(np.sqrt(1)) #pylint: disable=assignment-from-no-return
    nmax = 3
    theta = np.sqrt(0.5)
    phi = np.sqrt(0.5)*1j

    backend = setup_backend(2)

    backend.prepare_squeezed_state(r, 0, 0)
    backend.prepare_squeezed_state(r, 0, 1)
    backend.beamsplitter(theta, phi, 0, 1)

    if batch_size is None:
        bsize = 1
    else:
        bsize = batch_size

    gbs = np.empty((bsize, nmax, nmax))
    state = backend.state()

    for i in range(nmax):
        for j in range(nmax):
            gbs[:, i, j] = state.fock_prob(np.array([i, j]))

    n = np.arange(nmax)
    exact = np.diag((1/np.cosh(r)**2) * np.tanh(r)**(2*n))
    exact = np.tile(exact.flatten(), bsize)

    assert np.allclose(gbs.flatten(), exact.flatten(), atol=tol, rtol=0)

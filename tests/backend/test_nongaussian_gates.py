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

r"""Unit tests for non-Gaussian gates"""

import pytest

import numpy as np

from scipy.linalg import expm

KAPPAS = np.linspace(0, 2 * np.pi, 7)
GAMMAS = np.linspace(0, 6, 7)

@pytest.mark.backends("fock", "tf")
class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    @pytest.mark.parametrize("gamma", GAMMAS)
    def test_cubic_phase(self, setup_backend, gamma, cutoff, tol):
        """Tests if the Cubic phase gate has the right effect on states in the Fock basis"""

        backend = setup_backend(1)

        backend.prepare_ket_state(np.ones([cutoff]) / np.sqrt(cutoff), 0)
        backend.cubic_phase(gamma, 0)
        s = backend.state()
        if s.is_pure:
            numer_state = s.ket()
        else:
            numer_state = s.dm()

        #Create annihilation operator matrix
        ladder_vals = np.arange(1, cutoff)
        ladder_vals = np.sqrt(ladder_vals)
        a = np.zeros([cutoff, cutoff])
        np.fill_diagonal(a[:, 1:], ladder_vals)

        #Construct (unnormalized) x matrix, ie a+a^dag, and it's third power
        x = a + a.T
        x3 = x @ x @ x

        gate = expm(1j * gamma * x3 / 6)

        #state to be transformed
        ket = np.ones(cutoff) / np.sqrt(cutoff)

        ref_state = np.matmul(gate, ket)
        assert np.allclose(numer_state, ref_state, atol=tol, rtol=0.0)

    @pytest.mark.parametrize("kappa", KAPPAS)
    def test_kerr_interaction(self, setup_backend, kappa, cutoff, tol):
        """Tests if the Kerr interaction has the right effect on states in the Fock basis"""

        backend = setup_backend(1)

        backend.prepare_ket_state(np.ones([cutoff]) / cutoff, 0)
        backend.kerr_interaction(kappa, 0)
        s = backend.state()
        if s.is_pure:
            numer_state = s.ket()
        else:
            numer_state = s.dm()
        ref_state = np.exp(1j * kappa * np.arange(cutoff) ** 2) / cutoff
        assert np.allclose(numer_state, ref_state, atol=tol, rtol=0.0)

    @pytest.mark.parametrize("kappa", KAPPAS)
    def test_cross_kerr_interaction(self, setup_backend, kappa, cutoff, tol):
        """Tests if the cross-Kerr interaction has the right effect on states in the Fock basis"""

        backend = setup_backend(2)

        ket1 = np.array([1.0 for n in range(cutoff)])
        ket1 /= np.linalg.norm(ket1)
        ket = np.outer(ket1, ket1)

        backend.prepare_ket_state(ket, modes=[0, 1])
        backend.cross_kerr_interaction(kappa, 0, 1)
        s = backend.state()

        if s.is_pure:
            numer_state = s.ket()
        else:
            numer_state = s.dm()

        n1 = np.arange(cutoff).reshape(-1, 1)
        n2 = np.arange(cutoff).reshape(1, -1)
        ref_state = np.exp(1j * kappa * n1 * n2).flatten() / cutoff
        ref_state = np.reshape(ref_state, [cutoff] * 2)
        assert np.allclose(numer_state, ref_state, atol=tol, rtol=0.0)

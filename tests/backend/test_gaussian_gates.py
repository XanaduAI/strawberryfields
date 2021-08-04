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

r"""Unit tests for Gaussian gate"""

import pytest
import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group
from strawberryfields.backends.tfbackend.ops import choi_trick, n_mode_gate, single_mode_gate, two_mode_gate
from thewalrus.symplectic import sypmat

NUMBER_MODES = np.arange(3,6)
CUTOFF_LIST = np.arange(4,6)
SYMPLECTIC_MATRIX = sympmat(2)
DISPLACEMENT_VECTOR = np.random.random(2)

class TestUnitaryFunctionRelated:
    """Basic tests over new functions related to gaussian gates"""
    @pytest.mark.parametrize("num_mode", NUMBER_MODES)
    def test_choi_trick(self, tol):
        """Test if we can get correct C, mu, Sigma from S, d"""
        W = unitary_group.rvs(num_mode)
        V = unitary_group.rvs(num_mode)
        r = np.random.random(num_mode) # r needs to be real
        alpha = np.random.random(num_mode) + 1j * np.random.random(num_mode)
        _C = np.exp(-0.5 * np.sum(np.abs(alpha) ** 2) - 0.5 * np.conj(alpha).T @ W @ np.diag(np.tanh(r)) @ W.T @ np.conj(alpha)) / np.sqrt(np.prod(np.cosh(r)))
        _mu = np.block([ np.conj(alpha).T @ W @ np.diag(np.tanh(r)) @ W.T + alpha.T, -np.conj(alpha).T @ W @ np.diag(1/np.cosh(r)) @ V])
        tanhr = np.diag(np.tanh(r))
        sechr = np.diag(1 / np.cosh(r))
        _Sigma = np.block(
            [[W @ tanhr @ W.T, -W @ sechr @ V], [-V.T @ sechr @ W.T, -V.T @ tanhr @ V]])
        OW = np.block([[W.real, -W.imag], [W.imag, W.real]])
        OV = np.block([[V.real, -V.imag], [V.imag, V.real]])
        S = OW@np.diag(np.concatenate([np.exp(-r), np.exp(r)]))@OV
        expected_C, expected_mu, expected_Sigma = choi_trick(S, alpha, num_mode)
        assert np.allclose(_C, expected_C, atol=tol, rtol=0)
        assert np.allclose(_mu, expected_mu, atol=tol, rtol=0)
        assert np.allclose(_Sigma, expected_Sigma, atol=tol, rtol=0)
    
    @pytest.mark.parametrize("cutoff", CUTOFF_LIST)
    def test_n_mode_gate(tol):
        """Test if n_mode_gate is compatable with single and two-mode gate in tfbackend"""
        #single mode gate
        _pure = True
        matrix = np.random.random((cutoff,cutoff))
        mode = 1
        in_modes =  np.random.random((cutoff,cutoff,cutoff))
        _batched = False
        assert np.allclose(single_mode_gate(matrix, mode, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, mode, in_modes = in_modes, pure=_pure, batched=_batched))
        matrix = np.random.random((batch,cutoff,cutoff))
        mode = 0
        in_modes =  np.random.random((batch,cutoff,cutoff,cutoff,cutoff))
        _batched = True
        assert np.allclose(single_mode_gate(matrix, mode, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, mode, in_modes = in_modes, pure=_pure, batched=_batched))
        
        _pure = False
        matrix = np.random.random((cutoff,cutoff))
        mode = 1
        in_modes =  np.random.random((cutoff,cutoff,cutoff,cutoff))
        _batched = False
        assert np.allclose(single_mode_gate(matrix, mode, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, mode, in_modes = in_modes, pure=_pure, batched=_batched))
        matrix = np.random.random((batch,cutoff,cutoff))
        mode = 0
        in_modes =  np.random.random((batch,cutoff,cutoff,cutoff,cutoff))
        _batched = True
        assert np.allclose(single_mode_gate(matrix, mode, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, mode, in_modes = in_modes, pure=_pure, batched=_batched))
        #two mode gate
        _pure = True
        matrix = np.random.random((cutoff,cutoff,cutoff,cutoff))
        mode1 = 1
        mode2 = 2
        in_modes =  np.random.random((cutoff,cutoff,cutoff,cutoff))
        _batched = False
        assert np.allclose(two_mode_gate(matrix, mode1, mode2, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, mode1, mode2, in_modes = in_modes, pure=_pure, batched=_batched))
        batch = 3
        matrix = np.random.random((batch,cutoff,cutoff,cutoff,cutoff))
        in_modes =  np.random.random((batch,cutoff,cutoff,cutoff,cutoff,cutoff,cutoff))
        _batched = True
        assert np.allclose(two_mode_gate(matrix, mode1, mode2, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, mode1, mode2,in_modes = in_modes, pure=_pure, batched=_batched))
        
        _pure = False
        matrix = np.random.random((cutoff,cutoff,cutoff,cutoff))
        mode1 = 0
        mode2 = 1
        in_modes =  np.random.random((cutoff,cutoff,cutoff,cutoff))
        _batched = False
        assert np.allclose(two_mode_gate(matrix, mode1, mode2, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, mode1, mode2, in_modes = in_modes, pure=_pure, batched=_batched))
        batch = 3
        matrix = np.random.random((batch,cutoff,cutoff,cutoff,cutoff))
        in_modes =  np.random.random((batch,cutoff,cutoff,cutoff,cutoff,cutoff,cutoff))
        mode1 = 0
        mode2 = 2
        _batched = True
        assert np.allclose(two_mode_gate(matrix, mode1, mode2, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, mode1, mode2,in_modes = in_modes, pure=_pure, batched=_batched))
    
#from thewalrus.quantum.fock_tensors import fock_tensor
#@pytest.mark.backends("tf")
#class TestFockRepresentation:
#    @pytest.mark.parametrize("S", SYMPLECTIC_MATRIX)
#    @pytest.mark.parametrize("d", DISPLACEMENT_VECTOR)
#    @pytest.mark.parametrize("cutoff", CUTOFF_LIST)
#    def test_gaussian_gate(self, setup_backend, S, d, cutoff, tol):
#        """Test if the gaussian gate has the right effect on states in the Fock basis"""
#        backend = setup_backend(1)
#
#        backend.prepare_ket_state(np.ones([cutoff]) / np.sqrt(cutoff), 0)
#        backend.gaussian_gate(S, d, 0, 1)
#        s = backend.state()
#        if s.is_pure:
#            numer_state = s.ket()
#        else:
#            numer_state = s.dm()
#
#        #S,d -> gaussian state
#        ref_state = fock_tensor(S, d, cutoff)
#        assert np.allclose(numer_state, ref_state, atol=tol, rtol=0.0)


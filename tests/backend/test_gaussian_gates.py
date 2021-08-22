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
from strawberryfields.backends.tfbackend.ops import choi_trick, n_mode_gate, single_mode_gate, two_mode_gate, gaussian_gate


class TestUnitaryFunctionRelated:
    """Basic tests over new functions related to gaussian gates"""
    @pytest.mark.parametrize("num_mode", [4,5])
    def test_choi_trick(self, num_mode, tol):
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

    @pytest.mark.parametrize("cutoff", [5,7])
    def test_n_mode_gate_with_single_mode_gate(self, cutoff, tol):
        """Test if n_mode_gate is compatable with single_mode_gate in case of pure=True/False and batched=True/False"""
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


    @pytest.mark.parametrize("cutoff", [5,7])
    def test_n_mode_gate_with_two_mode_gate(self, cutoff, tol):
        """Test if n_mode_gate is compatable with two_mode_gate in case of pure=True/False and batched=True/False"""
        _pure = True
        matrix = np.random.random((cutoff,cutoff,cutoff,cutoff))
        mode1 = 1
        mode2 = 2
        in_modes =  np.random.random((cutoff,cutoff,cutoff,cutoff))
        _batched = False
        assert np.allclose(two_mode_gate(matrix, mode1, mode2, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, [mode1, mode2], in_modes = in_modes, pure=_pure, batched=_batched))
        batch = 3
        matrix = np.random.random((batch,cutoff,cutoff,cutoff,cutoff))
        in_modes =  np.random.random((batch,cutoff,cutoff,cutoff,cutoff,cutoff,cutoff))
        _batched = True
        assert np.allclose(two_mode_gate(matrix, mode1, mode2, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, [mode1, mode2],in_modes = in_modes, pure=_pure, batched=_batched))

        _pure = False
        matrix = np.random.random((cutoff,cutoff,cutoff,cutoff))
        mode1 = 0
        mode2 = 1
        in_modes =  np.random.random((cutoff,cutoff,cutoff,cutoff))
        _batched = False
        assert np.allclose(two_mode_gate(matrix, mode1, mode2, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, [mode1, mode2], in_modes = in_modes, pure=_pure, batched=_batched))
        batch = 3
        matrix = np.random.random((batch,cutoff,cutoff,cutoff,cutoff))
        in_modes =  np.random.random((batch,cutoff,cutoff,cutoff,cutoff,cutoff,cutoff))
        mode1 = 0
        mode2 = 2
        _batched = True
        assert np.allclose(two_mode_gate(matrix, mode1, mode2, in_modes, pure=_pure, batched=_batched),n_mode_gate(matrix, [mode1, mode2], in_modes = in_modes, pure=_pure, batched=_batched))

from thewalrus.quantum.fock_tensors import fock_tensor
from thewalrus.symplectic import sympmat

@pytest.mark.backends("tf")
class TestFockRepresentation:
    @pytest.mark.parametrize("cutoff", [4,5])
    @pytest.mark.parametrize("num_mode", [2,3])
    def test_gaussian_gate_matrix_with_fock_tensor(self, num_mode, cutoff, tol):
        """Test if the gaussian gate matrix has the right effect in the Fock basis"""
        S = sympmat(num_mode)
        d = np.random.random(num_mode)
        Ggate_matrix = gaussian_gate_matrix(S, d, cutoff)
        ref_state = fock_tensor(S, d, cutoff)
        assert np.allclose(Ggate_matrix, ref_state, atol=tol, rtol=0.0)
        #batched=True case for gaussian_gate_matrix
        batch = 3
        S = np.stack([S,S,S])
        d = np.random.random((batch,num_mode))
        Ggate_matrix = gaussian_gate_matrix(S, d, cutoff, batched = True)
        ref_state = fock_tensor(S[0], d[0], cutoff)
        assert np.allclose(Ggate_matrix[0], ref_state, atol=tol, rtol=0.0)
        ref_state = fock_tensor(S[2], d[2], cutoff)
        assert np.allclose(Ggate_matrix[2], ref_state, atol=tol, rtol=0.0)

    def test_gaussian_gate_output(self, num_mode, cutoff, tol):
        """Test if the output state of the gaussian gate has the right effect on states in the Fock basis"""
        S = sympmat(num_mode)
        d = np.random.random(num_mode)
        X = np.zeros((cutoff, cutoff))
        X[0,0] = 1
        Ggate_output = gaussian_gate(S, d, [0,1], in_modes = X, cutoff = cutoff, pure = True)
        ref_state = np.einsum("abcd,bd->ac",fock_tensor(S, d, cutoff), X)
        assert np.allclose(Ggate_output, ref_state, atol=tol, rtol=0.0)
        
    def test_gaussian_gate_with_backend(self, setup_backend, cutoff, tol):
        """Test the gaussian from the view of backend"""
        num_mode = 2
        S = sympmat(num_mode)
        d = np.random.random(num_mode)
        backend = setup_backend(2)
        backend.prepare_vacuum_state([0, 1])
        backend.gaussian_gate(S, d, (0, 1))
        s = backend.state().ket()
        X = np.zeros((cutoff, cutoff))
        X[0,0] = 1
        ref_state = np.einsum("abcd,bd->ac",fock_tensor(S, d, cutoff), X)
        assert np.allclose(s, ref_state, atol=tol, rtol=0.0)

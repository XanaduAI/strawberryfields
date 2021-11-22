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
r"""Unit tests for the GaussianUnitary class"""

import pytest
import numpy as np

import strawberryfields as sf
import strawberryfields.ops as ops
from strawberryfields.utils import random_symplectic
from strawberryfields.compilers.passive import (
    _apply_one_mode_gate,
    _apply_two_mode_gate,
    _beam_splitter_passive,
)

from scipy.stats import unitary_group

from thewalrus.symplectic import interferometer

pytestmark = pytest.mark.frontend

np.random.seed(42)


@pytest.mark.parametrize("depth", [1, 3, 6])
@pytest.mark.parametrize("width", [5, 10, 15])
def test_passive_program(tol, depth, width):
    """Tests that a circuit and its compiled version produce the same Gaussian state"""
    circuit = sf.Program(width)

    T_circuit = np.eye(width, dtype=np.complex128)
    with circuit.context as q:
        for _ in range(depth):
            U = unitary_group.rvs(width)
            T_circuit = U @ T_circuit
            ops.Interferometer(U) | q
            for i in range(width):
                ops.LossChannel(0.5) | q[i]
            T_circuit *= np.sqrt(0.5)

    compiled_circuit = circuit.compile(compiler="passive")
    T = compiled_circuit.circuit[0].op.p[0]
    assert np.allclose(T, T_circuit, atol=tol, rtol=0)


def test_all_passive_gates(hbar, tol):
    """test that all gates run and do not cause anything to crash"""

    eng = sf.LocalEngine(backend="gaussian")
    circuit = sf.Program(4)

    with circuit.context as q:
        for i in range(4):
            ops.Sgate(1, 0.3) | q[i]
        ops.Rgate(np.pi) | q[0]
        ops.PassiveChannel(np.ones((2, 2))) | (q[1], q[2])
        ops.LossChannel(0.9) | q[1]
        ops.MZgate(0.25 * np.pi, 0) | (q[2], q[3])
        ops.PassiveChannel(np.array([[0.83]])) | q[0]
        ops.sMZgate(0.11, -2.1) | (q[0], q[3])
        ops.Interferometer(np.array([[np.exp(1j * 2)]])) | q[1]
        ops.BSgate(0.8, 0.4) | (q[1], q[3])
        ops.Interferometer(0.5 ** 0.5 * np.fft.fft(np.eye(2))) | (q[0], q[2])
        ops.PassiveChannel(0.1 * np.ones((3, 3))) | (q[3], q[1], q[0])

    cov = eng.run(circuit).state.cov()

    circuit = sf.Program(4)
    with circuit.context as q:
        ops.Rgate(np.pi) | q[0]
        ops.PassiveChannel(np.ones((2, 2))) | (q[1], q[2])
        ops.LossChannel(0.9) | q[1]
        ops.MZgate(0.25 * np.pi, 0) | (q[2], q[3])
        ops.PassiveChannel(np.array([[0.83]])) | q[0]
        ops.sMZgate(0.11, -2.1) | (q[0], q[3])
        ops.Interferometer(np.array([[np.exp(1j * 2)]])) | q[1]
        ops.BSgate(0.8, 0.4) | (q[1], q[3])
        ops.Interferometer(0.5 ** 0.5 * np.fft.fft(np.eye(2))) | (q[0], q[2])
        ops.PassiveChannel(0.1 * np.ones((3, 3))) | (q[3], q[1], q[0])

    compiled_circuit = circuit.compile(compiler="passive")
    T = compiled_circuit.circuit[0].op.p[0]

    S_sq = np.eye(8, dtype=np.complex128)
    r = 1
    phi = 0.3
    for i in range(4):
        S_sq[i, i] = np.cosh(r) - np.sinh(r) * np.cos(phi)
        S_sq[i, i + 4] = -np.sinh(r) * np.sin(phi)
        S_sq[i + 4, i] = -np.sinh(r) * np.sin(phi)
        S_sq[i + 4, i + 4] = np.cosh(r) + np.sinh(r) * np.cos(phi)

    cov_sq = (hbar / 2) * S_sq @ S_sq.T
    mu = np.zeros(8)

    P = interferometer(T)
    L = (hbar / 2) * (np.eye(P.shape[0]) - P @ P.T)
    cov2 = P @ cov_sq @ P.T + L

    assert np.allclose(cov, cov2, atol=tol, rtol=0)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_modes_subset(depth):
    """Tests that the compiler recognizes which modes are not being modified and acts accordingly"""

    width = 10
    eng = sf.LocalEngine(backend="gaussian")
    eng1 = sf.LocalEngine(backend="gaussian")
    circuit = sf.Program(width)
    indices = (1, 4, 2, 6, 7)
    active_modes = len(indices)
    with circuit.context as q:
        for _ in range(depth):
            U = unitary_group.rvs(len(indices))
            ops.Interferometer(U) | tuple(q[i] for i in indices)

    compiled_circuit = circuit.compile(compiler="passive")

    assert len(compiled_circuit.circuit[0].reg) == 5
    indices = [compiled_circuit.circuit[0].reg[i].ind for i in range(5)]
    assert indices == sorted(list(indices))


@pytest.mark.parametrize("M", range(4, 8))
def test_one_mode_gate(M, tol):
    """test _apply_one_mode_gate applies correctly"""
    T = np.ones((M, M))
    a = np.random.random()

    _apply_one_mode_gate(a, T, M - 2)
    assert np.allclose(T[M - 2], a, atol=tol, rtol=0)


@pytest.mark.parametrize("M", range(4, 8))
def test_two_mode_gate(M, tol):
    """test _apply_two_mode_gate transforms operations correctly"""
    T = np.arange(M ** 2, dtype=np.complex128).reshape((M, M))
    G = np.random.random((2, 2)) + 1j * np.random.random((2, 2))

    _apply_two_mode_gate(G, T, M - 4, M - 2)

    G_expand = np.eye(M, dtype=np.complex128)
    modes = [M - 4, M - 2]
    G_expand[np.ix_(modes, modes)] = G
    T1 = G_expand @ np.arange(M ** 2).reshape((M, M))

    assert np.allclose(T, T1, atol=tol, rtol=0)


@pytest.mark.parametrize("theta", [0, 0.4, np.pi])
@pytest.mark.parametrize("phi", [0, 0.1, np.pi])
def test_beam_splitter_passive(tol, theta, phi):
    """test _beam_splitter_passive"""

    ct = np.cos(theta)
    st = np.sin(theta)
    eip = np.exp(1j * phi)
    U = np.array(
        [
            [ct, -eip.conj() * st],
            [eip * st, ct],
        ]
    )

    U2 = _beam_splitter_passive(theta, phi)

    assert np.allclose(U, U2, atol=tol, rtol=tol)

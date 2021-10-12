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
from strawberryfields.compilers.gaussian_unitary import (
    _apply_symp_one_mode_gate,
    _apply_symp_two_mode_gate,
)
from thewalrus.symplectic import expand, interferometer

pytestmark = pytest.mark.frontend

np.random.seed(42)


def random_params(size, sq_bound, disp_bound):
    """Returns random parameters of a Gaussian circuit

    Args:
        size (int): number of modes
        sq_bound (float): maximum value of the squeezing
        disp_bound (float): maximum value of the displacement in absolute value

    Returns:
        tuple: Gaussian circuit parameters

    """
    A = np.random.rand(size, size) + 1j * np.random.rand(size, size)
    U, s, V = np.linalg.svd(A)
    s = sq_bound * s / (np.max(s))
    alphas = disp_bound * ((np.random.rand(size) - 0.5) + 1j * (np.random.rand(size) - 0.5))
    return U, s, V, alphas


@pytest.mark.parametrize("depth", [1, 3, 6])
@pytest.mark.parametrize("width", [1, 2, 5, 10, 15])
@pytest.mark.parametrize("compiler", ["gaussian_unitary", "gaussian_merge"])
def test_gaussian_program(depth, width, compiler):
    """Tests that a circuit and its compiled version produce the same Gaussian state"""
    eng = sf.LocalEngine(backend="gaussian")
    eng1 = sf.LocalEngine(backend="gaussian")
    circuit = sf.Program(width)
    with circuit.context as q:
        for _ in range(depth):
            U, s, V, alphas = random_params(width, 2.0 / depth, 1.0)
            ops.Interferometer(U) | q
            for i in range(width):
                ops.Sgate(s[i]) | q[i]
            ops.Interferometer(V) | q
            for i in range(width):
                ops.Dgate(np.abs(alphas[i]), np.angle(alphas[i])) | q[i]
    compiled_circuit = circuit.compile(compiler=compiler)
    cv = eng.run(circuit).state.cov()
    mean = eng.run(circuit).state.means()

    cv1 = eng1.run(compiled_circuit).state.cov()
    mean1 = eng1.run(compiled_circuit).state.means()
    assert np.allclose(cv, cv1)
    assert np.allclose(mean, mean1)


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("width", [5, 10])
@pytest.mark.parametrize("compiler", ["gaussian_unitary", "gaussian_merge"])
def test_symplectic_composition(depth, width, compiler):
    """Tests that symplectic operations are composed correctly"""
    eng = sf.LocalEngine(backend="gaussian")
    eng1 = sf.LocalEngine(backend="gaussian")
    circuit = sf.Program(width)
    Snet = np.identity(2 * width)
    with circuit.context as q:
        for _ in range(depth):
            S = random_symplectic(width, scale=0.2)
            Snet = S @ Snet
            ops.GaussianTransform(S) | q
    compiled_circuit = circuit.compile(compiler=compiler)
    assert np.allclose(compiled_circuit.circuit[0].op.p[0], Snet)


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("compiler", ["gaussian_unitary", "gaussian_merge"])
def test_modes_subset(depth, compiler):
    """Tests that the compiler recognizes which modes are not being modified and acts accordingly"""

    width = 10
    eng = sf.LocalEngine(backend="gaussian")
    eng1 = sf.LocalEngine(backend="gaussian")
    circuit = sf.Program(width)
    indices = (1, 4, 2, 6, 7)
    active_modes = len(indices)
    with circuit.context as q:
        for _ in range(depth):
            U, s, V, _ = random_params(active_modes, 2.0 / depth, 1.0)
            ops.Interferometer(U) | tuple(q[i] for i in indices)
            for i, index in enumerate(indices):
                ops.Sgate(s[i]) | q[index]
            ops.Interferometer(V) | tuple(q[i] for i in indices)
    compiled_circuit = circuit.compile(compiler=compiler)
    cv = eng.run(circuit).state.cov()
    mean = eng.run(circuit).state.means()

    cv1 = eng1.run(compiled_circuit).state.cov()
    mean1 = eng1.run(compiled_circuit).state.means()
    assert np.allclose(cv, cv1)
    assert np.allclose(mean, mean1)
    assert len(compiled_circuit.circuit[0].reg) == 5
    indices = [compiled_circuit.circuit[0].reg[i].ind for i in range(5)]
    assert indices == sorted(list(indices))


@pytest.mark.parametrize("compiler", ["gaussian_unitary", "gaussian_merge"])
def test_non_primitive_gates(compiler):
    """Tests that the compiler is able to compile a number of non-primitive Gaussian gates"""

    width = 6
    eng = sf.LocalEngine(backend="gaussian")
    eng1 = sf.LocalEngine(backend="gaussian")
    circuit = sf.Program(width)
    A = np.random.rand(width, width) + 1j * np.random.rand(width, width)
    A = A + A.T
    valsA = np.linalg.svd(A, compute_uv=False)
    A = A / 2 * np.max(valsA)
    B = np.random.rand(width // 2, width // 2) + 1j * np.random.rand(width // 2, width // 2)
    valsB = np.linalg.svd(B, compute_uv=False)
    B = B / 2 * valsB
    B = np.block([[0 * B, B], [B.T, 0 * B]])
    with circuit.context as q:
        ops.GraphEmbed(A) | q
        ops.BipartiteGraphEmbed(B) | q
        ops.Pgate(0.1) | q[1]
        ops.CXgate(0.2) | (q[0], q[1])
        ops.MZgate(0.4, 0.5) | (q[2], q[3])
        ops.Fourier | q[0]
        ops.Xgate(0.4) | q[1]
        ops.Zgate(0.5) | q[3]
        ops.sMZgate(0.5, 0.2) | (q[1], q[2])
    compiled_circuit = circuit.compile(compiler=compiler)
    cv = eng.run(circuit).state.cov()
    mean = eng.run(circuit).state.means()

    cv1 = eng1.run(compiled_circuit).state.cov()
    mean1 = eng1.run(compiled_circuit).state.means()
    assert np.allclose(cv, cv1)
    assert np.allclose(mean, mean1)


@pytest.mark.parametrize("depth", [1, 3, 6])
@pytest.mark.parametrize("width", [5, 10, 15])
@pytest.mark.parametrize("compiler", ["gaussian_unitary", "gaussian_merge"])
def test_displacements_only(depth, width, compiler):
    """Tests that a circuit and its compiled version produce
    the same Gaussian state when there are only displacements"""
    eng = sf.LocalEngine(backend="gaussian")
    eng1 = sf.LocalEngine(backend="gaussian")
    circuit = sf.Program(width)
    with circuit.context as q:
        for _ in range(depth):
            alphas = np.random.rand(width) + 1j * np.random.rand(width)
            for i in range(width):
                ops.Dgate(np.abs(alphas[i]), np.angle(alphas[i])) | q[i]
    compiled_circuit = circuit.compile(compiler=compiler)
    cv = eng.run(circuit).state.cov()
    mean = eng.run(circuit).state.means()

    cv1 = eng1.run(compiled_circuit).state.cov()
    mean1 = eng1.run(compiled_circuit).state.means()
    assert np.allclose(cv, cv1)
    assert np.allclose(mean, mean1)


@pytest.mark.parametrize("M", range(4, 8))
def test_one_mode_gate_expand(M, tol):
    """test _apply_symp_one_mode_gate applies correctly on a larger matrices"""
    S = np.random.random((2 * M, 2 * M))
    r = np.random.random(2 * M)
    S_G = interferometer(np.exp(1j * 0.3))

    S1, r1 = _apply_symp_one_mode_gate(S_G, S.copy(), r.copy(), 1)

    S_G_expand = expand(S_G, [1], M)
    S2 = S_G_expand @ S
    r2 = S_G_expand @ r

    assert np.allclose(S1, S2, atol=tol, rtol=0)
    assert np.allclose(r1, r2, atol=tol, rtol=0)


@pytest.mark.parametrize("M", range(4, 8))
def test_two_mode_gate_expand(M, tol):
    """test _apply_symp_two_mode_gate applies correctly"""
    S = np.random.random((2 * M, 2 * M))
    r = np.random.random(2 * M)
    S_G = interferometer(0.5 ** 0.5 * np.fft.fft(np.eye(2)))

    S1, r1 = _apply_symp_two_mode_gate(S_G, S.copy(), r.copy(), 1, 3)

    S_G_expand = expand(S_G, [1, 3], M)
    S2 = S_G_expand @ S
    r2 = S_G_expand @ r

    assert np.allclose(S1, S2, atol=tol, rtol=0)
    assert np.allclose(r1, r2, atol=tol, rtol=0)

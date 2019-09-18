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
r"""Unit tests for the Strawberry Fields decompositions within the ops module"""
import pytest

import numpy as np
from scipy.linalg import qr, block_diag

import strawberryfields as sf
from strawberryfields import decompositions as dec
from strawberryfields.utils import (
    random_interferometer,
    random_symplectic,
    random_covariance,
    squeezed_state,
)
from strawberryfields import ops
from strawberryfields.backends.shared_ops import (
    haar_measure,
    changebasis,
    rotation_matrix as rot,
)


# make the test file deterministic
np.random.seed(42)


u1 = random_interferometer(3)
u2 = random_interferometer(3)
S = random_symplectic(3)


@pytest.fixture
def V_mixed(hbar):
    return random_covariance(3, hbar=hbar, pure=False)

@pytest.fixture
def V_pure(hbar):
    return random_covariance(3, hbar=hbar, pure=True)

@pytest.fixture
def r_means(hbar):
    return np.random.randn(6) * np.sqrt(hbar / 2)


A = np.array(
    [
        [1.28931633 + 0.75228801j, 1.45557375 + 0.96825143j, 1.53672608 + 1.465635j],
        [1.45557375 + 0.96825143j, 0.37611686 + 0.84964159j, 1.25122856 + 1.28071385j],
        [1.53672608 + 1.465635j, 1.25122856 + 1.28071385j, 1.88217983 + 1.70869293j],
    ]
)

A -= np.trace(A) * np.identity(3) / 3


Abp = np.array([[0, 0, 0, 0.11959425, 0.71324479, 0.76078505],
 [0, 0, 0, 0.5612772, 0.77096718, 0.4937956],
 [0, 0, 0, 0.52273283, 0.42754102, 0.02541913],
 [0.11959425, 0.5612772, 0.52273283, 0, 0, 0, ],
 [0.71324479, 0.77096718, 0.42754102, 0, 0, 0, ],
 [0.76078505, 0.4937956, 0.02541913, 0, 0, 0, ]])



@pytest.mark.backends("gaussian")
class TestGaussianBackendDecompositions:
    """Test that the frontend decompositions work on the Gaussian backend"""

    def test_gaussian_transform(self, setup_eng, hbar, tol):
        """Test applying a Gaussian symplectic transform"""
        eng, prog = setup_eng(3)

        with prog.context as q:
            ops.GaussianTransform(S) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), S @ S.T * hbar / 2, atol=tol)

    def test_graph_embed(self, setup_eng, tol):
        """Test that embedding a traceless adjacency matrix A
        results in the property Amat/A = c J, where c is a real constant,
        and J is the all ones matrix"""
        N = 3
        eng, prog = setup_eng(3)

        with prog.context as q:
            ops.GraphEmbed(A) | q

        state = eng.run(prog).state
        Amat = eng.backend.circuit.Amat()

        # check that the matrix Amat is constructed to be of the form
        # Amat = [[B^\dagger, 0], [0, B]]
        assert np.allclose(Amat[:N, :N], Amat[N:, N:].conj().T, atol=tol)
        assert np.allclose(Amat[:N, N:], np.zeros([N, N]), atol=tol)
        assert np.allclose(Amat[N:, :N], np.zeros([N, N]), atol=tol)

        ratio = np.real_if_close(Amat[N:, N:] / A)
        ratio /= ratio[0, 0]
        assert np.allclose(ratio, np.ones([N, N]), atol=tol)

    def test_graph_embed_identity(self, setup_eng, tol):
        """Test that nothing is done if the adjacency matrix is the identity"""
        prog = sf.Program(3)
        with prog.context as q:
            ops.GraphEmbed(np.identity(3)) | q

        assert len(prog) == 1
        prog = prog.compile('gaussian')
        assert len(prog) == 0

    def test_bipartite_graph_embed(self, setup_eng, tol):
        """Test that embedding a bipartite adjacency matrix A
        results in the property Amat/A = c J, where c is a real constant,
        and J is the all ones matrix"""
        N = 6
        eng, prog = setup_eng(6)

        with prog.context as q:
            ops.BipartiteGraphEmbed(Abp) | q

        state = eng.run(prog).state
        Amat = eng.backend.circuit.Amat()

        # check that the matrix Amat is constructed to be of the form
        # Amat = [[B^\dagger, 0], [0, B]]
        assert np.allclose(Amat[:N, :N], Amat[N:, N:].conj().T, atol=tol)
        assert np.allclose(Amat[:N, N:], np.zeros([N, N]), atol=tol)
        assert np.allclose(Amat[N:, :N], np.zeros([N, N]), atol=tol)

        # final Amat
        Amat = Amat[:N, :N]
        n = N // 2

        ratio = np.real_if_close(Amat[n:, :n] / Abp[n:, :n])
        ratio /= ratio[0, 0]
        assert np.allclose(ratio, np.ones([n, n]), atol=tol)

    def test_bipartite_graph_embed_edgeset(self, setup_eng, tol):
        """Test that embedding a bipartite edge set B
        results in the property Amat/A = c J, where c is a real constant,
        and J is the all ones matrix"""
        N = 6
        eng, prog = setup_eng(6)
        B = Abp[:N//2, N//2:]
        print(B)

        with prog.context as q:
            ops.BipartiteGraphEmbed(B, edges=True) | q

        state = eng.run(prog).state
        Amat = eng.backend.circuit.Amat()

        # check that the matrix Amat is constructed to be of the form
        # Amat = [[B^\dagger, 0], [0, B]]
        assert np.allclose(Amat[:N, :N], Amat[N:, N:].conj().T, atol=tol)
        assert np.allclose(Amat[:N, N:], np.zeros([N, N]), atol=tol)
        assert np.allclose(Amat[N:, :N], np.zeros([N, N]), atol=tol)

        # final Amat
        Amat = Amat[:N, :N]
        n = N // 2

        ratio = np.real_if_close(Amat[n:, :n] / B.T)
        ratio /= ratio[0, 0]
        assert np.allclose(ratio, np.ones([n, n]), atol=tol)


    def test_passive_gaussian_transform(self, setup_eng, tol):
        """Test applying a passive Gaussian symplectic transform,
        which is simply an interferometer"""
        eng, p1 = setup_eng(3)
        O = np.vstack([np.hstack([u1.real, -u1.imag]), np.hstack([u1.imag, u1.real])])

        with p1.context as q:
            ops.All(ops.Squeezed(0.5)) | q
        init = eng.run(p1).state

        p2 = sf.Program(p1)
        with p2.context as q:
            ops.GaussianTransform(O) | q

        state = eng.run(p2).state
        assert np.allclose(state.cov(), O @ init.cov() @ O.T, atol=tol)

    def test_active_gaussian_transform_on_vacuum(self, setup_eng, hbar, tol):
        """Test applying a passive Gaussian symplectic transform,
        which is simply squeezing and ONE interferometer"""
        eng, prog = setup_eng(3)

        with prog.context as q:
            ops.GaussianTransform(S, vacuum=True) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), S @ S.T * hbar / 2, atol=tol)

    def test_interferometer(self, setup_eng, tol):
        """Test applying an interferometer"""
        eng, p1 = setup_eng(3)

        with p1.context as q:
            ops.All(ops.Squeezed(0.5)) | q
        init = eng.run(p1).state

        p2 = sf.Program(p1)
        with p2.context as q:
            ops.Interferometer(u1) | q

        state = eng.run(p2).state
        O = np.vstack([np.hstack([u1.real, -u1.imag]), np.hstack([u1.imag, u1.real])])
        assert np.allclose(state.cov(), O @ init.cov() @ O.T, atol=tol)

    def test_identity_interferometer(self, setup_eng, tol):
        """Test that applying an identity interferometer does nothing"""
        prog = sf.Program(3)
        with prog.context as q:
            ops.Interferometer(np.identity(3)) | q

        assert len(prog) == 1
        prog = prog.compile('gaussian')
        assert len(prog) == 0


@pytest.mark.backends("gaussian")
class TestGaussianBackendPrepareState:
    """Test passing several Gaussian states directly to the Gaussian backend.
    This is allowed for backends that implement the prepare_gaussian_state method."""

    def test_vacuum(self, setup_eng, hbar, tol):
        """Testing a vacuum state"""
        eng, prog = setup_eng(3)
        cov = (hbar / 2) * np.identity(6)
        with prog.context as q:
            ops.Gaussian(cov, decomp=False) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), cov, atol=tol)
        assert np.all(state.means() == np.zeros([6]))
        assert np.allclose(state.fidelity_vacuum(), 1, atol=tol)

    def test_squeezed(self, setup_eng, hbar, tol):
        """Testing a squeezed state"""
        eng, prog = setup_eng(3)
        cov = (hbar / 2) * np.diag([np.exp(-0.1)] * 3 + [np.exp(0.1)] * 3)

        with prog.context as q:
            ops.Gaussian(cov, decomp=False) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), cov, atol=tol)

    def test_displaced_squeezed(self, setup_eng, hbar, tol):
        """Testing a displaced squeezed state"""
        eng, prog = setup_eng(3)
        cov = (hbar / 2) * np.diag([np.exp(-0.1)] * 3 + [np.exp(0.1)] * 3)
        means = np.array([0, 0.1, 0.2, -0.1, 0.3, 0])

        with prog.context as q:
            ops.Gaussian(cov, r=means, decomp=False) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), cov, atol=tol)
        assert np.allclose(state.means(), means, atol=tol)

    def test_thermal(self, setup_eng, hbar, tol):
        """Testing a thermal state"""
        eng, prog = setup_eng(3)
        cov = np.diag(hbar * (np.array([0.3, 0.4, 0.2] * 2) + 0.5))

        with prog.context as q:
            ops.Gaussian(cov, decomp=False) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), cov, atol=tol)

    def test_rotated_squeezed(self, setup_eng, hbar, tol):
        """Testing a rotated squeezed state"""
        eng, prog = setup_eng(3)

        r = 0.1
        phi = 0.2312
        v1 = (hbar / 2) * np.diag([np.exp(-r), np.exp(r)])
        A = changebasis(3)
        cov = A.T @ block_diag(*[rot(phi) @ v1 @ rot(phi).T] * 3) @ A

        with prog.context as q:
            ops.Gaussian(cov, decomp=False) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), cov, atol=tol)


@pytest.mark.backends("gaussian")
class TestGaussianBackendDecomposeState:
    """Test decomposing several Gaussian states for the Gaussian backend."""

    def test_vacuum(self, setup_eng, hbar, tol):
        """Testing decomposed vacuum state"""
        eng, prog = setup_eng(3)
        cov = (hbar / 2) * np.identity(6)
        with prog.context as q:
            ops.Gaussian(cov) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), cov, atol=tol)
        assert np.all(state.means() == np.zeros([6]))
        assert np.allclose(state.fidelity_vacuum(), 1, atol=tol)
        assert len(eng.run_progs[-1]) == 3

    def test_squeezed(self, setup_eng, hbar, tol):
        """Testing decomposed squeezed state"""
        eng, prog = setup_eng(3)
        cov = (hbar / 2) * np.diag([np.exp(-0.1)] * 3 + [np.exp(0.1)] * 3)

        with prog.context as q:
            ops.Gaussian(cov) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), cov, atol=tol)
        assert len(eng.run_progs[-1]) == 3

    def test_displaced_squeezed(self, setup_eng, hbar, tol):
        """Testing decomposed displaced squeezed state"""
        eng, prog = setup_eng(3)
        cov = (hbar / 2) * np.diag([np.exp(-0.1)] * 3 + [np.exp(0.1)] * 3)
        means = np.array([0, 0.1, 0.2, -0.1, 0.3, 0])

        with prog.context as q:
            ops.Gaussian(cov, means) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), cov, atol=tol)
        assert np.allclose(state.means(), means, atol=tol)
        assert len(eng.run_progs[-1]) == 7

    def test_thermal(self, setup_eng, hbar, tol):
        """Testing decomposed thermal state"""
        eng, prog = setup_eng(3)
        cov = np.diag(hbar * (np.array([0.3, 0.4, 0.2] * 2) + 0.5))

        with prog.context as q:
            ops.Gaussian(cov) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), cov, atol=tol)
        assert len(eng.run_progs[-1]) == 3

    def test_rotated_squeezed(self, setup_eng, hbar, tol):
        """Testing decomposed rotated squeezed state"""
        eng, prog = setup_eng(3)

        r = 0.1
        phi = 0.2312
        v1 = (hbar / 2) * np.diag([np.exp(-r), np.exp(r)])
        A = changebasis(3)
        cov = A.T @ block_diag(*[rot(phi) @ v1 @ rot(phi).T] * 3) @ A

        with prog.context as q:
            ops.Gaussian(cov) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), cov, atol=tol)
        assert len(eng.run_progs[-1]) == 3

    def test_random_state_mixed(self, setup_eng, V_mixed, r_means, tol):
        """Test applying a mixed covariance state"""
        eng, prog = setup_eng(3)
        with prog.context as q:
            ops.Gaussian(V_mixed, r_means) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), V_mixed, atol=tol, rtol=0)
        assert np.allclose(state.means(), r_means, atol=tol, rtol=0)
        assert len(eng.run_progs[-1]) == 30

    def test_random_state_pure(self, setup_eng, V_pure, r_means, tol):
        """Test applying a pure covariance state"""
        eng, prog = setup_eng(3)
        with prog.context as q:
            ops.Gaussian(V_pure, r_means) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), V_pure, atol=tol, rtol=0)
        assert np.allclose(state.means(), r_means, atol=tol, rtol=0)
        assert len(eng.run_progs[-1]) == 21


@pytest.mark.backends("tf", "fock")
class TestFockBackendDecomposeState:
    """Test decomposing several Gaussian states for the Fock backends,
    by measuring the fidelities."""

    def test_vacuum(self, setup_eng, hbar, tol):
        eng, prog = setup_eng(3)

        with prog.context as q:
            ops.Gaussian(np.identity(6) * hbar / 2) | q

        state = eng.run(prog).state
        assert np.allclose(state.fidelity_vacuum(), 1, atol=tol)
        assert len(eng.run_progs[-1]) == 3

    def test_squeezed(self, setup_eng, cutoff, hbar, tol):
        eng, prog = setup_eng(3)
        r = 0.05
        phi = 0
        cov = (hbar / 2) * np.diag([np.exp(-2 * r)] * 3 + [np.exp(2 * r)] * 3)
        in_state = squeezed_state(r, phi, basis="fock", fock_dim=cutoff)

        with prog.context as q:
            ops.Gaussian(cov) | q

        state = eng.run(prog).state
        assert len(eng.run_progs[-1]) == 3

        for n in range(3):
            assert np.allclose(state.fidelity(in_state, n), 1, atol=tol)

    def test_rotated_squeezed(self, setup_eng, cutoff, hbar, tol):
        eng, prog = setup_eng(3)

        r = 0.1
        phi = 0.2312
        in_state = squeezed_state(r, phi, basis="fock", fock_dim=cutoff)

        v1 = (hbar / 2) * np.diag([np.exp(-2 * r), np.exp(2 * r)])
        A = changebasis(3)
        cov = A.T @ block_diag(*[rot(phi) @ v1 @ rot(phi).T] * 3) @ A

        with prog.context as q:
            ops.Gaussian(cov) | q

        state = eng.run(prog).state
        assert len(eng.run_progs[-1]) == 3

        for n in range(3):
            assert np.allclose(state.fidelity(in_state, n), 1, atol=tol)

class TestDecompositionsGaussianGates:
    """Test the actions of several non-primitive Gaussian gates"""

    @pytest.mark.backends("gaussian")
    def test_Pgate(self, setup_eng, pure, hbar, tol):
        """Test the action of the P gate in phase space"""
        if not pure:
            pytest.skip("Test only runs on pure states")
        N = 1
        eng, prog = setup_eng(N)
        r = 3
        x1 = 2
        p1 = 1.3
        s = 0.5
        with prog.context as q:
            ops.Sgate(r) | q
            ops.Xgate(x1) | q
            ops.Zgate(p1) | q
            ops.Pgate(s) | q
        state = eng.run(prog).state

        Pmat = np.array([[1, 0], [s, 1]])
        Vexpected = 0.5 * hbar * Pmat @ np.diag(np.exp([-2 * r, 2 * r])) @ Pmat.T
        assert np.allclose(Vexpected, state.cov(), atol=tol, rtol=0)
        rexpected = Pmat @ np.array([x1, p1])
        assert np.allclose(rexpected, state.means(), atol=tol, rtol=0)

    @pytest.mark.backends("gaussian")
    def test_CXgate(self, setup_eng, pure, hbar, tol):
        """Test the action of the CX gate in phase space"""
        if not pure:
            pytest.skip("Test only runs on pure states")
        N = 2
        eng, prog = setup_eng(N)
        r = 3
        x1 = 2
        x2 = 1
        p1 = 1.37
        p2 = 2.71
        s = 0.5
        with prog.context as q:
            ops.Sgate(r) | q[0]
            ops.Xgate(x1) | q[0]
            ops.Zgate(p1) | q[0]
            ops.Sgate(r) | q[1]
            ops.Xgate(x2) | q[1]
            ops.Zgate(p2) | q[1]
            ops.CXgate(s) | q
        state = eng.run(prog).state
        CXmat = np.array([[1, 0, 0, 0], [s, 1, 0, 0], [0, 0, 1, -s], [0, 0, 0, 1]])
        Vexpected = 0.5 * hbar * CXmat @ np.diag(np.exp([-2 * r, -2 * r, 2 * r, 2 * r])) @ CXmat.T
        # Checks the covariance matrix is transformed correctly
        assert np.allclose(state.cov(), Vexpected, atol=tol, rtol=0)
        rexpected = CXmat @ np.array([x1, x2, p1, p2])
        # Checks the means are transformed correctly
        assert np.allclose(state.means(), rexpected, atol=tol, rtol=0)

    @pytest.mark.backends("gaussian")
    def test_CZgate(self, setup_eng, pure, hbar, tol):
        """Test the action of the CZ gate in phase space"""
        if not pure:
            pytest.skip("Test only runs on pure states")
        N = 2
        eng, prog = setup_eng(N)
        r = 3
        x1 = 2
        x2 = 1
        p1 = 1.37
        p2 = 2.71
        s = 0.5
        with prog.context as q:
            ops.Sgate(r) | q[0]
            ops.Xgate(x1) | q[0]
            ops.Zgate(p1) | q[0]
            ops.Sgate(r) | q[1]
            ops.Xgate(x2) | q[1]
            ops.Zgate(p2) | q[1]
            ops.CZgate(s) | q
        state = eng.run(prog).state
        CZmat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, s, 1, 0], [s, 0, 0, 1]])
        Vexpected = 0.5 * hbar * CZmat @ np.diag(np.exp([-2 * r, -2 * r, 2 * r, 2 * r])) @ CZmat.T
        # Checks the covariance matrix is transformed correctly
        assert np.allclose(state.cov(), Vexpected, atol=tol, rtol=0)
        rexpected = CZmat @ np.array([x1, x2, p1, p2])
        # Checks the means are transformed correctly
        assert np.allclose(state.means(), rexpected, atol=tol, rtol=0)

    @pytest.mark.parametrize('s', np.linspace(0., 0.6, 5))
    def test_Pgate_decomp_equal(self, setup_eng, s, tol):
        """Tests that the Pgate gives the same transformation as its decomposition."""
        eng, prog = setup_eng(1)

        r = np.arccosh(np.sqrt(1 + (s / 2) ** 2))
        theta = np.arctan(s / 2)
        phi = -np.sign(s) * np.pi / 2 - theta

        with prog.context as q:
            ops.Pgate(s) | q
            # run decomposition with reversed arguments
            ops.Rgate(-theta) | q
            ops.Sgate(r, phi + np.pi) | q

        eng.run(prog)
        assert np.all(eng.backend.is_vacuum(tol))

    @pytest.mark.parametrize('s', np.linspace(0., 0.6, 5))
    def test_CXgate_decomp_equal(self, setup_eng, s, tol):
        """Tests that the CXgate gives the same transformation as its decomposition."""
        eng, prog = setup_eng(2)

        r = np.arcsinh(-s / 2)
        y = -1 / np.cosh(r)
        x = -np.tanh(r)
        theta = np.arctan2(y, x) / 2

        with prog.context as q:
            ops.CXgate(s) | q
            # run decomposition with reversed arguments
            ops.BSgate(-(np.pi / 2 + theta), 0) | q
            ops.Sgate(r, np.pi) | q[0]
            ops.Sgate(r, 0) | q[1]
            ops.BSgate(-theta, 0) | q

        eng.run(prog)
        assert np.all(eng.backend.is_vacuum(tol))

    @pytest.mark.parametrize('s', np.linspace(0., 0.6, 5))
    def test_CZgate_decomp_equal(self, setup_eng, s, tol):
        """Tests that the CZgate gives the same transformation as its decomposition."""
        eng, prog = setup_eng(2)

        with prog.context as q:
            ops.CZgate(s) | q
            # run decomposition with reversed arguments
            ops.Rgate(-np.pi / 2) | q[1]
            ops.CXgate(-s) | q
            ops.Rgate(np.pi / 2) | q[1]

        eng.run(prog)
        assert np.all(eng.backend.is_vacuum(tol))

    @pytest.mark.parametrize('r', np.linspace(0., 0.6, 5))
    def test_S2gate_decomp_equal(self, setup_eng, r, tol):
        """Tests that the S2gate gives the same transformation as its decomposition."""
        eng, prog = setup_eng(2)

        r = 0.25
        phi = np.pi / 5

        with prog.context as q:
            ops.S2gate(r, phi) | q
            # run decomposition with reversed arguments
            ops.BSgate(np.pi / 4, 0) | q
            ops.Sgate(r, phi + np.pi) | q[0]
            ops.Sgate(r, phi) | q[1]
            ops.BSgate(-np.pi / 4, 0) | q

        eng.run(prog)
        assert np.all(eng.backend.is_vacuum(tol))


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


A = np.array(
    [
        [1.28931633 + 0.75228801j, 1.45557375 + 0.96825143j, 1.53672608 + 1.465635j],
        [1.45557375 + 0.96825143j, 0.37611686 + 0.84964159j, 1.25122856 + 1.28071385j],
        [1.53672608 + 1.465635j, 1.25122856 + 1.28071385j, 1.88217983 + 1.70869293j],
    ]
)

A -= np.trace(A) * np.identity(3) / 3


@pytest.mark.backends("gaussian")
class TestGaussianBackendDecompositions:
    """Test that the frontend decompositions work on the Gaussian backend"""

    def test_covariance_random_state_mixed(self, setup_eng, V_mixed, tol):
        """Test applying a mixed covariance state"""
        eng, prog = setup_eng(3)
        with prog.context as q:
            ops.Gaussian(V_mixed) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), V_mixed, atol=tol, rtol=0)

    def test_covariance_random_state_pure(self, setup_eng, V_pure, tol):
        """Test applying a pure covariance state"""
        eng, prog = setup_eng(3)
        with prog.context as q:
            ops.Gaussian(V_pure) | q

        state = eng.run(prog).state
        assert np.allclose(state.cov(), V_pure, atol=tol)

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
            ops.GraphEmbed(np.identity(6)) | q

        assert len(prog) == 1
        prog = prog.compile()
        assert len(prog) == 0

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
            ops.Interferometer(np.identity(6)) | q

        assert len(prog) == 1
        prog = prog.compile()
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
        assert len(eng.run_progs[-1]) == 0

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
            ops.Gaussian(cov, r=means) | q

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
        assert len(eng.run_progs[-1]) == 0

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

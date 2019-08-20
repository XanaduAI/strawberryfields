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

pytestmark = pytest.mark.frontend

import numpy as np
from thewalrus.quantum import Amat

import strawberryfields as sf
from strawberryfields import decompositions as dec
from strawberryfields.utils import (
    random_interferometer,
    random_symplectic,
    random_covariance,
)
from strawberryfields import ops

# make the test file deterministic
np.random.seed(42)


def expand(S, modes, num_modes):
    r"""Expands a Symplectic matrix S to act on the entire subsystem.

    Args:
        S (array): a :math:`2M\times 2M` Symplectic matrix
        modes (Sequence[int]): the modes that S acts on
        num_modes (int): total number of modes in the system

    Returns:
        array: the resulting :math:`2N\times 2N` Symplectic matrix
    """
    if num_modes == 1:
        # total number of modes is 1, simply return the matrix
        return S

    N = num_modes
    w = np.asarray(modes)

    M = len(S) // 2
    S2 = np.identity(2 * N)

    S2[w.reshape(-1, 1), w.reshape(1, -1)] = S[:M, :M].copy()  # XX
    S2[(w + N).reshape(-1, 1), (w + N).reshape(1, -1)] = S[M:, M:].copy()  # PP
    S2[w.reshape(-1, 1), (w + N).reshape(1, -1)] = S[:M, M:].copy()  # XP
    S2[(w + N).reshape(-1, 1), w.reshape(1, -1)] = S[M:, :M].copy()  # PX

    return S2


def _rotation(phi, mode, num_modes):
    r"""Utility function, returns the Heisenberg transformation of a phase rotation gate.

    Args:
        phi (float): rotation angle
        mode (int): mode it is applied to
        num_modes (int): total number of modes in the system

    Returns:
        array[float]: transformation matrix
    """
    c = np.cos(phi)
    s = np.sin(phi)
    S = np.array([[c, -s], [s, c]])

    return expand(S, mode, num_modes)


def _squeezing(r, phi, mode, num_modes):
    """Squeezing in the phase space.

    Args:
        r (float): squeezing magnitude
        phi (float): rotation parameter
        mode (int): mode it is applied to
        num_modes (int): total number of modes in the system

    Returns:
        array: symplectic transformation matrix
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ch = np.cosh(r)
    sh = np.sinh(r)

    S = np.array([[ch - cp * sh, -sp * sh], [-sp * sh, ch + cp * sh]])

    return expand(S, mode, num_modes)


def _two_mode_squeezing(r, phi, modes, num_modes):
    """Two mode squeezing in the phase space.

    Args:
        r (float): squeezing magnitude
        phi (float): rotation parameter
        modes (list[int]): modes it is applied to
        num_modes (int): total number of modes in the system

    Returns:
        array: symplectic transformation matrix
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ch = np.cosh(r)
    sh = np.sinh(r)

    S = np.array(
        [
            [ch, cp * sh, 0, sp * sh],
            [cp * sh, ch, sp * sh, 0],
            [0, sp * sh, ch, -cp * sh],
            [sp * sh, 0, -cp * sh, ch],
        ]
    )

    return expand(S, modes, num_modes)


def _beamsplitter(theta, phi, modes, num_modes):
    r"""Utility function, returns the Heisenberg transformation of a beamsplitter.

    Args:
        theta (float): beamsplitter angle.
        phi (float): phase angle.
        mode (list[int]): modes it is applied to
        num_modes (int): total number of modes in the system

    Returns:
        array[float]: transformation matrix
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)

    S = np.array(
        [
            [ct, -cp * st, 0, -st * sp],
            [cp * st, ct, -st * sp, 0],
            [0, st * sp, ct, -cp * st],
            [st * sp, 0, cp * st, ct],
        ]
    )

    return expand(S, modes, num_modes)


class TestInterferometer:
    """Tests for the interferometer quantum operation"""

    def test_merge(self, tol):
        """Test that two interferometers merge: U = U1 @ U2"""
        n = 3
        U1 = random_interferometer(n)
        U2 = random_interferometer(n)

        int1 = ops.Interferometer(U1)
        int1inv = ops.Interferometer(U1.conj().T)
        int2 = ops.Interferometer(U2)

        # an interferometer merged with its inverse is identity
        assert int1.merge(int1inv) is None

        # two merged unitaries are the same as their product
        assert np.allclose(int1.merge(int2).p[0].x, U2 @ U1, atol=tol, rtol=0)

    def test_identity(self):
        """Test that nothing is done if the unitary is the identity"""
        prog = sf.Program(2)

        G = ops.Interferometer(np.identity(6))
        # identity flag is correctly set
        assert G.identity

        # as a result, no gates are returned when decomposed
        assert not G.decompose(prog.register)

    def test_decomposition(self, tol):
        """Test that an interferometer is correctly decomposed"""
        n = 3
        prog = sf.Program(n)
        U = random_interferometer(n)

        G = ops.Interferometer(U)
        cmds = G.decompose(prog.register)

        S = np.identity(2 * n)

        # calculating the resulting decomposed symplectic
        for cmd in cmds:
            # all operations should be BSgates or Rgates
            assert isinstance(cmd.op, (ops.BSgate, ops.Rgate))

            # build up the symplectic transform
            modes = [i.ind for i in cmd.reg]

            if isinstance(cmd.op, ops.Rgate):
                S = _rotation(cmd.op.p[0].x, modes, n) @ S

            if isinstance(cmd.op, ops.BSgate):
                S = _beamsplitter(cmd.op.p[0].x, cmd.op.p[1].x, modes, n) @ S

        # the resulting applied unitary
        X1 = S[:n, :n]
        P1 = S[n:, :n]
        U_applied = X1 + 1j * P1

        assert np.allclose(U, U_applied, atol=tol, rtol=0)


class TestGraphEmbed:
    """Tests for the GraphEmbed quantum operation"""

    def test_identity(self, tol):
        """Test that nothing is done if the adjacency matrix is the identity"""
        G = ops.GraphEmbed(np.identity(6))
        assert G.identity

    def test_decomposition(self, hbar, tol):
        """Test that an graph is correctly decomposed"""
        n = 3
        prog = sf.Program(n)

        A = np.random.random([n, n]) + 1j * np.random.random([n, n])
        A += A.T
        A -= np.trace(A) * np.identity(n) / n

        sq, U = dec.graph_embed(A)

        G = ops.GraphEmbed(A)
        cmds = G.decompose(prog.register)

        assert np.all(sq == G.sq)
        assert np.all(U == G.U)

        S = np.identity(2 * n)

        # calculating the resulting decomposed symplectic
        for cmd in cmds:
            # all operations should be BSgates, Rgates, or Sgates
            assert isinstance(
                cmd.op, (ops.Interferometer, ops.BSgate, ops.Rgate, ops.Sgate)
            )

            # build up the symplectic transform
            modes = [i.ind for i in cmd.reg]

            if isinstance(cmd.op, ops.Sgate):
                S = _squeezing(cmd.op.p[0].x, cmd.op.p[1].x, modes, n) @ S

            if isinstance(cmd.op, ops.Rgate):
                S = _rotation(cmd.op.p[0].x, modes, n) @ S

            if isinstance(cmd.op, ops.BSgate):
                S = _beamsplitter(cmd.op.p[0].x, cmd.op.p[1].x, modes, n) @ S

            if isinstance(cmd.op, ops.Interferometer):
                U1 = cmd.op.p[0].x
                S_U = np.vstack(
                    [np.hstack([U1.real, -U1.imag]), np.hstack([U1.imag, U1.real])]
                )
                S = S_U @ S

        # the resulting covariance state
        cov = S @ S.T

        # calculate Hamilton's A matrix: A = X.(I-Q^{-1})*
        A_res = np.real_if_close(Amat(cov))

        # The bottom right corner of A_res should be identical to A,
        # up to some constant scaling factor. Check if the ratio
        # of all elements is one
        ratio = np.real_if_close(A_res[n:, n:] / A)
        ratio /= ratio[0, 0]

        assert np.allclose(ratio, np.ones([n, n]), atol=tol, rtol=0)

    def test_decomposition_interferometer_with_zero(self, tol):
        """Test that an graph is correctly decomposed when the interferometer
        has one zero somewhere in the unitary matrix, which is the case for the
        adjacency matrix below"""
        n = 6
        prog = sf.Program(n)

        A = np.array([
        [0, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 0],
        ]
        )
        sq, U = dec.graph_embed(A)
        assert not np.allclose(U, np.identity(n))

        G = ops.GraphEmbed(A)
        cmds = G.decompose(prog.register)
        last_op = cmds[-1].op
        param_val = last_op.p[0].x

        assert isinstance(last_op, ops.Interferometer)
        assert last_op.ns == n
        assert np.allclose(param_val, U, atol=tol, rtol=0)


class TestBipartiteGraphEmbed:
    """Tests for the BipartiteGraphEmbed quantum operation"""

    def test_not_bipartite(self, tol):
        """Test exception raised if the graph is not bipartite"""
        A = np.array([
                [0, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 1],
                [0, 1, 0, 1, 1, 0],
                [0, 0, 1, 0, 1, 0],
                [1, 1, 1, 1, 0, 1],
                [1, 1, 0, 0, 1, 0]
            ]
        )

        with pytest.raises(ValueError, match="does not represent a bipartite graph"):
            ops.BipartiteGraphEmbed(A)

        A = np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0]
            ]
        )

        with pytest.raises(ValueError, match="does not represent a bipartite graph"):
            ops.BipartiteGraphEmbed(A)

        A = np.array([
                [0, 0, 1, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0]
            ]
        )

        with pytest.raises(ValueError, match="does not represent a bipartite graph"):
            ops.BipartiteGraphEmbed(A)

    def test_decomposition(self, hbar, tol):
        """Test that a graph is correctly decomposed"""
        n = 3
        prog = sf.Program(2*n)

        A = np.zeros([2*n, 2*n])
        B = np.random.random([n, n])

        A[:n, n:] = B
        A += A.T

        sq, U, V = dec.bipartite_graph_embed(B)

        G = ops.BipartiteGraphEmbed(A)
        cmds = G.decompose(prog.register)

        S = np.identity(4 * n)

        # calculating the resulting decomposed symplectic
        for cmd in cmds:
            # all operations should be BSgates, Rgates, or S2gates
            assert isinstance(
                cmd.op, (ops.Interferometer, ops.S2gate)
            )

            # build up the symplectic transform
            modes = [i.ind for i in cmd.reg]

            if isinstance(cmd.op, ops.S2gate):
                # check that the registers are i, i+n
                assert len(modes) == 2
                assert modes[1] == modes[0] + n

                r, phi = [i.x for i in cmd.op.p]
                assert -r in sq
                assert phi == 0

                S = _two_mode_squeezing(r, phi, modes, 2*n) @ S

            if isinstance(cmd.op, ops.Interferometer):
                # check that each unitary only applies to half the modes
                assert len(modes) == n
                assert (modes == [0, 1, 2]) or (modes == [3, 4, 5])

                # check matrix is unitary
                U1 = cmd.op.p[0].x
                assert np.allclose(U1 @ U1.conj().T, np.identity(n), atol=tol, rtol=0)

                if modes[0] == 0:
                    assert np.allclose(U1, U, atol=tol, rtol=0)
                else:
                    assert modes[0] == 3
                    assert np.allclose(U1, V, atol=tol, rtol=0)

                S_U = np.vstack(
                    [np.hstack([U1.real, -U1.imag]), np.hstack([U1.imag, U1.real])]
                )

                S = expand(S_U, modes, 2*n) @ S

        # the resulting covariance state
        cov = S @ S.T
        A_res = Amat(cov)[:2*n, :2*n]

        # The bottom right corner of A_res should be identical to A,
        # up to some constant scaling factor. Check if the ratio
        # of all elements is one
        ratio = np.real_if_close(A_res[n:, :n] / B.T)
        ratio /= ratio[0, 0]

        assert np.allclose(ratio, np.ones([n, n]), atol=tol, rtol=0)


class TestGaussianTransform:
    """Tests for the GaussianTransform quantum operation"""

    def test_merge(self, hbar, tol):
        """Test that two symplectics merge: S = S2 @ S1"""
        n = 3
        S1 = random_symplectic(n)
        S2 = random_symplectic(n)

        G1 = ops.GaussianTransform(S1)
        G1inv = ops.GaussianTransform(np.linalg.inv(S1))
        G2 = ops.GaussianTransform(S2)

        # a symplectic merged with its inverse is identity
        assert G1.merge(G1inv) is None

        # two merged symplectics are the same as their product
        assert np.allclose(G1.merge(G2).p[0].x, S2 @ S1, atol=tol, rtol=0)

    def test_passive(self, tol):
        """Test that a passive decomposition is correctly flagged as requiring
        only a single interferometer"""
        G = ops.GaussianTransform(np.identity(6))

        assert not G.active
        assert hasattr(G, "U1")
        assert not hasattr(G, "Sq")
        assert not hasattr(G, "U2")

    def test_active(self, tol):
        """Test that an active decomposition is correctly flagged as requiring
        two interferometers and squeezing"""
        S1 = random_symplectic(3, passive=False)
        G = ops.GaussianTransform(S1)

        assert G.active
        assert hasattr(G, "U1")
        assert hasattr(G, "Sq")
        assert hasattr(G, "U2")

    def test_decomposition_active(self, hbar, tol):
        """Test that an active symplectic is correctly decomposed into
        two interferometers and squeezing"""
        n = 3
        S = random_symplectic(n, passive=False)

        O1, Sq, O2 = dec.bloch_messiah(S)
        X1 = O1[:n, :n]
        P1 = O1[n:, :n]
        X2 = O2[:n, :n]
        P2 = O2[n:, :n]
        U1 = X1 + 1j * P1
        U2 = X2 + 1j * P2

        prog = sf.Program(n)
        G = ops.GaussianTransform(S)
        cmds = G.decompose(prog.register)

        assert np.all(U1 == G.U1)
        assert np.all(U2 == G.U2)
        assert np.all(np.diag(Sq)[:n] == G.Sq)

        S = np.identity(2 * n)

        # command queue should have 2 interferometers, 3 squeezers
        assert len(cmds) == 5

        # calculating the resulting decomposed symplectic
        for cmd in cmds:
            # all operations should be Interferometers or Sgates
            assert isinstance(cmd.op, (ops.Interferometer, ops.Sgate))

            # build up the symplectic transform
            modes = [i.ind for i in cmd.reg]

            if isinstance(cmd.op, ops.Sgate):
                S = _squeezing(cmd.op.p[0].x, cmd.op.p[1].x, modes, n) @ S

            if isinstance(cmd.op, ops.Interferometer):
                U1 = cmd.op.p[0].x
                S_U = np.vstack(
                    [np.hstack([U1.real, -U1.imag]), np.hstack([U1.imag, U1.real])]
                )
                S = S_U @ S

        # the resulting covariance state
        cov = S @ S.T
        assert np.allclose(cov, S @ S.T, atol=tol, rtol=0)

    def test_decomposition_passive(self, hbar, tol):
        """Test that a passive symplectic is correctly decomposed into an interferometer"""
        n = 3
        S = random_symplectic(n, passive=True)
        X1 = S[:n, :n]
        P1 = S[n:, :n]
        U1 = X1 + 1j * P1

        prog = sf.Program(n)
        G = ops.GaussianTransform(S)
        cmds = G.decompose(prog.register)

        S = np.identity(2 * n)

        # command queue should have 1 interferometer
        assert len(cmds) == 1

        # calculating the resulting decomposed symplectic
        for cmd in cmds:
            # all operations should be Interferometers
            assert isinstance(cmd.op, ops.Interferometer)

            # build up the symplectic transform
            modes = [i.ind for i in cmd.reg]

            if isinstance(cmd.op, ops.Interferometer):
                U1 = cmd.op.p[0].x
                S_U = np.vstack(
                    [np.hstack([U1.real, -U1.imag]), np.hstack([U1.imag, U1.real])]
                )
                S = S_U @ S

        # the resulting covariance state
        cov = S @ S.T
        assert np.allclose(cov, S @ S.T, atol=tol, rtol=0)

    def test_active_on_vacuum(self, hbar, tol):
        """Test that an active symplectic applied to a vacuum is
        correctly decomposed into just squeezing and one interferometer"""
        n = 3
        S = random_symplectic(n, passive=False)

        O1, Sq, O2 = dec.bloch_messiah(S)
        X1 = O1[:n, :n]
        P1 = O1[n:, :n]
        X2 = O2[:n, :n]
        P2 = O2[n:, :n]

        U1 = X1 + 1j * P1
        U2 = X2 + 1j * P2

        prog = sf.Program(n)
        G = ops.GaussianTransform(S, vacuum=True)
        cmds = G.decompose(prog.register)

        S = np.identity(2 * n)

        # command queue should have 3 Sgates, 1 interferometer
        assert len(cmds) == 4

        # calculating the resulting decomposed symplectic
        for cmd in cmds:
            # all operations should be Interferometers or Sgates
            assert isinstance(cmd.op, (ops.Interferometer, ops.Sgate))

            # build up the symplectic transform
            modes = [i.ind for i in cmd.reg]

            if isinstance(cmd.op, ops.Sgate):
                S = _squeezing(cmd.op.p[0].x, cmd.op.p[1].x, modes, n) @ S

            if isinstance(cmd.op, ops.Interferometer):
                U1 = cmd.op.p[0].x
                S_U = np.vstack(
                    [np.hstack([U1.real, -U1.imag]), np.hstack([U1.imag, U1.real])]
                )
                S = S_U @ S

        # the resulting covariance state
        cov = S @ S.T

        assert np.allclose(cov, S @ S.T, atol=tol, rtol=0)


class TestGaussian:
    """Tests for the Gaussian quantum state preparation"""

    def test_merge(self, hbar):
        """Test that merging two Preparations only keeps the latter one."""
        n = 3
        V1 = random_covariance(n, pure=False, hbar=hbar)
        V2 = random_covariance(n, pure=True, hbar=hbar)
        r1 = np.random.randn(2*n)
        r2 = np.random.randn(2*n)

        G1 = ops.Gaussian(V1, r1)
        G2 = ops.Gaussian(V2, r2)

        # applying a second state preparation replaces the first
        assert G1.merge(G2) is G2

        # the same is true of all state preparations
        S = ops.Squeezed(2)
        assert S.merge(G2) is G2
        assert G2.merge(S) is S

    def test_incorrect_means_length(self, hbar):
        """Test that an exception is raised len(means)!=len(cov)"""
        cov = random_covariance(3, hbar=hbar)

        with pytest.raises(ValueError, match="must have the same length"):
            ops.Gaussian(cov, r=np.array([0]))

    def test_apply_decomp(self, hbar):
        """Test that the apply method, when decomp = False, calls the Backend directly."""
        prog = sf.Program(3)
        cov = random_covariance(3, hbar=hbar)

        class DummyBackend:
            """Dummy backend class"""
            def prepare_gaussian_state(*args):
                """Raises a syntax error when called"""
                raise SyntaxError

        G = ops.Gaussian(cov, decomp=False)
        with pytest.raises(SyntaxError):
            G._apply(prog.register, DummyBackend())

    def test_decomposition(self, hbar, tol):
        """Test that an arbitrary decomposition provides the right covariance matrix"""
        n = 3
        prog = sf.Program(n)
        cov = random_covariance(n)

        G = ops.Gaussian(cov)
        cmds = G.decompose(prog.register)

        S = np.identity(2 * n)
        cov_init = np.identity(2 * n) * hbar / 2

        # calculating the resulting decomposed symplectic
        for cmd in cmds:
            # all operations should be BSgates, Rgates, or Sgates
            assert isinstance(cmd.op, (ops.Vacuum, ops.Thermal, ops.GaussianTransform))

            # build up the symplectic transform
            modes = [i.ind for i in cmd.reg]

            if isinstance(cmd.op, ops.Thermal):
                cov_init[cmd.reg[0].ind, cmd.reg[0].ind] = (
                    (2 * cmd.op.p[0].x + 1) * hbar / 2
                )
                cov_init[cmd.reg[0].ind + n, cmd.reg[0].ind + n] = (
                    (2 * cmd.op.p[0].x + 1) * hbar / 2
                )

            if isinstance(cmd.op, ops.GaussianTransform):
                S = cmd.op.p[0].x @ S

        # the resulting covariance state
        cov_res = S @ cov_init @ S.T

        assert np.allclose(cov, cov_res, atol=tol, rtol=0)

    def test_thermal_decomposition(self, hbar, tol):
        """Test that a thermal covariance matrix decomposes into Thermal preparations."""
        n = 3
        prog = sf.Program(n)
        nbar = np.array([0.453, 0.23, 0.543])
        cov = np.diag(np.tile(2 * nbar + 1, 2)) * hbar / 2

        G = ops.Gaussian(cov)
        cmds = G.decompose(prog.register)

        assert len(cmds) == n

        # calculating the resulting decomposed symplectic
        for i, cmd in enumerate(cmds):
            assert isinstance(cmd.op, ops.Thermal)
            assert np.allclose(cmd.op.p[0].x, nbar[i], atol=tol, rtol=0)

    def test_squeezed_decomposition(self, hbar, tol):
        """Test that an axially squeezed covariance matrix decomposes into Squeezed preparations."""
        n = 3
        prog = sf.Program(n)

        sq_r = np.array([0.453, 0.23, 0.543])
        S = np.diag(np.exp(np.concatenate([-sq_r, sq_r])))
        cov = S @ S.T * (hbar / 2)

        G = ops.Gaussian(cov)
        cmds = G.decompose(prog.register)

        assert len(cmds) == n

        # calculating the resulting decomposed symplectic
        for i, cmd in enumerate(cmds):
            assert isinstance(cmd.op, ops.Squeezed)
            assert np.allclose(cmd.op.p[0].x, sq_r[i], atol=tol, rtol=0)
            assert cmd.op.p[1].x == 0

    def test_rotated_squeezed_decomposition(self, hbar, tol):
        """Test that a rotated squeezed covariance matrix decomposes into Squeezed preparations"""
        n = 3
        prog = sf.Program(n)

        sq_r = np.array([0.453, 0.23, 0.543])
        sq_phi = np.array([-0.123, 0.2143, 0.021])

        S = np.diag(np.exp(np.concatenate([-sq_r, sq_r])))

        for i, phi in enumerate(sq_phi):
            S = _rotation(phi / 2, i, n) @ S

        cov = S @ S.T * (hbar / 2)
        G = ops.Gaussian(cov)
        cmds = G.decompose(prog.register)

        assert len(cmds) == n

        # calculating the resulting decomposed symplectic
        for i, cmd in enumerate(cmds):
            assert isinstance(cmd.op, ops.Squeezed)
            assert np.allclose(cmd.op.p[0].x, sq_r[i], atol=tol, rtol=0)
            assert np.allclose(cmd.op.p[1].x, sq_phi[i], atol=tol, rtol=0)

    def test_degenerate_decomposition(self, hbar, tol):
        """Test that a decomposition involving no squeezing results in a Vacuum preparation."""
        n = 2
        prog = sf.Program(n)

        sq_r = np.array([0, 1.5])
        S = np.diag(np.exp(np.concatenate([-sq_r, sq_r])))
        cov = S @ S.T * (hbar / 2)

        G = ops.Gaussian(cov)
        cmds = G.decompose(prog.register)

        assert len(cmds) == 2

        for cmd in cmds[:1]:
            assert isinstance(cmd.op, ops.Vacuum)

        for cmd in cmds[1:]:
            assert isinstance(cmd.op, ops.Squeezed)
            assert np.allclose(cmd.op.p[0].x, sq_r[1], atol=tol, rtol=0)
            assert np.allclose(cmd.op.p[1].x, 0, atol=tol, rtol=0)

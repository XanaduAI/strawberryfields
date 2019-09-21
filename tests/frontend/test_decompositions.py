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
r"""Unit tests for the Strawberry Fields decompositions module"""
import pytest

pytestmark = pytest.mark.frontend

import numpy as np
import scipy as sp
from scipy.linalg import qr, block_diag

from strawberryfields import decompositions as dec

N_SAMPLES = 10


# fix the seed to make the test deterministic
np.random.seed(42)


def omega(n):
    """Returns the symplectic matrix for n modes"""
    idm = np.identity(n)
    O = np.concatenate(
        (
            np.concatenate((0 * idm, idm), axis=1),
            np.concatenate((-idm, 0 * idm), axis=1),
        ),
        axis=0,
    )
    return O


def haar_measure(n):
    """A Random matrix distributed with the Haar measure.

    For more details, see :cite:`mezzadri2006`.

    Args:
        n (int): matrix size
    Returns:
        array: an nxn random matrix
    """
    z = (sp.randn(n, n) + 1j * sp.randn(n, n)) / np.sqrt(2.0)
    q, r = qr(z)
    d = sp.diagonal(r)
    ph = d / np.abs(d)
    q = np.multiply(q, ph, q)
    return q


class TestTakagi:
    """Takagi decomposition tests"""

    def test_square_validation(self):
        """Test that the takagi decomposition raises exception if not square"""
        A = np.random.random([4, 5]) + 1j * np.random.random([4, 5])
        with pytest.raises(ValueError, match="matrix must be square"):
            dec.takagi(A)

    def test_symmetric_validation(self):
        """Test that the takagi decomposition raises exception if not symmetric"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        with pytest.raises(ValueError, match="matrix is not symmetric"):
            dec.takagi(A)

    def test_random_symm(self, tol):
        """Verify that the Takagi decomposition, applied to a random symmetric
        matrix, produced a decomposition that can be used to reconstruct the matrix."""
        A = np.random.random([6, 6]) + 1j * np.random.random([6, 6])
        A += A.T
        rl, U = dec.takagi(A)
        res = U @ np.diag(rl) @ U.T
        assert np.allclose(res, A, atol=tol, rtol=0)


class TestGraphEmbed:
    """graph_embed tests"""

    def test_square_validation(self):
        """Test that the graph_embed decomposition raises exception if not square"""
        A = np.random.random([4, 5]) + 1j * np.random.random([4, 5])
        with pytest.raises(ValueError, match="matrix is not square"):
            dec.graph_embed(A)

    def test_symmetric_validation(self):
        """Test that the graph_embed decomposition raises exception if not symmetric"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        with pytest.raises(ValueError, match="matrix is not symmetric"):
            dec.graph_embed(A)

    def test_max_mean_photon_deprecated(self, tol):
        """This test verifies that the maximum amount of squeezing used to encode
        the graph is indeed capped by the parameter max_mean_photon"""
        max_mean_photon = 2
        A = np.random.random([6, 6]) + 1j * np.random.random([6, 6])
        A += A.T
        sc, _ = dec.graph_embed_deprecated(A, max_mean_photon=max_mean_photon)
        res_mean_photon = np.sinh(np.max(np.abs(sc))) ** 2

        assert np.allclose(res_mean_photon, max_mean_photon, atol=tol, rtol=0)

    def test_make_traceless_deprecated(self, monkeypatch, tol):
        """Test that A is properly made traceless"""
        A = np.random.random([6, 6]) + 1j * np.random.random([6, 6])
        A += A.T

        assert not np.allclose(np.trace(A), 0, atol=tol, rtol=0)

        with monkeypatch.context() as m:
            # monkeypatch the takagi function to simply return A,
            # so that we can inspect it and make sure it is now traceless
            m.setattr(dec, "takagi", lambda A, tol: (np.ones([6]), A))
            _, A_out = dec.graph_embed_deprecated(A, make_traceless=True)

        assert np.allclose(np.trace(A_out), 0, atol=tol, rtol=0)

    def test_mean_photon(self, tol):
        """Test that the mean photon number is correct in graph_embed"""
        num_modes = 6
        A = np.random.random([num_modes, num_modes]) + 1j * np.random.random([num_modes, num_modes])
        A += A.T
        n_mean = 10.0 / num_modes
        sc, _ = dec.graph_embed(A, mean_photon_per_mode=n_mean)
        n_mean_calc = np.mean(np.sinh(sc) ** 2)

        assert np.allclose(n_mean, n_mean_calc, atol=tol, rtol=0)


class TestBipartiteGraphEmbed:
    """graph_embed tests"""

    def test_square_validation(self):
        """Test that the graph_embed decomposition raises exception if not square"""
        A = np.random.random([4, 5]) + 1j * np.random.random([4, 5])
        with pytest.raises(ValueError, match="matrix is not square"):
            dec.bipartite_graph_embed(A)

    @pytest.mark.parametrize("make_symmetric", [True, False])
    def test_mean_photon(self, tol, make_symmetric):
        """Test that the mean photon number is correct in graph_embed"""
        num_modes = 6
        A = np.random.random([num_modes, num_modes]) + 1j * np.random.random(
            [num_modes, num_modes]
        )
        if make_symmetric:
            A += A.T
        n_mean = 1.0
        sc, _, _ = dec.bipartite_graph_embed(A, mean_photon_per_mode=n_mean)
        n_mean_calc = np.sum(np.sinh(sc) ** 2) / (num_modes)
        assert np.allclose(n_mean, n_mean_calc, atol=tol, rtol=0)

    @pytest.mark.parametrize("make_symmetric", [True, False])
    def test_correct_graph(self, tol, make_symmetric):
        """Test that the graph is embeded correctly"""
        num_modes = 3
        A = np.random.random([num_modes, num_modes]) + 1j * np.random.random(
            [num_modes, num_modes]
        )
        U, l, V = np.linalg.svd(A)
        new_l = np.array(
            [np.tanh(np.arcsinh(np.sqrt(i))) for i in range(1, num_modes + 1)]
        )
        n_mean = 0.5 * (num_modes + 1)
        if make_symmetric:
            At = U @ np.diag(new_l) @ U.T
        else:
            At = U @ np.diag(new_l) @ V.T
        sqf, Uf, Vf = dec.bipartite_graph_embed(At, mean_photon_per_mode=n_mean)

        assert np.allclose(np.tanh(-np.flip(sqf)), new_l)
        assert np.allclose(Uf @ np.diag(np.tanh(-sqf)) @ Vf.T, At)


class TestRectangularDecomposition:
    """Tests for linear interferometer rectangular decomposition"""

    def test_unitary_validation(self):
        """Test that an exception is raised if not unitary"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        with pytest.raises(ValueError, match="matrix is not unitary"):
            dec.rectangular(A)

    def test_identity(self, tol):
        """This test checks the rectangular decomposition for an identity unitary.

        An identity unitary is decomposed via the rectangular decomposition of
        Clements et al. and the resulting beamsplitters are multiplied together.
        Test passes if the product matches identity.
        """
        # TODO: this test currently uses the T and Ti functions used to compute
        # Clements as the comparison. Probably should be changed.
        n = 20
        U = np.identity(n)

        tilist, diags, tlist = dec.rectangular(U)

        qrec = np.identity(n)

        for i in tilist:
            qrec = dec.T(*i) @ qrec

        qrec = np.diag(diags) @ qrec

        for i in reversed(tlist):
            qrec = dec.Ti(*i) @ qrec

        assert np.allclose(U, qrec, atol=tol, rtol=0)

    def test_random_unitary(self, tol):
        """This test checks the rectangular decomposition for a random unitary.

        A random unitary is drawn from the Haar measure, then is decomposed via
        the rectangular decomposition of Clements et al., and the resulting
        beamsplitters are multiplied together. Test passes if the product
        matches the drawn unitary.
        """
        # TODO: this test currently uses the T and Ti functions used to compute
        # Clements as the comparison. Probably should be changed.
        n = 20
        U = haar_measure(n)

        tilist, diags, tlist = dec.rectangular(U)

        qrec = np.identity(n)

        for i in tilist:
            qrec = dec.T(*i) @ qrec

        qrec = np.diag(diags) @ qrec

        for i in reversed(tlist):
            qrec = dec.Ti(*i) @ qrec

        assert np.allclose(U, qrec, atol=tol, rtol=0)

    def test_random_unitary_phase_end(self, tol):
        """This test checks the rectangular decomposition with phases at the end.

        A random unitary is drawn from the Haar measure, then is decomposed
        using Eq. 5 of the rectangular decomposition procedure of Clements et al,
        i.e., moving all the phases to the end of the interferometer. The
        resulting beamsplitters are multiplied together. Test passes if the
        product matches the drawn unitary.
        """
        n = 20
        U = haar_measure(n)

        tlist, diags, _ = dec.rectangular_phase_end(U)

        qrec = np.identity(n)

        for i in tlist:
            qrec = dec.T(*i) @ qrec

        qrec = np.diag(diags) @ qrec

        assert np.allclose(U, qrec, atol=tol, rtol=0)


class TestRectangularSymmetricDecomposition:
    """Tests for linear interferometer decomposition into rectangular grid of
    phase-shifters and pairs of symmetric beamsplitters"""

    def test_unitary_validation(self):
        """Test that an exception is raised if not unitary"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        with pytest.raises(ValueError, match="matrix is not unitary"):
            dec.rectangular_symmetric(A)

    @pytest.mark.parametrize('U', [
        pytest.param(np.identity(2), id='identity2'),
        pytest.param(np.identity(2)[::-1], id='antiidentity2'),
        pytest.param(haar_measure(2), id='random2'),
        pytest.param(np.identity(4), id='identity4'),
        pytest.param(np.identity(3)[::-1], id='antiidentity4'),
        pytest.param(haar_measure(4), id='random4'),
        pytest.param(np.identity(8), id='identity8'),
        pytest.param(np.identity(8)[::-1], id='antiidentity8'),
        pytest.param(haar_measure(8), id='random8'),
        pytest.param(np.identity(20), id='identity20'),
        pytest.param(np.identity(20)[::-1], id='antiidentity20'),
        pytest.param(haar_measure(20), id='random20')
        ])
    def test_decomposition(self, U, tol):
        """This test checks the function :func:`dec.rectangular_symmetric` for
        various unitary matrices.

        A given unitary (identity or random draw from Haar measure) is
        decomposed using the function :func:`dec.rectangular_symmetric`
        and the resulting beamsplitters are multiplied together.

        Test passes if the product matches identity.
        """
        nmax, mmax = U.shape
        assert nmax == mmax
        tlist, diags, _ = dec.rectangular_symmetric(U)
        qrec = np.identity(nmax)
        for i in tlist:
            assert i[2] >= 0 and i[2] < 2 * np.pi  # internal phase
            assert i[3] >= 0 and i[3] < 2 * np.pi  # external phase
            qrec = dec.mach_zehnder(*i) @ qrec
        qrec = np.diag(diags) @ qrec
        assert np.allclose(U, qrec, atol=tol, rtol=0)


class TestTriangularDecomposition:
    """Tests for linear interferometer triangular decomposition"""

    def test_unitary_validation(self):
        """Test that an exception is raised if not unitary"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        with pytest.raises(ValueError, match="matrix is not unitary"):
            dec.triangular(A)

    def test_identity(self, tol):
        """This test checks the rectangular decomposition for an identity unitary.

        An identity unitary is decomposed via the rectangular decomposition of
        Clements et al. and the resulting beamsplitters are multiplied together.
        Test passes if the product matches identity.
        """
        # TODO: this test currently uses the T and Ti functions used to compute
        # Clements as the comparison. Probably should be changed.
        n = 20
        U = np.identity(n)

        tlist, diags, _ = dec.triangular(U)

        qrec = np.diag(diags)

        for i in tlist:
            qrec = dec.Ti(*i) @ qrec

        assert np.allclose(U, qrec, atol=tol, rtol=0)

    def test_random_unitary(self, tol):
        """This test checks the rectangular decomposition for a random unitary.

        A random unitary is drawn from the Haar measure, then is decomposed via
        the rectangular decomposition of Clements et al., and the resulting
        beamsplitters are multiplied together. Test passes if the product
        matches the drawn unitary.
        """
        # TODO: this test currently uses the T and Ti functions used to compute
        # Clements as the comparison. Probably should be changed.
        n = 20
        U = haar_measure(n)

        tlist, diags, _ = dec.triangular(U)

        qrec = np.diag(diags)

        for i in tlist:
            qrec = dec.Ti(*i) @ qrec

        assert np.allclose(U, qrec, atol=tol, rtol=0)


class TestWilliamsonDecomposition:
    """Tests for the Williamson decomposition"""

    @pytest.fixture
    def create_cov(self, hbar, tol):
        """create a covariance state for use in testing.

        Args:
            nbar (array[float]): vector containing thermal state values

        Returns:
            tuple: covariance matrix and symplectic transform
        """

        def _create_cov(nbar):
            """wrapped function"""
            n = len(nbar)
            O = omega(n)

            # initial vacuum state
            cov = np.diag(2 * np.tile(nbar, 2) + 1) * hbar / 2

            # interferometer 1
            U1 = haar_measure(n)
            S1 = np.vstack(
                [np.hstack([U1.real, -U1.imag]), np.hstack([U1.imag, U1.real])]
            )

            # squeezing
            r = np.log(0.2 * np.arange(n) + 2)
            Sq = block_diag(np.diag(np.exp(-r)), np.diag(np.exp(r)))

            # interferometer 2
            U2 = haar_measure(n)
            S2 = np.vstack(
                [np.hstack([U2.real, -U2.imag]), np.hstack([U2.imag, U2.real])]
            )

            # final symplectic
            S_final = S2 @ Sq @ S1

            # final covariance matrix
            cov_final = S_final @ cov @ S_final.T

            # check valid symplectic transform
            assert np.allclose(S_final.T @ O @ S_final, O)

            # check valid state
            eigs = np.linalg.eigvalsh(cov_final + 1j * (hbar / 2) * O)
            eigs[np.abs(eigs) < tol] = 0
            assert np.all(eigs >= 0)

            if np.allclose(nbar, 0):
                # check pure
                assert np.allclose(np.linalg.det(cov_final), (hbar / 2) ** (2 * n))
            else:
                # check not pure
                assert not np.allclose(np.linalg.det(cov_final), (hbar / 2) ** (2 * n))

            return cov_final, S_final

        return _create_cov

    def test_square_validation(self):
        """Test that the graph_embed decomposition raises exception if not square"""
        A = np.random.random([4, 5]) + 1j * np.random.random([4, 5])
        with pytest.raises(ValueError, match="matrix is not square"):
            dec.williamson(A)

    def test_symmetric_validation(self):
        """Test that the graph_embed decomposition raises exception if not symmetric"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        with pytest.raises(ValueError, match="matrix is not symmetric"):
            dec.williamson(A)

    def test_even_validation(self):
        """Test that the graph_embed decomposition raises exception if not even number of rows"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        A += A.T
        with pytest.raises(
            ValueError, match="must have an even number of rows/columns"
        ):
            dec.williamson(A)

    def test_positive_definite_validation(self):
        """Test that the graph_embed decomposition raises exception if not positive definite"""
        A = np.diag([-2, 0.1, 2, 3])
        with pytest.raises(ValueError, match="matrix is not positive definite"):
            dec.williamson(A)

    def test_vacuum_state(self, tol, hbar):
        """Test vacuum state"""
        V = np.identity(4)
        Db, S = dec.williamson(V)
        assert np.allclose(Db, np.identity(4), atol=tol, rtol=0)
        assert np.allclose(S, np.identity(4), atol=tol, rtol=0)

    def test_pure_state(self, create_cov, hbar, tol):
        """Test pure state"""
        n = 3
        O = omega(n)

        cov, _ = create_cov(np.zeros([n]))

        Db, S = dec.williamson(cov)
        nbar = np.diag(Db) / hbar - 0.5

        # check decomposition is correct
        assert np.allclose(S @ Db @ S.T, cov, atol=tol, rtol=0)
        # check nbar = 0
        assert np.allclose(nbar, 0, atol=tol, rtol=0)
        # check S is symplectic
        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)

    def test_mixed_state(self, create_cov, hbar, tol):
        """Test mixed state"""
        n = 3
        O = omega(n)
        nbar_in = np.abs(np.random.random(n))

        cov, _ = create_cov(nbar_in)

        Db, S = dec.williamson(cov)
        nbar = np.diag(Db) / hbar - 0.5

        # check decomposition is correct
        assert np.allclose(S @ Db @ S.T, cov, atol=tol, rtol=0)
        # check nbar
        assert np.allclose(sorted(nbar[:n]), sorted(nbar_in), atol=tol, rtol=0)
        # check S is symplectic
        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)


class TestBlochMessiahDecomposition:
    """Tests for the Bloch-Messiah decomposition"""

    @pytest.fixture
    def create_transform(self):
        """create a symplectic transform for use in testing.

        Args:
            n (int): number of modes
            passive (bool): whether transform should be passive or not

        Returns:
            array: symplectic matrix
        """

        def _create_transform(n, passive=True):
            """wrapped function"""
            O = omega(n)

            # interferometer 1
            U1 = haar_measure(n)
            S1 = np.vstack(
                [np.hstack([U1.real, -U1.imag]), np.hstack([U1.imag, U1.real])]
            )

            Sq = np.identity(2 * n)
            if not passive:
                # squeezing
                r = np.log(0.2 * np.arange(n) + 2)
                Sq = block_diag(np.diag(np.exp(-r)), np.diag(np.exp(r)))

            # interferometer 2
            U2 = haar_measure(n)
            S2 = np.vstack(
                [np.hstack([U2.real, -U2.imag]), np.hstack([U2.imag, U2.real])]
            )

            # final symplectic
            S_final = S2 @ Sq @ S1

            # check valid symplectic transform
            assert np.allclose(S_final.T @ O @ S_final, O)
            return S_final

        return _create_transform

    def test_square_validation(self):
        """Test raises exception if not square"""
        A = np.random.random([4, 5]) + 1j * np.random.random([4, 5])
        with pytest.raises(ValueError, match="matrix is not square"):
            dec.bloch_messiah(A)

    def test_symmplectic(self):
        """Test raises exception if not symmetric"""
        A = np.random.random([6, 6]) + 1j * np.random.random([6, 6])
        A += A.T
        with pytest.raises(ValueError, match="matrix is not symplectic"):
            dec.bloch_messiah(A)

    def test_even_validation(self):
        """Test raises exception if not even number of rows"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        A += A.T
        with pytest.raises(
            ValueError, match="must have an even number of rows/columns"
        ):
            dec.bloch_messiah(A)

    def test_identity(self, tol):
        """Test identity"""
        n = 2
        S_in = np.identity(2 * n)
        O1, S, O2 = dec.bloch_messiah(S_in)

        assert np.allclose(O1 @ O2, np.identity(2 * n), atol=tol, rtol=0)
        assert np.allclose(S, np.identity(2 * n), atol=tol, rtol=0)

        # test orthogonality
        assert np.allclose(O1.T, O1, atol=tol, rtol=0)
        assert np.allclose(O2.T, O2, atol=tol, rtol=0)

        # test symplectic
        O = omega(n)
        assert np.allclose(O1 @ O @ O1.T, O, atol=tol, rtol=0)
        assert np.allclose(O2 @ O @ O2.T, O, atol=tol, rtol=0)

    def test_passive_transform(self, create_transform, tol):
        """Test passive transform has no squeezing.
        Note: this test also tests the case with degenerate symplectic values"""
        n = 3
        S_in = create_transform(3, passive=True)
        O1, S, O2 = dec.bloch_messiah(S_in)

        # test decomposition
        assert np.allclose(O1 @ S @ O2, S_in, atol=tol, rtol=0)

        # test no squeezing
        assert np.allclose(O1 @ O2, S_in, atol=tol, rtol=0)
        assert np.allclose(S, np.identity(2 * n), atol=tol, rtol=0)

        # test orthogonality
        assert np.allclose(O1.T @ O1, np.identity(2 * n), atol=tol, rtol=0)
        assert np.allclose(O2.T @ O2, np.identity(2 * n), atol=tol, rtol=0)

        # test symplectic
        O = omega(n)
        # TODO: BUG:
        # assert np.allclose(O1.T @ O @ O1, O, atol=tol, rtol=0)
        # assert np.allclose(O2.T @ O @ O2, O, atol=tol, rtol=0)
        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)

    def test_active_transform(self, create_transform, tol):
        """Test passive transform with squeezing"""
        n = 3
        S_in = create_transform(3, passive=False)
        O1, S, O2 = dec.bloch_messiah(S_in)

        # test decomposition
        assert np.allclose(O1 @ S @ O2, S_in, atol=tol, rtol=0)

        # test orthogonality
        assert np.allclose(O1.T @ O1, np.identity(2 * n), atol=tol, rtol=0)
        assert np.allclose(O2.T @ O2, np.identity(2 * n), atol=tol, rtol=0)

        # test symplectic
        O = omega(n)
        assert np.allclose(O1.T @ O @ O1, O, atol=tol, rtol=0)
        assert np.allclose(O2.T @ O @ O2, O, atol=tol, rtol=0)
        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)

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

from strawberryfields.utils.random_numbers_matrices import random_interferometer

pytestmark = pytest.mark.frontend

import networkx as nx
import numpy as np
import scipy as sp
from scipy.linalg import qr, block_diag
from thewalrus.symplectic import sympmat as omega

from strawberryfields import decompositions as dec
from strawberryfields.utils import random_interferometer as haar_measure

N_SAMPLES = 10



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
        A = np.random.random([num_modes, num_modes]) + 1j * np.random.random([num_modes, num_modes])
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
        A = np.random.random([num_modes, num_modes]) + 1j * np.random.random([num_modes, num_modes])
        U, l, V = np.linalg.svd(A)
        new_l = np.array([np.tanh(np.arcsinh(np.sqrt(i))) for i in range(1, num_modes + 1)])
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

    @pytest.mark.parametrize(
        "U",
        [
            pytest.param(np.identity(20), id="identity20"),
            pytest.param(np.identity(20)[::-1], id="antiidentity20"),
            pytest.param(haar_measure(20), id="random20"),
        ],
    )
    def test_rectangular(self, U, tol):
        """This test checks the function :func:`dec.rectangular` for
        various unitary matrices.

        A given unitary (identity or random draw from Haar measure) is
        decomposed using the function :func:`dec.rectangular`
        and the resulting beamsplitters are multiplied together.

        Test passes if the product matches the given unitary.
        """
        nmax, mmax = U.shape
        assert nmax == mmax

        tilist, diags, tlist = dec.rectangular(U)

        qrec = np.identity(nmax)

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

    @pytest.mark.parametrize(
        "U",
        [
            pytest.param(np.identity(20), id="identity20"),
            pytest.param(np.identity(20)[::-1], id="antiidentity20"),
            pytest.param(haar_measure(20), id="random20"),
        ],
    )
    def test_rectangular_MZ(self, U, tol):
        """This test checks the function :func:`dec.rectangular_MZ` for
        various unitary matrices.

        A given unitary (identity or random draw from Haar measure) is
        decomposed using the function :func:`dec.rectangular_MZ`
        and the resulting beamsplitters are multiplied together.

        Test passes if the product matches the given unitary.
        """
        nmax, mmax = U.shape
        assert nmax == mmax

        tilist, diags, tlist = dec.rectangular_MZ(U)

        qrec = np.identity(nmax)

        for i in tilist:
            qrec = dec.mach_zehnder(*i) @ qrec

        qrec = np.diag(diags) @ qrec

        for i in reversed(tlist):
            qrec = dec.mach_zehnder_inv(*i) @ qrec

        assert np.allclose(U, qrec, atol=tol, rtol=0)


class TestRectangularSymmetricDecomposition:
    """Tests for linear interferometer decomposition into rectangular grid of
    phase-shifters and pairs of symmetric beamsplitters"""

    def test_unitary_validation(self):
        """Test that an exception is raised if not unitary"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        with pytest.raises(ValueError, match="matrix is not unitary"):
            dec.rectangular_symmetric(A)

    @pytest.mark.parametrize(
        "U",
        [
            pytest.param(np.identity(2), id="identity2"),
            pytest.param(np.identity(2)[::-1], id="antiidentity2"),
            pytest.param(haar_measure(2), id="random2"),
            pytest.param(np.identity(4), id="identity4"),
            pytest.param(np.identity(3)[::-1], id="antiidentity4"),
            pytest.param(haar_measure(4), id="random4"),
            pytest.param(np.identity(8), id="identity8"),
            pytest.param(np.identity(8)[::-1], id="antiidentity8"),
            pytest.param(haar_measure(8), id="random8"),
            pytest.param(np.identity(20), id="identity20"),
            pytest.param(np.identity(20)[::-1], id="antiidentity20"),
            pytest.param(haar_measure(20), id="random20"),
        ],
    )
    def test_decomposition(self, U, tol):
        """This test checks the function :func:`dec.rectangular_symmetric` for
        various unitary matrices.

        A given unitary (identity or random draw from Haar measure) is
        decomposed using the function :func:`dec.rectangular_symmetric`
        and the resulting beamsplitters are multiplied together.

        Test passes if the product matches the given unitary.
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


def _rectangular_compact_recompose(phases):
    r"""Calculates the unitary of a rectangular compact interferometer,
    using the phases provided in phases dict.

    Args:
        phases (dict):
        where the keywords:

        * ``m``: the length of the matrix
        * ``phi_ins``: parameters for the phase-shifters
        * ``sigmas``: parameters for the sMZI
        * ``deltas``: parameters for the sMZI
        * ``phi_edges``: parameters for the edge phase shifters
        * ``phi_outs``: parameters for the phase-shifters

    Returns:
        array : unitary matrix of the interferometer
    """
    m = phases["m"]
    U = np.eye(m, dtype=np.complex128)
    for j in range(0, m - 1, 2):
        phi = phases["phi_ins"][j]
        U = dec.P(j, phi, m) @ U
    for layer in range(m):
        if (layer + m + 1) % 2 == 0:
            phi_bottom = phases["phi_edges"][m - 1, layer]
            U = dec.P(m - 1, phi_bottom, m) @ U
        for mode in range(layer % 2, m - 1, 2):
            delta = phases["deltas"][mode, layer]
            sigma = phases["sigmas"][mode, layer]
            U = dec.M(mode, sigma, delta, m) @ U
    for j, phi_j in phases["phi_outs"].items():
        U = dec.P(j, phi_j, m) @ U
    return U


class TestRectangularCompactDecomposition:
    """Tests for linear interferometer decomposition into rectangular grid of
    phase-shifters and pairs of symmetric beamsplitters"""

    def test_unitary_validation(self):
        """Test that an exception is raised if not unitary"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        with pytest.raises(ValueError, match="The input matrix is not unitary"):
            dec.rectangular_compact(A)

    @pytest.mark.parametrize(
        "U",
        [
            pytest.param(np.identity(2), id="identity2"),
            pytest.param(np.identity(2)[::-1], id="antiidentity2"),
            pytest.param(haar_measure(2), id="random2"),
            pytest.param(np.identity(4), id="identity4"),
            pytest.param(np.identity(4)[::-1], id="antiidentity4"),
            pytest.param(haar_measure(4), id="random4"),
            pytest.param(np.identity(8), id="identity8"),
            pytest.param(np.identity(8)[::-1], id="antiidentity8"),
            pytest.param(haar_measure(8), id="random8"),
            pytest.param(np.identity(20), id="identity20"),
            pytest.param(np.identity(20)[::-1], id="antiidentity20"),
            pytest.param(haar_measure(20), id="random20"),
            pytest.param(np.identity(7), id="identity7"),
            pytest.param(np.identity(7)[::-1], id="antiidentity7"),
            pytest.param(haar_measure(7), id="random7"),
        ],
    )
    def test_decomposition(self, U, tol):
        """This test checks the function :func:`dec.rectangular_symmetric` for
        various unitary matrices.

        A given unitary (identity or random draw from Haar measure) is
        decomposed using the function :func:`dec.rectangular_symmetric`
        and the resulting beamsplitters are multiplied together.

        Test passes if the product matches the given unitary.
        """
        nmax, mmax = U.shape
        assert nmax == mmax
        phases = dec.rectangular_compact(U)
        Uout = _rectangular_compact_recompose(phases)
        assert np.allclose(U, Uout, atol=tol, rtol=0)


def _triangular_compact_recompose(phases):
    r"""Calculates the unitary of a triangular compact interferometer,
    using the phases provided in phases dict.

    Args:
        phases (dict):
        where the keywords:

        * ``m``: the length of the matrix
        * ``phi_ins``: parameter of the phase-shifter at the beginning of the mode
        * ``sigmas``: parameter of the sMZI :math:`\frac{(\theta_1+\theta_2)}{2}`, where `\theta_{1,2}` are the values of the two internal phase-shifts of sMZI
        * ``deltas``: parameter of the sMZI :math:`\frac{(\theta_1-\theta_2)}{2}`, where `\theta_{1,2}` are the values of the two internal phase-shifts of sMZI
        * ``zetas``: parameter of the phase-shifter at the end of the mode

    Returns:
        U (array) : unitary matrix of the interferometer
    """
    m = phases["m"]
    U = np.identity(m, dtype=np.complex128)
    for j in range(m - 1):
        phi_j = phases["phi_ins"][j]
        U = dec.P(j + 1, phi_j, m) @ U
        for k in range(j + 1):
            n = j - k
            delta = phases["deltas"][n, k]
            sigma = phases["sigmas"][n, k]
            U = dec.M(n, sigma, delta, m) @ U
    for j in range(m):
        zeta = phases["zetas"][j]
        U = dec.P(j, zeta, m) @ U
    return U


class TestTriangularCompactDecomposition:
    """Tests for linear interferometer decomposition into rectangular grid of
    phase-shifters and pairs of symmetric beamsplitters"""

    def test_unitary_validation(self):
        """Test that an exception is raised if not unitary"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        with pytest.raises(ValueError, match="The input matrix is not unitary"):
            dec.triangular_compact(A)

    @pytest.mark.parametrize(
        "U",
        [
            pytest.param(np.identity(2), id="identity2"),
            pytest.param(np.identity(2)[::-1], id="antiidentity2"),
            pytest.param(haar_measure(2), id="random2"),
            pytest.param(np.identity(4), id="identity4"),
            pytest.param(np.identity(4)[::-1], id="antiidentity4"),
            pytest.param(haar_measure(4), id="random4"),
            pytest.param(np.identity(8), id="identity8"),
            pytest.param(np.identity(8)[::-1], id="antiidentity8"),
            pytest.param(haar_measure(8), id="random8"),
            pytest.param(np.identity(20), id="identity20"),
            pytest.param(np.identity(20)[::-1], id="antiidentity20"),
            pytest.param(haar_measure(20), id="random20"),
            pytest.param(np.identity(7), id="identity7"),
            pytest.param(np.identity(7)[::-1], id="antiidentity7"),
            pytest.param(haar_measure(7), id="random7"),
        ],
    )
    def test_decomposition(self, U, tol):
        """This test checks the function :func:`dec.rectangular_symmetric` for
        various unitary matrices.

        A given unitary (identity or random draw from Haar measure) is
        decomposed using the function :func:`dec.rectangular_symmetric`
        and the resulting beamsplitters are multiplied together.

        Test passes if the product matches the given unitary.
        """
        nmax, mmax = U.shape
        assert nmax == mmax
        phases = dec.triangular_compact(U)
        Uout = _triangular_compact_recompose(phases)
        assert np.allclose(U, Uout, atol=tol, rtol=0)


class TestSUnFactorization:
    """tests for the SU(n) factorization"""

    def _embed_su2(self, n, i, j, params):
        """Embed the SU(2) transformation given by params into modes i and j
        of an SU(n) matrix
            SU_ij(3) = [ e^(i(a+g)/2) cos(b/2)   -e^(i(a-g)/2) sin(b/2)
                        e^(-i(a-g)/2) sin(b/2)   e^(-i(a-g)/2) cos(b/2) ]
        Returns the full n-dimensional matrix.
        """
        a, b, g = params[0], params[1], params[2]

        # Create SU(2) element and scaled by loss if desired.
        Rij = np.array(
            [
                [
                    np.exp(1j * (a + g) / 2) * np.cos(b / 2),
                    -np.exp(1j * (a - g) / 2) * np.sin(b / 2),
                ],
                [
                    np.exp(-1j * (a - g) / 2) * np.sin(b / 2),
                    np.exp(-1j * (a + g) / 2) * np.cos(b / 2),
                ],
            ]
        )

        # Stuff it into modes i and j of SU(n)
        full_Rij = np.identity(n, dtype=complex)
        full_Rij[i : j + 1, i : j + 1] = Rij

        return full_Rij

    def _sun_reconstruction(self, n, parameters):
        """Reconstruct an SU(n) matrix using a list of transformations given as
        tuples ("i,j", [a, b, g])

        Args:
            n (int): dimension of the unitary special matrix
            parameters (list(tuple)): sequence of tranformation parameters with the
                form ("i,j", [a, b, g]) where i,j are the indices of the modes and
                a,b,g the SU(2) transformation parameters.

        Returns:
            array[complex]: the reconstructed SU(n) matrix
        """
        U = np.identity(n, dtype=complex)

        for modes, params in parameters:
            # Get the indices of the modes
            md1, md2 = int(modes[0]), int(modes[1])

            if md1 not in range(n) or md2 not in range(n):
                raise ValueError(
                    f"Mode combination {md1},{md2}  is invalid for a system of dimension {n}."
                )

            if md2 != md1 + 1:
                raise ValueError(
                    f"Mode combination {md1},{md2} is invalid.\n"
                    + "Currently only transformations on adjacent modes are implemented."
                )

            # Compute the next transformation and multiply
            next_trans = self._embed_su2(n, md1, md2, params)
            U = U @ next_trans

        return U

    def test_unitary_validation(self, tol):
        """Test that an exception is raised if not unitary"""
        A = np.random.random([5, 5]) + 1j * np.random.random([5, 5])
        with pytest.raises(ValueError, match="The input matrix is not unitary."):
            dec.sun_compact(A, tol)

    def test_unitary_size(self, tol):
        """test n=2 unitaries are not decomposed"""

        U = random_interferometer(2)
        with pytest.raises(
            ValueError, match="Input matrix for decomposition must be at least 3x3."
        ):
            dec.sun_compact(U, tol)

    @pytest.mark.parametrize("SU_matrix", [True, False])
    def test_global_phase(self, SU_matrix, tol):
        """test factorized phase from unitary matrix"""
        n = 3
        # Generate a random SU(n) matrix.
        U = random_interferometer(n)
        det = np.linalg.det(U)
        if SU_matrix:
            U /= det ** (1 / n)

        # get result from factorization
        _, global_phase = dec.sun_compact(U, tol)

        if SU_matrix:
            assert global_phase is None
        else:
            expected_phase = np.angle(det)
            assert np.allclose(global_phase, expected_phase, atol=tol)

    @pytest.mark.parametrize("n", range(3, 8))
    def test_SU_reconstruction(self, n, tol):
        """test numerical reconstruction of Special Unitary matrix equals the original matrix"""

        # Generate a random SU(n) matrix.
        U = random_interferometer(n)
        SU_expected = U / np.linalg.det(U) ** (1 / n)

        # get result from factorization
        factorization_params, global_phase = dec.sun_compact(SU_expected, tol)

        SU_reconstructed = self._sun_reconstruction(n, factorization_params)

        assert global_phase is None
        assert np.allclose(SU_expected, SU_reconstructed, atol=tol, rtol=0)

    @pytest.mark.parametrize("n", range(3, 8))
    def test_interferometer_reconstruction(self, n, tol):
        """test numerical reconstruction of Unitary matrix equals the original matrix"""

        U = random_interferometer(n)

        factorization_params, phase = dec.sun_compact(U, tol)
        det = np.exp(1j * phase) ** (1 / n)
        SU_reconstructed = self._sun_reconstruction(n, factorization_params)
        U_reconstructed = det * SU_reconstructed

        assert np.allclose(U_reconstructed, U)

    @pytest.mark.parametrize("phase", np.linspace(0, 2 * np.pi, 5))
    @pytest.mark.parametrize(
        "n,permutation",
        [
            (3, np.array([0, 1, 2])),
            (4, np.array([0, 1, 2, 3])),
            (4, np.array([2, 3, 0, 1])),
            (5, np.array([0, 1, 2, 3, 4])),
            (5, np.array([3, 1, 0, 2, 4])),
        ],
    )
    def test_embeded_unitary(self, n, permutation, phase, tol):
        """test factorization of U(n-1) transformations embeded on U(n) transformation"""

        # Embed U(4) on n=5 matrix
        U = np.zeros((n, n), dtype=complex)
        U[0, 0] = np.exp(1j * phase)
        U4 = random_interferometer(n - 1)
        U[1:, 1:] = U4

        # permute rows
        U = U[permutation, :]

        factorization_params, _ = dec.sun_compact(U, tol)
        _, first_params = factorization_params[0]

        assert first_params == [0.0, 0.0, 0.0]

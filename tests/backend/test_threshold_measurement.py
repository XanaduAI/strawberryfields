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

r"""Unit tests for measurements in the Fock basis"""
import pytest
import numpy as np
import numba
from thewalrus.quantum import Qmat, Xmat, Amat

NUM_REPEATS = 50

@pytest.mark.backends("gaussian")
class TestGaussianRepresentation:
    """Tests that make use of the Fock basis representation."""
    def tor(self, A, fsum=False):
        """Returns the Torontonian of a matrix.

        For more direct control, you may wish to call :func:`tor_real` or
        :func:`tor_complex` directly.

        The input matrix is cast to quadruple precision
        internally for a quadruple precision torontonian computation.

        Args:
            A (array): a np.complex128, square, symmetric array of even dimensions.
            fsum (bool): if ``True``, the `Shewchuck algorithm <https://github.com/achan001/fsum>`_
                for more accurate summation is performed. This can significantly increase
                the `accuracy of the computation <https://link.springer.com/article/10.1007%2FPL00009321>`_,
                but no casting to quadruple precision takes place, as the Shewchuck algorithm
                only supports double precision.

        Returns:
           np.float64 or np.complex128: the torontonian of matrix A.
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("Input matrix must be a NumPy array.")

        matshape = A.shape

        if matshape[0] != matshape[1]:
            raise ValueError("Input matrix must be square.")

        if A.dtype == np.complex128:
            if np.any(np.iscomplex(A)):
                return self.tor_complex(A, fsum=fsum)
            return self.tor_real(np.float64(A.real), fsum=fsum)

        return self.tor_real(A, fsum=fsum)

    @numba.jit(nopython=True)
    def numba_ix(self, arr, rows, cols):
        # pragma: no cover
        """Numba implementation of `np.ix_`.
        Required due to numba lacking support for advanced numpy indexing.

        Args:
            arr (array) : matrix to take submatrix of
            rows (array) : rows to be selected in submatrix
            cols (array) : columns to be selected in submatrix

        Return:
           array: selected submatrix of arr, of shape `(len(rows), len(cols))`

        """
        return self.arr[rows][:, cols]
    
    @numba.jit(nopython=True)
    def nb_block(self, X):  # pragma: no cover
        """Numba implementation of `np.block`.
        Only suitable for 2x2 blocks.
        Taken from: https://stackoverflow.com/a/57562911
        Args:
        X (array) : arrays for blocks of matrix
        Return:
        array : the block matrix from X
        """
        xtmp1 = np.concatenate(X[0], axis=1)
        xtmp2 = np.concatenate(X[1], axis=1)
        return np.concatenate(self, (xtmp1, xtmp2), axis=0)

    @numba.jit(nopython=True)
    def Qmat_numba(self, cov, hbar=2):  # pragma: no cover
        r"""Numba compatible version of `thewalrus.quantum.Qmat`
        Returns the :math:`Q` Husimi matrix of the Gaussian state.
        Args:
        cov (array): :math:`2N\times 2N xp-` Wigner covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
            Returns:
            array: the :math:`Q` matrix.
        """
        # number of modes
        N = len(cov) // 2
        I = np.identity(N)
        x = cov[:N, :N] * (2.0 / hbar)
        xp = cov[:N, N:] * (2.0 / hbar)
        p = cov[N:, N:] * (2.0 / hbar)
        # the (Hermitian) matrix elements <a_i^\dagger a_j>
        aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
        # the (symmetric) matrix elements <a_i a_j>
        aiaj = (x - p + 1j * (xp + xp.T)) / 4
        # calculate the covariance matrix sigma_Q appearing in the Q function:
        Q = self.nb_block(((aidaj, aiaj.conj()), (aiaj, aidaj.conj()))) + np.identity(2 * N)
        return Q

    @numba.jit(nopython=True)
    def powerset(self, parent_set):  # pragma: no cover
        """Generates the powerset, the set of all the subsets, of its input. Does not include the empty set.

        Args:
        parent_set (Sequence) : sequence to take powerset from

        Return:
        subset (tuple) : subset of parent_set
        """
        n = len(parent_set)
        for i in range(n + 1):
            for subset in self.combinations(self, parent_set, i):
                yield subset

    @numba.jit(nopython=True)
    def threshold_detection_prob_displacement(self, mu, cov, det_pattern, hbar=2):
        # pragma: no cover
        r"""Threshold detection probabilities for Gaussian states with displacement.
        Formula from Jake Bulmer and Stefano Paesani.
        Args:
            mu (1d array) : means of xp Gaussian Wigner function
            cov (2d array) : : xp Wigner covariance matrix
            det_pattern (1d numpy array) : array of {0,1} to describe the threshold detection outcome
            hbar (float): the value of :math:`\hbar` in the commutation relation :math:`[\x,\p]=i\hbar`.
        Returns:
           np.float64 : probability of detection pattern
           """
        det_pattern = np.asarray(det_pattern).astype(np.int8)

        m = len(cov)
        assert cov.shape == (m, m)
        assert m % 2 == 0
        n = m // 2

        means_x = mu[:n]
        means_p = mu[n:]
        avec = np.concatenate((means_x + 1j * means_p, means_x - 1j * means_p), axis=0) / np.sqrt(
            2 * hbar
        )

        Q = self.Qmat_numba(cov, hbar=hbar)

        if max(det_pattern) > 1:
            raise ValueError(
                "When using threshold detectors, the detection pattern can contain only 1s or 0s."
            )

        nonzero_idxs = np.where(det_pattern == 1)[0]
        zero_idxs = np.where(det_pattern == 0)[0]

        ii1 = np.concatenate((nonzero_idxs, nonzero_idxs + n), axis=0)
        ii0 = np.concatenate((zero_idxs, zero_idxs + n), axis=0)

        Qaa = self.numba_ix(Q, ii0, ii0)
        Qab = self.numba_ix(Q, ii0, ii1)
        Qba = self.numba_ix(Q, ii1, ii0)
        Qbb = self.numba_ix(Q, ii1, ii1)

        Qaa_inv = np.linalg.inv(Qaa)
        Qcond = Qbb - Qba @ Qaa_inv @ Qab

        avec_a = avec[ii0]
        avec_b = avec[ii1]
        avec_cond = avec_b - Qba @ Qaa_inv @ avec_a

        p0a_fact_exp = np.exp(avec_a @ Qaa_inv @ avec_a.conj() * (-0.5)).real
        p0a_fact_det = np.sqrt(np.linalg.det(Qaa).real)
        p0a = p0a_fact_exp / p0a_fact_det

        n_det = len(nonzero_idxs)
        p_sum = 1.0  # empty set is not included in the powerset function so we start at 1
        for z in self.powerset(np.arange(n_det)):
            Z = np.asarray(z)
            ZZ = np.concatenate((Z, Z + n_det), axis=0)

            avec0 = avec_cond[ZZ]
            Q0 = self.numba_ix(Qcond, ZZ, ZZ)
            Q0inv = np.linalg.inv(Q0)

            fact_exp = np.exp(avec0 @ Q0inv @ avec0.conj() * (-0.5)).real
            fact_det = np.sqrt(np.linalg.det(Q0).real)

            p_sum += ((-1) ** len(Z)) * fact_exp / fact_det
        return p0a * p_sum

    def threshold_detection_prob(self, mu, cov, det_pattern, hbar=2, atol=1e-10, rtol=1e-10):
        # pylint: disable=too-many-arguments
        r"""Threshold detection probabilities for Gaussian states.
        Formula from Jake Bulmer and Stefano Paesani.
        When state is displaced, threshold_detection_prob_displacement is called.
        Otherwise, tor is called.
        Args:
            mu (1d array) : means of xp Gaussian Wigner function
            cov (2d array) : : xp Wigner covariance matrix
            det_pattern (1d array) : array of {0,1} to describe the threshold detection outcome
            hbar (float): the value of :math:`\hbar` in the commutation relation :math:`[\x,\p]=i\hbar`.
            rtol (float): the relative tolerance parameter used in `np.allclose`
            atol (float): the absolute tolerance parameter used in `np.allclose
            `
            Returns:
               np.float64 : probability of detection pattern
               """
        if np.allclose(mu, 0, atol=atol, rtol=rtol):
            # no displacement
            n_modes = cov.shape[0] // 2
            Q = Qmat(cov, hbar)
            O = Xmat(n_modes) @ Amat(cov, hbar=hbar)
            rpt2 = np.concatenate((det_pattern, det_pattern))
            Os = self.reduction(O, rpt2)
            return self.tor(Os) / np.sqrt(np.linalg.det(Q))
        det_pattern = np.asarray(det_pattern).astype(np.int8)
        return self.threshold_detection_prob_displacement(mu, cov, det_pattern, hbar)
    
    def measure_threshold(self, setup_backend):
        """Tests that threshold_detection backend is implemented"""
        backend = setup_backend(3)
        backend.measure_threshold([0, 1], shots=5)

@pytest.mark.backends("gaussian")
class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    def test_two_mode_squeezed_measurements(self, setup_backend, pure):
        """Tests Threshold measurement on the two mode squeezed vacuum state."""
        for _ in range(NUM_REPEATS):
            backend = setup_backend(2)
            backend.reset(pure=pure)

            r = 0.25
            # Circuit to prepare two mode squeezed vacuum
            backend.squeeze(r, np.pi, 0)
            backend.squeeze(r, 0, 1)
            backend.beamsplitter(np.pi/4, np.pi, 0, 1)
            meas_modes = [0, 1]
            meas_results = backend.measure_threshold(meas_modes)
            assert np.all(meas_results[0][0] == meas_results[0][1])

    def test_vacuum_measurements(self, setup_backend, pure):
        """Tests Threshold measurement on the vacuum state."""
        backend = setup_backend(3)

        for _ in range(NUM_REPEATS):
            backend.reset(pure=pure)

            meas = backend.measure_threshold([0, 1, 2])[0]
            assert np.all(np.array(meas) == 0)


    def test_binary_outcome(self, setup_backend, pure):
        """Test that the outcomes of a threshold measurement is zero or one."""
        num_modes = 2
        for _ in range(NUM_REPEATS):
            backend = setup_backend(num_modes)
            backend.reset(pure=pure)

            r = 0.5
            backend.squeeze(r, 0, 0)
            backend.beamsplitter(np.pi/4, np.pi, 0, 1)
            meas_modes = [0, 1]
            meas_results = backend.measure_threshold(meas_modes)
            for i in range(num_modes):
                assert meas_results[0][i] == 0 or meas_results[0][i] == 1


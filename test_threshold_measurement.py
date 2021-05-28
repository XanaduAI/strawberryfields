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

    @numba.jit(nopython=True)
    def numba_ix(arr, rows, cols):

        return arr[rows][:, cols]

    @numba.jit(nopython=True)
    def threshold_detection_prob_displacement(mu, cov, det_pattern, hbar=2):

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

        Q = Qmat_numba(cov, hbar=hbar)

        if max(det_pattern) > 1:
            raise ValueError(
                "When using threshold detectors, the detection pattern can contain only 1s or 0s."
            )

        nonzero_idxs = np.where(det_pattern == 1)[0]
        zero_idxs = np.where(det_pattern == 0)[0]

        ii1 = np.concatenate((nonzero_idxs, nonzero_idxs + n), axis=0)
        ii0 = np.concatenate((zero_idxs, zero_idxs + n), axis=0)

        Qaa = numba_ix(Q, ii0, ii0)
        Qab = numba_ix(Q, ii0, ii1)
        Qba = numba_ix(Q, ii1, ii0)
        Qbb = numba_ix(Q, ii1, ii1)

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
        for z in powerset(np.arange(n_det)):
            Z = np.asarray(z)
            ZZ = np.concatenate((Z, Z + n_det), axis=0)

            avec0 = avec_cond[ZZ]
            Q0 = numba_ix(Qcond, ZZ, ZZ)
            Q0inv = np.linalg.inv(Q0)

            fact_exp = np.exp(avec0 @ Q0inv @ avec0.conj() * (-0.5)).real
            fact_det = np.sqrt(np.linalg.det(Q0).real)

            p_sum += ((-1) ** len(Z)) * fact_exp / fact_det
        return p0a * p_sum

    def threshold_detection_prob(mu, cov, det_pattern, hbar=2, atol=1e-10, rtol=1e-10):

        if np.allclose(mu, 0, atol=atol, rtol=rtol):
            # no displacement
            n_modes = cov.shape[0] // 2
            Q = Qmat(cov, hbar)
            O = Xmat(n_modes) @ Amat(cov, hbar=hbar)
            rpt2 = np.concatenate((det_pattern, det_pattern))
            Os = reduction(O, rpt2)
            return tor(Os) / np.sqrt(np.linalg.det(Q))
        det_pattern = np.asarray(det_pattern).astype(np.int8)
        return threshold_detection_prob_displacement(mu, cov, det_pattern, hbar)
    
    
    def measure_threshold(self, setup_backend):
        

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


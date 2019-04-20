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

r"""Unit tests for homodyne measurements."""
import numpy as np


N_MEAS = 300  # number of homodyne measurements to perform
NUM_STDS = 10.0
std_10 = NUM_STDS / np.sqrt(N_MEAS)


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    def test_mode_reset_vacuum(self, setup_backend, tol):
        """Tests that modes get reset to the vacuum after measurement."""
        backend = setup_backend(1)

        r = np.arcsinh(1.0)  # pylint: disable=assignment-from-no-return
        alpha = 0.8

        backend.squeeze(r, 0)
        backend.displacement(alpha, 0)
        backend.measure_homodyne(0, 0)

        assert np.all(backend.is_vacuum(tol))

    def test_mean_and_std_vacuum(self, setup_backend, pure, tol):
        """Tests that the mean and standard deviation estimates of many homodyne
        measurements are in agreement with the expected values for the
        vacuum state"""
        x = np.empty(0)

        backend = setup_backend(1)

        for _ in range(N_MEAS):
            backend.reset(pure=pure)
            meas_result = backend.measure_homodyne(0, 0)
            x = np.append(x, meas_result)

        assert np.allclose(x.mean(), 0.0, atol=std_10 + tol, rtol=0)
        assert np.allclose(x.std(), 1.0, atol=std_10 + tol, rtol=0)

    def test_mean_coherent(self, setup_backend, pure, tol):
        """Tests that the mean and standard deviation estimates of many homodyne
        measurements are in agreement with the expected values for a
        coherent state"""
        alpha = 1.0 + 1.0j
        x = np.empty(0)

        backend = setup_backend(1)

        for _ in range(N_MEAS):
            backend.reset(pure=pure)
            backend.prepare_coherent_state(alpha, 0)

            meas_result = backend.measure_homodyne(0, 0)
            x = np.append(x, meas_result)

        assert np.allclose(x.mean(), 2 * alpha.real, atol=std_10 + tol)

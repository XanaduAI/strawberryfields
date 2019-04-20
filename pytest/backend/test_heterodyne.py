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
r"""Unit tests for heterodyne measurements."""
import pytest

import numpy as np

mag_alphas = np.linspace(0, 0.8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)
squeeze_val = np.arcsinh(1.0)

n_meas = 300
disp_val = 1.0 + 1j * 1.0
num_stds = 10.0
std_10 = num_stds / np.sqrt(n_meas)


@pytest.mark.backends("gaussian")
class TestHeterodyne:
    """Basic implementation-independent tests."""

    def test_mode_reset_vacuum(self, setup_backend, tol):
        """Test heterodyne measurement resets mode to vacuum"""
        backend = setup_backend(1)
        backend.squeeze(squeeze_val, 0)
        backend.displacement(mag_alphas[-1], 0)
        backend.measure_homodyne(0, 0)
        assert np.all(backend.is_vacuum(tol))

    def test_mean_vacuum(self, setup_backend, pure, tol):
        """Test heterodyne provides the correct vacuum mean"""
        backend = setup_backend(1)
        x = np.empty(0)

        for i in range(n_meas):
            backend.reset(pure=pure)
            meas_result = backend.measure_heterodyne(0)
            x = np.append(x, meas_result)

        mean_diff = np.abs(x.mean() - 0)

        assert np.allclose(x.mean(), 0.0, atol=std_10 + tol, rtol=0)

    def test_mean_coherent(self, setup_backend, pure, tol):
        """Test heterodyne provides the correct coherent mean alpha"""
        backend = setup_backend(1)
        x = np.empty(0)

        for i in range(n_meas):
            backend.reset(pure=pure)
            backend.prepare_coherent_state(disp_val, 0)
            meas_result = backend.measure_heterodyne(0)
            x = np.append(x, meas_result)

        assert np.allclose(x.mean(), disp_val, atol=std_10 + tol, rtol=0)

    def test_std_vacuum(self, setup_backend, pure, tol):
        """Test heterodyne provides the correct standard deviation of the vacuum"""
        backend = setup_backend(1)
        x = np.empty(0)

        for i in range(n_meas):
            backend.reset(pure=pure)
            meas_result = backend.measure_heterodyne(0)
            x = np.append(x, meas_result)

        xr = x.real
        xi = x.imag
        xvar = xi.std() ** 2 + xr.std() ** 2

        assert np.allclose(np.sqrt(xvar), np.sqrt(0.5), atol=std_10 + tol, rtol=0)

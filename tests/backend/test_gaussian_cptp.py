# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Unit test for Gaussian CPTP map
"""


import pytest

import numpy as np

LOSS_TS = np.linspace(0.0, 1, 3, endpoint=True)
DISPS = np.linspace(0.0, 5, 3, endpoint=True)
PHIS = np.linspace(0.0, np.pi / 2, 3, endpoint=True)


@pytest.mark.backends("bosonic")
@pytest.mark.parametrize("r", DISPS)
@pytest.mark.parametrize("phi", PHIS)
class TestGaussianCPTP:
    """Tests that Gaussian CPTP channels applied to bosonic states work."""

    @pytest.mark.parametrize("T", LOSS_TS)
    def test_cptp_vs_loss(self, T, r, phi, setup_backend, tol):
        """Tests loss channel and associated Gaussian CPTP produce same output."""
        backend = setup_backend(1)
        backend.prepare_coherent_state(r, phi, 0)
        backend.loss(T, 0)
        state1 = backend.state(0)
        backend.reset()
        backend.prepare_coherent_state(r, phi, 0)
        X = np.sqrt(T) * np.identity(2)
        Y = (1 - T) * np.identity(2)
        backend.gaussian_cptp(0, X, Y)
        state2 = backend.state(0)

        assert np.allclose(state1.means(), state2.means(), atol=tol, rtol=0)
        assert np.allclose(state1.covs(), state2.covs(), atol=tol, rtol=0)

    def test_cptp_vs_phaseshift(self, r, phi, setup_backend, tol):
        """Tests phase shift and associated Gaussian CPTP produce same output."""
        backend = setup_backend(1)
        backend.prepare_coherent_state(r, 0, 0)
        backend.rotation(phi, 0)
        state1 = backend.state(0)
        backend.reset()
        backend.prepare_coherent_state(r, 0, 0)
        X = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        backend.gaussian_cptp(0, X)
        state2 = backend.state(0)

        assert np.allclose(state1.means(), state2.means(), atol=tol, rtol=0)
        assert np.allclose(state1.covs(), state2.covs(), atol=tol, rtol=0)

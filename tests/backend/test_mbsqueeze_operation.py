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
r"""
Unit tests for squeezing operation
Convention: The squeezing unitary is fixed to be
U(z) = \exp(0.5 (z^* \hat{a}^2 - z (\hat{a^\dagger}^2)))
where \hat{a} is the photon annihilation operator.
"""
# pylint: disable=too-many-arguments
import pytest

import numpy as np


PHASE = np.linspace(0, 2 * np.pi, 3, endpoint=False)
MAG = np.linspace(0.0, 1, 5, endpoint=False)
ETA = np.linspace(0.5, 1, 3)
ANCMAG = np.linspace(0.0, 1, 5, endpoint=False)


@pytest.mark.backends("bosonic")
class TestBosonicRepresentation:
    """Basic implementation-independent tests."""

    @pytest.mark.parametrize("r_anc", ANCMAG)
    @pytest.mark.parametrize("eta", ETA)
    def test_avg_zero_mbsqueezing(self, setup_backend, r_anc, eta, tol):
        """Tests displacement operation where the result should be a vacuum state."""
        backend = setup_backend(2)
        backend.mbsqueeze(0, 0, 0, r_anc, eta, True)
        backend.squeeze(0, 0, 1)
        assert np.all(abs(backend.circuit.covs[0, :2, :2] - backend.circuit.covs[0, 2:, 2:]) < tol)

    @pytest.mark.parametrize("r", MAG)
    @pytest.mark.parametrize("phi", PHASE)
    def test_avg_mbsqueezing(self, setup_backend, r, phi, tol):
        """Tests displacement operation where the result should be a vacuum state."""
        backend = setup_backend(2)
        backend.mbsqueeze(0, r, phi, 9, 1.0, True)
        backend.squeeze(r, phi, 1)
        assert np.all(abs(backend.circuit.covs[0, :2, :2] - backend.circuit.covs[0, 2:, 2:]) < tol)

    @pytest.mark.parametrize("r_anc", ANCMAG)
    @pytest.mark.parametrize("eta", ETA)
    def test_singleshot_zero_mbsqueezing(self, setup_backend, r_anc, eta, tol):
        """Tests displacement operation where the result should be a vacuum state."""
        backend = setup_backend(2)
        backend.mbsqueeze(0, 0, 0, r_anc, eta, False)
        backend.squeeze(0, 0, 1)
        assert np.all(abs(backend.circuit.covs[0, :2, :2] - backend.circuit.covs[0, 2:, 2:]) < tol)

    @pytest.mark.parametrize("r", MAG)
    @pytest.mark.parametrize("phi", PHASE)
    def test_singleshot_mbsqueezing(self, setup_backend, r, phi, tol):
        """Tests displacement operation where the result should be a vacuum state."""
        backend = setup_backend(2)
        backend.mbsqueeze(0, r, phi, 9, 1.0, False)
        backend.squeeze(r, phi, 1)
        assert np.all(abs(backend.circuit.covs[0, :2, :2] - backend.circuit.covs[0, 2:, 2:]) < tol)

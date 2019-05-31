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

"""
Unit tests for the is_vacuum() check.
"""

import pytest

import numpy as np


MAG_ALPHAS = np.linspace(0, 0.8, 3)

class TestIsVacuum:
    """Common tests for all the backends."""
    @pytest.mark.parametrize("r", MAG_ALPHAS)
    def test_coherent_state(self, setup_backend, r, tol):
        """Tests if a range of displaced states are within the expected tolerance from vacuum.
        """
        # fidelity of the displaced state with |0>
        fid = np.exp(-np.abs(r)**2)
        # expected tolerance limit for the test
        lim = np.abs(fid-1)

        backend = setup_backend(1)
        backend.displacement(r, 0)
        assert not np.any(backend.is_vacuum(lim-tol))
        assert np.all(backend.is_vacuum(lim+tol))

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
Tests for strawberryfields.gbs.vibronic
"""
import numpy as np
import pytest

from strawberryfields.gbs import vibronic

pytestmark = pytest.mark.gbs


wp = np.array([700.0, 600.0, 500.0])

S1 = [[1, 1, 0], [1, 0, 2]]
E1 = [1300.0, 1700.0]

S2 = [[1, 1, 0]]
E2 = [1300.0]

S3 = [1, 1, 0]
E3 = 1300.0

@pytest.mark.parametrize("sample, sample_energy", [(S1, E1), (S2, E2), (S3, E3)])
def test_energies(sample, sample_energy):
    r"""Tests the correctness of the energies generated for GBS samples"""
    assert np.allclose(vibronic.energies(sample, wp), sample_energy)
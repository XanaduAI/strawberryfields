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
Tests for strawberryfields.apps.qchem.utils
"""
import numpy as np
import pytest

from strawberryfields.apps.qchem import utils

pytestmark = pytest.mark.apps

Li_1 = np.array([[-0.0872], [0.0000], [0.0000], [0.9534], [-0.0000], [-0.0000]])
Lf_1 = np.array([[-0.0872], [0.0000], [0.0000], [0.9534], [-0.0000], [-0.0000]])
ri_1 = np.array([-0.0236, 0.0000, 0.0000, 1.2236, 0.0000, 0.0000])
rf_1 = np.array([0.0000, 0.0000, 0.0000, 1.4397, 0.0000, 0.0000])
wi_1 = np.array([2330.4])
wf_1 = np.array([1363.2])
m_1 = np.array([11.0093] * 3 + [1.0078] * 3)

U_1 = np.array([0.99977449])
delta_1 = np.array([-1.17623073])

Li_2 = np.array(
    [
        [0.1166, -0.0242, 0.0000, 0.0081, 0.0045, 0.0002],
        [0.1090, 0.0165, 0.0000, -0.0060, 0.0075, -0.0000],
        [0.0000, 0.0000, -0.0225, 0.0000, 0.0000, 0.0000],
        [-0.1224, -0.0194, 0.0000, 0.0036, -0.0522, -0.0500],
        [-0.1133, -0.0372, 0.0000, -0.0604, -0.0406, 0.0326],
        [0.0000, 0.0000, -0.0370, 0.0000, 0.0000, 0.0000],
        [-0.1499, 0.2708, 0.0000, -0.4782, 0.7999, -0.0057],
        [-0.1678, -0.4037, 0.0000, 0.7037, 0.5267, 0.0314],
        [0.0000, 0.0000, 0.9795, 0.0000, 0.0000, 0.0000],
        [-0.1049, 0.4954, 0.0000, 0.2673, -0.0571, 0.7949],
        [-0.0889, 0.6827, 0.0000, 0.3694, -0.0235, -0.5501],
        [0.0000, 0.0000, 0.0344, 0.0000, 0.0000, 0.0000],
    ]
)
Lf_2 = np.array(
    [
        [0.1158, -0.0192, 0.0000, 0.0000, 0.0005, 0.0000],
        [0.1100, 0.0319, 0.0000, -0.0009, 0.0005, 0.0000],
        [0.0000, 0.0000, -0.0166, 0.0000, 0.0000, 0.0000],
        [-0.1195, -0.0266, 0.0000, 0.0110, -0.0136, -0.0661],
        [-0.1157, -0.0421, 0.0000, -0.0656, -0.0500, 0.0048],
        [0.0000, 0.0000, -0.0422, 0.0000, 0.0000, 0.0000],
        [-0.1505, 0.4602, 0.0000, -0.5005, 0.6243, 0.3006],
        [-0.0917, -0.5166, 0.0000, 0.4846, 0.5890, 0.3286],
        [0.0000, 0.0000, 0.9790, 0.0000, 0.0000, 0.0000],
        [-0.1356, 0.3266, 0.0000, 0.3238, -0.4178, 0.7474],
        [-0.1470, 0.5840, 0.0000, 0.5757, 0.1953, -0.4066],
        [0.0000, 0.0000, 0.0064, 0.0000, 0.0000, 0.0000],
    ]
)
ri_2 = np.array(
    [1.690, 1.573, -0.000, -0.067, -0.116, -0.000, 0.721, 0.5959, 0.000, -0.844, 0.447, 0.000]
)
rf_2 = np.array(
    [2.000, 2.000, -0.000, 0.017, -0.050, -0.000, 0.690, 0.639, 0.000, -0.819, 0.423, 0.000]
)
wf_2 = np.array([80.3, 143.9, 527.4, 1625.3, 3771.7, 3935.9])
m_2 = np.array([18.9] * 3 + [15.9] * 3 + [1.0] * 6)

U_2 = np.array(
    [
        [0.98804024, -0.08463515, 0.0, 0.02641942, 0.0418268, 0.00652302],
        [0.08253868, 0.94550407, 0.0, -0.2482175, 0.11579147, -0.08123613],
        [0.0, 0.0, 0.99103607, 0.0, 0.0, 0.0],
        [0.0034773, 0.25739985, 0.0, 0.94329876, -0.14403899, -0.08398276],
        [-0.04727852, -0.10866938, 0.0, 0.1236643, 0.87254883, -0.43971077],
        [-0.02246042, 0.0589729, 0.0, 0.12867759, 0.43216521, 0.88142107],
    ]
)
delta_2 = np.array([-1.98471372, -0.04722818, 0.0, 0.21337783, 0.76966615, 0.53978258])

p1 = [Li_1, Lf_1, ri_1, rf_1, wf_1, m_1, U_1, delta_1]
p2 = [Li_2, Lf_2, ri_2, rf_2, wf_2, m_2, U_2, delta_2]


@pytest.mark.parametrize("p", [p1, p2])
class TestDuschinsky:
    """Tests for the function ``strawberryfields.apps.qchem.vibronic.duschinsky``"""

    def test_duschinsky_parameters(self, p):
        """Test if function outputs the correct duschinsky parameters."""

        Li, Lf, ri, rf, wf, m, U, delta = p

        assert np.allclose(utils.duschinsky(Li, Lf, ri, rf, wf, m)[0], U)
        assert np.allclose(utils.duschinsky(Li, Lf, ri, rf, wf, m)[1], delta)

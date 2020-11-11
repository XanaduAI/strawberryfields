# Copyright 2020 Xanadu Quantum Technologies Inc.

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
import os

import numpy as np
import pytest

from strawberryfields.apps.qchem import utils

pytestmark = pytest.mark.apps

dir_path = os.path.dirname(os.path.realpath(__file__))

Li_1 = np.array([[-0.28933191], [0.0], [0.0], [0.95711104], [0.0], [0.0]])
Lf_1 = np.array([[-0.28933191], [0.0], [0.0], [0.95711104], [0.0], [0.0]])
ri_1 = np.array([-0.0236, 0.0, 0.0, 1.2236, 0.0, 0.0])
rf_1 = np.array([0.0, 0.0, 0.0, 1.4397, 0.0, 0.0])
wi_1 = np.array([2330.4])
wf_1 = np.array([1363.2])
m_1 = np.array([11.0093] * 3 + [1.0078] * 3)

U_1 = np.array([0.99977449])
delta_1 = np.array([-1.17623073])

Li_2 = np.array(
    [
        [5.06908359e-01, -1.05207395e-01, 0.0, 3.52140455e-02, 1.95633586e-02, 8.69482605e-04],
        [4.73868020e-01, 7.17323149e-02, 0.0, -2.60844781e-02, 3.26055977e-02, -0.0],
        [0.0, 0.0, -9.78167930e-02, 0.0, 0.0, 0.0],
        [-4.88067602e-01, -7.73571199e-02, 0.0, 1.43549295e-02, -2.08146477e-01, -1.99374020e-01],
        [-4.51781530e-01, -1.48334271e-01, 0.0, -2.40843817e-01, -1.61891705e-01, 1.29991861e-01],
        [0.0, 0.0, -1.47536775e-01, 0.0, 0.0, 0.0],
        [-1.4990e-01, 2.7080e-01, 0.0, -4.7820e-01, 7.9990e-01, -5.7000e-03],
        [-1.6780e-01, -4.0370e-01, 0.0, 7.0370e-01, 5.2670e-01, 3.1400e-02],
        [0.0, 0.0, 9.7950e-01, 0.0, 0.0, 0.0],
        [-1.0490e-01, 4.9540e-01, 0.0, 2.6730e-01, -5.7100e-02, 7.9490e-01],
        [-8.8900e-02, 6.8270e-01, 0.0, 3.6940e-01, -2.3500e-02, -5.5010e-01],
        [0.0, 0.0, 3.4400e-02, 0.0, 0.0, 0.0],
    ]
)

Lf_2 = np.array(
    [
        [0.50343043, -0.08347033, 0.0, 0.0, 0.00217371, 0.0],
        [0.47821543, 0.13868248, 0.0, -0.00391267, 0.00217371, 0.0],
        [0.0, 0.0, -0.07216706, 0.0, 0.0, 0.0],
        [-0.47650391, -0.10606698, 0.0, 0.04386228, -0.05422973, -0.26357245],
        [-0.46135148, -0.16787293, 0.0, -0.26157871, -0.19937402, 0.01913991],
        [0.0, 0.0, -0.16827167, 0.0, 0.0, 0.0],
        [-0.1505, 0.4602, 0.0, -0.5005, 0.6243, 0.3006],
        [-0.0917, -0.5166, 0.0, 0.4846, 0.589, 0.3286],
        [0.0, 0.0, 0.979, 0.0, 0.0, 0.0],
        [-0.1356, 0.3266, 0.0, 0.3238, -0.4178, 0.7474],
        [-0.147, 0.584, 0.0, 0.5757, 0.1953, -0.4066],
        [0.0, 0.0, 0.0064, 0.0, 0.0, 0.0],
    ]
)

ri_2 = np.array([1.690, 1.573, 0.0, -0.067, -0.116, 0.0, 0.721, 0.5959, 0.0, -0.844, 0.447, 0.0])
rf_2 = np.array([2.000, 2.000, 0.0, 0.017, -0.050, 0.0, 0.690, 0.639, 0.0, -0.819, 0.423, 0.0])
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
    """Tests for the function ``strawberryfields.apps.qchem.utils.duschinsky``"""

    def test_duschinsky_parameters(self, p):
        """Test if function outputs the correct duschinsky parameters."""

        Li, Lf, ri, rf, wf, m, U, delta = p

        assert np.allclose(utils.duschinsky(Li, Lf, ri, rf, wf, m)[0], U)
        assert np.allclose(utils.duschinsky(Li, Lf, ri, rf, wf, m)[1], delta)


class TestReadGAMESS:
    """Tests for the function ``strawberryfields.apps.qchem.utils.read_gamess``"""

    r1 = np.array([[0.0, 0.0, 0.0], [1.2536039, 0.0, 0.0]])
    m1 = np.array([11.00931, 1.00782])
    w1 = np.array([19.74, 19.73, 0.0, 0.0, 0.0, 2320.32])
    l1 = np.array(
        [
            [-0.000000e0, -7.532200e-4, -8.727621e-2, 0.000000e0, 8.228090e-3, 9.533905e-1],
            [-0.000000e0, -8.727621e-2, 7.532200e-4, 0.000000e0, 9.533905e-1, -8.228090e-3],
            [2.884692e-1, -2.000000e-8, 2.000000e-8, 2.884692e-1, -2.000000e-8, 2.000000e-8],
            [2.000000e-8, 2.884692e-1, -2.000000e-8, 2.000000e-8, 2.884692e-1, -2.000000e-8],
            [-2.000000e-8, 2.000000e-8, 2.884692e-1, -2.000000e-8, 2.000000e-8, 2.884692e-1],
            [-8.727946e-2, 0.000000e0, 0.000000e0, 9.534260e-1, -0.000000e0, -0.000000e0],
        ]
    )

    r2 = np.array([[0.0, 0.0, -0.77087574], [0.0, -0.0, 0.77087574]])
    m2 = np.array([7.016, 1.00782])
    w2 = np.array([2.2290e01, 2.2290e01, 1.1100e00, 1.1100e00, 1.1100e00, 1.6899e03])
    l2 = np.array(
        [
            [-9.313476e-2, -9.744043e-2, -1.100000e-7, 6.429051e-1, 6.726266e-1, -1.100000e-7],
            [-9.744023e-2, 9.313474e-2, 8.000000e-8, 6.726268e-1, -6.429052e-1, 8.000000e-8],
            [4.876110e-3, 7.181940e-3, 3.529211e-1, 4.917520e-3, 7.243290e-3, 3.529206e-1],
            [3.207159e-1, -1.466350e-1, -1.450200e-3, 3.234367e-1, -1.478789e-1, -1.450200e-3],
            [1.465611e-1, 3.206387e-1, -8.568300e-3, 1.478046e-1, 3.233594e-1, -8.568290e-3],
            [-0.000000e0, -0.000000e0, -1.338001e-1, 0.000000e0, 0.000000e0, 9.314543e-1],
        ]
    )

    p1 = ["BH_frq.out", r1, m1, w1, l1]
    p2 = ["lih_hessian_ccsd.out", r2, m2, w2, l2]

    @pytest.mark.parametrize("params", [p1, p2])
    def test_parameters(self, params):
        """Test if function outputs the correct parameters."""

        file, ri, mi, wi, li = params

        r, m, w, l = utils.read_gamess(os.path.join(dir_path, file))

        assert np.allclose(r, ri)
        assert np.allclose(m, mi)
        assert np.allclose(w, wi)
        assert np.allclose(l, li)

    def test_no_coordinates(self):
        """Test if function raises a ``ValueError`` when the atomic coordinates array is empty."""
        with pytest.raises(ValueError, match="No atomic coordinates found in the output file"):
            utils.read_gamess(os.path.join(dir_path, "gamess_dummy_r.out"))

    def test_no_masses(self):
        """Test if function raises a ``ValueError`` when the atomic masses array is empty."""
        with pytest.raises(ValueError, match="No atomic masses found in the output file"):
            utils.read_gamess(os.path.join(dir_path, "gamess_dummy_m.out"))

    def test_no_frequencies(self):
        """Test if function raises a ``ValueError`` when the frequencies array is empty."""
        with pytest.raises(ValueError, match="No vibrational frequencies found in the output file"):
            utils.read_gamess(os.path.join(dir_path, "gamess_dummy_w.out"))


class TestMarginals:
    """Tests for the function ``strawberryfields.apps.qchem.utils.marginals``"""

    mu = np.array([0.00000000, 2.82842712, 0.00000000, 0.00000000, 0.00000000, 0.00000000])
    V = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    n = 10
    p = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                1.35335284e-01,
                2.70670567e-01,
                2.70670566e-01,
                1.80447044e-01,
                9.02235216e-02,
                3.60894085e-02,
                1.20298028e-02,
                3.43708650e-03,
                8.59271622e-04,
                1.90949249e-04,
            ],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    def test_correct_marginals(self):
        """Test if the function returns the correct probabilities"""
        assert np.allclose(utils.marginals(self.mu, self.V, self.n), self.p)

    def test_square_covariance(self):
        """Test if function raises a ``ValueError`` when the covariance matrix is not square."""
        with pytest.raises(ValueError, match="The covariance matrix must be a square matrix"):
            utils.marginals(self.mu, np.hstack((self.V, np.ones((6, 1)))), self.n)

    def test_incorrect_modes(self):
        """Test if function raises a ``ValueError`` when the number of modes in the displacement
        vector and the covariance matrix are different."""
        with pytest.raises(
            ValueError, match="The dimension of the displacement vector and the covariance"
        ):
            utils.marginals(np.append(self.mu, 0), self.V, self.n)

    def test_incorrect_states(self):
        """Test if function raises a ``ValueError`` when the number of vibrational states is not
        larger than zero."""
        with pytest.raises(ValueError, match="The number of vibrational states must be larger"):
            utils.marginals(self.mu, self.V, 0)


sample1 = [[0, 2], [1, 1], [0, 2], [2, 0], [1, 1], [0, 2], [1, 1], [1, 1], [1, 1], [0, 2]]
sample2 = [
    [0, 2, 0],
    [1, 0, 1],
    [0, 0, 2],
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
    [0, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [0, 2, 0],
]
is1 = [0, 2]
is2 = [0, 2, 0]
prob1 = 0.4
prob2 = 0.3

e1 = [sample1, is1, prob1]
e2 = [sample2, is2, prob2]


@pytest.mark.parametrize("e", [e1, e2])
class TestProb:
    """Tests for the function ``strawberryfields.apps.qchem.dynamics.prob``"""

    def test_correct_prob(self, e):
        """Test if the function returns the correct probability"""
        samples, state, pref = e
        p = utils.prob(samples, state)

        assert p == pref

    def test_empty_samples(self, e):
        """Test if function raises a ``ValueError`` when the samples list is empty."""
        with pytest.raises(ValueError, match="The samples list must not be empty"):
            _, state, _ = e
            utils.prob([], state)

    def test_empty_state(self, e):
        """Test if function raises a ``ValueError`` when the given state is empty."""
        with pytest.raises(ValueError, match="The excited state list must not be empty"):
            samples, _, _ = e
            utils.prob(samples, [])

    def test_n_modes(self, e):
        """Test if function raises a ``ValueError`` when the number of modes in the samples and
        the state are different."""
        with pytest.raises(
            ValueError, match="The number of modes in the samples and the excited state must be"
        ):
            samples, state, _ = e
            utils.prob(samples, state + [0])

    def test_negative_state(self, e):
        """Test if function raises a ``ValueError`` when the given excited state contains negative
        values."""
        with pytest.raises(ValueError, match="The excited state must not contain negative values"):
            samples, state, _ = e
            utils.prob(samples, [-i for i in state])

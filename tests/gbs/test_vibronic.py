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


class TestGBSParams:
    """Tests for the function ``strawberryfields.gbs.vibronic.gbs_params``"""

    dim = 7
    w = np.array([3765.2386, 3088.1826, 1825.1799, 1416.9512, 1326.4684, 1137.0490, 629.7144])
    wp = np.array([3629.9472, 3064.9143, 1566.4602, 1399.6554, 1215.3421, 1190.9077, 496.2845])
    Ud = np.array(
        [
            [0.9934, 0.0144, 0.0153, 0.0268, 0.0638, 0.0751, -0.0428],
            [-0.0149, 0.9931, 0.0742, 0.0769, -0.0361, -0.0025, 0.0173],
            [-0.0119, -0.0916, 0.8423, 0.1799, -0.3857, 0.3074, 0.0801],
            [0.0381, 0.0409, -0.3403, -0.5231, -0.6679, 0.3848, 0.1142],
            [-0.0413, -0.0342, -0.4004, 0.7636, -0.1036, 0.4838, 0.0941],
            [0.0908, -0.0418, -0.0907, 0.3151, -0.5900, -0.7193, 0.1304],
            [-0.0325, 0.0050, -0.0206, 0.0694, -0.2018, 0.0173, -0.9759],
        ]
    )
    d = np.array([0.2254, 0.1469, 1.5599, -0.3784, 0.4553, -0.3439, 0.0618])

    T = 300.0

    t = np.array([1.19894657e-04, 6.07995705e-04, 1.25672831e-02, 3.34594410e-02, 4.15755027e-02,
                  6.55350936e-02, 2.24605404e-01])

    c = 299792458
    h = 6.62607015e-34
    k = 1.380649e-23

    U1, S, U2, alpha, t = vibronic.gbs_params(w, wp, Ud, d, T)

    @pytest.mark.parametrize("unitary", [U1, U2])
    def test_unitary(self, unitary):
        """Test if function outputs the interferometer unitaries that are valid unitaries,
        i.e., satisfying U U^dag = I"""
        assert unitary.shape == (self.dim, self.dim)
        assert np.allclose(unitary @ unitary.T.conj(), np.identity(self.dim))

    def test_displacement(self):
        """Test if function returns correct displacement parameters"""
        assert np.allclose(self.alpha * np.sqrt(2), self.d)

    def test_squeezing(self):
        """Test if function returns squeezing parameters as a vector rather than a diagonal
        matrix"""
        assert self.S.shape == (self.dim,)

    def test_duschinsky(self):
        """Test if function returns interferometer unitary and squeezing parameters that
        correctly reconstruct the input Duschinsky matrix"""
        sigma = np.diag(np.exp(self.S))
        J = self.U2 @ sigma @ self.U1
        Ud = np.diag(self.wp ** -0.5) @ J @ np.diag(self.w ** 0.5)
        assert np.allclose(Ud, self.Ud)

    def test_invalid_temperature(self):
        """Test if function raises a ``ValueError`` when a negative temperature is given."""
        with pytest.raises(ValueError, match="Temperature must be zero or positive"):
            vibronic.gbs_params(self.w, self.wp, self.Ud, self.d, -1)

    def test_twomode(self):
        """Test if function returns two-mode squeezing parameters that correctly reconstruct the
        input normal mode frequencies."""
        w = - self.k * self.T / (0.5 * h * w * c * 100) * np.log(np.tanh(self.t))
        assert np.allclose(w, self.w)


wp = np.array([700.0, 600.0, 500.0])

S1 = [[1, 1, 0], [1, 0, 2]]
E1 = [1300.0, 1700.0]

S2 = [[1, 1, 0]]
E2 = [1300.0]

S3 = [1, 1, 0]
E3 = 1300.0


@pytest.mark.parametrize("sample, sample_energy", [(S1, E1), (S2, E2), (S3, E3)])
def test_energies(sample, sample_energy):
    r"""Tests the correctness of the energies generated for GBS samples in
    ``strawberryfields.gbs.vibronic.energies``."""
    assert np.allclose(vibronic.energies(sample, wp), sample_energy)

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
Tests for strawberryfields.apps.qchem.vibronic
"""
# pylint: disable=protected-access,no-self-use,unused-argument,redefined-outer-name
from unittest import mock

import numpy as np
import pytest
from scipy.constants import c, h, k

import strawberryfields as sf
from strawberryfields.apps.qchem import vibronic

pytestmark = pytest.mark.apps


class TestGBSParams:
    """Tests for the function ``strawberryfields.apps.qchem.vibronic.gbs_params``"""

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

    t, U1, S, U2, alpha = vibronic.gbs_params(w, wp, Ud, d, T)

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
        w = -k * self.T / (0.5 * h * c * 100) * np.log(np.tanh(self.t))
        assert np.allclose(w, self.w)

    def test_zero_temperature(self):
        """Test if function returns zero two-mode squeezing parameters when temperature is zero."""
        t, _, _, _, _ = vibronic.gbs_params(self.w, self.wp, self.Ud, self.d, 0.0)
        assert np.all(t == 0)


w = np.array([300.0, 200.0, 100.0])
wp = np.array([700.0, 600.0, 500.0])

S1 = [[1, 1, 0, 0, 0, 0], [1, 2, 0, 0, 1, 1]]
E1 = [1300.0, 1600.0]

S2 = [[1, 2, 0, 0, 1, 1]]
E2 = [1600.0]

S3 = [1, 2, 0, 0, 1, 1]
E3 = 1600.0


@pytest.mark.parametrize("sample, sample_energy", [(S1, E1), (S2, E2), (S3, E3)])
def test_energies(sample, sample_energy):
    r"""Tests the correctness of the energies generated for GBS samples in
    ``strawberryfields.apps.qchem.vibronic.energies``."""
    assert np.allclose(vibronic.energies(sample, w, wp), sample_energy)


t0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

t1 = np.array([0.01095008, 0.02466257, 0.11257409, 0.18496601, 0.20673787, 0.26162544, 0.51007465])

U1 = np.array(
    [
        [-0.07985219, 0.66041032, -0.19389188, 0.01340832, 0.70312675, -0.1208423, -0.10352726],
        [0.19216669, -0.12470466, -0.81320519, 0.52045174, -0.1066017, -0.06300751, -0.00376173],
        [0.60838109, 0.0835063, -0.14958816, -0.34291399, 0.06239828, 0.68753918, -0.07955415],
        [0.63690134, -0.03047939, 0.46585565, 0.50545897, 0.21194805, -0.20422433, 0.18516987],
        [0.34556293, 0.22562207, -0.1999159, -0.50280235, -0.25510781, -0.55793978, 0.40065893],
        [-0.03377431, -0.66280536, -0.14740447, -0.25725325, 0.6145946, -0.07128058, 0.29804963],
        [-0.24570365, 0.22402764, 0.003273, 0.19204683, -0.05125235, 0.3881131, 0.83623564],
    ]
)

r = np.array([0.09721339, 0.07017918, 0.02083469, -0.05974357, -0.07487845, -0.1119975, -0.1866708])

U2 = np.array(
    [
        [-0.07012006, 0.14489772, 0.17593463, 0.02431155, -0.63151781, 0.61230046, 0.41087368],
        [0.5618538, -0.09931968, 0.04562272, 0.02158822, 0.35700706, 0.6614837, -0.326946],
        [-0.16560687, -0.7608465, -0.25644606, -0.54317241, -0.12822903, 0.12809274, -0.00597384],
        [0.01788782, 0.60430409, -0.19831443, -0.73270964, -0.06393682, 0.03376894, -0.23038293],
        [0.78640978, -0.11133936, 0.03160537, -0.09188782, -0.43483738, -0.4018141, 0.09582698],
        [-0.13664887, -0.11196486, 0.86353995, -0.19608061, -0.12313513, -0.08639263, -0.40251231],
        [-0.12060103, -0.01169781, -0.33937036, 0.34662981, -0.49895371, 0.03257453, -0.70709135],
    ]
)

alpha = np.array(
    [0.15938187, 0.10387399, 1.10301587, -0.26756921, 0.32194572, -0.24317402, 0.0436992]
)

p0 = [t0, U1, r, U2, alpha]

p1 = [t1, U1, r, U2, alpha]


@pytest.mark.parametrize("p", [p0, p1])
class TestSample:
    """Tests for the function ``strawberryfields.apps.qchem.sample``"""

    def test_invalid_n_samples(self, p):
        """Test if function raises a ``ValueError`` when a number of samples less than one is
        requested."""
        with pytest.raises(ValueError, match="Number of samples must be at least one"):
            vibronic.sample(*p, -1)

    def test_invalid_loss(self, p):
        """Test if function raises a ``ValueError`` when the loss parameter is specified outside
        of range."""
        with pytest.raises(ValueError, match="Loss parameter must take a value between zero and"):
            vibronic.sample(*p, 1, loss=2)

    def test_loss(self, monkeypatch, p):
        """Test if function correctly creates the SF program for lossy GBS."""

        def save_hist(*args, **kwargs):
            call_history.append(args[1])
            return sf.engine.Result

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(sf.engine.Result, "samples", np.array([[0]]))
            m.setattr(sf.LocalEngine, "run", save_hist)
            vibronic.sample(*p, 1, loss=0.5)

        assert isinstance(call_history[0].circuit[-2].op, sf.ops.LossChannel)

    def test_no_loss(self, monkeypatch, p):
        """Test if function correctly creates the SF program for GBS without loss."""

        def save_hist(*args, **kwargs):
            call_history.append(args[1])
            return sf.engine.Result

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(sf.engine.Result, "samples", np.array([[0]]))
            m.setattr(sf.LocalEngine, "run", save_hist)
            vibronic.sample(*p, 1)

        assert not all([isinstance(op, sf.ops.LossChannel) for op in call_history[0].circuit])

    def test_all_loss(self, monkeypatch, p):
        """Test if function samples from the vacuum when maximum loss is applied. This test is
        only done for the nonzero temperature case"""
        if not np.allclose(p[0], np.zeros(7)):
            dim = len(alpha)
            mock_eng_run = mock.MagicMock()

            with monkeypatch.context() as m:
                m.setattr(sf.LocalEngine, "run", mock_eng_run)
                vibronic.sample(*p, 1, loss=1)
                p_func = mock_eng_run.call_args[0][0]

            eng = sf.LocalEngine(backend="gaussian")

            state = eng.run(p_func).state
            cov = state.cov()
            disp = state.displacement()

            assert np.allclose(cov, 0.5 * state.hbar * np.eye(4 * dim))
            assert np.allclose(disp, np.zeros(2 * dim))


@pytest.mark.parametrize("p", [p0, p1])
@pytest.mark.parametrize("integration_sample_number", [1, 2])
def test_sample_integration(p, integration_sample_number):
    """Integration test for the function ``strawberryfields.apps.qchem.sample`` to check if
    it returns samples of correct form, i.e., correct number of samples, correct number of
    modes, all non-negative integers."""
    samples = np.array(vibronic.sample(*p, n_samples=integration_sample_number))

    dims = samples.shape

    assert len(dims) == 2
    assert dims == (integration_sample_number, len(alpha) * 2)
    assert samples.dtype == "int"
    assert (samples >= 0).all()


class TestOperation:
    """Tests for the function ``strawberryfields.apps.qchem.vibronic.VibronicTransition``"""

    U1 = np.array([[-0.17905859, 0.98383841], [0.98383841, 0.17905859]])

    r = np.array([0.00387885, -0.20415111])

    U2 = np.array([[-0.33417234, 0.94251199], [0.94251199, 0.33417234]])

    alpha = np.array([0.38558533, 2.99728422])

    prob = np.array(
        [
            [1.49817854e-04, 1.27328834e-03, 5.43505231e-03, 1.55353201e-02],
            [2.26762512e-06, 2.68454082e-05, 1.52080070e-04, 5.56423985e-04],
            [2.80701727e-06, 2.52203270e-05, 1.14702166e-04, 3.51760396e-04],
            [1.14518212e-07, 1.37859685e-06, 7.96673530e-06, 2.98259165e-05],
        ]
    )

    p1 = [U1, r, U2, alpha, prob]

    @pytest.mark.parametrize("params", [p1])
    def test_op_prob(self, params):
        """Test if the function ``strawberryfields.apps.qchem.vibronic.VibronicTransition`` gives
        the correct probabilities of all possible Fock basis states when used in a circuit"""

        U1, r, U2, alpha, prob = params

        eng = sf.LocalEngine(backend="gaussian")

        gbs = sf.Program(len(U1))

        with gbs.context as q:

            vibronic.VibronicTransition(U1, r, U2, alpha) | q

            p = eng.run(gbs).state.all_fock_probs(cutoff=4)

        assert np.allclose(p, prob)

    @pytest.mark.parametrize("params", [p1])
    def test_op_order(self, params):
        """Test if function ``strawberryfields.apps.qchem.vibronic.VibronicTransition``correctly
        applies the operations"""

        U1, r, U2, alpha, prob = params

        gbs = sf.Program(len(U1))

        with gbs.context as q:

            vibronic.VibronicTransition(U1, r, U2, alpha) | q

        assert isinstance(gbs.circuit[0].op, sf.ops.Interferometer)
        assert isinstance(gbs.circuit[1].op, sf.ops.Sgate)
        assert isinstance(gbs.circuit[2].op, sf.ops.Sgate)
        assert isinstance(gbs.circuit[3].op, sf.ops.Interferometer)
        assert isinstance(gbs.circuit[4].op, sf.ops.Dgate)
        assert isinstance(gbs.circuit[5].op, sf.ops.Dgate)

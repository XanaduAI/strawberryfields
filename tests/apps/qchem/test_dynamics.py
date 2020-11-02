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
Tests for strawberryfields.apps.qchem.dynamics
"""
from unittest import mock

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields.apps.qchem import dynamics

pytestmark = pytest.mark.apps

sampling_func = [dynamics.sample_fock, dynamics.sample_coherent, dynamics.sample_tmsv]

t1 = 10.0
t2 = 100.0
a = 1.0 / np.sqrt(2.0)
U1 = np.array([[a, -a], [a, a]])
U2 = np.array(
    [
        [-0.06822734, -0.96080077, -0.26871343],
        [-0.32019600, -0.23400449, 0.91799587],
        [0.94489129, -0.14867339, 0.29167906],
    ]
)
w1 = np.array([3914.92, 3787.59])
w2 = np.array([3914.92, 3787.59, 1627.01])
p1 = np.array(
    [
        [0.00000000e00, 0.00000000e00, 2.04851621e-04, 0.00000000e00],
        [0.00000000e00, 2.82155738e-02, 0.00000000e00, 0.00000000e00],
        [9.71579575e-01, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
    ]
)
p2 = np.array(
    [
        [
            [0.00000000, 0.00000000, 0.00530293, 0.00000000],
            [0.00000000, 0.03023791, 0.00000000, 0.00000000],
            [0.04310503, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000],
        ],
        [
            [0.00000000, 0.10479863, 0.00000000, 0.00000000],
            [0.29878705, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000],
        ],
        [
            [0.51776846, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000],
        ],
        [
            [0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000],
        ],
    ]
)
cf = 5
is1 = [0, 2]
is2 = [0, 2, 0]
ns1 = 1
ns2 = 2
d1 = [is1, t1, U1, w1, ns1, cf]
d2 = [is2, t2, U2, w2, ns2, cf]
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
prob1 = 0.4
prob2 = 0.3


sq1 = [[0.2, 0.1], [0.8, 0.2]]
sq2 = [[0.2, 0.1], [0.8, 0.2], [0.2, 0.1]]

tmsv1 = [sq1, t1, U1, w1, ns1]
tmsv2 = [sq2, t2, U2, w2, ns2]

alpha1 = [[0.3, 0.5], [1.4, 0.1]]
alpha2 = [[0.3, 0.5], [1.4, 0.1], [0.3, 0.5]]

c1 = [alpha1, t1, U1, w1, ns1]
c2 = [alpha2, t2, U2, w2, ns2]


@pytest.mark.parametrize("time, unitary, frequency, prob", [(t1, U1, w1, p1), (t2, U2, w2, p2)])
def test_evolution(time, unitary, frequency, prob):
    """Test if the function ``strawberryfields.apps.qchem.dynamics.TimeEvolution`` gives the correct
    probabilities of all possible Fock basis states when used in a circuit"""

    modes = len(unitary)

    eng = sf.Engine("fock", backend_options={"cutoff_dim": 4})
    prog = sf.Program(modes)

    with prog.context as q:

        sf.ops.Fock(2) | q[0]

        sf.ops.Interferometer(unitary.T) | q

        dynamics.TimeEvolution(frequency, time) | q

        sf.ops.Interferometer(unitary) | q

        p = eng.run(prog).state.all_fock_probs()

    assert np.allclose(p, prob)


@pytest.mark.parametrize("time, frequency", [(t2, w2)])
def test_evolution_order(time, frequency):
    """Test if function ``strawberryfields.apps.qchem.dynamics.TimeEvolution``correctly applies the
    operations"""

    modes = len(frequency)

    prog = sf.Program(modes)

    with prog.context as q:

        dynamics.TimeEvolution(frequency, time) | q

    assert isinstance(prog.circuit[0].op, sf.ops.Rgate)
    assert isinstance(prog.circuit[1].op, sf.ops.Rgate)
    assert isinstance(prog.circuit[2].op, sf.ops.Rgate)


@pytest.mark.parametrize("d", [d1, d2])
class TestSampleFock:
    """Tests for the function ``strawberryfields.apps.qchem.dynamics.sample_fock``"""

    def test_invalid_n_samples(self, d):
        """Test if function raises a ``ValueError`` when a number of samples less than one is
        requested."""
        with pytest.raises(ValueError, match="Number of samples must be at least one"):
            in_state, t, U, w, ns, cf = d
            dynamics.sample_fock(in_state, t, U, w, 0, cf)

    def test_complex_unitary(self, d):
        """Test if function raises a ``ValueError`` when a complex unitary is given."""
        with pytest.raises(
            ValueError, match="The normal mode to local mode transformation matrix must be real"
        ):
            in_state, t, U, w, ns, cf = d
            dynamics.sample_fock(in_state, t, 1.0j * U, w, ns, cf)

    def test_invalid_mode(self, d):
        """Test if function raises a ``ValueError`` when the number of modes in the input state and
        the normal-to-local transformation matrix are different."""
        with pytest.raises(
            ValueError,
            match="Number of modes in the input state and the normal-to-local transformation",
        ):
            in_state, t, U, w, ns, cf = d
            dynamics.sample_fock(in_state + [0], t, U, w, ns, cf)

    def test_negative_input_state(self, d):
        """Test if function raises a ``ValueError`` when input state contains negative values."""
        with pytest.raises(ValueError, match="Input state must not contain negative values"):
            in_state, t, U, w, ns, cf = d
            dynamics.sample_fock([-i for i in in_state], t, U, w, ns, cf)

    def test_invalid_input_photon(self, d):
        """Test if function raises a ``ValueError`` when the number of photons in each input state
        mode is not smaller than cutoff."""
        with pytest.raises(
            ValueError,
            match="Number of photons in each input mode must be smaller than cutoff",
        ):
            in_state, t, U, w, ns, cf = d
            dynamics.sample_fock([i * 2 for i in in_state], t, U, w, ns, 3)

    def test_op_order(self, monkeypatch, d):
        """Test if function correctly applies the operations."""
        if len(d[0]) == 2:
            mock_eng_run = mock.MagicMock()

            with monkeypatch.context() as m:
                m.setattr(sf.LocalEngine, "run", mock_eng_run)
                dynamics.sample_fock(*d)
                p_func = mock_eng_run.call_args[0][0]

            assert isinstance(p_func.circuit[0].op, sf.ops.Fock)
            assert isinstance(p_func.circuit[1].op, sf.ops.Fock)
            assert isinstance(p_func.circuit[2].op, sf.ops.Interferometer)
            assert isinstance(p_func.circuit[3].op, sf.ops.Rgate)
            assert isinstance(p_func.circuit[4].op, sf.ops.Rgate)
            assert isinstance(p_func.circuit[5].op, sf.ops.Interferometer)
            assert isinstance(p_func.circuit[6].op, sf.ops.MeasureFock)


e1 = [sample1, is1, prob1]
e2 = [sample2, is2, prob2]


@pytest.mark.parametrize("e", [e1, e2])
class TestProb:
    """Tests for the function ``strawberryfields.apps.qchem.dynamics.prob``"""

    def test_correct_prob(self, e):
        """Test if the function returns the correct probability"""
        samples, state, pref = e
        p = dynamics.prob(samples, state)

        assert p == pref

    def test_empty_samples(self, e):
        """Test if function raises a ``ValueError`` when the samples list is empty."""
        with pytest.raises(ValueError, match="The samples list must not be empty"):
            _, state, _ = e
            dynamics.prob([], state)

    def test_empty_state(self, e):
        """Test if function raises a ``ValueError`` when the given state is empty."""
        with pytest.raises(ValueError, match="The excited state list must not be empty"):
            samples, _, _ = e
            dynamics.prob(samples, [])

    def test_n_modes(self, e):
        """Test if function raises a ``ValueError`` when the number of modes in the samples and
        the state are different."""
        with pytest.raises(
            ValueError, match="The number of modes in the samples and the excited state must be"
        ):
            samples, state, _ = e
            dynamics.prob(samples, state + [0])

    def test_negative_state(self, e):
        """Test if function raises a ``ValueError`` when the given excited state contains negative
        values."""
        with pytest.raises(ValueError, match="The excited state must not contain negative values"):
            samples, state, _ = e
            dynamics.prob(samples, [-i for i in state])


@pytest.mark.parametrize("c", [c1, c2])
class TestSampleCoherent:
    """Tests for the function ``strawberryfields.apps.qchem.dynamics.sample_coherent``"""

    def test_invalid_n_samples(self, c):
        """Test if function raises a ``ValueError`` when a number of samples less than one is
        requested."""
        with pytest.raises(ValueError, match="Number of samples must be at least one"):
            alpha, t, U, w, ns = c
            dynamics.sample_coherent(alpha, t, U, w, 0)

    def test_complex_unitary(self, c):
        """Test if function raises a ``ValueError`` when a complex unitary is given."""
        with pytest.raises(
            ValueError, match="The normal mode to local mode transformation matrix must be real"
        ):
            alpha, t, U, w, ns = c
            dynamics.sample_coherent(alpha, t, 1.0j * U, w, ns)

    def test_invalid_mode(self, c):
        """Test if function raises a ``ValueError`` when the number of displacement parameters and the
        number of modes in the normal-to-local transformation matrix are different."""
        with pytest.raises(
            ValueError,
            match="Number of displacement parameters and the number of modes in the normal-to-local",
        ):
            alpha, t, U, w, ns = c
            dynamics.sample_coherent(alpha + [0], t, U, w, ns)

    def test_op_order(self, monkeypatch, c):
        """Test if function correctly applies the operations."""
        if len(c[0]) == 2:
            mock_eng_run = mock.MagicMock()

            with monkeypatch.context() as m:
                m.setattr(sf.LocalEngine, "run", mock_eng_run)
                dynamics.sample_coherent(*c)
                p_func = mock_eng_run.call_args[0][0]

            assert isinstance(p_func.circuit[0].op, sf.ops.Dgate)
            assert isinstance(p_func.circuit[1].op, sf.ops.Dgate)
            assert isinstance(p_func.circuit[2].op, sf.ops.Interferometer)
            assert isinstance(p_func.circuit[3].op, sf.ops.Rgate)
            assert isinstance(p_func.circuit[4].op, sf.ops.Rgate)
            assert isinstance(p_func.circuit[5].op, sf.ops.Interferometer)
            assert isinstance(p_func.circuit[6].op, sf.ops.MeasureFock)


@pytest.mark.parametrize("func, par", list(zip(sampling_func, [d1, c1, tmsv1])))
class TestCommon:
    """Common tests for all sample functions in the dynamics module, namely:
    ``strawberryfields.apps.qchem.dynamics.sample_fock``,
    ``strawberryfields.apps.qchem.dynamics.sample_coherent``,
    ``strawberryfields.apps.qchem.dynamics.sample_tmsv``"""

    def test_loss(self, monkeypatch, func, par):
        """Test if function correctly creates the SF program for lossy circuits."""

        def save_hist(*args, **kwargs):
            call_history.append(args[1])
            return sf.engine.Result

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(sf.engine.Result, "samples", np.array([[0]]))
            m.setattr(sf.LocalEngine, "run", save_hist)
            func(*par, loss=0.5)

        assert isinstance(call_history[0].circuit[-2].op, sf.ops.LossChannel)

    def test_no_loss(self, monkeypatch, func, par):
        """Test if sample function correctly creates the SF program for circuits without loss."""

        def save_hist(*args, **kwargs):
            call_history.append(args[1])
            return sf.engine.Result

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(sf.engine.Result, "samples", np.array([[0]]))
            m.setattr(sf.LocalEngine, "run", save_hist)
            func(*par)

        assert not all([isinstance(op, sf.ops.LossChannel) for op in call_history[0].circuit])

    def test_rgate(self, monkeypatch, func, par):
        """Test if sample function correctly uses the rotation parameter in the rgates."""

        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.LocalEngine, "run", mock_eng_run)
            func(*par)
            p_func = mock_eng_run.call_args[0][0]

        assert np.allclose(p_func.circuit[3].op.p, -7.374345193888777)
        assert np.allclose(p_func.circuit[4].op.p, -7.13449983982334)

    def test_interferometer(self, monkeypatch, func, par):
        """Test if sample function correctly uses the interferometer unitaries."""

        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.LocalEngine, "run", mock_eng_run)
            func(*par)
            p_func = mock_eng_run.call_args[0][0]

        U = par[2]

        assert np.allclose(p_func.circuit[2].op.p, U.T)
        assert np.allclose(p_func.circuit[5].op.p, U)

    def test_output(self, func, par):
        """Test to check if sample function returns correct output; correct
        number of samples, correct number of modes, all non-negative integers."""
        samples = np.array(func(*par))

        dims = samples.shape

        assert len(dims) == 2
        if func == dynamics.sample_tmsv:
            assert dims == (par[4], 2 * len(par[2]))
        else:
            assert dims == (par[4], len(par[2]))
        assert samples.dtype == "int"
        assert (samples >= 0).all()


class TestMarginals:
    """Tests for the function ``strawberryfields.apps.qchem.dynamics.marginals``"""

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
        assert np.allclose(dynamics.marginals(self.mu, self.V, self.n), self.p)

    def test_square_covariance(self):
        """Test if function raises a ``ValueError`` when the covariance matrix is not square."""
        with pytest.raises(ValueError, match="The covariance matrix must be a square matrix"):
            dynamics.marginals(self.mu, np.hstack((self.V, np.ones((6, 1)))), self.n)

    def test_incorrect_modes(self):
        """Test if function raises a ``ValueError`` when the number of modes in the displacement
        vector and the covariance matrix are different."""
        with pytest.raises(
            ValueError, match="The dimension of the displacement vector and the covariance"
        ):
            dynamics.marginals(np.append(self.mu, 0), self.V, self.n)

    def test_incorrect_states(self):
        """Test if function raises a ``ValueError`` when the number of vibrational states is not
        larger than zero."""
        with pytest.raises(ValueError, match="The number of vibrational states must be larger"):
            dynamics.marginals(self.mu, self.V, 0)


@pytest.mark.parametrize("tmsv", [tmsv1, tmsv2])
class TestSampleTMSV:
    """Tests for the function ``strawberryfields.apps.qchem.dynamics.sample_tmsv``"""

    def test_invalid_n_samples(self, tmsv):
        """Test if function raises a ``ValueError`` when a number of samples less than one is
        requested."""
        with pytest.raises(ValueError, match="Number of samples must be at least one"):
            r, t, U, w, ns = tmsv
            dynamics.sample_tmsv(r, t, U, w, 0)

    def test_complex_unitary(self, tmsv):
        """Test if function raises a ``ValueError`` when a complex unitary is given."""
        with pytest.raises(
            ValueError, match="The normal mode to local mode transformation matrix must be real"
        ):
            r, t, U, w, ns = tmsv
            dynamics.sample_tmsv(r, t, 1.0j * U, w, ns)

    def test_invalid_mode(self, tmsv):
        """Test if function raises a ``ValueError`` when the number of modes in the input state and
        the normal-to-local transformation matrix are different."""
        with pytest.raises(
            ValueError,
            match="Number of squeezing parameters and the number of modes in the normal-to-local",
        ):
            r, t, U, w, ns = tmsv
            dynamics.sample_tmsv(r + [[0, 0]], t, U, w, ns)

    def test_op_order(self, monkeypatch, tmsv):
        """Test if function correctly applies the operations."""
        if len(tmsv[0]) == 2:
            mock_eng_run = mock.MagicMock()

            with monkeypatch.context() as m:
                m.setattr(sf.LocalEngine, "run", mock_eng_run)
                dynamics.sample_tmsv(*tmsv)
                p_func = mock_eng_run.call_args[0][0]

            assert isinstance(p_func.circuit[0].op, sf.ops.S2gate)
            assert isinstance(p_func.circuit[1].op, sf.ops.S2gate)
            assert isinstance(p_func.circuit[2].op, sf.ops.Interferometer)
            assert isinstance(p_func.circuit[3].op, sf.ops.Rgate)
            assert isinstance(p_func.circuit[4].op, sf.ops.Rgate)
            assert isinstance(p_func.circuit[5].op, sf.ops.Interferometer)
            assert isinstance(p_func.circuit[6].op, sf.ops.MeasureFock)

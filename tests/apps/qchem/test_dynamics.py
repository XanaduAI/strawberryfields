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
import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields.apps.qchem import dynamics

pytestmark = pytest.mark.apps


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
is1 = [0, 2]
is2 = [0, 2, 0]
ns1 = 1
ns2 = 2
d1 = [is1, t1, U1, w1, ns1]
d2 = [is2, t2, U2, w2, ns2]


@pytest.mark.parametrize("time, unitary, frequency, prob", [(t1, U1, w1, p1), (t2, U2, w2, p2)])
def test_evolution(time, unitary, frequency, prob):
    """Test if the function ``strawberryfields.apps.qchem.dynamics.evolution`` gives the correct
    probabilities of all possible Fock basis states when used in a circuit"""

    modes = len(unitary)
    op = dynamics.evolution(modes)

    eng = sf.Engine("fock", backend_options={"cutoff_dim": 4})
    gbs = sf.Program(modes)

    with gbs.context as q:

        sf.ops.Fock(2) | q[0]

        op(time, unitary, frequency) | q

        p = eng.run(gbs).state.all_fock_probs()

    assert np.allclose(p, prob)


@pytest.mark.parametrize("d", [d1, d2])
class TestSampleFock:
    """Tests for the function ``strawberryfields.apps.qchem.dynamics.sample_fock``"""

    def test_invalid_n_samples(self, d):
        """Test if function raises a ``ValueError`` when a number of samples less than one is
        requested."""
        with pytest.raises(ValueError, match="Number of samples must be at least one"):
            in_state, t, U, w, ns = d
            dynamics.sample_fock(in_state, t, U, w, 0)

    def test_invalid_time(self, d):
        """Test if function raises a ``ValueError`` when a negative time is given."""
        with pytest.raises(ValueError, match="Time must be zero or positive"):
            in_state, t, U, w, ns = d
            dynamics.sample_fock(in_state, -t, U, w, ns)

    def test_negative_frequency(self, d):
        """Test if function raises a ``ValueError`` when negative frequencies are given."""
        with pytest.raises(ValueError, match="Vibrational frequencies must be larger than zero"):
            in_state, t, U, w, ns = d
            dynamics.sample_fock(in_state, t, U, -w, ns)

    def test_zero_frequency(self, d):
        """Test if function raises a ``ValueError`` when zero frequencies are given."""
        with pytest.raises(ValueError, match="Vibrational frequencies must be larger than zero"):
            in_state, t, U, w, ns = d
            dynamics.sample_fock(in_state, t, U, 0.0 * w, ns)

    def test_complex_unitary(self, d):
        """Test if function raises a ``ValueError`` when a complex unitary is given."""
        with pytest.raises(
            ValueError, match="The normal mode to local mode transformation matrix must be real"
        ):
            in_state, t, U, w, ns = d
            dynamics.sample_fock(in_state, t, 1.0j * U, w, ns)

    def test_invalid_loss(self, d):
        """Test if function raises a ``ValueError`` when the loss parameter is specified outside
        of range."""
        with pytest.raises(
            ValueError, match="Loss parameter must take a value between zero and one"
        ):
            in_state, t, U, w, ns = d
            dynamics.sample_fock(in_state, t, U, w, ns, w[-1])

    def test_loss(self, monkeypatch, d):
        """Test if function correctly creates the SF program for lossy circuits."""

        def save_hist(*args):
            call_history.append(args[1])
            return sf.engine.Result

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(sf.engine.Result, "samples", np.array([[0]]))
            m.setattr(sf.LocalEngine, "run", save_hist)
            dynamics.sample_fock(*d, loss=0.5)

    def test_no_loss(self, monkeypatch, d):
        """Test if function correctly creates the SF program for circuits without loss."""

        def save_hist(*args):
            call_history.append(args[1])
            return sf.engine.Result

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(sf.engine.Result, "samples", np.array([[0]]))
            m.setattr(sf.LocalEngine, "run", save_hist)
            dynamics.sample_fock(*d)


@pytest.mark.parametrize("d", [d1, d2])
def test_fock_integration(d):
    """Integration test for the function ``strawberryfields.apps.qchem.dynamics.sample_fock`` to
    check if it returns samples of correct form, i.e., correct number of samples, correct number of
    modes, all non-negative integers."""
    samples = np.array(dynamics.sample_fock(*d))

    dims = samples.shape

    assert len(dims) == 2
    assert dims == (d[4], len(d[2]))
    assert samples.dtype == "int"
    assert (samples >= 0).all()

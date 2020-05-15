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
Unit tests for strawberryfields.apps.sample
"""
# pylint: disable=no-self-use,unused-argument,protected-access
from unittest import mock

import networkx as nx
import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields.apps import sample

pytestmark = pytest.mark.apps

adj_dim_range = range(2, 6)

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


@pytest.mark.parametrize("dim", [4])
class TestSample:
    """Tests for the function ``strawberryfields.apps.sample.sample``"""

    def test_invalid_adjacency(self, dim):
        """Test if function raises a ``ValueError`` for a matrix that is not symmetric"""
        with pytest.raises(ValueError, match="Input must be a NumPy array"):
            adj_asym = np.triu(np.ones((dim, dim)))
            sample.sample(A=adj_asym, n_mean=1.0)

    def test_invalid_n_samples(self, adj):
        """Test if function raises a ``ValueError`` when a number of samples less than one is
        requested """
        with pytest.raises(ValueError, match="Number of samples must be at least one"):
            sample.sample(A=adj, n_mean=1.0, n_samples=0)

    def test_invalid_n_mean(self, adj):
        """Test if function raises a ``ValueError`` when the mean photon number is specified to
        be negative."""
        with pytest.raises(ValueError, match="Mean photon number must be non-negative"):
            sample.sample(A=adj, n_mean=-1.0, n_samples=1)

    def test_invalid_loss(self, adj):
        """Test if function raises a ``ValueError`` when the loss parameter is specified outside
        of range"""
        with pytest.raises(ValueError, match="Loss parameter must take a value between zero and"):
            sample.sample(A=adj, n_mean=1.0, n_samples=1, loss=2)

    def test_threshold(self, monkeypatch, adj):
        """Test if function correctly creates the SF program for threshold GBS."""
        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.LocalEngine, "run", mock_eng_run)
            sample.sample(A=adj, n_mean=1, threshold=True)
            p_func = mock_eng_run.call_args[0][0]

        assert isinstance(p_func.circuit[-1].op, sf.ops.MeasureThreshold)

    def test_pnr(self, monkeypatch, adj):
        """Test if function correctly creates the SF program for photon-number resolving GBS."""
        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.LocalEngine, "run", mock_eng_run)
            sample.sample(A=adj, n_mean=1, threshold=False)
            p_func = mock_eng_run.call_args[0][0]

        assert isinstance(p_func.circuit[-1].op, sf.ops.MeasureFock)

    def test_loss(self, monkeypatch, adj):
        """Test if function correctly creates the SF program for lossy GBS."""
        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.LocalEngine, "run", mock_eng_run)
            sample.sample(A=adj, n_mean=1, threshold=False, loss=0.5)
            p_func = mock_eng_run.call_args[0][0]

        assert isinstance(p_func.circuit[-2].op, sf.ops.LossChannel)

    def test_no_loss(self, monkeypatch, adj):
        """Test if function correctly creates the SF program for GBS without loss."""
        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.LocalEngine, "run", mock_eng_run)
            sample.sample(A=adj, n_mean=1, threshold=False)
            p_func = mock_eng_run.call_args[0][0]

        assert not all([isinstance(op, sf.ops.LossChannel) for op in p_func.circuit])

    def test_all_loss(self, monkeypatch, adj, dim):
        """Test if function samples from the vacuum when maximum loss is applied."""
        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.LocalEngine, "run", mock_eng_run)
            sample.sample(A=adj, n_mean=1, threshold=False, loss=1)
            p_func = mock_eng_run.call_args[0][0]

        eng = sf.LocalEngine(backend="gaussian")

        state = eng.run(p_func).state
        cov = state.cov()
        disp = state.displacement()

        assert np.allclose(cov, 0.5 * state.hbar * np.eye(2 * dim))
        assert np.allclose(disp, np.zeros(dim))


@pytest.mark.parametrize("dim", adj_dim_range)
@pytest.mark.parametrize("integration_sample_number", [1, 2])
class TestSampleIntegration:
    """Integration tests for the function ``strawberryfields.apps.sample.sample``"""

    def test_pnr_integration(self, adj, integration_sample_number):
        """Integration test to check if function returns samples of correct form, i.e., correct
        number of samples, correct number of modes, all non-negative integers """
        samples = np.array(
            sample.sample(A=adj, n_mean=1.0, n_samples=integration_sample_number, threshold=False)
        )

        dims = samples.shape

        assert len(dims) == 2
        assert dims == (integration_sample_number, len(adj))
        assert samples.dtype == "int"
        assert (samples >= 0).all()

    def test_threshold_integration(self, adj, integration_sample_number):
        """Integration test to check if function returns samples of correct form, i.e., correct
        number of samples, correct number of modes, all integers of zeros and ones """
        samples = np.array(
            sample.sample(A=adj, n_mean=1.0, n_samples=integration_sample_number, threshold=True)
        )

        dims = samples.shape

        assert len(dims) == 2
        assert dims == (integration_sample_number, len(adj))
        assert samples.dtype == "int"
        assert (samples >= 0).all()
        assert (samples <= 1).all()


@pytest.mark.parametrize("dim", [4])
def test_seed(dim, adj):
    """Test for the function ``strawberryfields.apps.sample.seed``. Checks that samples are
    identical after repeated initialization of ``seed``."""

    sample.seed(1968)
    q_s_1 = sample.sample(A=adj, n_mean=2, n_samples=10, threshold=False)

    sample.seed(1968)
    q_s_2 = sample.sample(A=adj, n_mean=2, n_samples=10, threshold=False)

    assert np.array_equal(q_s_1, q_s_2)


@pytest.mark.parametrize("dim", [6])
class TestToSubgraphs:
    """Tests for the function ``sample.to_subgraphs``"""

    quantum_samples = [
        [0, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
    ]

    subgraphs = [
        [1, 2, 3, 4, 5],
        [0, 3],
        [0, 1, 2, 3, 4, 5],
        [2, 5],
        [1, 2, 3],
        [4, 5],
        [4, 5],
        [0, 1, 3, 4, 5],
        [0, 3],
        [4, 5],
    ]

    def test_graph(self, graph):
        """Test if function returns correctly processed subgraphs given input samples of the list
        ``quantum_samples``."""
        assert sample.to_subgraphs(self.quantum_samples, graph) == self.subgraphs

    def test_graph_mapped(self, graph):
        """Test if function returns correctly processed subgraphs given input samples of the list
        ``quantum_samples``. Note that graph nodes are numbered in this test as [0, 1, 4, 9,
        ...] (i.e., squares of the usual list) as a simple mapping to explore that the optimised
        subgraph returned is still a valid subgraph."""
        graph = nx.relabel_nodes(graph, lambda x: x ** 2)
        graph_nodes = list(graph.nodes)
        subgraphs_mapped = [
            sorted([graph_nodes[i] for i in subgraph]) for subgraph in self.subgraphs
        ]

        assert sample.to_subgraphs(self.quantum_samples, graph) == subgraphs_mapped


def test_modes_from_counts():
    """Test if the function ``strawberryfields.apps.sample.modes_from_counts`` returns the correct
    mode samples when input a set of photon count samples."""

    counts = [[0, 0, 0, 0], [1.0, 0.0, 0.0, 2.0], [1, 1, 1, 0], [1, 2, 1, 0], [0, 1, 0, 2, 4]]

    modes = [[], [0, 3, 3], [0, 1, 2], [0, 1, 1, 2], [1, 3, 3, 4, 4, 4, 4]]

    assert [sample.modes_from_counts(s) for s in counts] == modes


def test_postselect():
    """Test if the function ``strawberryfields.apps.sample.postselect`` correctly postselects on
    minimum number of photons or clicks."""
    counts_pnr = [
        [0, 0, 0, 0],
        [1, 1, 0, 2],
        [1, 1, 1, 0],
        [1, 2, 1, 0],
        [1, 0, 0, 1],
        [5, 0, 0, 0],
        [1, 2, 1, 2],
    ]
    counts_pnr_ps_4_5 = [[1, 1, 0, 2], [1, 2, 1, 0], [5, 0, 0, 0]]

    counts_threshold = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ]
    counts_threshold_ps_3_3 = [[1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 0]]

    assert sample.postselect(counts_pnr, 4, 5) == counts_pnr_ps_4_5
    assert sample.postselect(counts_threshold, 3, 3) == counts_threshold_ps_3_3


@pytest.mark.parametrize("p", [p0, p1])
class TestVibronic:
    """Tests for the function ``strawberryfields.apps.sample.vibronic``"""

    def test_invalid_n_samples(self, p):
        """Test if function raises a ``ValueError`` when a number of samples less than one is
        requested."""
        with pytest.raises(ValueError, match="Number of samples must be at least one"):
            sample.vibronic(*p, -1)

    def test_invalid_loss(self, p):
        """Test if function raises a ``ValueError`` when the loss parameter is specified outside
        of range."""
        with pytest.raises(ValueError, match="Loss parameter must take a value between zero and"):
            sample.vibronic(*p, 1, loss=2)

    def test_loss(self, monkeypatch, p):
        """Test if function correctly creates the SF program for lossy GBS."""
        def save_hist(*args, **kwargs):
            call_history.append(args[1])
            return sf.engine.Result

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(sf.engine.Result, "samples", np.array([[0]]))
            m.setattr(sf.LocalEngine, "run", save_hist)
            sample.vibronic(*p, 1, loss=0.5)

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
            sample.vibronic(*p, 1)

        assert not all([isinstance(op, sf.ops.LossChannel) for op in call_history[0].circuit])

    def test_all_loss(self, monkeypatch, p):
        """Test if function samples from the vacuum when maximum loss is applied. This test is
        only done for the zero temperature case"""
        if p == "p0":
            dim = len(alpha)
            mock_eng_run = mock.MagicMock()

            with monkeypatch.context() as m:
                m.setattr(sf.LocalEngine, "run", mock_eng_run)
                sample.vibronic(*p, 1, loss=1)
                p_func = mock_eng_run.call_args[0][0]

            eng = sf.LocalEngine(backend="gaussian")

            state = eng.run(p_func).state
            cov = state.cov()
            disp = state.displacement()

            assert np.allclose(cov, 0.5 * state.hbar * np.eye(2 * dim))
            assert np.allclose(disp, np.zeros(dim))


@pytest.mark.parametrize("p", [p0, p1])
@pytest.mark.parametrize("integration_sample_number", [1, 2])
def test_vibronic_integration(p, integration_sample_number):
    """Integration test for the function ``strawberryfields.apps.sample.vibronic`` to check if
    it returns samples of correct form, i.e., correct number of samples, correct number of
    modes, all non-negative integers."""
    samples = np.array(sample.vibronic(*p, n_samples=integration_sample_number))

    dims = samples.shape

    assert len(dims) == 2
    assert dims == (integration_sample_number, len(alpha) * 2)
    assert samples.dtype == "int"
    assert (samples >= 0).all()


class TestWawMatrix:
    """Tests for the function ``strawberryfields.apps.sample.waw_matrix``"""

    adjs = [
        np.array(
            [[0, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 0, 0, 1, 0], [1, 1, 1, 0, 1], [1, 1, 0, 1, 0]]
        ),
        np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]),
        np.array([[0, 1, 2], [1, 0, 0.5], [2, 0.5, 0]]),
    ]

    wvecs = [np.array([1, 1, 3, 1, 0.5]), np.array([1, -2, 3, -4]), np.array([0, 0.5, 4])]

    resc_adjs = [
        np.array(
            [
                [0, 1, 0, 1, 0.5],
                [1, 0, 0, 1, 0.5],
                [0, 0, 0, 3, 0],
                [1, 1, 3, 0, 0.5],
                [0.5, 0.5, 0, 0.5, 0],
            ]
        ),
        np.array([[0, -2, 0, -4], [-2, 0, -6, 0], [0, -6, 0, -12], [-4, 0, -12, 0]]),
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    ]

    def test_invalid_adjacency(self):
        """Test if function raises a ``ValueError`` for a matrix that is not symmetric."""
        adj_asym = np.triu(np.ones((4, 4)))
        with pytest.raises(ValueError, match="Input must be a NumPy array"):
            sample.sample(A=adj_asym, n_mean=1.0)

    @pytest.mark.parametrize("inst", zip(adjs, wvecs, resc_adjs))
    def test_valid_w(self, inst):
        """Test if function returns the correct answer on some pre-calculated instances."""
        assert np.allclose(sample.waw_matrix(inst[0], inst[1]), inst[2])

    @pytest.mark.parametrize("inst", zip(adjs, wvecs, resc_adjs))
    def test_valid_w_list(self, inst):
        """Test if function returns the correct answer on some pre-calculated instances,
        when w is a list"""
        assert np.allclose(sample.waw_matrix(inst[0], list(inst[1])), inst[2])

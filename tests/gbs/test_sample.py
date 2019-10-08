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
Unit tests for strawberryfields.gbs.sample
"""
# pylint: disable=no-self-use,unused-argument,protected-access
from unittest import mock

import blackbird
import networkx as nx
import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields.gbs import sample

pytestmark = pytest.mark.gbs

integration_sample_number = 2
adj_dim_range = range(2, 6)


@pytest.mark.parametrize("dim", [4])
class TestSample:
    """Tests for the function ``strawberryfields.gbs.sample.sample``"""

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

    @pytest.mark.parametrize("n_mean", [1, 2, 3])
    def test_threshold(self, monkeypatch, dim, adj, n_mean):
        """Test if function correctly creates the SF program for threshold GBS."""
        mean_photon_per_mode = n_mean / float(dim)
        p = sf.Program(dim)

        p.append(
            sf.ops.GraphEmbed(adj, mean_photon_per_mode=mean_photon_per_mode), list(range(dim))
        )
        p.append(sf.ops.MeasureThreshold(), list(range(dim)))

        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.Engine, "run", mock_eng_run)
            sample.sample(A=adj, n_mean=1, threshold=True)
            p_func = mock_eng_run.call_args[0][0]

        b = blackbird.dumps(sf.io.to_blackbird(p))
        b_func = blackbird.dumps(sf.io.to_blackbird(p_func))

        assert b == b_func

    @pytest.mark.parametrize("n_mean", [1, 2, 3])
    def test_pnr(self, monkeypatch, dim, adj, n_mean):
        """Test if function correctly creates the SF program for photon-number resolving GBS."""
        mean_photon_per_mode = n_mean / float(dim)
        p = sf.Program(dim)

        p.append(
            sf.ops.GraphEmbed(adj, mean_photon_per_mode=mean_photon_per_mode), list(range(dim))
        )
        p.append(sf.ops.MeasureFock(), list(range(dim)))

        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.Engine, "run", mock_eng_run)
            sample.sample(A=adj, n_mean=1, threshold=False)
            p_func = mock_eng_run.call_args[0][0]

        b = blackbird.dumps(sf.io.to_blackbird(p))
        b_func = blackbird.dumps(sf.io.to_blackbird(p_func))

        assert b == b_func


@pytest.mark.parametrize("dim", adj_dim_range)
class TestSampleIntegration:
    """Integration tests for the function ``strawberryfields.gbs.sample.sample``"""

    def test_pnr_integration(self, adj):
        """Integration test to check if function returns samples of correct form, i.e., correct
        number of samples, correct number of modes, all non-negative integers """
        samples = np.array(
            sample.sample(A=adj, n_mean=1.0, n_samples=integration_sample_number, threshold=False)
        )

        dims = samples.shape

        assert len(dims) == 2
        assert dims[0] == integration_sample_number
        assert dims[1] == len(adj)
        assert samples.dtype == "int"
        assert (samples >= 0).all()

    def test_threshold_integration(self, adj):
        """Integration test to check if function returns samples of correct form, i.e., correct
        number of samples, correct number of modes, all integers of zeros and ones """
        samples = np.array(
            sample.sample(A=adj, n_mean=1.0, n_samples=integration_sample_number, threshold=True)
        )

        dims = samples.shape

        assert len(dims) == 2
        assert dims[0] == integration_sample_number
        assert dims[1] == len(adj)
        assert samples.dtype == "int"
        assert (samples >= 0).all()
        assert (samples <= 1).all()


@pytest.mark.parametrize("dim", [4])
def test_seed(dim, adj):
    """Test for the function ``strawberryfields.gbs.sample.seed``. Checks that samples are identical
    after repeated initialization of ``seed``."""

    sample.seed(1968)
    q_s_1 = sample.sample(A=adj, n_mean=2, samples=10, backend_options={"threshold": False})

    sample.seed(1968)
    q_s_2 = sample.sample(A=adj, n_mean=2, samples=10, backend_options={"threshold": False})

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
        assert sample.to_subgraphs(graph, samples=self.quantum_samples) == self.subgraphs

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

        assert sample.to_subgraphs(graph, samples=self.quantum_samples) == subgraphs_mapped


def test_modes_from_counts():
    """Test if the function ``strawberryfields.gbs.sample.modes_from_counts`` returns the correct
    mode samples when input a set of photon count samples."""

    counts = [[0, 0, 0, 0], [1, 0, 0, 2], [1, 1, 1, 0], [1, 2, 1, 0], [0, 1, 0, 2, 4]]

    modes = [[], [0, 3, 3], [0, 1, 2], [0, 1, 1, 2], [1, 3, 3, 4, 4, 4, 4]]

    assert [sample.modes_from_counts(s) for s in counts] == modes

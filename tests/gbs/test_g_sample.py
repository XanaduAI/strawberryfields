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
Unit tests for strawberryfields.gbs.graph.sample
"""
# pylint: disable=no-self-use,unused-argument,too-many-arguments
import networkx as nx
import pytest

from strawberryfields.gbs import g_sample

pytestmark = pytest.mark.gbs

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

integration_sample_number = 2


@pytest.mark.parametrize("dim", [6])
def test_sample_subgraphs_invalid_distribution(graph):
    """Tests if function ``sample.sample_subgraphs`` raises a ``ValueError`` for an
    invalid sampling distribution"""
    with pytest.raises(ValueError, match="Invalid distribution selected"):
        g_sample.sample_subgraphs(graph, nodes=2, samples=10, sample_options={"distribution": ""})


@pytest.mark.parametrize(
    "dim, nodes, samples", [(6, 4, integration_sample_number), (8, 4, integration_sample_number)]
)
@pytest.mark.parametrize("distribution", ("uniform", "gbs"))
def test_sample_subgraphs_integration(graph, nodes, samples, distribution):
    """Integration tests for the function ``sample.sample_subgraphs``"""

    graph = nx.relabel_nodes(graph, lambda x: x ** 2)
    graph_nodes = set(graph.nodes)
    output_samples = g_sample.sample_subgraphs(
        graph=graph, nodes=nodes, samples=samples, sample_options={"distribution": distribution}
    )

    assert len(output_samples) == samples
    assert all(set(sample).issubset(graph_nodes) for sample in output_samples)


@pytest.mark.parametrize("dim", [6])
class TestToSubgraphs:
    """Tests for the function ``sample.to_subgraphs``"""

    def test_graph(self, graph):
        """Test if function returns correctly processed subgraphs given input samples of the list
        ``quantum_samples``."""
        assert g_sample.to_subgraphs(graph, samples=quantum_samples) == subgraphs

    def test_graph_mapped(self, graph):
        """Test if function returns correctly processed subgraphs given input samples of the list
        ``quantum_samples``. Note that graph nodes are numbered in this test as [0, 1, 4, 9,
        ...] (i.e., squares of the usual list) as a simple mapping to explore that the optimised
        subgraph returned is still a valid subgraph."""
        graph = nx.relabel_nodes(graph, lambda x: x ** 2)
        graph_nodes = list(graph.nodes)
        subgraphs_mapped = [sorted([graph_nodes[i] for i in subgraph]) for subgraph in subgraphs]

        assert g_sample.to_subgraphs(graph, samples=quantum_samples) == subgraphs_mapped

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
Unit tests for strawberryfields.gbs.graph.utils
"""
# pylint: disable=no-self-use,protected-access
import itertools

import networkx as nx
import numpy as np
import pytest

from strawberryfields.gbs import utils

pytestmark = pytest.mark.gbs

adj_dim_range = range(2, 6)


@pytest.mark.parametrize("dim", adj_dim_range)
class TestIsUndirected:
    """Tests for the function ``strawberryfields.gbs.graph.utils.is_undirected``"""

    def test_valid_input(self, adj):
        """Test if function returns ``True`` for a symmetric matrix"""
        assert utils.is_undirected(adj)

    def test_valid_input_complex(self, adj):
        """Test if function returns ``False`` for a complex symmetric matrix"""
        assert utils.is_undirected(1j * adj)

    def test_invalid_nonsymmetric(self, adj):
        """Test if function returns ``False`` for a non-symmetric matrix"""
        adj[1, 0] = 0

        assert not utils.is_undirected(adj)

    def test_invalid_sizes(self, adj, dim):
        """Test if function returns ``False`` for an input non-square matrix"""
        assert not utils.is_undirected(adj[: dim - 1])

    def test_invalid_dimension(self, adj, dim):
        """Test if function returns ``False`` for a (1, 1)-dimensional NumPy array"""
        if dim == 2:
            assert not utils.is_undirected(adj[0:1, 0:1])

    def test_invalid_rank(self, adj):
        """Test if function returns ``False`` for an input array with dimension not equal to 2"""
        assert not utils.is_undirected(np.expand_dims(adj, axis=0))

    def test_invalid_type(self, adj):
        """Test if function raises a ``TypeError`` when the input is not a NumPy array"""
        with pytest.raises(TypeError, match="Input matrix must be a NumPy array"):
            utils.is_undirected(adj.tolist())


@pytest.mark.parametrize("dim", [4, 5, 6])
class TestSubgraphAdjacency:
    """Tests for the function ``strawberryfields.gbs.graph.utils.subgraph_adjacency``"""

    def test_input(self, dim, graph):
        """Test if function returns the correct adjacency matrix of a subgraph. This test
        iterates over all possible subgraphs of size int(dim/2). Note that graph nodes are
        numbered as [0, 1, 4, 9, ...] (i.e., squares of the usual list) as a simple mapping to
        explore that the optimised subgraph returned is still a valid subgraph."""

        subgraphs = np.array(list(itertools.combinations(range(dim), int(dim / 2))))
        subgraphs = list(map(lambda x: x ** 2, subgraphs))
        graph = nx.relabel_nodes(graph, lambda x: x ** 2)

        assert np.allclose(
            [nx.to_numpy_array(graph.subgraph(s)) for s in subgraphs],
            [utils.subgraph_adjacency(graph, s) for s in subgraphs],
        )

    def test_input_bad_nodes(self, graph, dim):
        """Test if function raises a ``ValueError`` if a list of nodes that cannot be contained
        within the graph is given"""
        nodes = range(dim + 1)

        with pytest.raises(ValueError, match="Must input a list of subgraph"):
            utils.subgraph_adjacency(graph, nodes)


@pytest.mark.parametrize("dim", [4, 5, 6])
class TestIsSubgraph:
    """Tests for the function ``strawberryfields.gbs.graph.utils.is_subgraph``"""

    def test_invalid_type(self, graph):
        """Test if function raises a ``TypeError`` when fed an invalid subgraph type (i.e.,
        not iterable)."""
        with pytest.raises(TypeError, match="subgraph and graph.nodes must be iterable"):
            utils.is_subgraph(None, graph)

    def test_valid_subgraphs(self, graph, dim):
        """Test if function returns ``True`` when fed valid subgraphs. Note that graph nodes are
        numbered as [0, 1, 4, 9, ...] (i.e., squares of the usual list)."""
        graph = nx.relabel_nodes(graph, lambda x: x ** 2)
        subgraphs = itertools.combinations(list(graph.nodes), int(dim / 2))

        assert all([utils.is_subgraph(list(s), graph) for s in subgraphs])

    def test_invalid_subgraphs(self, graph, dim):
        """Test if function returns ``False`` when fed invalid subgraphs. Note that graph nodes are
        numbered as [0, 1, 4, 9, ...] (i.e., squares of the usual list)."""
        graph = nx.relabel_nodes(graph, lambda x: x ** 2)
        subgraphs = itertools.combinations(range(dim), int(dim / 2))

        assert not all([utils.is_subgraph(list(s), graph) for s in subgraphs])

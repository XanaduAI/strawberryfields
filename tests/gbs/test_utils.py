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
class TestValidateGraph:
    """Tests for the function ``strawberryfields.gbs.graph.utils.validate_graph``"""

    def test_valid_adjacency(self, adj, monkeypatch):
        """Test if function returns the NetworkX Graph class corresponding to the input adjacency
        matrix """
        with monkeypatch.context() as m:
            m.setattr(utils, "is_undirected", lambda _: True)
            graph = utils.validate_graph(adj)
        assert np.allclose(adj, nx.to_numpy_array(graph))

    def test_valid_adjacency_complex_dtype(self, adj, monkeypatch):
        """Test if function returns the NetworkX Graph class corresponding to the input adjacency
        matrix when the input dtype is complex """
        with monkeypatch.context() as m:
            m.setattr(utils, "is_undirected", lambda _: True)
            adj = adj.astype(complex)
            graph = utils.validate_graph(adj)
        assert np.allclose(adj, nx.to_numpy_array(graph))

    def test_valid_adjacency_int_dtype(self, adj, monkeypatch):
        """Test if function returns the NetworkX Graph class corresponding to the input adjacency
        matrix when the input dtype is int """
        with monkeypatch.context() as m:
            m.setattr(utils, "is_undirected", lambda _: True)
            adj = adj.astype(int)
            graph = utils.validate_graph(adj)
        assert np.allclose(adj, nx.to_numpy_array(graph))

    def test_invalid_adjacency_nonsymmetric(self, adj, monkeypatch):
        """Test if function raises a ``ValueError`` for a non-symmetric matrix"""
        adj[1, 0] = 0
        with monkeypatch.context() as m:
            m.setattr(utils, "is_undirected", lambda _: False)
            with pytest.raises(Exception, match="Input NumPy arrays must be real and symmetric"):
                utils.validate_graph(adj)

    def test_invalid_adjacency_complex(self, adj, monkeypatch):
        """Test if function raises a ``ValueError`` for a complex symmetric matrix"""
        adj = adj * 1j
        with monkeypatch.context() as m:
            m.setattr(utils, "is_undirected", lambda _: True)
            with pytest.raises(Exception, match="Input NumPy arrays must be real and symmetric"):
                utils.validate_graph(adj)

    def test_valid_graph(self, adj):
        """Test if function returns the same as input when a NetworkX Graph class is the input"""
        nx_graph = nx.Graph(adj)
        graph = utils.validate_graph(nx_graph)
        assert graph is nx_graph

    def test_invalid_type(self, adj):
        """Test if function raises a ``TypeError`` when the input is not a NumPy array or
        NetworkX graph """
        with pytest.raises(TypeError, match="Graph is not of valid type"):
            utils.validate_graph(adj.tolist())


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

    def test_input(self, dim, graph, monkeypatch):
        """Test if function returns the correct adjacency matrix of a subgraph. This test
        iterates over all possible subgraphs of size int(dim/2). Note that graph nodes are
        numbered as [0, 1, 4, 9, ...] (i.e., squares of the usual list) as a simple mapping to
        explore that the optimised subgraph returned is still a valid subgraph."""

        with monkeypatch.context() as m:
            m.setattr(utils, "validate_graph", lambda x: x)
            subgraphs = np.array(list(itertools.combinations(range(dim), int(dim / 2))))
            subgraphs = list(map(lambda x: x ** 2, subgraphs))
            graph = nx.relabel_nodes(graph, lambda x: x ** 2)

            assert np.allclose(
                [nx.to_numpy_array(graph.subgraph(s)) for s in subgraphs],
                [utils.subgraph_adjacency(graph, s) for s in subgraphs],
            )

    def test_input_bad_nodes(self, graph, dim, monkeypatch):
        """Test if function raises a ``ValueError`` if a list of nodes that cannot be contained
        within the graph is given"""
        with monkeypatch.context() as m:
            m.setattr(utils, "validate_graph", lambda x: x)
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


@pytest.mark.parametrize("dim", range(2, 10))
class TestIsClique:
    """Tests for the function `strawberryfields.gbs.graph.utils.is_clique` """

    def test_no_false_negatives(self, dim):
        """Tests that cliques are labelled as such"""
        g = nx.complete_graph(dim)
        assert utils.is_clique(g)

    def test_no_false_positives(self, dim):
        """Tests that non-cliques are labelled as such"""
        g = nx.empty_graph(dim)
        assert not utils.is_clique(g)


@pytest.mark.parametrize("dim", range(2, 10))
class TestC0:
    """Tests function :math:`c_0` that generates the set of nodes connected to all nodes in a
    clique"""

    def test_correct_c_0(self, dim):
        """Tests that :math:`c_0` is generated correctly for the lollipop graph"""
        g = nx.lollipop_graph(dim, 1)
        s = set(range(int(dim / 2)))
        res = set(utils.c_0(s, g))

        assert res not in s
        assert {dim + 1} not in res
        assert res | s == set(range(dim))

    def test_c_0_comp_graph(self, dim):
        """ Tests that the set :math:`c_0` for a node in a clique consists of all remaining nodes"""
        A = nx.complete_graph(dim)
        S = [dim - 1]
        K = utils.c_0(S, A)

        assert K == list(range(dim - 1))

    def test_c_0_error_on_not_clique_input(self, dim):
        """Tests if function raises a ``ValueError`` when input is not a clique"""
        A = np.ones((dim, dim)) - np.eye(dim)
        A = nx.Graph(A)
        A.remove_edge(0, 1)
        S = [0, 1]

        with pytest.raises(ValueError, match="Input subgraph is not a clique"):
            utils.c_0(S, A)


@pytest.mark.parametrize("dim", range(4, 10))
class TestC1:
    """Tests function :math:`c_1` that generates the set of nodes connected to all *but one* of
    the nodes
    in a clique"""

    def test_c_1_comp_graph(self, dim):
        """Tests that :math:`c_1` set is correctly generated for an almost-complete graph, where
        edge (0, 1) is removed """
        A = nx.complete_graph(dim)
        A.remove_edge(0, 1)
        S = [i for i in range(1, dim)]
        c1 = utils.c_1(S, A)

        assert c1 == [(1, 0)]

    def test_c_1_swap_to_clique(self, dim):
        """Tests that :math:`c_1` set gives a valid clique after swapping """
        A = nx.complete_graph(dim)
        A.remove_edge(0, 1)
        S = [i for i in range(1, dim)]
        c1 = utils.c_1(S, A)
        swap_nodes = c1[0]
        S.remove(swap_nodes[0])
        S.append(swap_nodes[1])

        assert utils.is_clique(A.subgraph(S))

    def test_c_1_error_on_not_clique_input(self, dim):
        """Tests if function raises a ``ValueError`` when input is not a clique"""
        A = np.ones((dim, dim)) - np.eye(dim)
        A = nx.Graph(A)
        A.remove_edge(0, 1)
        S = [0, 1]

        with pytest.raises(ValueError, match="Input subgraph is not a clique"):
            utils.c_1(S, A)

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
Unit tests for clique
"""
# pylint: disable=no-self-use,unused-argument
import functools

import networkx as nx
import numpy as np
import pytest

from strawberryfields.gbs import clique, utils

pytestmark = pytest.mark.gbs


def patch_random_choice(x, element):
    """Dummy function for ``np.random.choice`` to make output deterministic. This dummy function
    just returns the element of ``x`` specified by ``element``."""
    if isinstance(x, int):  # np.random.choice can accept an int or 1D array-like input
        return element

    return x[element]


def patch_resize(c, graph, node_select):
    """Dummy function for ``grow`` and ``swap`` to make output unchanged."""
    return c


class TestLocalSearch:
    """Tests for the function ``clique.search``"""

    def test_bad_iterations(self):
        """Test if function raises a ``ValueError`` when a non-positive number of iterations is
        specified"""
        with pytest.raises(ValueError, match="Number of iterations must be a positive int"):
            clique.search(clique=[0, 1, 2, 3], graph=nx.complete_graph(5), iterations=-1)

    def test_max_iterations(self, monkeypatch):
        """Test if function stops after 5 iterations despite not being in a dead end (i.e., when
        ``grow != swap``). This is achieved by monkeypatching the ``np.random.choice`` call in
        ``grow`` and ``swap`` so that for a 5-node wheel graph starting as a [0,
        1] node clique, the algorithm simply oscillates between [0, 1, 2] and [0, 2, 3] for each
        iteration of growth & swap. For odd iterations, we get [0, 2, 3]."""

        graph = nx.wheel_graph(5)
        c = [0, 1]

        with monkeypatch.context() as m:
            p = functools.partial(patch_random_choice, element=0)
            m.setattr(np.random, "choice", p)
            result = clique.search(c, graph, iterations=5)

        assert result == [0, 2, 3]

    def test_dead_end(self, monkeypatch):
        """Test if function stops when in a dead end (i.e., ``grow == swap``) such that no
        swapping is possible. This is achieved by monkeypatching ``grow`` and
        ``swap`` so that they simply return the same clique as fed in, which should cause
        the local search algorithm to conclude that it is already in a dead end and return the
        same clique."""

        graph = nx.complete_graph(5)
        c = [0, 1, 2]

        with monkeypatch.context() as m:
            m.setattr(clique, "grow", patch_resize)
            m.setattr(clique, "swap", patch_resize)
            result = clique.search(c, graph, iterations=100)

        assert result == c

    def test_expected_growth(self):
        """Test if function performs a growth and swap phase, followed by another growth and
        attempted swap phase. This is carried out by starting with a 4+1 node lollipop graph,
        adding a connection from node 4 to node 2 and starting with the [3, 4] clique. The local
        search algorithm should first grow to [2, 3, 4] and then swap to either [0, 2, 3] or [1,
        2, 3], and then grow again to [0, 1, 2, 3] and being unable to swap, hence returning [0,
        1, 2, 3]"""

        graph = nx.lollipop_graph(4, 1)
        graph.add_edge(4, 2)

        c = [3, 4]
        result = clique.search(c, graph, iterations=100)
        assert result == [0, 1, 2, 3]


def patch_random_shuffle(x, reverse):
    """Dummy function for ``np.random.shuffle`` to make output deterministic. This dummy function
    reverses ``x`` in place if ``reverse == True`` and does nothing otherwise."""
    if reverse:
        x.reverse()


@pytest.mark.parametrize("dim", range(4, 10))
class TestCliqueGrow:
    """Tests for the function ``strawberryfields.clique.grow``"""

    def test_grow_maximal(self, dim):
        """Test if function grows to expected maximal graph and then stops. The chosen graph is
        composed of two fully connected graphs joined together at one node. Starting from the
        first node, ``grow`` is expected to grow to be the first fully connected graph."""
        graph = nx.barbell_graph(dim, 0)
        s = [0]
        assert set(clique.grow(s, graph)) == set(range(dim))

    def test_grow_maximal_degree(self, dim):
        """Test if function grows to expected maximal graph when degree-based node selection is
        used. The chosen graph is a fully connected graph with only the final node being
        connected to an additional node. Furthermore, the ``dim - 2`` node is disconnected from
        the ``dim - 1`` node. Starting from the first ``dim - 3`` nodes, one can either add in
        the ``dim - 2`` node or the ``dim - 1`` node. The ``dim - 1`` node has a higher degree
        due to the lollipop graph structure, and hence should be selected."""
        graph = nx.lollipop_graph(dim, 1)
        graph.remove_edge(dim - 2, dim - 1)
        s = set(range(dim - 2))
        target = s | {dim - 1}
        assert set(clique.grow(s, graph, node_select="degree")) == target

    def test_grow_maximal_degree_tie(self, dim, monkeypatch):
        """Test if function grows using randomness to break ties during degree-based node
        selection. The chosen graph is a fully connected graph with the ``dim - 2`` and ``dim -
        1`` nodes then disconnected. Starting from the first ``dim - 3`` nodes, one can add
        either of the ``dim - 2`` and ``dim - 1`` nodes. As they have the same degree, they should
        be selected randomly with equal probability. This function monkeypatches the
        ``np.random.choice`` call to guarantee that one of the nodes is picked during one run of
        ``grow`` and the other node is picked during the next run."""
        graph = nx.complete_graph(dim)
        graph.remove_edge(dim - 2, dim - 1)
        s = set(range(dim - 2))

        patch_random_choice_1 = functools.partial(patch_random_choice, element=0)
        patch_random_choice_2 = functools.partial(patch_random_choice, element=1)

        with monkeypatch.context() as m:
            m.setattr(np.random, "choice", patch_random_choice_1)
            c1 = clique.grow(s, graph, node_select="degree")

        with monkeypatch.context() as m:
            m.setattr(np.random, "choice", patch_random_choice_2)
            c2 = clique.grow(s, graph, node_select="degree")

        assert c1 != c2

    def test_input_not_clique(self, dim):
        """Tests if function raises a ``ValueError`` when input is not a clique"""
        with pytest.raises(ValueError, match="Input subgraph is not a clique"):
            clique.grow([0, 1], nx.empty_graph(dim))

    def test_bad_node_select(self, dim):
        """Tests if function raises a ``ValueError`` when input an invalid ``node_select``
        argument"""
        graph = nx.barbell_graph(dim, 0)
        s = [0]
        with pytest.raises(ValueError, match="Node selection method not recognized"):
            clique.grow(s, graph, node_select="")

    def test_input_not_subgraph(self, dim):
        """Test if function raises a ``ValueError`` when input is not a subgraph"""
        with pytest.raises(ValueError, match="Input is not a valid subgraph"):
            clique.grow([dim + 1], nx.empty_graph(dim))


@pytest.mark.parametrize("dim", range(5, 10))
class TestCliqueSwap:
    """Tests for the function ``strawberryfields.clique.swap``"""

    def test_swap(self, dim):
        """Test if function performs correct swap operation. Input is a complete graph with a
        connection between node ``0`` and ``dim - 1`` removed. An input clique of the first
        ``dim - 1`` nodes is then input, with the result being a different clique with the first
        node removed and the ``dim`` node added."""
        graph = nx.complete_graph(dim)
        graph.remove_edge(0, dim - 1)
        s = list(range(dim - 1))
        assert set(clique.swap(s, graph)) == set(range(1, dim))

    def test_swap_degree(self, dim):
        """Test if function performs correct swap operation when degree-based node selection is
        used. Input graph is a lollipop graph, consisting of a fully connected graph with a
        single additional node connected to just one of the nodes in the graph. Additionally,
        a connection between node ``0`` and ``dim - 1`` is removed as well as a connection
        between node ``0`` and ``dim - 2``. A clique of the first ``dim - 2`` nodes is then
        input. In this case, C1 consists of nodes ``dim - 1`` and ``dim - 2``. However,
        node ``dim - 1`` has greater degree due to the extra node in the lollipop graph. This
        test confirms that the resultant swap is performed correctly."""
        graph = nx.lollipop_graph(dim, 1)
        graph.remove_edge(0, dim - 1)
        graph.remove_edge(0, dim - 2)
        s = list(range(dim - 2))
        result = set(clique.swap(s, graph, node_select="degree"))
        expected = set(range(1, dim - 2)) | {dim - 1}
        assert result == expected

    def test_swap_degree_tie(self, dim, monkeypatch):
        """Test if function performs correct swap operation using randomness to break ties during
        degree-based node selection. Input graph is a fully connected graph. Additionally,
        a connection between node ``0`` and ``dim - 1`` is removed as well as a connection
        between node ``0`` and ``dim - 2``. A clique of the first ``dim - 2`` nodes is then
        input. In this case, C1 consists of nodes ``dim - 1`` and ``dim - 2``. As they have the
        same degree, they should be selected randomly with equal probability. This function
        monkeypatches the ``np.random.choice`` call to guarantee that one of the nodes is picked
        during one run of ``swap`` and the other node is picked during the next run.
        """
        graph = nx.complete_graph(dim)
        graph.remove_edge(0, dim - 1)
        graph.remove_edge(0, dim - 2)
        s = set(range(dim - 2))

        patch_random_choice_1 = functools.partial(patch_random_choice, element=0)
        patch_random_choice_2 = functools.partial(patch_random_choice, element=1)

        with monkeypatch.context() as m:
            m.setattr(np.random, "choice", patch_random_choice_1)
            c1 = clique.swap(s, graph, node_select="degree")

        with monkeypatch.context() as m:
            m.setattr(np.random, "choice", patch_random_choice_2)
            c2 = clique.swap(s, graph, node_select="degree")

        assert c1 != c2

    def test_input_not_clique(self, dim):
        """Tests if function raises a ``ValueError`` when input is not a clique"""
        with pytest.raises(ValueError, match="Input subgraph is not a clique"):
            clique.swap([0, 1], nx.empty_graph(dim))

    def test_input_valid_subgraph(self, dim):
        """Tests if function raises a ``ValueError`` when input is not a clique"""
        with pytest.raises(ValueError, match="Input is not a valid subgraph"):
            clique.swap([0, dim], nx.empty_graph(dim))

    def test_bad_node_select(self, dim):
        """Tests if function raises a ``ValueError`` when input an invalid ``node_select``
        argument"""
        graph = nx.barbell_graph(dim, 0)
        s = [0]
        with pytest.raises(ValueError, match="Node selection method not recognized"):
            clique.swap(s, graph, node_select="")


@pytest.mark.parametrize("dim", range(6, 10))
class TestCliqueShrink:
    """Tests for the function ``clique.shrink``"""

    def test_is_output_clique(self, dim):
        """Test that the output subgraph is a valid clique, in this case the maximum clique
        in a lollipop graph"""
        graph = nx.lollipop_graph(dim, dim)
        subgraph = list(range(2 * dim))  # subgraph is the entire graph
        resized = clique.shrink(subgraph, graph)
        assert utils.is_clique(graph.subgraph(resized))
        assert resized == list(range(dim))

    def test_input_clique_then_output_clique(self, dim):
        """Test that if the input is already a clique, then the output is the same clique. """
        graph = nx.lollipop_graph(dim, dim)
        subgraph = list(range(dim))  # this is a clique, the "candy" of the lollipop

        assert clique.shrink(subgraph, graph) == subgraph

    def test_degree_relative_to_subgraph(self, dim):
        """Test that function removes nodes of small degree relative to the subgraph,
        not relative to the entire graph. This is done by creating an unbalanced barbell graph,
        with one "bell" larger than the other. The input subgraph is the small bell (a clique) plus
        a node from the larger bell. The function should remove only the node from the larger
        bell, since this has a low degree within the subgraph, despite having high degree overall"""
        g = nx.disjoint_union(nx.complete_graph(dim), nx.complete_graph(dim + 1))
        g.add_edge(dim, dim - 1)
        subgraph = list(range(dim + 1))
        assert clique.shrink(subgraph, g) == list(range(dim))

    def test_wheel_graph(self, dim):
        """Test that output is correct for a wheel graph, whose largest cliques have dimension
        3. The cliques always include the central spoke and two consecutive nodes in the outer
        wheel."""
        graph = nx.wheel_graph(dim)
        subgraph = graph.nodes()  # subgraph is the entire graph
        subgraph = clique.shrink(subgraph, graph)

        assert len(subgraph) == 3
        assert subgraph[0] == 0
        assert subgraph[1] + 1 == subgraph[2] or (subgraph[1] == 1 and subgraph[2] == dim - 1)

    def test_wheel_graph_tie(self, dim, monkeypatch):
        """Test that output is correct for a wheel graph, whose largest cliques have dimension
        3. The cliques always include the central spoke and two consecutive nodes in the outer
        wheel. Since the function uses randomness in node selection when there is a tie (which
        occurs in the case of the wheel graph), the resultant shrunk cliques are expected to be
        different each time ``shrink`` is run. This function monkeypatches the
        ``np.random.shuffle`` call so that different nodes are removed during each run."""
        graph = nx.wheel_graph(dim)
        subgraph = graph.nodes()  # subgraph is the entire graph

        patch_random_shuffle_1 = functools.partial(patch_random_shuffle, reverse=False)
        patch_random_shuffle_2 = functools.partial(patch_random_shuffle, reverse=True)

        with monkeypatch.context() as m:
            m.setattr(np.random, "shuffle", patch_random_shuffle_1)
            c1 = clique.shrink(subgraph, graph)
        with monkeypatch.context() as m:
            m.setattr(np.random, "shuffle", patch_random_shuffle_2)
            c2 = clique.shrink(subgraph, graph)

        assert c1 != c2

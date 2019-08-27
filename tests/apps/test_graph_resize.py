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
Unit tests for strawberryfields.apps.graph.resize
"""
# pylint: disable=no-self-use,unused-argument
import networkx as nx
import numpy as np
import pytest

from strawberryfields.apps.graph import utils, resize

pytestmark = pytest.mark.apps

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


@pytest.fixture()
def patch_is_subgraph(monkeypatch):
    """dummy function for ``utils.is_subgraph``"""
    monkeypatch.setattr(utils, "is_subgraph", lambda v1, v2: True)


@pytest.mark.parametrize("dim", [5])
class TestResizeSubgraphs:
    """Tests for the function ``resize.resize_subgraphs``"""

    def test_target_wrong_type(self, graph):
        """Test if function raises a ``TypeError`` when incorrect type given for ``target`` """
        with pytest.raises(TypeError, match="target must be an integer"):
            resize.resize_subgraphs(subgraphs=[[0, 1]], graph=graph, target=[])

    def test_target_small(self, graph):
        """Test if function raises a ``ValueError`` when a too small number is given for
        ``target`` """
        with pytest.raises(ValueError, match="target must be greater than two and less than"):
            resize.resize_subgraphs(subgraphs=[[0, 1]], graph=graph, target=1)

    def test_target_big(self, graph):
        """Test if function raises a ``ValueError`` when a too large number is given for
        ``target`` """
        with pytest.raises(ValueError, match="target must be greater than two and less than"):
            resize.resize_subgraphs(subgraphs=[[0, 1]], graph=graph, target=5)

    def test_callable_input(self, graph):
        """Tests if function returns the correct output given a custom method set by the user"""
        objective_return = (0, [0])

        def custom_method(*args, **kwargs):
            """Mockup of custom-method function fed to ``find_dense``"""
            return objective_return

        result = resize.resize_subgraphs(
            subgraphs=[[0, 1]], graph=graph, target=4, resize_options={"method": custom_method}
        )

        assert result == objective_return

    @pytest.mark.parametrize("methods", resize.METHOD_DICT)
    def test_valid_input(self, graph, monkeypatch, methods):
        """Tests if function returns the correct output under normal conditions. The resizing
        method is here monkey patched to return a known result."""
        objective_return = (0, [0])

        def custom_method(*args, **kwargs):
            """Mockup of custom-method function fed to ``resize_subgraphs``"""
            return objective_return

        with monkeypatch.context() as m:
            m.setattr(resize, "METHOD_DICT", {methods: custom_method})

            result = resize.resize_subgraphs(
                subgraphs=[[0, 1]], graph=graph, target=4, resize_options={"method": methods}
            )

        assert result == objective_return


@pytest.mark.parametrize("dim, target", [(6, 4), (8, 5)])
@pytest.mark.parametrize("methods", resize.METHOD_DICT)
def test_resize_subgraphs_integration(graph, target, methods):
    """Test if function returns resized subgraphs of the correct form given an input list of
    variable sized subgraphs specified by ``subgraphs``. The output should be a list of ``len(
    10)``, each element containing itself a list corresponding to a subgraph that should be of
    ``len(target)``. Each element in the subraph list should correspond to a node of the input
    graph. Note that graph nodes are numbered in this test as [0, 1, 4, 9, ...] (i.e., squares of
    the usual list) as a simple mapping to explore that the resized subgraph returned is still a
    valid subgraph."""
    graph = nx.relabel_nodes(graph, lambda x: x ** 2)
    graph_nodes = set(graph.nodes)
    s_relabeled = [(np.array(s) ** 2).tolist() for s in subgraphs]
    resized = resize.resize_subgraphs(
        subgraphs=s_relabeled, graph=graph, target=target, resize_options={"method": methods}
    )
    resized = np.array(resized)
    dims = resized.shape
    assert len(dims) == 2
    assert dims[0] == len(s_relabeled)
    assert dims[1] == target
    assert all(set(sample).issubset(graph_nodes) for sample in resized)


@pytest.mark.usefixtures("patch_is_subgraph")
@pytest.mark.parametrize("dim", [5])
class TestGreedyDensity:
    """Tests for the function ``resize.greedy_density``"""

    def test_invalid_subgraph(self, graph, monkeypatch):
        """Test if function raises an ``Exception`` when an element of ``subgraphs`` is not
        contained within nodes of the graph """
        with monkeypatch.context() as m:
            m.setattr(utils, "is_subgraph", lambda v1, v2: False)
            with pytest.raises(Exception, match="Input is not a valid subgraph"):
                resize.greedy_density(subgraphs=[[0, 9]], graph=graph, target=3)

    def test_normal_conditions_grow(self, graph):
        """Test if function returns correct subgraph under normal conditions, where one needs to
        grow to a larger subgraph.  We consider the 5-node checkerboard graph starting with a
        subgraph of the nodes [0, 1, 4] and aiming to grow to 4 nodes. We can see that there are
        two subgraphs of size 4: [0, 1, 2, 4] with 3 edges and [0, 1, 3, 4] with 4 edges,
        so we hence expect the second option as the returned solution."""
        subgraph = resize.greedy_density(subgraphs=[[0, 1, 4]], graph=graph, target=4)[0]
        assert np.allclose(subgraph, [0, 1, 3, 4])

    def test_normal_conditions_shrink(self, graph):
        """Test if function returns correct subgraph under normal conditions, where one needs to
        shrink to a smaller subgraph. We consider a 5-node graph given by the below adjacency
        matrix, with a starting subgraph of nodes [1, 2, 3, 4] and the objective to shrink to 3
        nodes. We can see that there are 4 candidate subgraphs: [1, 2, 3] with 3 edges, and [1,
        2, 4], [1, 3, 4], and [2, 3, 4] all with 2 edges, so we hence expect the first option as
        the returned solution."""
        adj = ((0, 1, 0, 0, 0), (1, 0, 1, 1, 0), (0, 1, 0, 1, 0), (0, 1, 1, 0, 1), (0, 0, 0, 1, 0))
        graph = nx.Graph(0.5 * np.array(adj))  # multiply by 0.5 to follow weightings of adj fixture
        subgraph = resize.greedy_density(subgraphs=[[1, 2, 3, 4]], graph=graph, target=3)[0]
        assert np.allclose(subgraph, [1, 2, 3])


@pytest.mark.usefixtures("patch_is_subgraph")
@pytest.mark.parametrize("dim", [5])
class TestGreedyDegree:
    """Tests for the function ``resize.greedy_degree``"""

    def test_invalid_subgraph(self, graph, monkeypatch):
        """Test if function raises an ``Exception`` when an element of ``subgraphs`` is not
        contained within nodes of the graph """
        with monkeypatch.context() as m:
            m.setattr(utils, "is_subgraph", lambda v1, v2: False)
            with pytest.raises(Exception, match="Input is not a valid subgraph"):
                resize.greedy_degree(subgraphs=[[0, 9]], graph=graph, target=3)

    def test_normal_conditions_grow(self, graph):
        """Test if function returns correct subgraph under normal conditions, where one needs to
        grow to a larger subgraph.  We consider the 7-node graph given by the below adjacency
        matrix starting with a subgraph of the nodes [0, 1, 4] and aiming to grow to 4 nodes. We
        can see that there are four candidate nodes to add in (corresponding degree shown in
        brackets): node 2 (degree 4), node 3 (degree 3), node 5 (degree 1) and node 6 (degree 1).
        Hence, since node 2 has the greatest degree, we expect the returned solution to be
        [0, 1, 2, 4]."""
        adj = (
            (0, 1, 0, 0, 0, 0, 0),
            (1, 0, 1, 1, 0, 0, 0),
            (0, 1, 0, 1, 0, 1, 1),
            (0, 1, 1, 0, 1, 0, 0),
            (0, 0, 0, 1, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0),
        )
        graph = nx.Graph(0.5 * np.array(adj))  # multiply by 0.5 to follow weightings of adj fixture

        subgraph = resize.greedy_degree(subgraphs=[[0, 1, 4]], graph=graph, target=4)[0]

        assert np.allclose(subgraph, [0, 1, 2, 4])

    def test_normal_conditions_shrink(self, graph):
        """Test if function returns correct subgraph under normal conditions, where one needs to
        shrink to a smaller subgraph. We consider the 7-node graph given by the below adjacency
        matrix starting with a subgraph of the nodes [0, 1, 2, 3] and aiming to shrink to 3
        nodes. We can see that there are four candidate nodes to remove (corresponding degree
        shown in brackets): node 0 (degree 3), node 1 (degree 3), node 2 (degree 2) and node 4 (
        degree 3). Hence, since node 2 has the lowest degree, we expect the returned solution
        to be [0, 1, 3]."""
        adj = (
            (0, 1, 0, 0, 0, 1, 1),
            (1, 0, 1, 1, 0, 0, 0),
            (0, 1, 0, 1, 0, 0, 0),
            (0, 1, 1, 0, 1, 0, 0),
            (0, 0, 0, 1, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0),
        )
        graph = nx.Graph(0.5 * np.array(adj))  # multiply by 0.5 to follow weightings of adj fixture
        subgraph = resize.greedy_degree(subgraphs=[[0, 1, 2, 3]], graph=graph, target=3)
        assert np.allclose(subgraph, [0, 1, 3])


@pytest.mark.parametrize("dim", range(6, 10))
class TestCliqueShrink:
    """Tests for the function ``resize.clique_shrink``"""

    def test_is_output_clique(self, dim):
        """Test that the output subgraph is a valid clique, in this case the the maximum clique
        in a lollipop graph"""
        graph = nx.lollipop_graph(dim, dim)
        subgraph = list(range(2 * dim))  # subgraph is the entire graph
        resized = resize.clique_shrink(subgraph, graph)
        assert utils.is_clique(graph.subgraph(resized))
        assert resized == list(range(dim))

    def test_input_clique_then_output_clique(self, dim):
        """Test that if the input is already a clique, then the output is the same clique. """
        graph = nx.lollipop_graph(dim, dim)
        subgraph = list(range(dim))  # this is a clique, the "candy" of the lollipop

        assert resize.clique_shrink(subgraph, graph) == subgraph

    def test_degree_relative_to_subgraph(self, dim):
        """Test that function removes nodes of small degree relative to the subgraph,
        not relative to the entire graph. This is done by creating an unbalanced barbell graph,
        with one "bell" larger than the other. The input subgraph is the small bell (a clique) plus
        a node from the larger bell. The function should remove only the node from the larger
        bell, since this has a low degree within the subgraph, despite having high degree overall"""
        g = nx.disjoint_union(nx.complete_graph(dim), nx.complete_graph(dim + 1))
        g.add_edge(dim, dim - 1)
        subgraph = list(range(dim + 1))
        assert resize.clique_shrink(subgraph, g) == list(range(dim))

    def test_wheel_graph(self, dim):
        """Test that output is correct for a wheel graph, whose largest cliques have dimension
        3. The cliques always include the central spoke and two consecutive nodes in the outer
        wheel."""
        graph = nx.wheel_graph(dim)
        subgraph = graph.nodes()  # subgraph is the entire graph
        clique = resize.clique_shrink(subgraph, graph)

        assert len(clique) == 3
        assert clique[0] == 0
        assert clique[1] + 1 == clique[2] or (clique[1] == 1 and clique[2] == dim - 1)

    def test_wheel_graph_tie(self, dim):
        """Test that output is correct for a wheel graph, whose largest cliques have dimension
        3. The cliques always include the central spoke and two consecutive nodes in the outer
        wheel. Since the function uses randomness in node selection when there is a tie (which
        occurs in the case of the wheel graph), this test checks that, with 100 repetitions,
        multiple 3-node cliques are returned."""
        graph = nx.wheel_graph(dim)
        subgraph = graph.nodes()  # subgraph is the entire graph

        np.random.seed(0)  # set random seed for reproducible results

        cliques = [resize.clique_shrink(subgraph, graph) for _ in range(100)]

        nodes_returned = set([node for clique in cliques for node in clique])

        assert len(nodes_returned) > 3

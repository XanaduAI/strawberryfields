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
Unit tests for strawberryfields.gbs.graph.resize
"""
# pylint: disable=no-self-use,unused-argument
import networkx as nx
import numpy as np
import pytest

from strawberryfields.gbs import resize, utils

pytestmark = pytest.mark.gbs

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

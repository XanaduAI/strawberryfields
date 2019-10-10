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
Unit tests for strawberryfields.gbs.subgraph
"""
# pylint: disable=no-self-use,unused-argument
import functools

import networkx as nx
import numpy as np
import pytest

from strawberryfields.gbs import sample, subgraph

pytestmark = pytest.mark.gbs


samples_subgraphs = np.array(
    [
        [1, 2, 3, 4],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 5],
        [0, 1, 2, 3],
        [0, 1, 4, 5],
        [0, 1, 4, 5],
        [0, 1, 3, 4],
        [0, 1, 2, 3],
        [0, 1, 4, 5],
    ]
)


@pytest.mark.parametrize("dim", [5])
class TestSearch:
    """Tests for the function ``subgraph.search``"""

    def test_callable_input(self, adj):
        """Tests if function returns the correct output given a custom method set by the user"""
        objective_return = (0, [0])

        def custom_method(*args, **kwargs):
            """Mockup of custom-method function fed to ``search``"""
            return objective_return

        result = subgraph.search(
            graph=adj, nodes=3, iterations=10, options={"heuristic": {"method": custom_method}}
        )

        assert result == objective_return

    @pytest.mark.parametrize("methods", subgraph.METHOD_DICT)
    def test_valid_input(self, graph, monkeypatch, methods):
        """Tests if function returns the correct output under normal conditions. The method is here
        monkey patched to return a known result."""
        objective_return = (0, [0])

        def custom_method(*args, **kwargs):
            """Mockup of custom-method function fed to ``search``"""
            return objective_return

        with monkeypatch.context() as m:
            m.setattr(subgraph, "METHOD_DICT", {methods: custom_method})

            result = subgraph.search(
                graph=graph, nodes=3, iterations=10, options={"heuristic": {"method": methods}}
            )

        assert result == objective_return


@pytest.mark.parametrize("dim", [5])
class TestRandomSearch:
    """Tests for the function ``subgraph.random_search``"""

    def sampler(self, *args, **kwargs):
        """Dummy function for sampling subgraphs"""

        return (samples_subgraphs ** 2).tolist()

    def test_returns_densest(self, graph, monkeypatch):
        """Tests if function returns the densest subgraph from a fixed set of subgraph samples. The
        samples are given by ``samples_subgraphs`` and are defined with respect to the adjacency
        matrix given below. Note that graph nodes are numbered in this test as [0, 1, 4, 9,
        ...] (i.e., squares of the usual list) as a simple mapping to explore that the optimised
        subgraph returned is still a valid subgraph. Among the samples, only one ([0, 1, 9,
        16]) corresponds to a subgraph that is a complete graph. We therefore expect the return
        value to be (1.0, [0, 1, 9, 16])."""
        adj = (
            (0, 1, 0, 1, 1, 0),
            (1, 0, 1, 1, 1, 0),
            (0, 1, 0, 1, 0, 0),
            (1, 1, 1, 0, 1, 0),
            (1, 1, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 0),
        )
        graph = nx.Graph(0.5 * np.array(adj))  # multiply by 0.5 to follow weightings of adj fixture
        optimal_density = 1.0
        optimal_sample = [0, 1, 9, 16]
        graph = nx.relabel_nodes(graph, lambda x: x ** 2)

        def patch_resize(s, graph, sizes):
            return {sizes[0]: s}

        with monkeypatch.context() as m:
            m.setattr(sample, "sample", self.sampler)
            m.setattr(sample, "to_subgraphs", self.sampler)
            m.setattr(subgraph, "resize", patch_resize)

            result = subgraph.random_search(graph=graph, nodes=4, iterations=10)

        assert result == (optimal_density, optimal_sample)


class TestResize:
    """Tests for the function ``subgraph.resize``"""

    def test_input_not_subgraph(self):
        """Test if function raises a ``ValueError`` when input is not a subgraph"""
        dim = 5
        with pytest.raises(ValueError, match="Input is not a valid subgraph"):
            subgraph.resize([dim + 1], nx.complete_graph(dim))

    def test_invalid_size(self):
        """Test if function raises a ``ValueError`` when an invalid size is requested"""
        dim = 5
        with pytest.raises(ValueError, match="Requested sizes must be within size range of graph"):
            subgraph.resize([0, 1], nx.complete_graph(dim), sizes=[-1, dim + 1])

    @pytest.mark.parametrize("dim", [5, 6, 7])
    def test_full_range(self, dim):
        """Test if function correctly resizes to full range of sizes when the ``sizes`` argument
        is not specified."""
        g = nx.complete_graph(dim)
        resized = subgraph.resize([0, 1, 2], g)

        assert set(resized.keys()) == set(range(1, dim))

        subgraph_sizes = [len(resized[i]) for i in range(1, dim)]

        assert subgraph_sizes == list(range(1, dim))

    def test_correct_resize(self):
        """Test if function correctly resizes on a fixed example where the ideal resizing is
        known. The example is a lollipop graph of 6 fully connected nodes and 2 nodes on the
        lollipop stick. An edge between node zero and node ``dim - 1`` (the lollipop node
        connecting to the stick) is removed and a starting subgraph of nodes ``[2, 3, 4, 5,
        6]`` is selected. The objective is to add-on 1 and 2 nodes and remove 1 node."""
        dim = 6
        g = nx.lollipop_graph(dim, 2)
        g.remove_edge(0, dim - 1)

        s = [2, 3, 4, 5, 6]
        sizes = [4, 6, 7]

        ideal = {4: [2, 3, 4, 5], 6: [1, 2, 3, 4, 5, 6], 7: [0, 1, 2, 3, 4, 5, 6]}
        resized = subgraph.resize(s, g, sizes)

        assert ideal == resized

    def test_tie(self, monkeypatch):
        """Test if function correctly settles ties with random selection. This test starts with
        the problem of a 6-dimensional complete graph and a 4-node subgraph, with the objective
        of shrinking down to 3 nodes. In this case, any of the 4 nodes in the subgraph is an
        equally good candidate for removal since all nodes have the same degree. The test
        monkeypatches the ``np.random.shuffle`` function to simply permute the list according to
        an offset, e.g. for an offset of 2 the list [0, 1, 2, 3] gets permuted to [2, 3, 0,
        1]. Using this we expect the node to be removed by ``resize`` in this case to have label
        equal to the offset."""
        dim = 6
        g = nx.complete_graph(dim)
        s = [0, 1, 2, 3]

        def permute(l, offset):
            d = len(l)
            ll = l.copy()
            for i in range(d):
                l[i] = ll[(i + offset) % d]

        for i in range(4):
            permute_i = functools.partial(permute, offset=i)
            with monkeypatch.context() as m:
                m.setattr(np.random, "shuffle", permute_i)
                resized = subgraph.resize(s, g, [3])
                resized_subgraph = resized[3]
                removed_node = list(set(s) - set(resized_subgraph))[0]
                assert removed_node == i

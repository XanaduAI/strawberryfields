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
Unit tests for strawberryfields.gbs.graph.dense
"""
# pylint: disable=no-self-use,unused-argument
import networkx as nx
import numpy as np
import pytest

from strawberryfields.gbs import dense, g_sample, resize

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
class TestFindDense:
    """Tests for the function ``dense.find_dense``"""

    def test_callable_input(self, adj):
        """Tests if function returns the correct output given a custom method set by the user"""
        objective_return = (0, [0])

        def custom_method(*args, **kwargs):
            """Mockup of custom-method function fed to ``find_dense``"""
            return objective_return

        result = dense.find_dense(
            graph=adj, nodes=3, iterations=10, options={"heuristic": {"method": custom_method}}
        )

        assert result == objective_return

    @pytest.mark.parametrize("methods", dense.METHOD_DICT)
    def test_valid_input(self, graph, monkeypatch, methods):
        """Tests if function returns the correct output under normal conditions. The method is here
        monkey patched to return a known result."""
        objective_return = (0, [0])

        def custom_method(*args, **kwargs):
            """Mockup of custom-method function fed to ``find_dense``"""
            return objective_return

        with monkeypatch.context() as m:
            m.setattr(dense, "METHOD_DICT", {methods: custom_method})

            result = dense.find_dense(
                graph=graph, nodes=3, iterations=10, options={"heuristic": {"method": methods}}
            )

        assert result == objective_return


@pytest.mark.parametrize("dim", [5])
class TestRandomSearch:
    """Tests for the function ``dense.random_search``"""

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

        with monkeypatch.context() as m:
            m.setattr(g_sample, "sample_subgraphs", self.sampler)
            # The monkeypatch above is not necessary given the one below, but simply serves to
            # speed up testing by not requiring a call to ``sample_subgraphs``, which is a
            # bottleneck

            m.setattr(resize, "resize_subgraphs", self.sampler)

            result = dense.random_search(graph=graph, nodes=4, iterations=10)

        assert result == (optimal_density, optimal_sample)

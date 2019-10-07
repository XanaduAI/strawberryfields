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
Unit tests for strawberryfields.gbs.plot
"""
# pylint: disable=no-self-use,unused-argument,too-many-arguments,protected-access
import networkx as nx
import numpy as np
import pytest

from strawberryfields.gbs import plot

pytestmark = pytest.mark.gbs


class TestNodeCoords:
    """Test for the internal function strawberryfields.gbs.plot._node_coords"""

    @pytest.mark.parametrize("dim", [4, 5, 6])
    def test_complete_graph(self, dim):
        """Tests that nodes in a complete graph are located in the unit circle when using the
        Kamada-Kawai layout"""
        graph = nx.complete_graph(dim)
        layout = nx.kamada_kawai_layout(graph)
        coords = plot._node_coords(graph, layout)
        x = coords["x"]
        y = coords["y"]
        radii = [np.sqrt(x[i] ** 2 + y[i] ** 2) for i in range(dim)]

        assert np.allclose(radii, np.ones(dim), atol=1e-5)

    def test_custom_layout(self):
        """Tests that nodes in a complete graph are correctly located based on a custom layout"""
        graph = nx.complete_graph(4)
        layout = {0: [1, 1], 1: [1, -1], 2: [-1, 1], 3: [-1, -1]}
        coords = plot._node_coords(graph, layout)
        x = coords["x"]
        y = coords["y"]

        assert x == [1, 1, -1, -1]
        assert y == [1, -1, 1, -1]


class TestEdgeCoords:
    """Test for the internal function strawberryfields.gbs.plot._edge_coords"""

    @pytest.mark.parametrize("dim", [4, 5, 6])
    def test_cycle_graph(self, dim):
        """Tests that the length of edges in a circular cycle graph are all of equal length."""
        graph = nx.cycle_graph(dim)
        layout = nx.kamada_kawai_layout(graph)
        coords = plot._edge_coords(graph, layout)
        x = coords["x"]
        y = coords["y"]
        dists = [
            np.sqrt((x[3 * k] - x[3 * k + 1]) ** 2 + (y[3 * k] - y[3 * k + 1]) ** 2)
            for k in range(dim)
        ]
        first_dist = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)

        assert np.allclose(dists, first_dist)

    def test_custom_layout(self):
        """Tests that edges in a complete graph are correctly located based on a custom layout"""
        graph = nx.complete_graph(2)
        layout = {0: [1, 1], 1: [-1, -1]}
        coords = plot._edge_coords(graph, layout)
        x = coords["x"]
        y = coords["y"]
        print(x)
        print(y)

        assert x[:2] == [1, -1]
        assert y[:2] == [1, -1]

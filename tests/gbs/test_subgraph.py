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

from strawberryfields.gbs import data, sample, subgraph

pytestmark = pytest.mark.gbs

p = data.Planted()
g = nx.Graph(p.adj)


'''class TestSearch:
    """Tests for the function ``subgraph.search``"""

    def test_bad_subgraph(self):
        """Test if function prints a warning when an invalid subgraph is included in
        ``subgraphs`` and continues on to the next subgraph"""
        samps = 100

        s = sample.to_subgraphs(p[:samps], g)
        s.append([31])  # invalid sample

        res = subgraph.search(s, g, 10, 20)
        print(res.keys())'''

class TestAddToList:
    """Tests for the function ``subgraph._add_to_list``"""

    l = [(0.09284148990494756, [1, 2, 5, 9]), (0.16013443604938415, [0, 1, 2, 5]), (0.38803386506300297, [0, 2, 3, 6]), (0.4132105226731848, [0, 1, 3, 6]), (0.5587798561345368, [3, 6, 8, 9]), (0.7017410327600858, [
0, 4, 7, 9]), (0.8360847995330193, [2, 3, 5, 9]), (0.8768364522184167, [0, 1, 4, 8]), (0.9011700496489199, [2, 5, 6, 9]), (0.9183222696574376, [0, 3, 4, 6])]


    def test_already_contained(self):
        """Test if function does not act on list if fed a subgraph that is already there"""
        l = self.l.copy()
        subgraph._add_to_list(l, [1, 2, 5, 9], 0.09284148990494756, 15)
        assert l == self.l

    def test_simple_add(self):
        """Test if function simply adds a new tuple if ``len(l)`` does not exceed ``top_count``"""
        s = [0, 1, 2, 3]
        d = 0.7017410327600858
        l = self.l.copy()
        l_ideal = sorted(l + [(d, s)])
        subgraph._add_to_list(l, [0, 1, 2, 3], 0.7017410327600858, 15)
        assert l_ideal == l




class TestResize:
    """Tests for the function ``subgraph.resize``"""

    def test_input_not_subgraph(self):
        """Test if function raises a ``ValueError`` when input is not a subgraph"""
        dim = 5
        with pytest.raises(ValueError, match="Input is not a valid subgraph"):
            subgraph.resize([dim + 1], nx.complete_graph(dim), 2, 3)

    def test_invalid_min_size(self):
        """Test if function raises a ``ValueError`` when an invalid min_size is requested"""
        dim = 5
        with pytest.raises(ValueError, match="min_size must be at least 1"):
            subgraph.resize([0, 1], nx.complete_graph(dim), 0, 3)

    def test_invalid_max_size(self):
        """Test if function raises a ``ValueError`` when an invalid max_size is requested"""
        dim = 5
        with pytest.raises(ValueError, match="max_size must be less than number of nodes in graph"):
            subgraph.resize([0, 1], nx.complete_graph(dim), 2, dim)

    def test_invalid_max_vs_min(self):
        """Test if function raises a ``ValueError`` when max_size is less than min_size"""
        dim = 5
        with pytest.raises(ValueError, match="max_size must not be less than min_size"):
            subgraph.resize([0, 1], nx.complete_graph(dim), 4, 3)

    @pytest.mark.parametrize("dim", [7, 8])
    @pytest.mark.parametrize(
        "min_size,max_size",
        [(1, 4), (1, 5), (2, 6), (1, 6), (4, 6), (1, 2), (1, 1), (3, 3), (5, 5), (1, 3), (3, 6)],
    )
    def test_full_range(self, dim, min_size, max_size):
        """Test if function correctly resizes to full range of requested sizes"""
        g = nx.complete_graph(dim)
        resized = subgraph.resize([0, 1, 2], g, min_size, max_size)
        r = range(min_size, max_size + 1)

        assert set(resized.keys()) == set(r)

        subgraph_sizes = [len(resized[i]) for i in r]

        assert subgraph_sizes == list(r)

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
        min_size = 4
        max_size = 7

        ideal = {
            4: [2, 3, 4, 5],
            5: [2, 3, 4, 5, 6],
            6: [1, 2, 3, 4, 5, 6],
            7: [0, 1, 2, 3, 4, 5, 6],
        }
        resized = subgraph.resize(s, g, min_size, max_size)

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
            for _i in range(d):
                l[_i] = ll[(_i + offset) % d]

        for i in range(4):
            permute_i = functools.partial(permute, offset=i)
            with monkeypatch.context() as m:
                m.setattr(np.random, "shuffle", permute_i)
                resized = subgraph.resize(s, g, min_size=3, max_size=3)
                resized_subgraph = resized[3]
                removed_node = list(set(s) - set(resized_subgraph))[0]
                assert removed_node == i

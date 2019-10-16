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
# for testing private functions with pylint: disable=protected-access
# for fixtures with pylint: disable=redefined-outer-name
import functools

import networkx as nx
import numpy as np
import pytest

from strawberryfields.gbs import data, subgraph

pytestmark = pytest.mark.gbs

p_planted = data.Planted()
g_planted = nx.Graph(p_planted.adj)


@pytest.fixture()
def process_planted(min_size, max_size, max_count, n_samples):
    """Fixture for loading samples from the Planted dataset"""
    samples = p_planted[:n_samples]
    d = subgraph.search(samples, g_planted, min_size, max_size, max_count)
    return d


@pytest.mark.parametrize("min_size", [4, 5])
@pytest.mark.parametrize("max_size", [6, 7])
@pytest.mark.parametrize("max_count", [2, 4])
@pytest.mark.parametrize("n_samples", [200])
@pytest.mark.usefixtures("process_planted")
class TestSearch:
    """Tests for the function ``subgraph.search``"""

    def test_size_range(self, min_size, max_size, process_planted):
        """Test if function searches the full range of sizes specified by ``min_size`` and
        ``max_size``"""
        assert set(process_planted.keys()) == set(range(min_size, max_size + 1))

    def test_max_count(self, max_count, process_planted):
        """Test if function only keeps track of at most ``max_count`` subgraphs for each size in
        the specified range"""
        assert all([len(l) <= max_count for l in process_planted.values()])

    def test_ordered(self, process_planted):
        """Test if function always returns dictionary values that are sorted lists in descending
        order"""
        assert all(
            [
                all((l == sorted(l, reverse=True), isinstance(l, list)))
                for l in process_planted.values()
            ]
        )

    def test_tuple(self, process_planted):
        """Test if function always returns dictionary values that are lists composed of tuples
        with the first value being a float and the second being a list"""
        assert all(
            [
                all(
                    (
                        isinstance(t, tuple),
                        isinstance(t[0], float),
                        isinstance(t[1], list),
                        len(t) == 2,
                    )
                )
                for l in process_planted.values()
                for t in l
            ]
        )

    def test_density(self, process_planted):
        """Test if function returns dictionary values that are lists of tuples where the first
        element is the density of the subgraph specified by the second element"""
        assert all(
            [
                t[0] == nx.density(g_planted.subgraph(t[1]))
                for l in process_planted.values()
                for t in l
            ]
        )


def test_update_dict(monkeypatch):
    """Test if the function ``subgraph._update_dict`` correctly combines a hard-coded dictionary
    with another dictionary of candidate subgraph tuples."""

    d = {
        3: [
            (0.8593365162004337, [1, 4, 6]),
            (0.7834607649769199, [0, 1, 9]),
            (0.6514468663852714, [1, 8, 9]),
        ],
        4: [(0.5738321520630872, [1, 5, 7, 9]), (0.4236138075085395, [5, 6, 7, 8])],
    }

    d_new = {
        3: (0.4509829558474371, [2, 3, 7]),
        4: (0.13609407922395578, [4, 5, 7, 8]),
        5: (0.7769987961311593, [4, 6, 7, 8, 9]),
    }

    d_ideal = {
        3: [
            (0.8593365162004337, [1, 4, 6]),
            (0.7834607649769199, [0, 1, 9]),
            (0.6514468663852714, [1, 8, 9]),
            (0.4509829558474371, [2, 3, 7]),
        ],
        4: [
            (0.5738321520630872, [1, 5, 7, 9]),
            (0.4236138075085395, [5, 6, 7, 8]),
            (0.13609407922395578, [4, 5, 7, 8]),
        ],
        5: [(0.7769987961311593, [4, 6, 7, 8, 9])],
    }

    def patch_update_subgraphs_list(l, t, _max_count):
        if l != [t]:
            l.append(t)

    with monkeypatch.context() as m:
        m.setattr(subgraph, "_update_subgraphs_list", patch_update_subgraphs_list)
        subgraph._update_dict(d, d_new, max_count=10)

    assert d == d_ideal


class TestUpdateSubgraphsList:
    """Tests for the function ``subgraph._update_subgraphs_list``"""

    l = [
        (0.9183222696574376, [0, 3, 4, 6]),
        (0.9011700496489199, [2, 5, 6, 9]),
        (0.8768364522184167, [0, 1, 4, 8]),
        (0.8360847995330193, [2, 3, 5, 9]),
        (0.7017410327600858, [0, 4, 7, 9]),
        (0.5587798561345368, [3, 6, 8, 9]),
        (0.4132105226731848, [0, 1, 3, 6]),
        (0.38803386506300297, [0, 2, 3, 6]),
        (0.16013443604938415, [0, 1, 2, 5]),
        (0.09284148990494756, [1, 2, 5, 9]),
    ]

    l_shuffled = [
        (0.4132105226731848, [0, 1, 3, 6]),
        (0.5587798561345368, [3, 6, 8, 9]),
        (0.9183222696574376, [0, 3, 4, 6]),
        (0.8768364522184167, [0, 1, 4, 8]),
        (0.9011700496489199, [2, 5, 6, 9]),
        (0.38803386506300297, [0, 2, 3, 6]),
        (0.16013443604938415, [0, 1, 2, 5]),
        (0.8360847995330193, [2, 3, 5, 9]),
        (0.7017410327600858, [0, 4, 7, 9]),
        (0.09284148990494756, [1, 2, 5, 9]),
    ]

    t_above_min_density = [
        (0.9744200893790599, [10, 11, 12, 13]),
        (0.7356617723720428, [14, 15, 16, 17]),
        (0.1020993710922522, [18, 19, 20, 21]),
    ]

    t_min_density = (0.09284148990494756, [22, 23, 24, 25])

    t_below_min_density = (0.05835994410622691, [26, 27, 28, 29])

    @pytest.mark.parametrize("elem", list(range(10)))
    def test_already_contained(self, elem):
        """Test if function does not add a subgraph tuple that is already contained in the list"""
        l = self.l.copy()
        subgraph._update_subgraphs_list(l, l[elem], max_count=20)
        assert l == self.l

    def test_add_below_max_count(self):
        """Test if function adds subgraph tuples whenever len(l) < max_count. This test starts
        with an empty list and tries to add on elements one at a time from ``l_shuffled``. Since
        ``max_count = 10`` and ``len(l) = 10``, we expect to be able to build up our list to be
        equal to ``l``."""
        l = []
        for t in self.l_shuffled:
            subgraph._update_subgraphs_list(l, t, max_count=10)

        assert l == self.l

    def test_add_above_max_count_high_density(self):
        """Test if function adds subgraph tuples with density exceeding the minimum value of
        ``l`` even though the length of ``l`` is at its maximum"""
        max_count = 10

        for t in self.t_above_min_density:
            l = self.l.copy()
            l_ideal = sorted(l + [t], reverse=True)

            subgraph._update_subgraphs_list(l, t, max_count=max_count)
            del l_ideal[-1]
            assert len(l) == max_count
            assert l == l_ideal

    def test_add_above_max_count_equal_density(self, monkeypatch):
        """Test if function randomly adds in a subgraph tuple with density equal to the minimum
        value of ``l``, even though the length of ``l`` is at its maximum. This test
        monkeypatches the ``np.random.choice`` call used in the function to decide whether to
        keep the original subgraph or add in the new one. In one case, ``np.random.choice`` is
        monkeypatched to leave ``l`` unchanged, and in another case ``np.random.choice`` is
        monkeypatched to swap the minimum value of ``l`` with ``t_min_density``."""
        max_count = 10

        with monkeypatch.context() as m:
            m.setattr(np.random, "choice", lambda x: 0)
            l = self.l.copy()
            subgraph._update_subgraphs_list(l, self.t_min_density, max_count=max_count)
            assert len(l) == max_count
            assert l == self.l

        with monkeypatch.context() as m:
            m.setattr(np.random, "choice", lambda x: 1)
            l = self.l.copy()
            subgraph._update_subgraphs_list(l, self.t_min_density, max_count=max_count)

            l_ideal = self.l.copy()
            del l_ideal[-1]
            l_ideal.append(self.t_min_density)
            assert len(l) == max_count
            assert l == l_ideal

    def test_add_above_max_count_low_density(self):
        """Test if function does not add in a subgraph tuple of density lower than the minimum
        value of ``l`` when the length of ``l`` is at its maximum"""
        max_count = 10
        l = self.l.copy()

        subgraph._update_subgraphs_list(l, self.t_below_min_density, max_count=max_count)
        assert l == self.l


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

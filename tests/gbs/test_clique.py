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
Unit tests for strawberryfields.gbs.clique
"""
# pylint: disable=no-self-use,unused-argument
import networkx as nx
import numpy as np
import pytest

from strawberryfields.gbs import clique, resize

pytestmark = pytest.mark.gbs


def patch_random_choice(x):
    """Dummy function for ``np.random.choice`` to make output deterministic. This dummy function
    just returns the element of ``x`` specified by ``element``."""
    if isinstance(x, int):  # np.random.choice can accept an int or 1D array-like input
        return 0

    return x[0]


def patch_clique_resize(c, graph, node_select):
    """Dummy function for ``clique_grow`` and ``clique_swap`` to make output unchanged."""
    return c


class TestLocalSearch:
    """Tests for the function ``max_clique.local_search``"""

    def test_bad_iterations(self):
        """Test if function raises a ``ValueError`` when a non-positive number of iterations is
        specified"""
        with pytest.raises(ValueError, match="Number of iterations must be a positive int"):
            clique.local_search(clique=[0, 1, 2, 3], graph=nx.complete_graph(5), iterations=-1)

    def test_max_iterations(self, monkeypatch):
        """Test if function stops after 5 iterations despite not being in a dead end (i.e., when
        ``grow != swap``). This is achieved by monkeypatching the ``np.random.choice`` call in
        ``clique_grow`` and ``clique_swap`` so that for a 5-node wheel graph starting as a [0,
        1] node clique, the algorithm simply oscillates between [0, 1, 2] and [0, 2, 3] for each
        iteration of growth & swap. For odd iterations, we get [0, 2, 3]."""

        graph = nx.wheel_graph(5)
        c = [0, 1]

        with monkeypatch.context() as m:
            m.setattr(np.random, "choice", patch_random_choice)
            result = clique.local_search(c, graph, iterations=5)

        assert result == [0, 2, 3]

    def test_dead_end(self, monkeypatch):
        """Test if function stops when in a dead end (i.e., ``grow == swap``) such that no
        swapping is possible. This is achieved by monkeypatching ``clique_grow`` and
        ``clique_swap`` so that they simply return the same clique as fed in, which should cause
        the local search algorithm to conclude that it is already in a dead end and return the
        same clique."""

        graph = nx.complete_graph(5)
        c = [0, 1, 2]

        with monkeypatch.context() as m:
            m.setattr(resize, "clique_grow", patch_clique_resize)
            m.setattr(resize, "clique_swap", patch_clique_resize)
            result = clique.local_search(c, graph, iterations=100)

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
        result = clique.local_search(c, graph, iterations=100)
        assert result == [0, 1, 2, 3]

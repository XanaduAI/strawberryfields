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
Dense subgraph identification
=============================

**Module name:** :mod:`strawberryfields.gbs.subgraph`

.. currentmodule:: strawberryfields.gbs.subgraph

This module provides tools for users to find dense subgraphs of a range of different sizes.
The search heuristic in this module works by resizing a collection of starting subgraphs and
keeping track of the densest identified. The starting subgraphs can be selected by sampling from
GBS, resulting in candidates that are likely to be dense :cite:`arrazola2018using`.

An accompanying tutorial can be found :ref:`here <gbs-subgraph-tutorial>`.

Algorithm
---------

The heuristic algorithm provided proceeds as follows. Each starting subgraph :math:`s` is resized
to a range of sizes :math:`\{k\}_{k_{\min}}^{k_{\max}}`, resulting in the subgraphs
:math:`s_{k}`. For a given size :math:`k`, a collection of the densest :math:`n` subgraphs
identified is recorded, meaning that :math:`s_{k}` is added to the collection only if it has
sufficient density.

.. autosummary::
    search

This algorithm returns a dictionary over the range of sizes specified, with each value being the
collection of densest-:math:`k` subgraphs. This collection is a list of tuple pairs specifying
the subgraph density and nodes.

Subgraph resizing
-----------------

The key element of the :func:`search` algorithm is the resizing of each subgraph, allowing a
range of subgraph sizes to be tracked. Resizing proceeds by greedily adding or removing nodes to
a subgraph one-at-a-time. Node selection is carried out by picking the node with the greatest
or least degree with respect to to the subgraph, with ties settled uniformly at random.

.. autosummary::
    resize

This function returns a dictionary over the range of sizes specified, with each value being the
corresponding resized subgraph.

Code details
^^^^^^^^^^^^
"""
import networkx as nx
import numpy as np


def search(
    subgraphs: list, graph: nx.Graph, min_size: int, max_size: int, max_count: int = 10
) -> dict:
    """Search for dense subgraphs within an input size range.

    For each subgraph from ``subgraphs``, this function resizes using :func:`resize` to the input
    range specified by ``min_size`` and ``max_size``, resulting in a range of differently sized
    subgraphs. This function loops over all elements of ``subgraphs`` and keeps track of the
    ``max_count`` number of densest subgraphs identified for each size.

    **Example usage:**

    >>> s = data.Planted()
    >>> g = nx.Graph(s.adj)
    >>> s = sample.postselect(s, 16, 30)
    >>> s = sample.to_subgraphs(s, g)
    >>> search(s, g, 8, 9, max_count=3)
    {9: [(0.9722222222222222, [21, 22, 23, 24, 25, 26, 27, 28, 29]),
      (0.9722222222222222, [20, 21, 22, 24, 25, 26, 27, 28, 29]),
      (0.9444444444444444, [20, 21, 22, 23, 24, 25, 26, 27, 29])],
     8: [(1.0, [21, 22, 24, 25, 26, 27, 28, 29]),
      (1.0, [21, 22, 23, 24, 25, 26, 27, 28]),
      (1.0, [20, 21, 22, 24, 25, 26, 27, 29])]}

    Args:
        subgraphs (list[list[int]]): a list of subgraphs specified by their nodes
        graph (nx.Graph): the input graph
        min_size (int): minimum size to search for dense subgraphs
        max_size (int): maximum size to search for dense subgraphs
        max_count (int): maximum number of densest subgraphs to keep track of for each size

    Returns:
        dict[int, list[tuple[float, list[int]]]]: a dictionary of different sizes, each containing a
        list of densest subgraphs reported as a tuple of subgraph density and subgraph nodes,
        sorted in non-increasing order of density
    """
    dense = {}

    for s in subgraphs:
        r = resize(s, graph, min_size, max_size)

        for size, subgraph in r.items():
            r[size] = (nx.density(graph.subgraph(subgraph)), subgraph)

        _update_dict(dense, r, max_count)

    return dense


def _update_dict(d: dict, d_new: dict, max_count: int) -> None:
    """Updates dictionary ``d`` with subgraph tuples contained in ``d_new``.

    Subgraph tuples are a pair of values: a float specifying the subgraph density and a list of
    integers specifying the subgraph nodes. Both ``d`` and ``d_new`` are dictionaries over
    different subgraph sizes. The values of ``d`` are lists of subgraph tuples containing the top
    densest subgraphs for a given size, with maximum length ``max_count``. The values of
    ``d_new`` are candidate subgraph tuples that can be the result of resizing an input subgraph
    over a range using :func:`resize`. We want to add these candidates to the list of subgraph
    tuples in ``d`` to build up our collection of dense subgraphs.

    Args:
        d (dict[int, list[tuple[float, list[int]]]]): dictionary of subgraph sizes and
            corresponding list of subgraph tuples
        d_new (dict[int, tuple[float, list[int]]]): dictionary of subgraph sizes and corresponding
            subgraph tuples that are candidates to be added to the list
        max_count (int):  the maximum length of every subgraph tuple list

    Returns:
        None: this function modifies the dictionary ``d`` in place
    """
    for size, t in d_new.items():
        l = d.setdefault(size, [t])
        _update_subgraphs_list(l, t, max_count)


def _update_subgraphs_list(l: list, t: tuple, max_count: int) -> None:
    """Updates list of top subgraphs with a candidate.

    Here, the list ``l`` to be updated is a list of tuples with each tuple being a pair of
    values: a float specifying the subgraph density and a list of integers specifying the
    subgraph nodes. For example, ``l`` may be:

    ``[(0.8, [0, 5, 9, 10]), (0.5, [1, 2, 5, 6]), (0.3, [0, 4, 6, 9])]``

    We want to update ``l`` with a candidate tuple ``t``, which should be a pair specifying a
    subgraph density and corresponding subgraph nodes. For example, we might want to add:

    ``(0.4, [1, 4, 9, 10])``

    This function checks:

    - if ``t`` is already an element of ``l``, do nothing (i.e., so that ``l`` never has
      repetitions)

    - if ``len(l) < max_count``, add ``t``

    - otherwise, if the density of ``t`` exceeds the minimum density of ``l`` , add ``t`` and
      remove the element with the minimum density

    - otherwise, if the density of ``t`` equals the minimum density of ``l``, flip a coin and
      randomly swap in ``t`` with the minimum element of ``l``.

    The list ``l`` is also sorted so that its first element is the subgraph with the highest
    density.

    Args:
        l (list[tuple[float, list[int]]]): the list of subgraph tuples to be updated
        t (tuple[float, list[int]): the candidate subgraph tuple
        max_count (int): the maximum length of ``l``

    Returns:
        None: this function modifies ``l`` in place
    """
    t = (t[0], sorted(set(t[1])))

    for _d, s in l:
        if t[1] == s:
            return

    if len(l) < max_count:
        l.append(t)
        l.sort(reverse=True)
        return

    l_min = l[-1][0]

    if t[0] > l_min:
        l.append(t)
        l.sort(reverse=True)
        del l[-1]
    elif t[0] == l_min:
        if np.random.choice(2):
            del l[-1]
            l.append(t)
            l.sort(reverse=True)

    return


def resize(subgraph: list, graph: nx.Graph, min_size: int, max_size: int) -> dict:
    """Resize a subgraph to a range of input sizes.

    This function uses a greedy approach to iteratively add or remove nodes one at a time to an
    input subgraph to reach the range of sizes specified by ``min_size`` and ``max_size``.

    When growth is required, the algorithm examines all nodes from the remainder of the graph as
    candidates and adds the single node with the highest degree relative to the rest of the
    subgraph. This results in a graph that is one node larger, and if growth is still required,
    the algorithm performs the procedure again.

    When shrinking is required, the algorithm examines all nodes from within the subgraph as
    candidates and removes the single node with lowest degree relative to the subgraph. In both
    growth and shrink phases, ties for addition/removal with nodes of equal degree are settled by
    uniform random choice.

    **Example usage:**

    >>> s = data.Planted()
    >>> g = nx.Graph(s.adj)
    >>> s = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    >>> resize(s, g, 8, 12)
    {10: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
     11: [11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
     12: [0, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
     9: [20, 21, 22, 24, 25, 26, 27, 28, 29],
     8: [20, 21, 22, 24, 25, 26, 27, 29]}

    Args:
        subgraph (list[int]): a subgraph specified by a list of nodes
        graph (nx.Graph): the input graph
        min_size (int): minimum size for subgraph to be resized to
        max_size (int): maximum size for subgraph to be resized to

    Returns:
        dict[int, list[int]]: a dictionary of different sizes with corresponding subgraph
    """
    nodes = graph.nodes()
    subgraph = set(subgraph)

    if not subgraph.issubset(nodes):
        raise ValueError("Input is not a valid subgraph")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if max_size >= len(nodes):
        raise ValueError("max_size must be less than number of nodes in graph")
    if max_size < min_size:
        raise ValueError("max_size must not be less than min_size")

    starting_size = len(subgraph)

    if min_size <= starting_size <= max_size:
        resized = {starting_size: sorted(subgraph)}
    else:
        resized = {}

    if max_size > starting_size:

        grow_subgraph = graph.subgraph(subgraph).copy()

        while grow_subgraph.order() < max_size:
            grow_nodes = grow_subgraph.nodes()
            complement_nodes = nodes - grow_nodes

            degrees = [
                (c, graph.subgraph(list(grow_nodes) + [c]).degree()[c]) for c in complement_nodes
            ]
            np.random.shuffle(degrees)

            to_add = max(degrees, key=lambda x: x[1])
            grow_subgraph.add_node(to_add[0])

            new_size = grow_subgraph.order()

            if min_size <= new_size <= max_size:
                resized[new_size] = sorted(grow_subgraph.nodes())

    if min_size < starting_size:

        shrink_subgraph = graph.subgraph(subgraph).copy()

        while shrink_subgraph.order() > min_size:
            degrees = list(shrink_subgraph.degree())
            np.random.shuffle(degrees)

            to_remove = min(degrees, key=lambda x: x[1])
            shrink_subgraph.remove_node(to_remove[0])

            new_size = shrink_subgraph.order()

            if min_size <= new_size <= max_size:
                resized[new_size] = sorted(shrink_subgraph.nodes())

    return resized

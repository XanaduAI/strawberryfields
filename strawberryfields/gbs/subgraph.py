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

This module provides tools for users to idensity dense subgraphs.

The :func:`search` function provides a heuristic algorithm for finding dense regions and proceeds
by greedily resizing input subgraphs and keeping track of the densest found.

.. autosummary::
    search

Subgraph resizing
-----------------

Subgraphs sampled from GBS are not guaranteed to be of size :math:`k`, even if this is the mean
photon number. On the other hand, the densest-:math:`k` subgraph problem requires graphs of fixed
size to be considered. This means that heuristic algorithms at some point must resize the sampled
subgraphs. Resizing functionality is provided by the following function.

.. autosummary::
    resize

Code details
^^^^^^^^^^^^
"""
import networkx as nx
import numpy as np


def search(
    subgraphs: list, graph: nx.Graph, min_size: int, max_size: int, top_count: int = 10
) -> dict:
    """Search for dense subgraphs within an input size range.

    For each subgraph, this function resizes to the input range specified by ``min_size`` and
    ``max_size`` and keeps track of the ``top_count`` number of densest subgraphs identified for
    each size.

    Args:
        subgraphs (list[list[int]]): a list of subgraphs specified by their nodes
        graph (nx.Graph): the input graph
        min_size (int): minimum size for subgraph to be resized to
        max_size (int): maximum size for subgraph to be resized to
        top_count (int): maximum number of densest subgraphs to keep track of for each size

    Returns:
        dict[int, tuple(float, list[int])]: a dictionary of different sizes, each containing a
        list of subgraphs reported as a tuple of subgraph density and subgraph nodes
    """
    nodes = graph.nodes()
    size_range = range(min_size, max_size + 1)

    dense = {}

    for s in subgraphs:
        s = set(s)
        if not s.issubset(nodes):
            continue

        r = resize(s, graph, min_size, max_size)

        _combine_dicts(dense, r, graph, top_count)

    return dense


def _combine_dicts(main: dict, new: dict, graph: nx.Graph, top_count: int) -> None:

    for size, candidate in new.items():
        current = main.get(size)

        sub = graph.subgraph(candidate)
        sub_dens = nx.density(sub)

        if current is None:
            current = [(sub_dens, candidate)]
            main[size] = current
        else:
            _add_to_list(current, candidate, sub_dens)


def _add_candidate_subgraph(l: list, t: tuple, max_count: int) ->


def _add_to_list(l: list, subgraph: list, density: float, top_count: int) -> None:

    current_subgraphs = set(tuple(elem[1]) for elem in l)
    subgraph = sorted(set(subgraph))

    if not {tuple(subgraph)}.issubset(current_subgraphs):

        if len(l) < top_count:
            l.append((density, subgraph))
            l.sort()
        elif density > min(l)[0]:
            l.append((density, subgraph))
            l.sort()
            del l[-1]


def resize(subgraph: list, graph: nx.Graph, min_size: int, max_size: int) -> dict:
    """Resize a subgraph to a range of input sizes.

    This function uses a greedy approach to iteratively add or remove nodes one at a time to an
    input subgraph to reach the range of sizes specified by ``min_size`` and ``max_size``.

    When growth is required, the algorithm examines all nodes from the remainder of the graph as
    candidates and adds-in the single node with the highest degree relative to the rest of the
    subgraph. This results in a graph that is one node larger,  and if growth is still required
    the algorithm performs the above procedure again.

    When shrinking is required, the algorithm examines all nodes from within the subgraph as
    candidates and removes the single node with lowest degree relative to the subgraph. In both
    growth and shrink phases, ties for addition/removal with nodes of equal degree are settled by
    uniform random choice.

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

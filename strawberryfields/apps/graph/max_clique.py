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
Maximum clique
==============

**Module name:** :mod:`strawberryfields.apps.graph.max_clique`

.. currentmodule:: strawberryfields.apps.graph.max_clique

This module provides tools for users to identify large cliques in graphs. A clique is a subgraph
where all nodes are connected to each other. The maximum clique problem is to identify the
largest clique in a graph. The problem is NP-Hard in a worst-case setting, which is why it is
valuable to develop better algorithms to tackle it.

The core of this module is a local search algorithm that proceeds as follows:

#. A small clique in the graph is identified. The initial clique can be a single node, or it can
   be obtained by shrinking a random subgraph, for example obtained from GBS.
#. The clique is grown as much as possible by adding nodes connected to all nodes in the clique.
#. When no more growth is possible, a node in the clique is swapped with a node outside of it.

Steps 2-3 are repeated until a pre-determined number of steps have been completed, or until no
more growth or swaps are possible.

Summary
-------

.. autosummary::
    local_search

Code details
^^^^^^^^^^^^
"""
import networkx as nx

from strawberryfields.apps.graph import resize


def local_search(clique: list, graph: nx.Graph, iterations, node_select: str = "uniform") -> list:
    """Use local search algorithm to identify large cliques.

    This function implements a version of the local search algorithm given in
    :ref:`pullan2006dynamic` and :ref:`pullan2006phased`. It proceeds by iteratively applying
    phases of greedy growth and plateau search to the input clique subgraph. Growth is achieved
    using the :func:`~.clique_grow` function, which repeatedly evaluates the set :math:`C_0` of
    nodes in the remainder of the graph that are connected to all nodes in the clique,
    and selects one candidate node from :math:`C_0` to make a larger clique. Plateau search is
    performed with the :func:`~.clique_swap` function, which evaluates the set :math:`C_1` of
    nodes in the remainder of the graph that are connected to all but one of the nodes in the
    clique, and selects one candidate from :math:`C_1` and its corresponding unconnected node in
    the clique to swap.

    The iterative applications of growth and search are applied until either the specified number
    of iterations has been carried out or when a dead end has been reached. The function reaches
    a dead end when it is unable to perform any new swaps or growth. Whenever the sets
    :math:`C_0` and :math:`C_1` used during growth and swapping have more than one element,
    there must be a choice of which node to add or swap. This choice is specified with the
    ``node_select`` argument, with node selection based on uniform randomness and node degree
    supported. Degree-based node selection involves picking the node with the greatest degree,
    with ties settled by uniform random choice.

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph
        iterations (int): number of steps in the algorithm
        node_select (str): method of selecting nodes during swap and growth. Can be either
        ``"uniform"`` for uniform random selection or ``"degree"`` for degree-based selection.
        Defaults to ``"uniform"``

    Returns:
       list[int]: the largest cliques found by the algorithm
    """
    if iterations < 1:
        raise ValueError("Number of iterations must be a positive int")

    grow = resize.clique_grow(clique, graph, node_select=node_select)
    swap = resize.clique_swap(grow, graph, node_select=node_select)

    iterations -= 1

    if grow == swap or iterations == 0:
        return swap

    return local_search(swap, graph, iterations, node_select)

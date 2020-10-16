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
Tools for users to identify large cliques in graphs.

A clique is a subgraph where all nodes are connected to each other. The maximum clique problem is
to identify the largest clique in a graph. It has been shown that samples from GBS can be used to
select dense subgraphs as a starting seed for heuristic algorithms :cite:`banchi2019molecular`.

.. seealso::

    :ref:`apps-clique-tutorial`

Algorithm
^^^^^^^^^

This module provides :func:`search`, a variant of the local search heuristics described in
:cite:`pullan2006dynamic` and :cite:`pullan2006phased`. The algorithm proceeds as follows:

#. A small clique in the graph is identified. The initial clique can be a single node, or it can
   be obtained by shrinking a random subgraph, for example obtained from GBS.
#. The clique is grown as much as possible by adding nodes connected to all nodes in the clique.
#. When no more growth is possible, a node in the clique is swapped with a node outside of it.

Steps 2-3 are repeated until a pre-determined number of steps have been completed, or until no
more growth or swaps are possible.

Clique growth and swapping
^^^^^^^^^^^^^^^^^^^^^^^^^^

A clique can be grown by evaluating the set :math:`C_0` of nodes in the remainder of the graph that
are connected to all nodes in the clique. A single node from :math:`C_0` is selected and added to
the clique. This process is repeated until :math:`C_0` becomes empty. This is provided with the
:func:`grow` function.

Searching the local space around a clique can be achieved by swapping a node from the clique with a
node in the remainder of the graph. The first step is to evaluate the set :math:`C_1` of nodes in
the remainder of the graph that are connected to *all but one* of the nodes in the current
clique. A swap is then performed by adding the node into the clique and removing the node in the
clique that is not connected to it. This is provided with the :func:`swap` function.

Whenever the sets :math:`C_0` and :math:`C_1` used during growth and swapping have more than
one element, there must be a choice of which node to add or swap. The supported choices are:

- Select among candidate nodes uniformly at random;
- Select the candidate node with the greatest degree, settling remaining ties uniformly at random;
- Select the candidate node with the greatest node weight, settling remaining ties uniformly at
  random.

Using GBS to find a starting clique
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Samples from GBS correspond to subgraphs that are likely to be dense.
These subgraphs may not be cliques, which is the required input to the :func:`search` algorithm.
To reconcile this, a subgraph may be shrunk using the :func:`shrink` function by removing nodes
until the remainder forms a clique. This can be achieved by selecting the node with the lowest
degree relative to the rest of subgraph and removing the node, repeating the process until a
clique is found.
"""
from typing import Union

import networkx as nx
import numpy as np


def search(
    clique: list, graph: nx.Graph, iterations, node_select: Union[str, np.ndarray, list] = "uniform"
) -> list:
    """Local search algorithm for identifying large cliques.

    This function implements a version of the local search algorithm given in
    :cite:`pullan2006dynamic` and :cite:`pullan2006phased`. It proceeds by iteratively applying
    phases of greedy growth and plateau search to the input clique subgraph.

    **Growth phase**

    Growth is achieved using the :func:`~.grow` function, which repeatedly evaluates the
    set :math:`C_0` of nodes in the remainder of the graph that are connected to all nodes in the
    clique, and selects one candidate node from :math:`C_0` to make a larger clique.

    **Search phase**

    Plateau search is performed with the :func:`~.swap` function, which evaluates the set
    :math:`C_1` of nodes in the remainder of the graph that are connected to all but one of the
    nodes in the clique. The function then proceeds to select one candidate from :math:`C_1` and
    swap it with its corresponding node in the clique.

    Whenever the sets :math:`C_0` and :math:`C_1` used during growth and swapping have more than
    one element, there must be a choice of which node to add or swap. This choice is specified
    with the ``node_select`` argument, which can be any of the following:

    - ``"uniform"`` (default): choose a node from the candidates uniformly at random;
    - ``"degree"``: choose the node from the candidates with the greatest degree, settling ties
      by uniform random choice;
    - A list or array: specifying the node weights of the graph, resulting in choosing the node
      from the candidates with the greatest weight, settling ties by uniform random choice.

    **Stopping**

    The iterative applications of growth and search are applied until either the specified number
    of iterations has been carried out or when a dead end has been reached. The function reaches
    a dead end when it is unable to perform any new swaps or growth.

    **Example usage:**

    >>> graph = nx.lollipop_graph(4, 1)
    >>> graph.add_edge(2, 4)
    >>> clique = [3, 4]
    >>> search(clique, graph, iterations=10)
    [0, 1, 2, 3]

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph
        iterations (int): number of steps in the algorithm
        node_select (str, list or array): method of selecting nodes during swap and growth. Can
            be ``"uniform"`` (default), ``"degree"``, or a NumPy array or list.

    Returns:
       list[int]: the largest clique found by the algorithm
    """
    if iterations < 1:
        raise ValueError("Number of iterations must be a positive int")

    grown = grow(clique, graph, node_select=node_select)
    swapped = swap(grown, graph, node_select=node_select)

    iterations -= 1

    if set(grown) == set(swapped) or iterations == 0:
        return swapped

    return search(swapped, graph, iterations, node_select)


def grow(
    clique: list, graph: nx.Graph, node_select: Union[str, np.ndarray, list] = "uniform"
) -> list:
    """Iteratively adds new nodes to the input clique to generate a larger clique.

    Each iteration involves calculating the set :math:`C_0` (provided by the function
    :func:`c_0`) with respect to the current clique. This set represents the nodes in the rest of
    the graph that are connected to all of the nodes in the current clique. Therefore, adding any of
    the nodes in :math:`C_0` will create a larger clique. This function proceeds by repeatedly
    evaluating :math:`C_0` and selecting and adding a node from this set to add to the current
    clique. Growth is continued until :math:`C_0` becomes empty.

    Whenever there are multiple nodes within :math:`C_0`, one must choose which node to add to
    the growing clique. This function allows a method of choosing nodes to be set with the
    ``node_select`` argument, which can be any of the following:

    - ``"uniform"`` (default): choose a node from the candidates uniformly at random;
    - ``"degree"``: choose the node from the candidates with the greatest degree, settling ties
      by uniform random choice;
    - A list or array: specifying the node weights of the graph, resulting in choosing the node
      from the candidates with the greatest weight, settling ties by uniform random choice.

    **Example usage:**

    >>> graph = nx.complete_graph(10)
    >>> clique = [0, 1, 2, 3, 4]
    >>> grow(clique, graph)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph
        node_select (str, list or array): method of selecting nodes from :math:`C_0` during
            growth. Can be ``"uniform"`` (default), ``"degree"``, or a NumPy array or list.

    Returns:
        list[int]: a new clique subgraph of equal or larger size than the input
    """

    if not set(clique).issubset(graph.nodes):
        raise ValueError("Input is not a valid subgraph")

    if not is_clique(graph.subgraph(clique)):
        raise ValueError("Input subgraph is not a clique")

    if isinstance(node_select, (list, np.ndarray)):
        if len(node_select) != graph.number_of_nodes():
            raise ValueError("Number of node weights must match number of nodes")
        w = {n: node_select[i] for i, n in enumerate(graph.nodes)}
        node_select = "weight"

    clique = set(clique)
    _c_0 = sorted(c_0(clique, graph))

    while _c_0:
        if node_select == "uniform":
            clique.add(np.random.choice(_c_0))
        elif node_select == "degree":
            degrees = np.array([graph.degree(n) for n in _c_0])
            to_add_index = np.random.choice(np.where(degrees == degrees.max())[0])
            to_add = _c_0[to_add_index]
            clique.add(to_add)
        elif node_select == "weight":
            weights = np.array([w[n] for n in _c_0])
            to_add_index = np.random.choice(np.where(weights == weights.max())[0])
            to_add = _c_0[to_add_index]
            clique.add(to_add)
        else:
            raise ValueError("Node selection method not recognized")

        _c_0 = sorted(c_0(clique, graph))

    return sorted(clique)


def swap(
    clique: list, graph: nx.Graph, node_select: Union[str, np.ndarray, list] = "uniform"
) -> list:
    """If possible, generates a new clique by swapping a node in the input clique with a node
    outside the clique.

    Proceeds by calculating the set :math:`C_1` of nodes in the rest of the graph that are
    connected to all but one of the nodes in the clique. If this set is not empty, this function
    randomly picks a node and swaps it with the corresponding node in the clique that is not
    connected to it. The set :math:`C_1` and corresponding nodes in the clique are provided by the
    :func:`c_1` function.

    Whenever there are multiple nodes within :math:`C_1`, one must choose which node to add to
    the growing clique. This function allows a method of choosing nodes to be set with the
    ``node_select`` argument, which can be any of the following:

    - ``"uniform"`` (default): choose a node from the candidates uniformly at random;
    - ``"degree"``: choose the node from the candidates with the greatest degree, settling ties
      by uniform random choice;
    - A list or array: specifying the node weights of the graph, resulting in choosing the node
      from the candidates with the greatest weight, settling ties by uniform random choice.

    **Example usage:**

    >>> graph = nx.wheel_graph(5)
    >>> graph.remove_edge(0, 4)
    >>> clique = [0, 1, 2]
    >>> swap(clique, graph)
    [0, 2, 3]

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph
        node_select (str, list or array): method of selecting incoming nodes from :math:`C_1`
            during swapping. Can be ``"uniform"`` (default), ``"degree"``, or a NumPy array or list.

    Returns:
        list[int]: a new clique subgraph of equal size as the input
    """

    if not set(clique).issubset(graph.nodes):
        raise ValueError("Input is not a valid subgraph")

    if not is_clique(graph.subgraph(clique)):
        raise ValueError("Input subgraph is not a clique")

    if isinstance(node_select, (list, np.ndarray)):
        if len(node_select) != graph.number_of_nodes():
            raise ValueError("Number of node weights must match number of nodes")
        w = {n: node_select[i] for i, n in enumerate(graph.nodes)}
        node_select = "weight"

    clique = set(clique)
    _c_1 = c_1(clique, graph)

    if _c_1:
        if node_select == "uniform":
            swap_index = np.random.choice(len(_c_1))
            swap_nodes = _c_1[swap_index]
        elif node_select == "degree":
            degrees = np.array([graph.degree(n[1]) for n in _c_1])
            to_swap_index = np.random.choice(np.where(degrees == degrees.max())[0])
            swap_nodes = _c_1[to_swap_index]
        elif node_select == "weight":
            weights = np.array([w[n[1]] for n in _c_1])
            to_swap_index = np.random.choice(np.where(weights == weights.max())[0])
            swap_nodes = _c_1[to_swap_index]
        else:
            raise ValueError("Node selection method not recognized")

        clique.remove(swap_nodes[0])
        clique.add(swap_nodes[1])

    return sorted(clique)


def shrink(
    subgraph: list, graph: nx.Graph, node_select: Union[str, np.ndarray, list] = "uniform"
) -> list:
    """Shrinks an input subgraph until it forms a clique.

    Proceeds by removing nodes in the input subgraph one at a time until the result is a clique
    that satisfies :func:`is_clique`. Upon each iteration, this function selects the node with
    lowest degree relative to the subgraph and removes it.

    In some instances, there may be multiple nodes of minimum degree as candidates to remove from
    the subgraph. The method of selecting which of these nodes to remove is specified by the
    ``node_select`` argument, which can be either:

    - ``"uniform"`` (default): choose a node from the candidates uniformly at random;
    - A list or array: specifying the node weights of the graph, resulting in choosing the node
      from the candidates with the lowest weight, settling ties by uniform random choice.

    **Example usage:**

    >>> graph = nx.barbell_graph(4, 0)
    >>> subgraph = [0, 1, 2, 3, 4, 5]
    >>> shrink(subgraph, graph)
    [0, 1, 2, 3]

    Args:
        subgraph (list[int]): a subgraph specified by a list of nodes
        graph (nx.Graph): the input graph
        node_select (str, list or array): method of settling ties when more than one node of
            equal degree can be removed. Can be ``"uniform"`` (default), or a NumPy array or list.

    Returns:
        list[int]: a clique of size smaller than or equal to the input subgraph
    """

    if not set(subgraph).issubset(graph.nodes):
        raise ValueError("Input is not a valid subgraph")

    if isinstance(node_select, (list, np.ndarray)):
        if len(node_select) != graph.number_of_nodes():
            raise ValueError("Number of node weights must match number of nodes")
        w = {n: node_select[i] for i, n in enumerate(graph.nodes)}
        node_select = "weight"

    subgraph = graph.subgraph(subgraph).copy()  # A copy is required to be able to modify the
    # structure of the subgraph (https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.Graph.subgraph.html)

    while not is_clique(subgraph):
        degrees = np.array(subgraph.degree)
        degrees_min = np.argwhere(degrees[:, 1] == degrees[:, 1].min()).flatten()

        if node_select == "uniform":
            to_remove_index = np.random.choice(degrees_min)
        elif node_select == "weight":
            weights = np.array([w[degrees[n][0]] for n in degrees_min])
            to_remove_index = np.random.choice(np.where(weights == weights.min())[0])
        else:
            raise ValueError("Node selection method not recognized")

        to_remove = degrees[to_remove_index][0]
        subgraph.remove_node(to_remove)

    return sorted(subgraph.nodes())


def is_clique(graph: nx.Graph) -> bool:
    """Determines if the input graph is a clique. A clique of :math:`n` nodes has exactly :math:`n(
    n-1)/2` edges.

    **Example usage:**

    >>> graph = nx.complete_graph(10)
    >>> is_clique(graph)
    True

    Args:
        graph (nx.Graph): the input graph

    Returns:
        bool: ``True`` if input graph is a clique and ``False`` otherwise
    """
    edges = graph.edges
    nodes = graph.order()

    return len(edges) == nodes * (nodes - 1) / 2


def c_0(clique: list, graph: nx.Graph):
    """Generates the set :math:`C_0` of nodes that are connected to all nodes in the input
    clique subgraph.

    The set :math:`C_0` is defined in :cite:`pullan2006phased` and is used to determine nodes
    that can be added to the current clique to grow it into a larger one.

    **Example usage:**

    >>> graph = nx.complete_graph(10)
    >>> clique = [0, 1, 2, 3, 4]
    >>> c_0(clique, graph)
    [5, 6, 7, 8, 9]

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph

    Returns:
        list[int]: a list containing the :math:`C_0` nodes for the clique

    """
    if not is_clique(graph.subgraph(clique)):
        raise ValueError("Input subgraph is not a clique")

    clique = set(clique)
    c_0_nodes = []
    non_clique_nodes = set(graph.nodes) - clique

    for i in non_clique_nodes:
        if clique.issubset(graph.neighbors(i)):
            c_0_nodes.append(i)

    return c_0_nodes


def c_1(clique: list, graph: nx.Graph):
    """Generates the set :math:`C_1` of nodes that are connected to all but one of the nodes in
    the input clique subgraph.

    The set :math:`C_1` is defined in :cite:`pullan2006phased` and is used to determine outside
    nodes that can be swapped with clique nodes to create a new clique.

    **Example usage:**

    >>> graph = nx.wheel_graph(5)
    >>> clique = [0, 1, 2]
    >>> c_1(clique, graph)
    [(1, 3), (2, 4)]

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph

    Returns:
       list[tuple[int]]: A list of tuples. The first node in the tuple is the node in the clique
       and the second node is the outside node it can be swapped with.
    """
    if not is_clique(graph.subgraph(clique)):
        raise ValueError("Input subgraph is not a clique")

    clique = set(clique)
    c_1_nodes = []
    non_clique_nodes = set(graph.nodes) - clique

    for i in non_clique_nodes:
        neighbors_in_subgraph = clique.intersection(graph.neighbors(i))

        if len(neighbors_in_subgraph) == len(clique) - 1:
            to_swap = clique - neighbors_in_subgraph
            (i_clique,) = to_swap
            c_1_nodes.append((i_clique, i))

    return c_1_nodes

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
Graph resizing
==============

**Module name:** :mod:`strawberryfields.apps.graph.resize`

.. currentmodule:: strawberryfields.apps.graph.resize

This module provides functionality for resizing of subgraphs. Samples from
:func:`~strawberryfields.apps.sample.quantum_sampler` have a variable number of clicks (
equivalently, the number of ones for samples with threshold detection). This results in subgraphs of
a random size in :func:`~strawberryfields.apps.graph.sample.dense_subgraph_sampler_gbs`.
However, some algorithms may want to work with subgraphs of a fixed size, necessitating the
resizing of sample subgraphs.

Resizing functionality is provided by the :func:`resize_subgraphs` function, allowing the user to
make use of different methods for resizing. The available methods are specified in
:func:`resize_subgraphs`.

Summary
-------

.. autosummary::
    METHOD_DICT
    RESIZE_DEFAULTS
    resize_subgraphs
    greedy_density
    greedy_degree
    clique_grow
    clique_swap
    clique_shrink

Code details
^^^^^^^^^^^^
"""

from typing import Iterable, Optional
import itertools

import networkx as nx
import numpy as np

from strawberryfields.apps.graph import utils

RESIZE_DEFAULTS = {"method": "greedy-density"}
"""dict[str, Any]: Dictionary to specify default parameters of options for resizing
"""


def resize_subgraphs(
    subgraphs: Iterable, graph: nx.Graph, target: int, resize_options: Optional[dict] = None
) -> list:
    """Resize subgraphs to a given size.

    This function resizes subgraphs to size ``target``. The resizing method can be fully
    specified by the optional ``resize_options`` argument, which should be a dict containing any
    of the following:

    .. glossary::

        key: ``"method"``, value: *str* or *callable*
            Value can be either a string selecting from a range of available methods or a
            customized callable function. Options include:

            - ``"greedy-density"``: resize subgraphs by iteratively adding/removing nodes that
              result in the densest subgraph among all candidates (default)
            - ``"greedy-degree"``: resize subgraphs by iteratively adding/removing nodes that
              have the highest/lowest degree.
            - *callable*: accepting ``(subgraphs: list, graph: nx.Graph,
              target: int, resize_options: dict)`` as arguments and returning a ``list``
              containing the resized subgraphs as lists of nodes, see :func:`greedy_density` for
              an example signature

    Args:
        subgraphs (iterable[list[int]]): an iterable over subgraphs, where each subgraph is given
            by a list of nodes
        graph (nx.Graph): the input graph
        target (int): the target number of nodes (must be larger than 1)
        resize_options (dict[str, Any]): dictionary specifying options used during resizing;
            defaults to :const:`RESIZE_DEFAULTS`

    Returns:
        list[list[int]]: a list of resized subgraphs, where each subgraph is given by a list of
        nodes
    """
    resize_options = {**RESIZE_DEFAULTS, **(resize_options or {})}

    try:
        target = int(target)
    except TypeError:
        raise TypeError("target must be an integer")

    if not 2 < target < graph.number_of_nodes():
        raise ValueError(
            "target must be greater than two and less than the number of nodes in "
            "the input graph"
        )

    method = resize_options["method"]

    if not callable(method):
        method = METHOD_DICT[method]

    return method(subgraphs=subgraphs, graph=graph, target=target, resize_options=resize_options)


def greedy_density(
    subgraphs: Iterable, graph: nx.Graph, target: int, resize_options: Optional[dict] = None
) -> list:
    """Method to greedily resize subgraphs based upon density.

    This function uses a greedy approach to iteratively add/remove nodes to an input subgraph.
    Suppose the input subgraph has :math:`M` nodes. When shrinking, the algorithm considers
    all :math:`M-1` node subgraphs of the input subgraph, and picks the one with the greatest
    density. It then considers all :math:`M-2` node subgraphs of the chosen :math:`M-1` node
    subgraph, and continues until the desired subgraph size has been reached.

    When growing, the algorithm considers all :math:`M+1` node subgraphs of the input graph given by
    combining the input :math:`M` node subgraph with another node from the remainder of the input
    graph, and picks the one with the greatest density. It then considers all :math:`M+2` node
    subgraphs given by combining the chosen :math:`M+1` node subgraph with nodes remaining from
    the overall graph, and continues until the desired subgraph size has been reached.

    ``resize_options`` is not currently in use in :func:`greedy_density`.

    Args:
        subgraphs (iterable[list[int]]): an iterable over subgraphs, where each subgraph is given
            by a list of nodes
        graph (nx.Graph): the input graph
        target (int): the target number of nodes in the subgraph (must be larger than 1)
        resize_options (dict[str, Any]): dictionary specifying options used during resizing;
            defaults to :const:`RESIZE_DEFAULTS`

    Returns:
        list[list[int]]: a list of resized subgraphs, where each subgraph is given by a list of
        nodes
    """
    # Note that ``resize_options`` will become non-trivial once randomness is added to methods
    resize_options = {**RESIZE_DEFAULTS, **(resize_options or {})}

    nodes = graph.nodes

    resized = []

    for s in subgraphs:
        s = set(s)

        if not utils.is_subgraph(s, graph):
            raise ValueError("Input is not a valid subgraph")

        while len(s) != target:
            if len(s) < target:
                s_candidates = [s | {n} for n in set(nodes) - s]
            else:
                s_candidates = list(itertools.combinations(s, len(s) - 1))

            _d, s = max((nx.density(graph.subgraph(n)), n) for n in s_candidates)

        resized.append(sorted(s))

    return resized


def greedy_degree(
    subgraphs: Iterable, graph: nx.Graph, target: int, resize_options: Optional[dict] = None
) -> list:
    """Method to greedily resize subgraphs based upon vertex degree.

    This function uses a greedy approach to iteratively add/remove nodes to an input subgraph.
    Suppose the input subgraph has :math:`M` nodes. When shrinking, the algorithm considers
    all the :math:`M` nodes and removes the one with the lowest degree (number of incident
    edges). It then repeats the process for the resultant :math:`M-1`-node subgraph,
    and continues until the desired subgraph size has been reached.

    When growing, the algorithm considers all :math:`N-M` nodes that are within the
    :math:`N`-node graph but not within the :math:`M` node subgraph, and then picks the node with
    the highest degree to create an :math:`M+1`-node subgraph. It then considers all
    :math:`N-M-1` nodes in the complement of the :math:`M+1`-node subgraph and adds the node with
    the highest degree, continuing until the desired subgraph size has been reached.

    ``resize_options`` is not currently in use in :func:`greedy_degree`.

    Args:
        subgraphs (iterable[list[int]]): an iterable over subgraphs, where each subgraph is given
            by a list of nodes
        graph (nx.Graph): the input graph
        target (int): the target number of nodes in the subgraph (must be
            larger than 1)
        resize_options (dict[str, Any]): dictionary specifying options used during resizing;
            defaults to :const:`RESIZE_DEFAULTS`

    Returns:
        list[list[int]]: a list of resized subgraphs, where each subgraph is given by a list of
        nodes
    """
    # Note that ``resize_options`` will become non-trivial once randomness is added to methods
    resize_options = {**RESIZE_DEFAULTS, **(resize_options or {})}

    nodes = graph.nodes

    resized = []

    for s in subgraphs:
        s = set(s)

        if not utils.is_subgraph(s, graph):
            raise ValueError("Input is not a valid subgraph")

        while len(s) != target:
            if len(s) < target:
                n_candidates = graph.degree(set(nodes) - s)
                n_opt = max(n_candidates, key=lambda x: x[1])[0]
                s |= {n_opt}
            else:
                n_candidates = graph.degree(s)
                n_opt = min(n_candidates, key=lambda x: x[1])[0]
                s -= {n_opt}

        resized.append(sorted(s))

    return resized


METHOD_DICT = {"greedy-density": greedy_density, "greedy-degree": greedy_degree}
"""dict[str, func]: Included methods for resizing subgraphs. The dictionary keys are strings
describing the method, while the dictionary values are callable functions corresponding to the
method."""


def clique_grow(clique: list, graph: nx.Graph, node_select: str = "uniform") -> list:
    """Iteratively adds new nodes to the input clique to generate a larger clique.

    Each iteration involves calculating the set :math:`C_0` (provided by the function
    :func:`~strawberryfields.apps.graph.utils.c_0`) with respect to the current clique. This set
    represents the nodes in the rest of the graph that are connected to all of the nodes in the
    current clique. Therefore, adding any of the nodes in :math:`C_0` will create a larger clique.
    This function proceeds by repeatedly evaluating :math:`C_0` and selecting and adding a node
    from this set to add to the current clique. Growth is continued until :math:`C_0` becomes empty.

    Whenever there are multiple nodes within :math:`C_0`, one must choose which node to add to
    the growing clique. This function allows a method of choosing nodes to be set with the
    ``node_select`` argument, with node selection based on uniform randomness and node degree
    supported. Degree-based node selection involves picking the node with the greatest degree,
    with ties settled by uniform random choice.

    Example usage:

    >>> graph = nx.complete_graph(10)
    >>> clique = [0, 1, 2, 3, 4]
    >>> clique_grow(clique, graph)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> graph = nx.lollipop_graph(5, 1)
    >>> graph.remove_edge(3, 4)
    >>> clique = [0, 1, 2]
    >>> clique_grow(clique, graph, node_select="degree")
    [0, 1, 2, 4]

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph
        node_select (str): method of selecting nodes from :math:`C_0` during growth. Can be either
            ``"uniform"`` for uniform random selection or ``"degree"`` for degree-based selection.
            Defaults to ``"uniform"``.

    Returns:
        list[int]: a new clique subgraph of equal or larger size than the input
    """

    if not utils.is_subgraph(clique, graph):
        raise ValueError("Input is not a valid subgraph")

    if not utils.is_clique(graph.subgraph(clique)):
        raise ValueError("Input subgraph is not a clique")

    clique = set(clique)
    c_0 = utils.c_0(clique, graph)

    while c_0:
        if node_select == "uniform":
            clique.add(np.random.choice(c_0))
        elif node_select == "degree":
            degrees = np.array([graph.degree(n) for n in c_0])
            to_add_index = np.random.choice(np.where(degrees == degrees.max())[0])
            to_add = c_0[to_add_index]
            clique.add(to_add)
        else:
            raise ValueError("Node selection method not recognized")

        c_0 = utils.c_0(clique, graph)

    return sorted(clique)


def clique_swap(clique: list, graph: nx.Graph, node_select: str = "uniform") -> list:
    """If possible, generates a new clique by swapping a node in the input clique with a node
    outside the clique.

    Proceeds by calculating the set :math:`C_1` of nodes in the rest of the graph that are
    connected to all but one of the nodes in the clique. If this set is not empty, this function
    randomly picks a node and swaps it with the corresponding node in the clique that is not
    connected to it. The set :math:`C_1` and corresponding nodes in the clique are provided by the
    :func:`~strawberryfields.apps.graph.utils.c_1` function.

    Whenever there are multiple nodes within :math:`C_1`, one must choose which node to add to
    the growing clique. This function allows a method of choosing nodes to be set with the
    ``node_select`` argument, with node selection based on uniform randomness and node degree
    supported. Degree-based node selection involves picking the node with the greatest degree,
    with ties settled by uniform random choice.

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph
        node_select (str): method of selecting nodes from :math:`C_0` during growth. Can be either
            ``"uniform"`` for uniform random selection or ``"degree"`` for degree-based selection.
            Defaults to ``"uniform"``.

       Returns:
           list[int]: a new clique subgraph of equal size as the input
       """

    if not utils.is_subgraph(clique, graph):
        raise ValueError("Input is not a valid subgraph")

    if not utils.is_clique(graph.subgraph(clique)):
        raise ValueError("Input subgraph is not a clique")

    clique = set(clique)
    c_1 = utils.c_1(clique, graph)

    if c_1:
        if node_select == "uniform":
            swap_index = np.random.choice(len(c_1))
            swap_nodes = c_1[swap_index]
        elif node_select == "degree":
            degrees = np.array([graph.degree(n[1]) for n in c_1])
            to_swap_index = np.random.choice(np.where(degrees == degrees.max())[0])
            swap_nodes = c_1[to_swap_index]
        else:
            raise ValueError("Node selection method not recognized")

        clique.remove(swap_nodes[0])
        clique.add(swap_nodes[1])

    return sorted(clique)


def clique_shrink(subgraph: list, graph: nx.Graph) -> list:
    """Shrinks an input subgraph until it forms a clique.

    Proceeds by removing nodes in the input subgraph one at a time until the result is a clique
    that satisfies :func:`~strawberryfields.apps.graph.utils.is_clique`. Upon each iteration,
    this function selects the node with the lowest degree relative to the subgraph and removes it.

    Example usage:

    >>> graph = nx.barbell_graph(4, 0)
    >>> subgraph = [0, 1, 2, 3, 4, 5]
    >>> clique_shrink(subgraph, graph)
    [0, 1, 2, 3]

    Args:
        subgraph (list[int]): a subgraph specified by a list of nodes
        graph (nx.Graph): the input graph

    Returns:
        list[int]: a clique of size smaller than or equal to the input subgraph
    """

    if not utils.is_subgraph(subgraph, graph):
        raise ValueError("Input is not a valid subgraph")

    subgraph = graph.subgraph(subgraph).copy()  # A copy is required to be able to modify the
    # structure of the subgraph (https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.Graph.subgraph.html)

    while not utils.is_clique(subgraph):
        degrees = list(subgraph.degree())
        np.random.shuffle(degrees)  # used to make sure selection of node with lowest degree is not
        # deterministic in case of a tie (https://docs.python.org/3/library/functions.html#min)

        to_remove = min(degrees, key=lambda x: x[1])
        subgraph.remove_node(to_remove[0])

    return sorted(subgraph.nodes())

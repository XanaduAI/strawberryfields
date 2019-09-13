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
Graph functions
===============

**Module name:** :mod:`strawberryfields.gbs.utils`

.. currentmodule:: strawberryfields.gbs.utils

This module provides some ancillary functions for dealing with graphs. This includes graph
validation functions such as :func:`is_undirected`, :func:`validate_graph` and :func:`is_subgraph`;
clique-based utility functions :func:`is_clique`, :func:`c_0` and :func:`c_1`; and a utility
function to return the adjacency matrix of a subgraph, :func:`subgraph_adjacency`.

Furthermore, ``graph_type = Union[nx.Graph, np.ndarray]`` is defined here as an input type that
can be either a NetworkX graph or an adjacency matrix in the form of a NumPy array. Some
functions in this module are able to accept either input.

Summary
-------

.. autosummary::
    is_undirected
    subgraph_adjacency
    validate_graph
    is_subgraph
    is_clique
    c_0
    c_1

Code details
^^^^^^^^^^^^
"""
from typing import Iterable, Union

import networkx as nx
import numpy as np

graph_type = Union[nx.Graph, np.ndarray]


def validate_graph(graph: graph_type) -> nx.Graph:
    """Checks if input graph is valid

    Checks if input Numpy array is a valid adjacency matrix, i.e., real and symmetric,
    and returns the corresponding NetworkX graph. If a NetworkX graph is input, this function
    simply returns the input.

    Args:
        graph (graph_type): input graph to be processed

    Returns:
        nx.Graph: the NetworkX graph corresponding to the input
    """
    if isinstance(graph, np.ndarray):
        if (
            graph.dtype is np.dtype("complex")
            and np.allclose(graph, graph.conj())
            and is_undirected(graph)
        ):
            graph = nx.Graph(graph.real.astype("float"))
        elif (graph.dtype is np.dtype("float") or graph.dtype is np.dtype("int")) and is_undirected(
            graph
        ):
            graph = nx.Graph(graph)
        else:
            raise ValueError(
                "Input NumPy arrays must be real and symmetric matrices and of dtype float or "
                "complex"
            )

    elif not isinstance(graph, nx.Graph):
        raise TypeError("Graph is not of valid type")

    return graph


def is_undirected(mat: np.ndarray) -> bool:
    """Checks if input is a symmetric NumPy array corresponding to an undirected graph

    Args:
        mat (array): input matrix to be checked

    Returns:
        bool: returns ``True`` if input array is symmetric and ``False`` otherwise
    """

    if not isinstance(mat, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array")

    dims = mat.shape

    conditions = len(dims) == 2 and dims[0] == dims[1] and dims[0] > 1 and np.allclose(mat, mat.T)

    return conditions


def subgraph_adjacency(graph: graph_type, nodes: list) -> np.ndarray:
    """Give adjacency matrix of a subgraph.

    Given a list of nodes selecting a subgraph, this function returns the corresponding adjacency
    matrix.

    Args:
        graph (graph_type): the input graph
        nodes (list): a list of nodes used to select the subgraph

    Returns:
        array: the adjacency matrix of the subgraph
    """
    graph = validate_graph(graph)
    all_nodes = graph.nodes

    if not set(nodes).issubset(all_nodes):
        raise ValueError(
            "Must input a list of subgraph nodes that is contained within the nodes of the input "
            "graph"
        )

    return nx.to_numpy_array(graph.subgraph(nodes))


def is_subgraph(subgraph: Iterable, graph: nx.Graph):
    """Checks if input is a valid subgraph.

    A valid subgraph is a set of nodes that are contained within the nodes of the graph.

    Args:
        subgraph (iterable): a collection of nodes
        graph (nx.Graph): the input graph

    Returns:
        bool: returns ``True`` only if input subgraph is valid
    """
    try:
        return set(subgraph).issubset(graph.nodes)
    except TypeError:
        raise TypeError("subgraph and graph.nodes must be iterable")


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
       list[tuple(int)]: A list of tuples. The first node in the tuple is the node in the clique
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

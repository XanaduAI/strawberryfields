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

**Module name:** :mod:`strawberryfields.apps.graph.utils`

.. currentmodule:: strawberryfields.apps.graph.utils

This module provides some ancillary functions for dealing with graphs. This includes
:func:`is_undirected` to check if an input matrix corresponds to an undirected graph (i.e.,
is symmetric) and :func:`subgraph_adjacency` to return the adjacency matrix of a subgraph when an
input graph and subset of nodes is specified.

Furthermore, the frontend :func:`~strawberryfields.apps.graph.dense_subgraph.find_dense`
function allows users to input graphs both as a `NumPy <https://www.numpy.org/>`__ array
containing the adjacency matrix and as a `NetworkX <https://networkx.github.io/>`__ ``Graph``
object. The :func:`to_networkx_graph` function allows both inputs to be processed into a NetworkX
Graph for ease of processing.

Summary
-------

.. autosummary::
    is_undirected
    subgraph_adjacency
    to_networkx_graph
    is_subgraph
    update_options

Code details
------------
"""
from typing import Union, Iterable

import networkx as nx
import numpy as np

graph_type = Union[nx.Graph, np.ndarray]


def to_networkx_graph(graph: graph_type) -> nx.Graph:
    """Converts input into a NetworkX ``Graph`` (i.e., an undirected graph).

    Given an input undirected graph of type ``graph_type = Union[nx.Graph, np.ndarray]``,
    this function outputs a NetworkX graph of type ``nx.Graph``. The input ``np.ndarray`` must be a
    matrix that is symmetric (i.e., satisfy :func:`is_undirected`) and also real.

    Args:
        graph (graph_type): input graph to be processed

    Returns:
        graph: the NetworkX graph corresponding to the input
    """
    if isinstance(graph, np.ndarray):
        if (
            graph.dtype is np.dtype("complex")
            and np.allclose(graph, graph.conj())
            and is_undirected(graph)
        ):
            graph = nx.Graph(graph.real.astype("float"))
        elif (
            graph.dtype is np.dtype("float") or graph.dtype is np.dtype("int")
        ) and is_undirected(graph):
            graph = nx.Graph(graph)
        else:
            raise Exception(
                "Graphs input as a NumPy array must be real, symmetric and of dtype "
                "float or complex"
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

    conditions = (
        len(dims) == 2
        and dims[0] == dims[1]
        and dims[0] > 1
        and np.allclose(mat, mat.T)
    )

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
    graph = to_networkx_graph(graph)
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

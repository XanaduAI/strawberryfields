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
validation functions such as :func:`is_undirected`, and :func:`is_subgraph`;
clique-based utility functions :func:`is_clique`, :func:`c_0` and :func:`c_1`; and a utility
function to return the adjacency matrix of a subgraph, :func:`subgraph_adjacency`.

Summary
-------

.. autosummary::
    is_undirected
    subgraph_adjacency
    is_subgraph

Code details
^^^^^^^^^^^^
"""
from typing import Iterable

import networkx as nx
import numpy as np


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


def subgraph_adjacency(graph: nx.Graph, nodes: list) -> np.ndarray:
    """Give adjacency matrix of a subgraph.

    Given a list of nodes selecting a subgraph, this function returns the corresponding adjacency
    matrix.

    Args:
        graph (nx.Graph): the input graph
        nodes (list): a list of nodes used to select the subgraph

    Returns:
        array: the adjacency matrix of the subgraph
    """
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

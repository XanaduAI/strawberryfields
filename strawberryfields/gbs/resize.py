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

**Module name:** :mod:`strawberryfields.gbs.resize`

.. currentmodule:: strawberryfields.gbs.resize

This module provides functionality for resizing of subgraphs.

Summary
-------

.. autosummary::
    METHOD_DICT
    RESIZE_DEFAULTS
    resize_subgraphs
    greedy_density
    greedy_degree

Code details
^^^^^^^^^^^^
"""

import itertools
from typing import Iterable, Optional

import networkx as nx

from strawberryfields.gbs import utils

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

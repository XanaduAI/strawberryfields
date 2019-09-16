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

**Module name:** :mod:`strawberryfields.gbs.dense`

.. currentmodule:: strawberryfields.gbs.dense

This module provides tools for users to identify dense subgraphs. Current functionality focuses
on the densest-:math:`k` subgraph problem :cite:`arrazola2018using`, which is NP-hard. This
problem considers an undirected graph :math:`G = (V, E)` of :math:`N` nodes :math:`V` and a list
of edges :math:`E`, and sets the objective of finding a :math:`k`-vertex subgraph with the
greatest density. In this setting, subgraphs :math:`G[S]` are defined by nodes :math:`S \subset
V` and corresponding edges :math:`E_{S} \subseteq E` that have both endpoints in :math:`S`. The
density of a subgraph is given by

.. math:: d(G[S]) = \frac{2 |E_{S}|}{|S|(|S|-1)},

where :math:`|\cdot|` denotes cardinality, and the densest-:math:`k` subgraph problem can be
written succinctly as

.. math:: {\rm argmax}_{S \in \mathcal{S}_{k}} d(G[S])

with :math:`\mathcal{S}_{k}` the set of all possible :math:`k` node subgraphs. This problem grows
combinatorially with :math:`{N \choose k}` and is NP-hard in the worst case.

Heuristics
----------

The :func:`search` function provides access to heuristic algorithms for finding
approximate solutions. At present, random search is the heuristic algorithm provided, accessible
through the :func:`random_search` function. This algorithm proceeds by randomly generating a set
of :math:`k` vertex subgraphs and selecting the densest. Sampling of subgraphs can be achieved
both uniformly at random and also with a quantum sampler programmed to be biased toward
outputting dense subgraphs.

.. autosummary::
    search
    random_search
    OPTIONS_DEFAULTS
    METHOD_DICT

Subgraph resizing
-----------------

Subgraphs sampled from GBS are not guaranteed to be of size :math:`k`, even if this is the mean
photon number. On the other hand, the densest-:math:`k` subgraph problem requires graphs of fixed
size to be considered. This means that heuristic algorithms at some point must resize the sampled
subgraphs. Resizing functionality is provided by the following functions.

.. autosummary::
    resize_subgraphs
    greedy_density
    greedy_degree
    RESIZE_DEFAULTS
    RESIZE_DICT

Code details
^^^^^^^^^^^^
"""
import itertools
from typing import Iterable, Optional, Tuple

import networkx as nx

from strawberryfields.gbs import g_sample, utils
from strawberryfields.gbs.sample import BACKEND_DEFAULTS
from strawberryfields.gbs.utils import graph_type


def search(
    graph: graph_type, nodes: int, iterations: int = 1, options: Optional[dict] = None
) -> Tuple[float, list]:
    """Find a dense subgraph of a given size.

    This function returns the densest `node-induced subgraph
    <http://mathworld.wolfram.com/Vertex-InducedSubgraph.html>`__ of size ``nodes`` after
    multiple repetitions. It uses heuristic optimization that combines search space exploration
    with local searching. The heuristic method can be set with the ``options`` argument. Methods
    can contain stochastic elements, where randomness can come from distributions including GBS.

    All elements of the heuristic can be controlled with the ``options`` argument, which should be a
    dict-of-dicts where the first level specifies the option type as a string-based key,
    with corresponding value being a dictionary of options for that type. The option types are:

    - ``"heuristic"``: specifying options used by optimization heuristic; corresponding
      dictionary of options explained further :ref:`below <heuristic>`
    - ``"backend"``: specifying options used by backend quantum samplers; corresponding
      dictionary of options explained further in :mod:`~strawberryfields.gbs.sample`
    - ``"resize"``: specifying options used by resizing method; corresponding dictionary of
      options explained further in :mod:`~strawberryfields.gbs.resize`
    - ``"sample"``: specifying options used in sampling; corresponding dictionary of options
      explained further in :mod:`~strawberryfields.gbs.sample`

    If unspecified, a default set of options is adopted for a given option type.

    .. _heuristic:

    The options dictionary corresponding to ``"heuristic"`` can contain any of the following:

    .. glossary::

        key: ``"method"``, value: *str* or *callable*
            Value can be either a string selecting from a range of available methods or a
            customized callable function. Options include:

            - ``"random-search"``: a simple random search algorithm where many subgraphs are
              selected and the densest one is chosen (default)
            - *callable*: accepting ``(graph: nx.Graph, nodes: int, iterations: int, options:
              dict)`` as arguments and returning ``Tuple[float, list]`` corresponding to the
              density and list of nodes of the densest subgraph found, see :func:`random_search`
              for an example

    Args:
        graph (graph_type): the input graph
        nodes (int): the size of desired dense subgraph
        iterations (int): number of iterations to use in algorithm
        options (dict[str, dict[str, Any]]): dict-of-dicts specifying options in different parts
            of heuristic search algorithm; defaults to :const:`OPTIONS_DEFAULTS`

    Returns:
        tuple[float, list]: the density and list of nodes corresponding to the densest subgraph
        found
    """
    options = {**OPTIONS_DEFAULTS, **(options or {})}

    method = options["heuristic"]["method"]

    if not callable(method):
        method = METHOD_DICT[method]

    return method(
        graph=utils.validate_graph(graph), nodes=nodes, iterations=iterations, options=options
    )


def random_search(
    graph: nx.Graph, nodes: int, iterations: int = 1, options: Optional[dict] = None
) -> Tuple[float, list]:
    """Random search algorithm for finding dense subgraphs of a given size.

    The algorithm proceeds by sampling subgraphs according to the
    :func:`~strawberryfields.gbs.sample.sample_subgraphs`. The resultant subgraphs
    are resized using :func:`~strawberryfields.gbs.dense.resize_subgraphs` to
    be of size ``nodes``. The densest subgraph is then selected among all the resultant
    subgraphs. Specified``options`` must be of the form given in :func:`search`.

    Args:
        graph (nx.Graph): the input graph
        nodes (int): the size of desired dense subgraph
        iterations (int): number of iterations to use in algorithm
        options (dict[str, dict[str, Any]]): dict-of-dicts specifying options in different parts
            of heuristic search algorithm; defaults to :const:`OPTIONS_DEFAULTS`

    Returns:
        tuple[float, list]: the density and list of nodes corresponding to the densest subgraph
        found
    """
    options = {**OPTIONS_DEFAULTS, **(options or {})}

    samples = g_sample.sample_subgraphs(
        graph=graph,
        nodes=nodes,
        samples=iterations,
        sample_options=options["sample"],
        backend_options=options["backend"],
    )

    samples = resize_subgraphs(
        subgraphs=samples, graph=graph, target=nodes, resize_options=options["resize"]
    )

    density_and_samples = [(nx.density(graph.subgraph(s)), s) for s in samples]

    return max(density_and_samples)


METHOD_DICT = {"random-search": random_search}
"""dict[str, func]: Included methods for finding dense subgraphs. The dictionary keys are strings
describing the method, while the dictionary values are callable functions corresponding to the
method."""

RESIZE_DEFAULTS = {"method": "greedy-density"}
"""dict[str, Any]: Dictionary to specify default parameters of options for resizing
"""

OPTIONS_DEFAULTS = {
    "heuristic": {"method": random_search},
    "backend": BACKEND_DEFAULTS,
    "resize": RESIZE_DEFAULTS,
    "sample": g_sample.SAMPLE_DEFAULTS,
}
"""dict[str, dict[str, Any]]: Options for dense subgraph identification heuristics. Composed of a
dictionary of dictionaries with the first level specifying the option type, selected from keys
``"heuristic"``, ``"backend"``, ``"resize"``, and ``"sample"``, with the corresponding value a
dictionary of options for that type.
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
        method = RESIZE_DICT[method]

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


RESIZE_DICT = {"greedy-density": greedy_density, "greedy-degree": greedy_degree}
"""dict[str, func]: Included methods for resizing subgraphs. The dictionary keys are strings
describing the method, while the dictionary values are callable functions corresponding to the
method."""

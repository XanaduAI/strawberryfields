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
of :math:`k` vertex subgraphs and selecting the densest.

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
subgraphs. Resizing functionality is provided by the following function.

.. autosummary::
    resize

Code details
^^^^^^^^^^^^
"""
from typing import Optional, Tuple

import networkx as nx
import numpy as np

from strawberryfields.gbs import sample


def search(
        graph: nx.Graph, nodes: int, iterations: int = 1, options: Optional[dict] = None
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

    method = options["heuristic"]["method"]

    if not callable(method):
        method = METHOD_DICT[method]

    return method(graph=graph, nodes=nodes, iterations=iterations, options=options)


def random_search(
        graph: nx.Graph, nodes: int, iterations: int = 1, options: Optional[dict] = None
) -> Tuple[float, list]:
    """Random search algorithm for finding dense subgraphs of a given size.

    The algorithm proceeds by sampling subgraphs according to the
    :func:`~strawberryfields.gbs.sample.subgraphs` function. The resultant subgraphs
    are resized using :func:`resize` to be of size ``nodes``. The densest subgraph is then
    selected among all the resultant subgraphs. Specified``options`` must be of the form given in
    :func:`search`.

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

    samples = sample.sample(
        nx.to_numpy_array(graph), n_mean=nodes, n_samples=iterations, threshold=True
    )

    samples = sample.to_subgraphs(samples, graph)
    samples = [resize(s, graph, [nodes])[nodes] for s in samples]

    density_and_samples = [(nx.density(graph.subgraph(s)), s) for s in samples]

    return max(density_and_samples)


METHOD_DICT = {"random-search": random_search}
"""dict[str, func]: Included methods for finding dense subgraphs. The dictionary keys are strings
describing the method, while the dictionary values are callable functions corresponding to the
method."""

OPTIONS_DEFAULTS = {"heuristic": {"method": random_search}}
"""dict[str, dict[str, Any]]: Options for dense subgraph identification heuristics. Composed of a
dictionary of dictionaries with the first level specifying the option type, selected from keys
``"heuristic"`` and ``"resize"``, with the corresponding value a
dictionary of options for that type.
"""


def resize(subgraph: list, graph: nx.Graph, sizes: Optional[list] = None) -> list:
    """Resize a subgraph to a range of input sizes.

    This function uses a greedy approach to iteratively add or remove nodes one at a time to an
    input subgraph to reach the range of sizes specified by ``sizes``.

    When growth is required, the algorithm picks individual nodes from the remainder of the graph
    as candidates to add into the subgraph and picks the node with the highest degree relative to
    the rest of the subgraph. This results in a graph that is one node larger, and if growth is
    still required the algorithm performs the above procedure again.

    When shrinking is required, the algorithm picks individual nodes from within the subgraph to
    remove, choosing the nodes with lowest degree relative to the subgraph. In both growth and
    shrink phases, ties for addition/removal with nodes of equal degree are settled by uniform
    random choice.

    Args:
        subgraph (list[int]): a subgraph specified by a list of nodes
        graph (nx.Graph): the input graph
        sizes (list[int]): a list of desired sized for the subgraph to be resized to; defaults to
            all possible nonzero sizes

    Returns:
        dict[int, list[int]]: a dictionary of different sizes with corresponding subgraph
    """
    subgraph = set(subgraph)
    if not subgraph.issubset(graph.nodes):
        raise ValueError("Input is not a valid subgraph")

    nodes = graph.nodes()
    all_sizes = set(range(1, len(nodes)))

    if sizes is None:
        sizes = all_sizes
    else:
        sizes = set(sizes)

        if not sizes.issubset(all_sizes):
            raise ValueError("Requested sizes must be within size range of graph")

    starting_size = len(subgraph)
    max_size = max(sizes)
    min_size = min(sizes)

    if starting_size in sizes:
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

            if new_size in sizes:
                resized[new_size] = sorted(grow_subgraph.nodes())

    if min_size < starting_size:

        shrink_subgraph = graph.subgraph(subgraph).copy()

        while shrink_subgraph.order() > min_size:
            degrees = list(shrink_subgraph.degree())
            np.random.shuffle(degrees)

            to_remove = min(degrees, key=lambda x: x[1])
            shrink_subgraph.remove_node(to_remove[0])

            new_size = shrink_subgraph.order()

            if new_size in sizes:
                resized[new_size] = sorted(shrink_subgraph.nodes())

    return resized

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
Graph sampling
==============

**Module name:** :mod:`strawberryfields.gbs.sample`

.. currentmodule:: strawberryfields.gbs.sample

This module provides functionality for sampling of subgraphs from undirected graphs. The
:func:`sample_subgraphs` function generates raw samples from
:mod:`strawberryfields.gbs.sample` and converts them to subgraphs using
:func:`to_subgraphs`.

Summary
-------

.. autosummary::
    SAMPLE_DEFAULTS
    sample_subgraphs
    to_subgraphs

Code details
^^^^^^^^^^^^
"""
from typing import Optional

import networkx as nx
import numpy as np

from strawberryfields.gbs import sample

SAMPLE_DEFAULTS = {"distribution": "gbs", "postselect_ratio": 0.75}
"""dict[str, Any]: Dictionary to specify default parameters of options in :func:`sample_subgraphs`.
"""


def sample_subgraphs(
    graph: nx.Graph,
    nodes: int,
    samples: int = 1,
    sample_options: Optional[dict] = None,
    backend_options: Optional[dict] = None,
) -> list:
    """Functionality for sampling subgraphs.

    The optional ``sample_options`` argument can be used to specify the type of sampling. It
    should be a dict that contains any of the following:

    .. glossary::

        key: ``"distribution"``, value: *str*
            Subgraphs can be sampled according to the following distributions:

            - ``"gbs"``: for generating subgraphs according to the Gaussian boson sampling
              distribution. In this distribution, subgraphs are sampled with a variable size (
              default).
            - ``"uniform"``: for generating subgraphs uniformly at random. When using this
              distribution, subgraphs are of a fixed size. Note that ``backend_options`` are not
              used and that remote sampling is unavailable due to the simplicity of the
              distribution.

        key: ``"postselect_ratio"``, value: *float*
            Ratio of ``nodes`` used to determine the minimum size of subgraph sampled; defaults
            to 0.75.

    Args:
        graph (nx.Graph): the input graph
        nodes (int): the mean size of subgraph samples
        samples (int): number of samples; defaults to 1
        sample_options (dict[str, Any]): dictionary specifying options used by ``sample_subgraphs``;
            defaults to :const:`SAMPLE_DEFAULTS`
        backend_options (dict[str, Any]): dictionary specifying options used by backends during
            sampling; defaults to ``None``

    Returns:
        list[list[int]]: a list of length ``samples`` whose elements are subgraphs given by a
        list of nodes
    """
    sample_options = {**SAMPLE_DEFAULTS, **(sample_options or {})}

    distribution = sample_options["distribution"]

    if distribution == "uniform":
        s = sample.uniform_sampler(modes=graph.order(), sampled_modes=nodes, samples=samples)
    elif distribution == "gbs":
        postselect = int(sample_options["postselect_ratio"] * nodes)
        backend_options = {**(backend_options or {}), "postselect": postselect}

        s = sample.quantum_sampler(
            A=nx.to_numpy_array(graph),
            n_mean=nodes,
            samples=samples,
            backend_options=backend_options,
        )
    else:
        raise ValueError("Invalid distribution selected")

    return to_subgraphs(graph, s)


def to_subgraphs(graph: nx.Graph, samples: list) -> list:
    """Converts a list of samples to a list of subgraphs.

    Given a list of samples, with each sample of ``len(nodes)`` being a list of zeros and ones,
    this function returns a list of subgraphs selected by, for each sample, picking the nodes
    corresponding to ones and creating the induced subgraph. The subgraph is specified as a list
    of selected nodes. For example, given an input 6-node graph, a sample :math:`[0, 1, 1, 0, 0,
    1]` is converted to a subgraph :math:`[1, 2, 5]`.

    Args:
        graph (nx.Graph): the input graph
        samples (list): a list of samples, each a binary sequence of ``len(nodes)``

    Returns:
        list[list[int]]: a list of subgraphs, where each subgraph is represented by a list of its
        nodes
    """

    graph_nodes = list(graph.nodes)
    node_number = len(graph_nodes)

    def to_subgraph(s: list) -> list:
        """Convert a single sample to a subgraph.

        Args:
           s (list): a binary sample of ``len(nodes)``

        Returns:
            list: a subgraph specified by its nodes
        """
        return list(np.nonzero(s)[0])

    subgraph_samples = [to_subgraph(s) for s in samples]

    if graph_nodes != list(range(node_number)):
        return [sorted([graph_nodes[i] for i in s]) for s in subgraph_samples]

    return subgraph_samples

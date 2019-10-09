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
Sampling functions
==================

**Module name:** :mod:`strawberryfields.gbs.sample`

.. currentmodule:: strawberryfields.gbs.sample

This module provides functionality for generating samples from GBS.

Gaussian boson sampling (GBS)
-----------------------------

The :func:`sample` function allows users to simulate Gaussian boson sampling (GBS) by
choosing a symmetric input matrix to sample from :cite:`bradler2018gaussian`. For a symmetric
:math:`N \times N` input matrix :math:`A`, corresponding to an undirected graph,
an :math:`N`-mode GBS device with threshold detectors generates samples that are binary strings
of length ``N``. Various problems can be encoded in the matrix :math:`A` so that the output
samples are informative and can be used as a form of randomness to improve solvers, as outlined
in Refs. :cite:`arrazola2018using` and :cite:`arrazola2018quantum`.

A utility function :func:`modes_from_counts` is also provided to convert between two ways to
represent samples from the device.

.. autosummary::
    sample
    seed

Subgraph sampling through GBS
-----------------------------

This module also provides functionality for sampling of subgraphs from undirected graphs. The
:func:`to_subgraphs` function can be used to convert samples to subgraphs.

.. autosummary::
    to_subgraphs
    modes_from_counts

Code details
^^^^^^^^^^^^
"""
from typing import Optional

import networkx as nx
import numpy as np

import strawberryfields as sf


def sample(A: np.ndarray, n_mean: float, n_samples: int = 1, threshold: bool = True) -> list:
    r"""Generate samples from GBS by encoding a symmetric matrix :math:`A`.

    Args:
        A (array): the symmetric matrix to sample from
        n_mean (float): mean photon number
        n_samples (int): number of samples
        threshold (bool): perform GBS with threshold detectors if ``True`` or photon-number
            resolving detectors if ``False``

    Returns:
        list[list[int]]: a list of samples from a simulation of GBS with respect to the input
        symmetric matrix
    """
    if not np.allclose(A, A.T):
        raise ValueError("Input must be a NumPy array corresponding to a symmetric matrix")
    if n_samples < 1:
        raise ValueError("Number of samples must be at least one")
    if n_mean < 0:
        raise ValueError("Mean photon number must be non-negative")

    nodes = len(A)

    p = sf.Program(nodes)
    eng = sf.LocalEngine(backend="gaussian")

    mean_photon_per_mode = n_mean / float(nodes)

    # pylint: disable=expression-not-assigned,pointless-statement
    with p.context as q:
        sf.ops.GraphEmbed(A, mean_photon_per_mode=mean_photon_per_mode) | q

        if threshold:
            sf.ops.MeasureThreshold() | q
        else:
            sf.ops.MeasureFock() | q

    return eng.run(p, run_options={"shots": n_samples}).samples.tolist()


def seed(value: Optional[int]) -> None:
    """Seed for random number generators.

    Wrapper function for `numpy.random.seed <https://docs.scipy.org/doc/numpy//reference/generated
    /numpy.random.seed.html>`_ to seed all NumPy-based random number generators. This allows for
    repeatable sampling.

    Args:
        value (int): random seed
    """
    np.random.seed(value)


def to_subgraphs(samples: list, graph: nx.Graph) -> list:
    """Converts lists of samples to lists of subgraphs.

    For a list of samples, with each sample of ``len(nodes)`` being a list of zeros and ones,
    this function returns a list of subgraphs selected by, for each sample, picking the nodes
    with nonzero entries and creating the induced subgraph. The subgraph is specified as a list
    of selected nodes. For example, given an input 6-node graph, a sample :math:`[0, 1, 1, 0, 0,
    1]` is converted to a subgraph :math:`[1, 2, 5]`.

    Args:
        samples (list[list[int]]): a list of samples
        graph (nx.Graph): the input graph

    Returns:
        list[list[int]]: a list of subgraphs, where each subgraph is represented by a list of its
        nodes
    """

    graph_nodes = list(graph.nodes)
    node_number = len(graph_nodes)

    subgraph_samples = [list(set(modes_from_counts(s))) for s in samples]

    if graph_nodes != list(range(node_number)):
        return [sorted([graph_nodes[i] for i in s]) for s in subgraph_samples]

    return subgraph_samples


def modes_from_counts(s: list) -> list:
    r"""Convert a sample of photon counts to a list of modes where photons are detected.

    There are two main methods of representing a sample. In the first, the number of photons
    detected in each mode is specified. In the second, the modes in which each photon was detected
    are listed. Since there are typically fewer photons than modes, the second method gives a
    shorter representation.

    This function converts from the first representation to the second. Given an :math:`N` mode
    sample :math:`s=\{s_{1},s_{2},\ldots,s_{N}\}` of photon counts in each mode with total photon
    number :math:`k`, this function returns a list of modes :math:`m=\{m_{1},m_{2},\ldots,
    m_{k}\}` where photons are detected.

    **Example usage:**

    >>> modes_from_counts([0, 1, 0, 1, 2, 0])
    [1, 3, 4, 4]

    Args:
       s (list[int]): a sample of photon counts

    Returns:
        list[int]: a list of modes where photons are detected, sorted in non-decreasing order
    """
    modes = []
    for i, c in enumerate(s):
        modes += [i] * c
    return sorted(modes)


def postselect(samples: list, min_count: int, max_count: int) -> list:
    """Postselect samples by imposing a minimum and maximum number of photons or clicks.

    Args:
        samples (list[list[int]]): a list of samples
        min_count (int): minimum number of photons or clicks for a sample to be included
        max_count (int): maximum number of photons or clicks for a sample to be included

    Returns:
        list[list[int]]: the postselected samples
    """
    return [s for s in samples if min_count <= sum(s) <= max_count]

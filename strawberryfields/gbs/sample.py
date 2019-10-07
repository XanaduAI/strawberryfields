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

This module provides functionality for generating samples from both a quantum device and the
uniform distribution.

Gaussian boson sampling (GBS)
-----------------------------

The :func:`sample` function allows users to simulate Gaussian boson sampling (GBS) by
choosing a symmetric input matrix to sample from :cite:`bradler2018gaussian`. For a symmetric
:math:`N \times N` input matrix :math:`A`, corresponding to an undirected graph,
an :math:`N`-mode GBS device with threshold detectors generates samples that are binary strings
of length ``N``. Various problems can be encoded in the matrix :math:`A` so that the output
samples are informative and can be used as a form of randomness to improve solvers, as outlined
in Refs. :cite:`arrazola2018using` and :cite:`arrazola2018quantum`.

On the other hand, the :func:`uniform` function allows users to generate samples where a
subset of modes are selected using the uniform distribution. A utility function
:func:`modes_from_counts` is also provided to convert between two ways to represent samples from
the device.

.. autosummary::
    sample
    uniform
    seed
    modes_from_counts

Subgraph sampling through GBS
-----------------------------

This module also provides functionality for sampling of subgraphs from undirected graphs. The
:func:`subgraphs` function generates raw samples from :mod:`strawberryfields.gbs.sample`
and converts them to subgraphs using :func:`to_subgraphs`. Sampling can be generated both from
GBS and by using the uniform distribution.

.. autosummary::
    SAMPLE_DEFAULTS
    subgraphs
    to_subgraphs

Code details
^^^^^^^^^^^^
"""
from typing import Optional

import networkx as nx
import numpy as np

import strawberryfields as sf


def sample(
    A: np.ndarray, n_mean: float, n_samples: int = 1, threshold: bool=True
) -> list:
    r"""Generate samples from GBS by encoding a symmetric matrix :math:`A`.

    Args:
        A (array): the :math:`N \times N` symmetric matrix to sample from
        n_mean (float): mean photon number
        n_samples (int): number of samples
        threshold (bool): perform GBS with threshold detectors if ``True`` or photon-number
            resolving detectors if ``False``

    Returns:
        list[list[int]]: a list of length ``samples`` whose elements are length :math:`N` lists of
        integers indicating events (e.g., photon numbers or clicks) in each of the :math:`N`
        modes of the detector
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
            sf.ops.MeasureThreshold | q
        else:
            sf.ops.MeasureFock | q

    return eng.run(p, run_options={"shots": n_samples}).samples.tolist()


def uniform(modes: int, sampled_modes: int, samples: int = 1) -> list:
    """Perform classical sampling using the uniform distribution to randomly select a subset of
    modes of a given size.

    Args:
        modes (int): number of modes to sample from
        sampled_modes (int): number of modes to sample
        samples (int): number of samples; defaults to 1

    Returns:
        list[list[int]]: a ``len(samples)`` list whose elements are ``len(modes)`` lists which
        contain zeros and ones. The number of ones is ``samples_modes``, corresponding to the modes
        selected.
    """

    if modes < 1 or sampled_modes < 1 or sampled_modes > modes:
        raise ValueError(
            "Modes and sampled_modes must be greater than zero and sampled_modes must not exceed "
            "modes "
        )
    if samples < 1:
        raise ValueError("Number of samples must be greater than zero")

    base_sample = [1] * sampled_modes + [0] * (modes - sampled_modes)

    output_samples = [list(np.random.permutation(base_sample)) for _ in range(samples)]

    return output_samples


def seed(value: Optional[int]) -> None:
    """Seed for random number generators.

    Wrapper function for `numpy.random.seed <https://docs.scipy.org/doc/numpy//reference/generated
    /numpy.random.seed.html>`_ to seed all NumPy-based random number generators. This allows for
    repeatable sampling.

    Args:
        value (int): random seed
    """
    np.random.seed(value)


SAMPLE_DEFAULTS = {"distribution": "gbs", "postselect_ratio": 0.75}
"""dict[str, Any]: Dictionary to specify default parameters of options in :func:`subgraphs`.
"""


def subgraphs(
    graph: nx.Graph,
    nodes: int,
    samples: int = 1,
    sample_options: Optional[dict] = None,
    backend_options: Optional[dict] = None,
) -> list:
    """Samples subgraphs from an input graph

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
        samples (int): number of samples
        sample_options (dict[str, Any]): dictionary specifying options used by :func:`subgraphs`;
            defaults to :const:`SAMPLE_DEFAULTS`
        backend_options (dict[str, Any]): dictionary specifying options used by backends during
            sampling

    Returns:
        list[list[int]]: a list of length ``samples`` whose elements are subgraphs given by a
        list of nodes
    """
    sample_options = {**SAMPLE_DEFAULTS, **(sample_options or {})}

    distribution = sample_options["distribution"]

    if distribution == "uniform":
        s = uniform(modes=graph.order(), sampled_modes=nodes, samples=samples)
    elif distribution == "gbs":
        postselect = int(sample_options["postselect_ratio"] * nodes)
        backend_options = {**(backend_options or {}), "postselect": postselect}

        s = sample(
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

    subgraph_samples = [modes_from_counts(s) for s in samples]

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

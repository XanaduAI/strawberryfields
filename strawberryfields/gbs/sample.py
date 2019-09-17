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
subset of modes are selected using the uniform distribution.

.. autosummary::
    QUANTUM_BACKENDS
    BACKEND_DEFAULTS
    sample
    uniform
    seed

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
from strawberryfields.gbs import utils

QUANTUM_BACKENDS = ("gaussian",)
"""tuple[str]: Available quantum backends for sampling."""

BACKEND_DEFAULTS = {"remote": False, "backend": "gaussian", "threshold": True, "postselect": 0}
"""dict[str, Any]: Dictionary to specify default parameters of options in backend sampling for
:func:`sample`.
"""


def sample(
    A: np.ndarray, n_mean: float, samples: int = 1, backend_options: Optional[dict] = None
) -> list:
    r"""Generate samples from GBS

    Perform quantum sampling of adjacency matrix :math:`A` using the Gaussian boson sampling
    algorithm.

    Sampling can be controlled with the optional ``backend_options`` argument, which should be a
    dict containing any of the following:

    .. glossary::

        key: ``"remote"``, value: *bool*
            Performs sampling on a remote server if ``True``. Remote sampling is
            required for sampling on hardware. If not specified, sampling will be performed locally.

        key: ``"backend"``, value: *str*
            Requested backend for sampling. Available backends are listed in
            ``QUANTUM_BACKENDS``, these are:

            - ``"gaussian"``: for simulating the output of a GBS device using the
              :mod:`strawberryfields.backends.gaussianbackend`

        key: ``"threshold"``, value: *bool*
            Performs sampling using threshold (on/off click) detectors if ``True``
            or using photon number resolving detectors if ``False``. Defaults to ``True`` for
            threshold sampling.

        key: ``"postselect"``, value: *int*
            Causes samples with a photon number or click number less than the
            specified value to be filtered out. Defaults to ``0`` if unspecified, resulting in no
            postselection. Note that the number of samples returned in :func:`sample` is
            still equal to the ``samples`` parameter.

    Args:
        A (array): the (real or complex) :math:`N \times N` adjacency matrix to sample from
        n_mean (float): mean photon number
        samples (int): number of samples; defaults to 1
        backend_options (dict[str, Any]): dictionary specifying options used by backends during
            sampling; defaults to :const:`BACKEND_DEFAULTS`

    Returns:
        list[list[int]]: a list of length ``samples`` whose elements are length :math:`N` lists of
        integers indicating events (e.g., photon numbers or clicks) in each of the :math:`N`
        modes of the detector
    """
    backend_options = {**BACKEND_DEFAULTS, **(backend_options or {})}

    if not utils.is_undirected(A):
        raise ValueError("Input must be a NumPy array corresponding to a symmetric matrix")

    if samples < 1:
        raise ValueError("Number of samples must be at least one")

    nodes = len(A)
    p = sf.Program(nodes)

    mean_photon_per_mode = n_mean / float(nodes)

    # pylint: disable=expression-not-assigned,pointless-statement
    with p.context as q:
        sf.ops.GraphEmbed(A, mean_photon_per_mode=mean_photon_per_mode) | q
        sf.ops.Measure | q

    p = p.compile("gbs")

    postselect = backend_options["postselect"]
    threshold = backend_options["threshold"]

    if postselect == 0:
        s = _sample_sf(p, shots=samples, backend_options=backend_options)

        if threshold:
            s[s >= 1] = 1

        s = s.tolist()

    elif postselect > 0:
        s = []

        while len(s) < samples:
            samp = np.array(_sample_sf(p, shots=1, backend_options=backend_options))

            if threshold:
                samp[samp >= 1] = 1

            counts = np.sum(samp)

            if counts >= postselect:
                s.append(samp.tolist())

    else:
        raise ValueError("Can only postselect on nonnegative values")

    return s


def _sample_sf(p: sf.Program, shots: int = 1, backend_options: Optional[dict] = None) -> np.ndarray:
    """Generate samples from Strawberry Fields.

    Args:
        p (sf.Program): the program to sample from
        shots (int): the number of samples; defaults to 1
        backend_options (dict[str, Any]): dictionary specifying options used by backends during
        sampling; defaults to :const:`BACKEND_DEFAULTS`

    Returns:
        array: an array of ``len(shots)`` samples, with each sample the result of running a
        :class:`~strawberryfields.program.Program` specified by ``p``
    """
    backend_options = {**BACKEND_DEFAULTS, **(backend_options or {})}

    if not backend_options["backend"] in QUANTUM_BACKENDS:
        raise ValueError("Invalid backend selected")

    if backend_options["remote"]:
        raise ValueError("Remote sampling not yet available")

    eng = sf.LocalEngine(backend=backend_options["backend"])

    return np.array(eng.run(p, run_options={"shots": shots}).samples)


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


def seed(value: int = None) -> None:
    """Seed for random number generators.

    Wrapper function for `numpy.random.seed <https://docs.scipy.org/doc/numpy//reference/generated
    /numpy.random.seed.html>`_ to seed NumPy-based random number generators used in the GBS
    applications layer. This allows for repeatable sampling.

    Args:
        value (int): random seed; defaults to ``None``
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
        sample_options (dict[str, Any]): dictionary specifying options used by ``subgraphs``;
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

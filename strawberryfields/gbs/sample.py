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

This module provides functionality for generating GBS samples using classical simulators.

An accompanying tutorial can be found :ref:`here <gbs-sample-tutorial>`.

Generating samples
------------------

An :math:`M` mode GBS device can be programmed by specifying an :math:`(M \times M)`-dimensional
symmetric matrix :math:`A` :cite:`bradler2018gaussian`. Running this device results in samples
that carry relevant information about the encoded matrix :math:`A`. When sampling, one must also
specify the mean number of photons in the device, the form of detection used at the output:
threshold detection or photon-number-resolving (PNR) detection, as well as the amount of loss.

The :func:`sample` function provides a simulation of sampling from GBS:

.. autosummary::
    sample

Here, each output sample is an :math:`M`-dimensional list. If threshold detection is used
(``threshold = True``), each element of a sample is either a zero (denoting no photons detected)
or a one (denoting one or more photons detected), conventionally called a "click".
If photon-number resolving (PNR) detection is used (``threshold = False``) then elements of a
sample are non-negative integers counting the number of photons detected in each mode. The ``loss``
parameter allows for simulation of photon loss in the device, which is the most common form of
noise in quantum photonics. Here, ``loss = 0`` describes ideal loss-free GBS, while generally
``0 <= loss <= 1`` describes the proportion of photons lost while passing through the device.

Samples can be postselected based upon their total number of photons or clicks and the ``numpy``
random seed used to generate samples can be fixed:

.. autosummary::
    postselect
    seed

.. _decomposition:

The :func:`vibronic` function allows users to sample Gaussian states for computing molecular
vibronic spectra.

.. autosummary::
    vibronic

Generating subgraphs
--------------------

There are two forms to represent a sample from a GBS device:

1. In the form returned by :func:`sample`, each sample is a length :math:`M` list of counts in
   each mode, e.g., the sample ``[2, 1, 2, 0, 1]`` denotes two photons in mode 0,
   1 photon in mode 1, and so forth.

2. The alternative representation is a list of modes where photons or clicks were observed, e.g.,
   the above sample can be written alternatively as ``[0, 0, 1, 2, 2, 4]``.

Converting from the former representation to the latter can be achieved with:

.. autosummary::
    modes_from_counts

Graphs can be encoded into GBS by setting the adjacency matrix to be :math:`A`. Resultant samples
can then be used to pick out nodes from the graph to form a subgraph. If threshold detection is
used, any nodes that click are selected as part of the subgraph. For example, if a sample is
``[1, 1, 1, 0, 1]`` then the corresponding subgraph has nodes ``[0, 1, 2, 4]``. This can be found
using:

>>> modes_from_counts([1, 1, 1, 0, 1])
[0, 1, 2, 4]

A collection of GBS samples from a graph, given by :func:`sample` in the first form, can be
converted to subgraphs using:

.. autosummary::
    to_subgraphs

A typical workflow would be:

>>> g = nx.erdos_renyi_graph(5, 0.7)
>>> a = nx.to_numpy_array(g)
>>> s = sample(a, 3, 4)
>>> s = to_subgraphs(s, g)
[[0, 2], [0, 1, 2, 4], [1, 2, 3, 4], [1]]

The subgraphs sampled from GBS are likely to be dense :cite:`arrazola2018using`, motivating their
use within heuristics for problems such as maximum clique (see :mod:`~.gbs.clique`).

Code details
^^^^^^^^^^^^
"""
import warnings
from typing import Optional

import networkx as nx
import numpy as np

import strawberryfields as sf


def sample(
    A: np.ndarray, n_mean: float, n_samples: int = 1, threshold: bool = True, loss: float = 0.0
) -> list:
    r"""Generate simulated samples from GBS encoded with a symmetric matrix :math:`A`.

    **Example usage:**

    >>> g = nx.erdos_renyi_graph(5, 0.7)
    >>> a = nx.to_numpy_array(g)
    >>> sample(a, 3, 4)
    [[1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]]

    Args:
        A (array): the symmetric matrix to sample from
        n_mean (float): mean photon number
        n_samples (int): number of samples
        threshold (bool): perform GBS with threshold detectors if ``True`` or photon-number
            resolving detectors if ``False``
        loss (float): fraction of generated photons that are lost while passing through device.
            Parameter should range from ``loss=0`` (ideal noise-free GBS) to ``loss=1``.

    Returns:
        list[list[int]]: a list of samples from GBS with respect to the input symmetric matrix
    """
    if not np.allclose(A, A.T):
        raise ValueError("Input must be a NumPy array corresponding to a symmetric matrix")
    if n_samples < 1:
        raise ValueError("Number of samples must be at least one")
    if n_mean < 0:
        raise ValueError("Mean photon number must be non-negative")
    if not 0 <= loss <= 1:
        raise ValueError("Loss parameter must take a value between zero and one")

    nodes = len(A)

    p = sf.Program(nodes)
    eng = sf.LocalEngine(backend="gaussian")

    mean_photon_per_mode = n_mean / float(nodes)

    # pylint: disable=expression-not-assigned,pointless-statement
    with p.context as q:
        sf.ops.GraphEmbed(A, mean_photon_per_mode=mean_photon_per_mode) | q

        if loss:
            for _q in q:
                sf.ops.LossChannel(1 - loss) | _q

        if threshold:
            sf.ops.MeasureThreshold() | q
        else:
            sf.ops.MeasureFock() | q

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Cannot simulate non-")
        s = eng.run(p, run_options={"shots": n_samples}).samples

    if n_samples == 1:
        return [s]

    return s.tolist()


def postselect(samples: list, min_count: int, max_count: int) -> list:
    """Postselect samples by imposing a minimum and maximum number of photons or clicks.

    **Example usage:**

    >>> s = [[1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]]
    >>> postselect(s, 2, 4)
    [[1, 1, 0, 1, 1], [1, 0, 0, 0, 1]]

    Args:
        samples (list[list[int]]): a list of samples
        min_count (int): minimum number of photons or clicks for a sample to be included
        max_count (int): maximum number of photons or clicks for a sample to be included

    Returns:
        list[list[int]]: the postselected samples
    """
    return [s for s in samples if min_count <= sum(s) <= max_count]


def seed(value: Optional[int]) -> None:
    """Seed for random number generators.

    Wrapper function for `numpy.random.seed <https://docs.scipy.org/doc/numpy//reference/generated
    /numpy.random.seed.html>`_ to seed all NumPy-based random number generators. This allows for
    repeatable sampling.

    **Example usage:**

    >>> g = nx.erdos_renyi_graph(5, 0.7)
    >>> a = nx.to_numpy_array(g)
    >>> seed(1967)
    >>> sample(a, 3, 4)
    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0]]
    >>> seed(1967)
    >>> sample(a, 3, 4)
    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0]]

    Args:
        value (int): random seed
    """
    np.random.seed(value)


def modes_from_counts(s: list) -> list:
    r"""Convert a list of counts to a list of modes where photons or clicks were observed.

    Consider an :math:`M`-mode sample :math:`s=\{s_{0},s_{1},\ldots,s_{M-1}\}` of counts in each
    mode, i.e., so that mode :math:`i` has :math:`s_{i}` photons/clicks. If :math:`k = \sum_{
    i=0}^{M-1} s_{i}` is the total number of photons or clicks, this function returns a list of
    modes :math:`m=\{m_{1},m_{2},\ldots, m_{k}\}` where photons or clicks were observed, i.e.,
    so that photon/click :math:`i` is in mode :math:`m_{i}`. Since there are typically fewer
    photons than modes, the latter form often gives a shorter representation.

    **Example usage:**

    >>> modes_from_counts([0, 1, 0, 1, 2, 0])
    [1, 3, 4, 4]

    Args:
       s (list[int]): a sample of counts

    Returns:
        list[int]: a list of modes where photons or clicks were observed, sorted in
        non-decreasing order
    """
    modes = []
    for i, c in enumerate(s):
        modes += [i] * int(c)
    return sorted(modes)


def to_subgraphs(samples: list, graph: nx.Graph) -> list:
    """Converts samples to their subgraph representation.

    Input samples are a list of counts that are processed into subgraphs by selecting the nodes
    where a click occured.

    **Example usage:**

    >>> g = nx.erdos_renyi_graph(5, 0.7)
    >>> s = [[1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]]
    >>> to_subgraphs(s, g)
    [[0, 1, 2, 3, 4], [0, 1, 3, 4], [], [0, 4]]

    Args:
        samples (list[list[int]]): a list of samples, each sample being a list of counts
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


def vibronic(
    t: np.ndarray,
    U1: np.ndarray,
    r: np.ndarray,
    U2: np.ndarray,
    alpha: np.ndarray,
    n_samples: int,
    loss: float = 0.0,
) -> list:
    """Generate samples for computing vibronic spectra. The following gates are applied to input
    vacuum states:

    1. Two-mode squeezing on all :math:`2N` modes with parameters ``t``
    2. Interferometer ``U1`` on the first :math:`N` modes
    3. Squeezing on the first :math:`N` modes with parameters ``r``
    4. Interferometer ``U2`` on the first :math:`N` modes
    5. Displacement on the first :math:`N` modes with parameters ``alpha``

    A sample is generated by measuring the number of photons in each of the :math:`2N` modes. In
    the special case that all of the two-mode squeezing parameters ``t`` are zero, only :math:`N`
    modes are considered, which speeds up calculations.

    **Example usage:**

>>> t = np.array([0., 0., 0., 0., 0., 0., 0.])
>>> U1 = np.array(
>>>     [[-0.07985219, 0.66041032, -0.19389188, 0.01340832, 0.70312675, -0.1208423, -0.10352726],
>>>     [0.19216669, -0.12470466, -0.81320519, 0.52045174, -0.1066017, -0.06300751, -0.00376173],
>>>     [0.60838109, 0.0835063, -0.14958816, -0.34291399, 0.06239828, 0.68753918, -0.07955415],
>>>     [0.63690134, -0.03047939, 0.46585565, 0.50545897, 0.21194805, -0.20422433, 0.18516987],
>>>     [0.34556293, 0.22562207, -0.1999159, -0.50280235, -0.25510781, -0.55793978, 0.40065893],
>>>     [-0.03377431, -0.66280536, -0.14740447, -0.25725325, 0.6145946, -0.07128058, 0.29804963],
>>>     [-0.24570365, 0.22402764, 0.003273, 0.19204683, -0.05125235, 0.3881131, 0.83623564]])
>>> r = np.array(
>>>     [0.09721339, 0.07017918, 0.02083469, -0.05974357, -0.07487845, -0.1119975, -0.1866708])
>>> U2 = np.array(
>>>     [[-0.07012006, 0.14489772, 0.17593463, 0.02431155, -0.63151781, 0.61230046, 0.41087368],
>>>     [0.5618538, -0.09931968, 0.04562272, 0.02158822, 0.35700706, 0.6614837, -0.326946],
>>>     [-0.16560687, -0.7608465, -0.25644606, -0.54317241, -0.12822903, 0.12809274, -0.00597384],
>>>     [0.01788782, 0.60430409, -0.19831443, -0.73270964, -0.06393682, 0.03376894, -0.23038293],
>>>     [0.78640978, -0.11133936, 0.03160537, -0.09188782, -0.43483738, -0.4018141, 0.09582698],
>>>     [-0.13664887, -0.11196486, 0.86353995, -0.19608061, -0.12313513, -0.08639263, -0.40251231],
>>>     [-0.12060103, -0.01169781, -0.33937036, 0.34662981, -0.49895371, 0.03257453, -0.70709135]])
>>> alpha = np.array(
>>>     [0.15938187, 0.10387399, 1.10301587, -0.26756921, 0.32194572, -0.24317402, 0.0436992])
>>> vibronic(t, U1, S, U2, alpha, 2, 0.0)
[[0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    Args:
        t (array): two-mode squeezing parameters
        U1 (array): unitary matrix for the first interferometer
        r (array): squeezing parameters
        U2 (array): unitary matrix for the second interferometer
        alpha (array): displacement parameters
        n_samples (int): number of samples to be generated
        loss (float): loss parameter denoting the fraction of generated photons that are lost


    Returns:
        list[list[int]]: a list of samples from GBS
    """
    if n_samples < 1:
        raise ValueError("Number of samples must be at least one")
    if not 0 <= loss <= 1:
        raise ValueError("Loss parameter must take a value between zero and one")

    n_modes = len(t)

    eng = sf.LocalEngine(backend="gaussian")

    if np.any(t != 0):
        gbs = sf.Program(n_modes * 2)
    else:
        gbs = sf.Program(n_modes)

    # pylint: disable=expression-not-assigned,pointless-statement
    with gbs.context as q:

        if np.any(t != 0):
            for i in range(n_modes):
                sf.ops.S2gate(t[i]) | (q[i], q[i + n_modes])

        sf.ops.Interferometer(U1) | q[:n_modes]

        for i in range(n_modes):
            sf.ops.Sgate(r[i]) | q[i]

        sf.ops.Interferometer(U2) | q[:n_modes]

        for i in range(n_modes):
            sf.ops.Dgate(alpha[i]) | q[i]

        if loss:
            for _q in q:
                sf.ops.LossChannel(1 - loss) | _q

        sf.ops.MeasureFock() | q

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Cannot simulate non-")

        s = eng.run(gbs, run_options={"shots": n_samples}).samples

    s = np.array(s).tolist()  # convert all generated samples to list

    if n_samples == 1:
        s = [s]
    if np.any(t == 0):
        s = np.pad(s, ((0, 0), (0, n_modes))).tolist()
    return s

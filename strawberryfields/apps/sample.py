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
Functionality for generating GBS samples using classical simulators.

.. seealso::

    :ref:`apps-sample-tutorial`

Generating samples
------------------

An :math:`M` mode GBS device can be programmed by specifying an :math:`(M \times M)`-dimensional
symmetric matrix :math:`A` :cite:`bradler2018gaussian`. Running this device results in samples
that carry relevant information about the encoded matrix :math:`A`. When sampling, one must also
specify the mean number of photons in the device, the form of detection used at the output:
threshold detection or photon-number-resolving (PNR) detection, as well as the amount of loss.

The :func:`sample` function provides a simulation of sampling from GBS.

Here, each output sample is an :math:`M`-dimensional list. If threshold detection is used
(``threshold = True``), each element of a sample is either a zero (denoting no photons detected)
or a one (denoting one or more photons detected), conventionally called a "click".
If photon-number resolving (PNR) detection is used (``threshold = False``) then elements of a
sample are non-negative integers counting the number of photons detected in each mode. The ``loss``
parameter allows for simulation of photon loss in the device, which is the most common form of
noise in quantum photonics. Here, ``loss = 0`` describes ideal loss-free GBS, while generally
``0 <= loss <= 1`` describes the proportion of photons lost while passing through the device.

Samples can be postselected based upon their total number of photons or clicks and the ``numpy``
random seed used to generate samples can be fixed.

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

Graphs can be encoded into GBS through their adjacency matrix. Resultant samples
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
use within heuristics for problems such as maximum clique (see :mod:`~.apps.clique`).

Including node weights
----------------------

Some graphs are composed of nodes with weights. These weights can correspond to relevant
information in an optimization problem and it is desirable to encode them into GBS along with the
graph's adjacency matrix. One canonical approach to doing this is to use the :math:`WAW` encoding
:cite:`banchi2019molecular`, which rescales the adjacency matrix according to:

.. math::

    A \rightarrow WAW,

with :math:`W=w_{i}\delta_{ij}` the diagonal matrix formed by the weighted nodes :math:`w_{i}`. The
rescaled adjacency matrix can be passed to :func:`sample`, resulting in a distribution that
increases the probability of observing a node proportionally to its weight.
"""
import warnings
from typing import Optional, Union

import networkx as nx
import numpy as np

import strawberryfields as sf
from strawberryfields.apps.qchem.vibronic import sample as vibronic  # pylint: disable=unused-import


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
        s = eng.run(p, shots=n_samples).samples
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
    where a click occurred.

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


def waw_matrix(A: np.ndarray, w: Union[np.ndarray, list]) -> np.ndarray:
    r"""Rescale adjacency matrix to account for node weights.

    Given a graph with adjacency matrix :math:`A` and a vector :math:`w` of weighted nodes,
    this function rescales the adjacency matrix according to :cite:`banchi2019molecular`:

    .. math::

        A \rightarrow WAW,

    with :math:`W=w_{i}\delta_{ij}` the diagonal matrix formed by the weighted nodes :math:`w_{i}`.
    The rescaled adjacency matrix can be passed to :func:`sample`, resulting in a distribution that
    increases the probability of observing a node proportionally to its weight.

    **Example usage:**

    >>> g = nx.erdos_renyi_graph(5, 0.7)
    >>> a = nx.to_numpy_array(g)
    >>> a
    array([[0., 1., 1., 1., 1.],
           [1., 0., 1., 1., 1.],
           [1., 1., 0., 1., 0.],
           [1., 1., 1., 0., 0.],
           [1., 1., 0., 0., 0.]])
    >>> w = [10, 1, 0, 1, 1]
    >>> a = waw_matrix(a, w)
    >>> a
    array([[ 0., 10.,  0., 10., 10.],
           [10.,  0.,  0.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [10.,  1.,  0.,  0.,  0.],
           [10.,  1.,  0.,  0.,  0.]])
    >>> sample(a, 3, 4)
    [[1, 0, 0, 1, 1], [1, 1, 0, 0, 0], [1, 1, 0, 1, 1], [1, 0, 0, 1, 0]]

    Args:
        A (array): adjacency matrix to rescale
        w (array or list): vector of real node weights

    Returns:
        array: matrix rescaled according to the WAW encoding
    """
    if not np.allclose(A, A.T):
        raise ValueError("Input must be a NumPy array corresponding to a symmetric matrix")

    if isinstance(w, list):
        w = np.array(w)

    return (w * A).T * w

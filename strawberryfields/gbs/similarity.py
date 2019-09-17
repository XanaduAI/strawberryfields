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
Graph similarity
================

**Module name:** :mod:`strawberryfields.gbs.similarity`

.. currentmodule:: strawberryfields.gbs.similarity

Functionality for calculating feature vectors of graphs using GBS.

Summary
-------

.. autosummary::
    sample_to_orbit
    sample_to_event
    orbit_to_sample
    event_to_sample
    orbits
    orbit_cardinality
    event_cardinality
    p_orbit_mc
    p_event_mc

Code details
^^^^^^^^^^^^
"""
from collections import Counter
from typing import Generator, Union

import networkx as nx
import numpy as np
from scipy.special import factorial

import strawberryfields as sf


def sample_to_orbit(sample: list) -> list:
    """Provides the orbit corresponding to a given sample.

    Orbits are simply a sorting of integer photon number samples in non-increasing order with the
    zeros at the end removed.

    **Example usage:**

    >>> sample = [1, 2, 0, 0, 1, 1, 0, 3]
    >>> sample_to_orbit(sample)
    [3, 2, 1, 1, 1]

    Args:
        sample (list[int]): a sample from GBS

    Returns:
        list[int]: the orbit of the sample
    """
    return sorted(filter(None, sample), reverse=True)


def sample_to_event(sample: list, max_count_per_mode: int) -> Union[int, None]:
    r"""Provides the event corresponding to a given sample.

    An event is a combination of orbits. For a fixed photon number, orbits are combined into an
    event if the photon count in any mode does not exceed ``max_count_per_mode``, i.e.,
    an orbit :math:`o=\{o_{1},o_{2},\ldots\}` with a total photon number :math:`\sum_{i}o_{i}=k`
    is included in the event :math:`E_{k}` if :math:`\max_{i} \{o_{i}\}` is not larger than
    ``max_count_per_mode``.

    **Example usage:**

    >>> sample = [1, 2, 0, 0, 1, 1, 0, 3]
    >>> sample_to_event(sample, 4)
    8

    >>> sample_to_event(sample, 2)
    None

    Args:
        sample (list[int]): a sample from GBS
        max_count_per_mode (int): the maximum number of photons counted in any given mode for a
            sample to be categorized as an event. Samples with counts exceeding this value are
            attributed the event ``None``.

    Returns:
        list[int]: the orbit of the sample
    """
    if max(sample) <= max_count_per_mode:
        return sum(sample)

    return None


def orbits(photon_number: int) -> Generator[list, None, None]:
    """Generate all the possible orbits for a given photon number.

    Provides a generator over the integer partitions of ``photon_number``.
    Code derived from `website <http://jeromekelleher.net/generating-integer-partitions.html>`__
    of Jerome Kelleher's, which is based upon an algorithm from Ref. :cite:`kelleher2009generating`.

    **Example usage:**

    >>> o = orbits(5)
    >>> list(o)
    [[1, 1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1], [2, 2, 1], [4, 1], [3, 2], [5]]

    Args:
        photon_number (int): number of photons to generate orbits from

    Returns:
        Generator[list[int]]: orbits with total photon number adding up to ``photon_number``
    """
    a = [0] * (photon_number + 1)
    k = 1
    y = photon_number - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield sorted(a[: k + 2], reverse=True)
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield sorted(a[: k + 1], reverse=True)


def orbit_to_sample(orbit: list, modes: int) -> list:
    """Generates a sample selected uniformly at random from the specified orbit.

    An orbit has a number of constituting samples, which are given by taking all permutations
    over the orbit. For a given orbit and number of modes, this function produces a sample
    selected uniformly at random among all samples in the orbit.

    **Example usage**:

    >>> orbit_to_sample([2, 1, 1], 6)
    [0, 1, 2, 0, 1, 0]

    Args:
        orbit (list[int]): orbit to generate a sample from
        modes (int): number of modes in the sample

    Returns:
        list[int]: a sample in the orbit
    """
    if modes < len(orbit):
        raise ValueError("Number of modes cannot be smaller than length of orbit")

    sample = list(orbit) + [0] * (modes - len(orbit))
    np.random.shuffle(sample)
    return sample


def event_to_sample(photon_number: int, max_count_per_mode: int, modes: int) -> list:
    """Generates a sample selected uniformly at random from the specified event.

    An event has a number of constituting samples, which are given by combining samples within all
    orbits with a fixed photon number whose photon count in any mode does not exceed
    ``max_count_per_mode``. This function produces a sample selected uniformly at random among
    all samples in the event.

    **Example usage**:

    >>> event_to_sample(4, 2, 6)
    [0, 1, 0, 0, 2, 1]

    Args:
        photon_number (int): number of photons in the event
        max_count_per_mode (int): maximum number of photons per mode in the event
        modes (int): number of modes in the sample

    Returns:
        list[int]: a sample in the event
    """
    if max_count_per_mode < 0:
        raise ValueError("Maximum number of photons must be non-negative")

    if max_count_per_mode * modes < photon_number:
        raise ValueError(
            "No valid samples can be generated. Consider increasing the "
            "max_count_per_mode or reducing the number of photons."
        )

    sample = [0] * modes
    available_modes = list(range(modes))

    for _ in range(photon_number):
        j = np.random.choice(available_modes)
        sample[j] += 1
        if sample[j] == max_count_per_mode:
            available_modes.remove(j)

    return sample


def orbit_cardinality(orbit: list, modes: int) -> int:
    """Gives the number of samples belonging to the input orbit.

    **Example usage**:

    >>> orbit_cardinality([2, 1, 1], 4)
    12

    Args:
        orbit (list[int]): orbit to count number of samples
        modes (int): number of modes in the sample

    Returns:
        int: number of samples in the orbit
    """

    sample = list(orbit) + [0] * (modes - len(orbit))
    counts = list(Counter(sample).values())
    return int(factorial(modes) / np.prod(factorial(counts)))


def event_cardinality(photon_number: int, max_count_per_mode: int, modes: int) -> int:
    """Gives the number of samples belonging to the input event.

    **Example usage**:

    >>> event_cardinality(5, 2, 6)
    126

    Args:
        photon_number (int): number of photons in the event
        max_count_per_mode (int): maximum number of photons per mode in the event
        modes (int): number of modes in the samples

    Returns:
        int: number of samples in the event
    """
    cardinality = 0

    for orb in orbits(photon_number):
        if max(orb) <= max_count_per_mode:
            cardinality += orbit_cardinality(orb, modes)

    return cardinality


def p_orbit_mc(graph: nx.Graph, orbit: list, n_mean: float = 5, samples: int = 1000) -> float:
    """Gives a Monte Carlo estimate of the probability of a given orbit with respect to a graph.

    To make this estimate, several samples from the orbit are drawn uniformly at random using
    :func:`orbit_to_sample`.

    For each sample, this function calculates the probability of observing that sample from a GBS
    device programmed according to the input graph and mean photon number. The sum of the
    probabilities is then rescaled according to the cardinality of the orbit and the total number of
     samples. The estimate is the sample mean of the rescaled probabilities.

    **Example usage**:

    >>> graph = nx.complete_graph(8)
    >>> p_orbit_mc(graph, [2, 1, 1])
    0.03744

    Args:
        orbit (list[int]): orbit for which to estimate the probability
        graph (nx.Graph): input graph encoded in the GBS device
        n_mean (float): total mean photon number of the GBS device
        samples (int): number of samples used in the Monte Carlo estimation

    Returns:
        float: estimated orbit probability
    """

    modes = graph.order()
    photons = sum(orbit)
    A = nx.to_numpy_array(graph)
    mean_photon_per_mode = n_mean / float(modes)

    p = sf.Program(modes)

    # pylint: disable=expression-not-assigned
    with p.context as q:
        sf.ops.GraphEmbed(A, mean_photon_per_mode=mean_photon_per_mode) | q

    eng = sf.LocalEngine(backend="gaussian")
    result = eng.run(p)

    prob = 0

    for _ in range(samples):
        sample = orbit_to_sample(orbit, modes)
        prob += result.state.fock_prob(sample, cutoff=photons + 1)

    prob = prob * orbit_cardinality(orbit, modes) / samples

    return prob


def p_event_mc(
    graph: nx.Graph,
    photon_number: int,
    max_count_per_mode: int,
    n_mean: float = 5,
    samples: int = 1000,
) -> float:
    """Gives a Monte Carlo estimation of the probability that a sample belongs to the given
    event.

    To make this estimate, several samples from the event are drawn uniformly at random. For each
    sample, we calculate the probability of observing that sample from a GBS programmed according to
    the input graph and mean photon number. These probabilities are then rescaled according to the
    cardinality of the event. The estimate is the sample mean of the rescaled probabilities.

    **Example usage**:

    >>> graph = nx.complete_graph(8)
    >>> p_event_mc(graph, 4, 2)
    0.1395

    Args:
        graph (nx.Graph): the input graph encoded in the GBS device
        photon_number (int): number of photons in the event
        max_count_per_mode (int): maximum number of photons per mode in the event
        n_mean (float): the total mean photon number of the GBS device
        samples (int): the number of samples used in the Monte Carlo estimation

    Returns:
        float: the estimated probability
    """

    modes = graph.order()
    A = nx.to_numpy_array(graph)
    mean_photon_per_mode = n_mean / float(modes)

    p = sf.Program(modes)

    # pylint: disable=expression-not-assigned
    with p.context as q:
        sf.ops.GraphEmbed(A, mean_photon_per_mode=mean_photon_per_mode) | q

    eng = sf.LocalEngine(backend="gaussian")
    result = eng.run(p)

    prob = 0

    for _ in range(samples):
        sample = event_to_sample(photon_number, max_count_per_mode, modes)
        prob += result.state.fock_prob(sample, cutoff=np.sum(sample) + 1)

    prob = prob * event_cardinality(photon_number, max_count_per_mode, modes) / samples

    return prob

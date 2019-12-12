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
Tools to construct graph kernels from GBS.

The graph kernel is built by mapping GBS samples from each graph to a feature vector. Similarity
between graphs can then be determined by calculating the overlap between these vectors.

The functionality here is based upon the research papers:
:cite:`bradler2018graph,schuld2019quantum,bradler2019duality`.

.. seealso::

    :ref:`apps-sim-tutorial`

Coarse-graining GBS samples
---------------------------

GBS feature vectors can be composed of probabilities of coarse-grained combinations of elementary
samples. We consider two coarse grainings:

- **Orbits:** Combine all samples that can be made identical under permutation. Orbits are
  written simply as a sorting of integer photon number samples in non-increasing order with the
  zeros at the end removed. For example, ``[1, 1, 2, 0]`` and ``[2, 1, 0, 1]`` both belong to the
  ``[2, 1, 1]`` orbit.

- **Events:** Combine all :math:`k`-photon orbits where the maximum photon count in any mode does
  not exceed a fixed value :math:`n_{\max}` into an event :math:`E_{k, n_{\max}}`. For example,
  orbits ``[2, 1]``, ``[1, 1, 1]`` are part of the :math:`E_{k=3, n_{\max}=2}` event, while
  orbit ``[3]`` is not.

This module provides the following tools for dealing with coarse-grained orbits and events.

Creating a feature vector
-------------------------

A feature vector of a graph can be created by choosing a collection of orbits or events and
evaluating their probabilities with respect to GBS with the embedded graph. These
probabilities are then selected to be elements of the feature vector. Evaluating the
probabilities of orbits or events can be achieved through two approaches:

- **Direct sampling:** infer the probability of orbits or events from a set of sample data.

- **Monte Carlo approximation:** generate samples within a given orbit or event and use them
  to approximate the probability.

In the direct sampling approach, :math:`N` samples are taken from GBS with the embedded graph.
The number that fall within a given orbit or event :math:`E` are counted, resulting in the count
:math:`c_{E}`. The probability :math:`p(E)` of an orbit or event is then approximated as:

.. math::
    p(E) \approx \frac{c_{E}}{N}.

To perform a Monte Carlo estimation of the probability of an orbit or event :math:`E`,
several samples from :math:`E` are drawn uniformly at random using :func:`orbit_to_sample` or
:func:`event_to_sample`. Suppose :math:`N` samples :math:`\{S_{1}, S_{2}, \ldots , S_{N}\}` are
generated. For each sample, this function calculates the probability :math:`p(S_i)` of observing
that sample from a GBS device programmed according to the input graph and mean photon number. The
sum of the probabilities is then rescaled according to the cardinality :math:`|E|` and the total
number of samples:

.. math::
    p(E) \approx \frac{1}{N}\sum_{i=1}^N p(S_i) |E|.

The sample mean of this sum is an estimate of the rescaled probability :math:`p(E)`.

This module provides functions to evaluate the probability of orbits and events through Monte Carlo
approximation.

Similar to the :func:`~.apps.sample.sample` function, the MC estimators include a ``loss`` argument
to specify the proportion of photons lost in the simulated GBS device.

One canonical method of constructing a feature vector is to pick event probabilities
:math:`p_{k} := p_{E_{k, n_{\max}}}` with all events :math:`E_{k, n_{\max}}` having a maximum
photon count :math:`n_{\max}` in each mode. If :math:`\mathbf{k}` is the vector of selected events,
the resultant feature vector is

.. math::
    f_{\mathbf{k}} = (p_{k_{1}}, p_{k_{2}}, \ldots).

This module allows for such feature vectors to be calculated using both the direct sampling and
Monte Carlo approximation methods.
"""
from collections import Counter
from typing import Generator, Union

import networkx as nx
import numpy as np
from scipy.special import factorial

import strawberryfields as sf


def sample_to_orbit(sample: list) -> list:
    """Provides the orbit corresponding to a given sample.

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

    For an input ``max_count_per_mode``, events are expressed here simply by the total photon
    number :math:`k`.

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
        int or None: the event of the sample
    """
    if max(sample) <= max_count_per_mode:
        return sum(sample)

    return None


def orbit_to_sample(orbit: list, modes: int) -> list:
    """Generates a sample selected uniformly at random from the specified orbit.

    **Example usage:**

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

    sample = orbit + [0] * (modes - len(orbit))
    np.random.shuffle(sample)
    return sample


def event_to_sample(photon_number: int, max_count_per_mode: int, modes: int) -> list:
    """Generates a sample selected uniformly at random from the specified event.

    **Example usage:**

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
        raise ValueError("Maximum number of photons per mode must be non-negative")

    if max_count_per_mode * modes < photon_number:
        raise ValueError(
            "No valid samples can be generated. Consider increasing the "
            "max_count_per_mode or reducing the number of photons."
        )

    cards = []
    orbs = []

    for orb in orbits(photon_number):
        if max(orb) <= max_count_per_mode:
            cards.append(orbit_cardinality(orb, modes))
            orbs.append(orb)

    norm = sum(cards)
    prob = [c / norm for c in cards]

    orbit = orbs[np.random.choice(len(prob), p=prob)]

    return orbit_to_sample(orbit, modes)


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


def orbit_cardinality(orbit: list, modes: int) -> int:
    """Gives the number of samples belonging to the input orbit.

    For example, there are three possible samples in the orbit [2, 1, 1] with three modes: [1, 1,
    2], [1, 2, 1], and [2, 1, 1]. With four modes, there are 12 samples in total.

    **Example usage:**

    >>> orbit_cardinality([2, 1, 1], 4)
    12

    Args:
        orbit (list[int]): orbit; we count how many samples are contained in it
        modes (int): number of modes in the samples

    Returns:
        int: number of samples in the orbit
    """
    sample = orbit + [0] * (modes - len(orbit))
    counts = list(Counter(sample).values())
    return int(factorial(modes) / np.prod(factorial(counts)))


def event_cardinality(photon_number: int, max_count_per_mode: int, modes: int) -> int:
    r"""Gives the number of samples belonging to the input event.

    For example, for three modes, there are six samples in an :math:`E_{k=2, n_{\max}=2}` event:
    [1, 1, 0], [1, 0, 1], [0, 1, 1], [2, 0, 0], [0, 2, 0], and [0, 0, 2].

    **Example usage:**

    >>> event_cardinality(2, 2, 3)
    6

    Args:
        photon_number (int): number of photons in the event
        max_count_per_mode (int): maximum number of photons per mode in the event
        modes (int): number of modes in counted samples

    Returns:
        int: number of samples in the event
    """
    cardinality = 0

    for orb in orbits(photon_number):
        if max(orb) <= max_count_per_mode:
            cardinality += orbit_cardinality(orb, modes)

    return cardinality


def prob_orbit_mc(
    graph: nx.Graph, orbit: list, n_mean: float = 5, samples: int = 1000, loss: float = 0.0
) -> float:
    r"""Gives a Monte Carlo estimate of the GBS probability of a given orbit according to the input
    graph.

    To make this estimate, several samples from the orbit are drawn uniformly at random using
    :func:`orbit_to_sample`. The GBS probabilities of these samples are then calculated and the
    sum is used to create an estimate of the orbit probability.

    **Example usage:**

    >>> graph = nx.complete_graph(8)
    >>> prob_orbit_mc(graph, [2, 1, 1])
    0.03744

    Args:
        graph (nx.Graph): input graph encoded in the GBS device
        orbit (list[int]): orbit for which to estimate the probability
        n_mean (float): total mean photon number of the GBS device
        samples (int): number of samples used in the Monte Carlo estimation
        loss (float): fraction of photons lost in GBS

    Returns:
        float: estimated orbit probability
    """
    if samples < 1:
        raise ValueError("Number of samples must be at least one")
    if n_mean < 0:
        raise ValueError("Mean photon number must be non-negative")
    if not 0 <= loss <= 1:
        raise ValueError("Loss parameter must take a value between zero and one")

    modes = graph.order()
    photons = sum(orbit)
    A = nx.to_numpy_array(graph)
    mean_photon_per_mode = n_mean / float(modes)

    p = sf.Program(modes)

    # pylint: disable=expression-not-assigned
    with p.context as q:
        sf.ops.GraphEmbed(A, mean_photon_per_mode=mean_photon_per_mode) | q

        if loss:
            for _q in q:
                sf.ops.LossChannel(1 - loss) | _q

    eng = sf.LocalEngine(backend="gaussian")
    result = eng.run(p)

    prob = 0

    for _ in range(samples):
        sample = orbit_to_sample(orbit, modes)
        prob += result.state.fock_prob(sample, cutoff=photons + 1)

    prob = prob * orbit_cardinality(orbit, modes) / samples

    return prob


def prob_event_mc(
    graph: nx.Graph,
    photon_number: int,
    max_count_per_mode: int,
    n_mean: float = 5,
    samples: int = 1000,
    loss: float = 0.0,
) -> float:
    r"""Gives a Monte Carlo estimate of the GBS probability of a given event according to the input
    graph.

    To make this estimate, several samples from the event are drawn uniformly at random using
    :func:`event_to_sample`. The GBS probabilities of these samples are then calculated and the
    sum is used to create an estimate of the event probability.

    **Example usage:**

    >>> graph = nx.complete_graph(8)
    >>> prob_event_mc(graph, 4, 2)
    0.1395

    Args:
        graph (nx.Graph): input graph encoded in the GBS device
        photon_number (int): number of photons in the event
        max_count_per_mode (int): maximum number of photons per mode in the event
        n_mean (float): total mean photon number of the GBS device
        samples (int): number of samples used in the Monte Carlo estimation
        loss (float): fraction of photons lost in GBS

    Returns:
        float: estimated orbit probability
    """
    if samples < 1:
        raise ValueError("Number of samples must be at least one")
    if n_mean < 0:
        raise ValueError("Mean photon number must be non-negative")
    if not 0 <= loss <= 1:
        raise ValueError("Loss parameter must take a value between zero and one")
    if photon_number < 0:
        raise ValueError("Photon number must not be below zero")
    if max_count_per_mode < 0:
        raise ValueError("Maximum number of photons per mode must be non-negative")

    modes = graph.order()
    A = nx.to_numpy_array(graph)
    mean_photon_per_mode = n_mean / float(modes)

    p = sf.Program(modes)

    # pylint: disable=expression-not-assigned
    with p.context as q:
        sf.ops.GraphEmbed(A, mean_photon_per_mode=mean_photon_per_mode) | q

        if loss:
            for _q in q:
                sf.ops.LossChannel(1 - loss) | _q

    eng = sf.LocalEngine(backend="gaussian")
    result = eng.run(p)

    prob = 0

    for _ in range(samples):
        sample = event_to_sample(photon_number, max_count_per_mode, modes)
        prob += result.state.fock_prob(sample, cutoff=photon_number + 1)

    prob = prob * event_cardinality(photon_number, max_count_per_mode, modes) / samples

    return prob


def feature_vector_sampling(
    samples: list, event_photon_numbers: list, max_count_per_mode: int = 2
) -> list:
    r"""Calculates feature vector with respect to input samples.

    The feature vector is composed of event probabilities with a fixed maximum photon count in
    each mode but a range of total photon numbers specified by ``event_photon_numbers``.

    Probabilities are reconstructed by measuring the occurrence of events in the input ``samples``.

    **Example usage:**

    >>> from strawberryfields.apps import data
    >>> samples = data.Mutag0()
    >>> feature_vector_sampling(samples, [2, 4, 6])
    [0.19035, 0.2047, 0.1539]

    Args:
        samples (list[list[int]]): a list of samples
        event_photon_numbers (list[int]): a list of events described by their total photon number
        max_count_per_mode (int): maximum number of photons per mode for all events

    Returns:
        list[float]: a feature vector of event probabilities in the same order as
        ``event_photon_numbers``
    """
    if min(event_photon_numbers) < 0:
        raise ValueError("Cannot request events with photon number below zero")

    if max_count_per_mode < 0:
        raise ValueError("Maximum number of photons per mode must be non-negative")

    n_samples = len(samples)

    e = (sample_to_event(s, max_count_per_mode) for s in samples)
    count = Counter(e)

    return [count[p] / n_samples for p in event_photon_numbers]


def feature_vector_mc(
    graph: nx.Graph,
    event_photon_numbers: list,
    max_count_per_mode: int = 2,
    n_mean: float = 5,
    samples: int = 1000,
    loss: float = 0.0,
) -> list:
    r"""Calculates feature vector using Monte Carlo estimation of event probabilities according to the
    input graph.

    The feature vector is composed of event probabilities with a fixed maximum photon count in
    each mode but a range of total photon numbers specified by ``event_photon_numbers``.

    Probabilities are reconstructed using Monte Carlo estimation.

    **Example usage:**

    >>> graph = nx.complete_graph(8)
    >>> feature_vector_mc(graph, [2, 4, 6], 2)
    [0.2115, 0.1457, 0.09085]

    Args:
        graph (nx.Graph): input graph
        event_photon_numbers (list[int]): a list of events described by their total photon number
        max_count_per_mode (int): maximum number of photons per mode for all events
        n_mean (float): total mean photon number of the GBS device
        samples (int): number of samples used in the Monte Carlo estimation
        loss (float): fraction of photons lost in GBS

    Returns:
        list[float]: a feature vector of event probabilities in the same order as
        ``event_photon_numbers``
    """
    if samples < 1:
        raise ValueError("Number of samples must be at least one")
    if n_mean < 0:
        raise ValueError("Mean photon number must be non-negative")
    if not 0 <= loss <= 1:
        raise ValueError("Loss parameter must take a value between zero and one")
    if min(event_photon_numbers) < 0:
        raise ValueError("Cannot request events with photon number below zero")
    if max_count_per_mode < 0:
        raise ValueError("Maximum number of photons per mode must be non-negative")

    return [
        prob_event_mc(graph, photons, max_count_per_mode, n_mean, samples, loss)
        for photons in event_photon_numbers
    ]

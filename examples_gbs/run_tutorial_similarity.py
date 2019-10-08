# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
"""
Graph Similarity Tutorial
=========================

This tutorial looks at how to use GBS to construct a similarity measure between graphs,
what is known as a graph kernel :cite:`schuld2019quantum`. Kernels can be applied to graph-based
data for machine learning tasks such as classification in machine learning using a support vector
machine.

Graph data
----------

Let's use the MUTAG dataset of graphs for this tutorial
:cite:`debnath1991structure,kriege2012subgraph`. This is a dataset of 188 different graphs that
each correspond to the structure of a chemical compound. Our objective is to use GBS samples from
these graphs to measure their similarity.

The :mod:`~.gbs.data` module provides pre-calculated GBS samples from four graphs in the
MUTAG dataset. We'll start by loading these sample sets and visualizing the corresponding graphs.
"""

from strawberryfields import gbs

m0 = gbs.data.Mutag0()
m1 = gbs.data.Mutag1()
m2 = gbs.data.Mutag2()
m3 = gbs.data.Mutag3()

##############################################################################
# These datasets contain both the adjacency matrix of the graph and the samples generated through
# GBS. We can access the adjacency matrix through:

m0_a = m0.adj
m1_a = m1.adj
m2_a = m2.adj
m3_a = m3.adj

##############################################################################
# We can now plot the four graphs using the :mod:`~gbs.plot` module. To use this module,
# we need to convert the adjacency matrices into NetworkX Graphs:

import networkx as nx
import plotly

plot_mutag_0 = gbs.plot.plot_graph(nx.Graph(m0_a))
plot_mutag_1 = gbs.plot.plot_graph(nx.Graph(m1_a))
plot_mutag_2 = gbs.plot.plot_graph(nx.Graph(m2_a))
plot_mutag_3 = gbs.plot.plot_graph(nx.Graph(m3_a))

plotly.offline.plot(plot_mutag_0, filename="MUTAG_0.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/MUTAG_0.html

plotly.offline.plot(plot_mutag_1, filename="MUTAG_1.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/MUTAG_1.html

plotly.offline.plot(plot_mutag_2, filename="MUTAG_2.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/MUTAG_2.html

plotly.offline.plot(plot_mutag_3, filename="MUTAG_3.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/MUTAG_3.html
#
# By visual inspection, we see that the graphs of ``m1_a`` and ``m2_a`` look very similar. In fact,
# it turns out that they are *isomorphic* to each other, which means that the graphs can be made
# identical by permuting their node labels.
#
# Samples from these graphs can be accessed by indexing:

print(m0[0])

##############################################################################
# Creating a feature vector
# -------------------------
#
# Following :cite:`schuld2019quantum`, we can create a *feature vector* to describe each graph.
# These feature vectors contain information about the graphs and can be viewed as a mapping to a
# high dimensional feature space, a technique often used in machine learning that allows us to
# find ways of separating points in the space for classification.
#
# The feature vector of a graph can be composed in a variety of ways. One approach is to built it
# using statistics from a GBS device configured to sample from the graph, as we now discuss.
#
# We begin by defining the concept of an *orbit*, which contains all the GBS samples that are
# equivalent under permutation of the modes. A sample can be converted to its corresponding orbit
# using the :func:`~.sample_to_orbit` function. For example, the first sample of ``m0`` is ``[0,
# 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]`` and has orbit:

from strawberryfields.gbs import similarity

print(similarity.sample_to_orbit(m0[0]))

##############################################################################
# Other samples can be randomly generated from the ``[1, 1]`` orbit using:

print(similarity.orbit_to_sample([1, 1], modes=m0.modes))

##############################################################################
# Orbits provide a useful way to coarse grain the samples from GBS into outcomes that are
# statistically more likely to be observed. However, we are interested in coarse graining further
# into **events**, which correspond to a combination of orbits with the same photon number such
# that the number of photons counted in each mode does not exceed a fixed value
# ``max_count_per_mode``. To understand this, let's look at all of the orbits with a photon
# number of 5:

print(list(similarity.orbits(5)))

##############################################################################
# All 5-photon samples fall into one of the orbits above. A 5-photon event with
# ``max_count_per_mode = 3`` means that we combine the orbits: ``[[1, 1, 1, 1, 1], [2, 1, 1, 1],
# [3, 1, 1], [2, 2, 1], [3, 2]]`` and ignore the orbits ``[[4, 1], [5]]``. For example,
# the sample ``[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0]`` is a 5-photon event:

print(similarity.sample_to_event([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0], 3))

##############################################################################
# Samples with more than ``max_count_per_mode`` in any mode are not counted as part of the event:

print(similarity.sample_to_event([0, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3))

##############################################################################
# Now that we have mastered orbits and events, how can we make a feature vector? It was shown in
# :cite:`schuld2019quantum` that one way of making a feature vector of a graph is to measure the
# probabilities of events. Specifically, for a :math:`k` photon event :math:`E_{k, n_{\max}}`
# with maximum count per mode :math:`n_{\max}` and corresponding probability :math:`p_{k,
# n_{\max}}:=p_{E_{k, n_{\max}}}(G)` with respect to a graph :math:`G`, a feature vector can be
# written as
#
# .. math::
#     f_{\mathbf{k}, n_{\max}} = (p_{k_{1}, n_{\max}}, p_{k_{2}, n_{\max}}, \ldots , p_{k_{K}, n_{\max}}),
#
# where :math:`\mathbf{k} := (k_{1}, k_{2}, \ldots , k_{K})` is a list of different total photon
# numbers.
#
# Calculating a feature vector
# ----------------------------
#
# We provide two methods for calculating a feature vector of GBS event probabilities in
# Strawberry Fields:
#
# 1. Through sampling
# 2. Using a Monte Carlo estimate of the probability
#
# For the first method, all one needs to do is generate some GBS samples from the graph of
# interest and feed through:

print(similarity.feature_vector_sampling(m0, event_photon_numbers=[2, 4, 6], max_count_per_mode=2))

##############################################################################
# For the second method, suppose we want to calculate the event probabilities exactly rather than
# through sampling. To do this, we consider the event probability :math:`p_{k, n_{\max}}` as the
# sum over all sample probabilities in the event. As discussed in the GBS `tutorial
# <gaussian_boson_tutorial>`, each sample probability is determined by the hafnian of a relevant
# sub-adjacency matrix. While this is tough to calculate, what makes calculating :math:`p_{k,
# n_{\max}}` really tough is the number of samples the corresponding event contains! For example,
# the 17-photon event :math:`E_{k=6, n_{\max}=2}` contains the following number of samples:

print(similarity.event_cardinality(6, 2, 17))

##############################################################################
# To avoid calculating a large number of sample probabilities, an alternative for calculating the
# event probability is to perform a Monte Carlo approximation. Here, samples within an event are
# generated uniformly at random and their resultant probabilities are calculated. If :math:`N`
# samples :math:`\{S_{1}, S_{2}, \ldots , S_{N}\}` are generated, then the event probability can
# be approximated as
#
# .. math::
#     p(E_{k, n_{\max}}) \approx \frac{1}{N}\sum_{i=1}^N p(S_i) |E_{k, n_{\max}}|,
#
# with :math:`|E_{k, n_{\max}}|` denoting the cardinality of the event.
#
# This method can be accessed using the :func:`prob_event_mc` function. The 4-photon event is
# approximated as:

print(similarity.prob_event_mc(nx.Graph(m0_a), 4, max_count_per_mode=2, n_mean=6))

##############################################################################
# The feature vector can then be calculated through Monte Carlo sampling using
# :func:`feature_vector_mc`.
#
# .. note::
#     The results of :func:`prob_event_mc` and :func:`feature_vector_mc` are probabilistic and
#     may vary between runs. Increasing the optional ``samples`` parameter will increase accuracy
#     but slow down calculation.
#
# The second method of MC approximation is intended for use in scenarios where it is
# computationally intensive to pre-calculate a statistically significant dataset of samples from
# GBS.
#
# Machine learning with GBS graph kernels
# ---------------------------------------
#
# We have seen how GBS can be used to provide a mapping of graphs into a feature space. This
# mapping can be used for machine learning tasks such as classification: by viewing each graph as
# a point in the high dimensional space, we can use standard approaches from machine learning
# such as `support vector machines <https://en.wikipedia.org/wiki/Support-vector_machine>`__.
#
# Let's build this up a bit more by creating two-dimensional feature vectors of our four MUTAG
# graphs.

events = [8, 10]
max_count = 2

f1 = similarity.feature_vector_sampling(m0, events, max_count)
f2 = similarity.feature_vector_sampling(m1, events, max_count)
f3 = similarity.feature_vector_sampling(m2, events, max_count)
f4 = similarity.feature_vector_sampling(m3, events, max_count)

print(f1)
print(f2)
print(f3)
print(f4)

##############################################################################
# A plot of these points gives:


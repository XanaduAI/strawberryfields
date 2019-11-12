# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports,invalid-name
"""
.. _gbs-sim-tutorial:

Graph Similarity Tutorial
=========================

This tutorial looks at how to use GBS to construct a similarity measure between graphs,
known as a graph kernel :cite:`schuld2019quantum`. Kernels can be applied to graph-based
data for machine learning tasks such as classification using a support vector machine.

The main functionality for this tutorial is provided in the :mod:`~.gbs.similarity` module.

Graph data
----------

We begin by fixing a dataset of graphs to consider and loading GBS samples from these graphs,
which will be needed in the following.

Let's use the MUTAG dataset of graphs for this tutorial
:cite:`debnath1991structure,kriege2012subgraph`. This is a dataset of 188 different graphs that
each correspond to the structure of a chemical compound. Our goal is to use GBS samples from
these graphs to measure their similarity.

The :mod:`~.gbs.data` module provides pre-calculated GBS samples for selected graphs in the MUTAG
dataset. Each set of samples is generated by encoding the graph into a GBS device, and collecting
photon click events. We'll start by loading four sets of samples and visualizing the
corresponding graphs.
"""

from strawberryfields.gbs import data, plot, similarity

m0 = data.Mutag0()
m1 = data.Mutag1()
m2 = data.Mutag2()
m3 = data.Mutag3()

##############################################################################
# These datasets contain both the adjacency matrix of the graph and the samples generated through
# GBS. We can access the adjacency matrix through:

m0_a = m0.adj
m1_a = m1.adj
m2_a = m2.adj
m3_a = m3.adj

##############################################################################
# Samples from these graphs can be accessed by indexing:

print(m0[0])

##############################################################################
# We can now plot the four graphs using the :mod:`~.gbs.plot` module. To use this module,
# we need to convert the adjacency matrices into NetworkX Graphs:

import networkx as nx
import plotly

plot_mutag_0 = plot.graph(nx.Graph(m0_a))
plot_mutag_1 = plot.graph(nx.Graph(m1_a))
plot_mutag_2 = plot.graph(nx.Graph(m2_a))
plot_mutag_3 = plot.graph(nx.Graph(m3_a))

plotly.offline.plot(plot_mutag_0, filename="MUTAG_0.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/MUTAG_0.html
#
# .. note::
#     The command ``plotly.offline.plot()`` is used to display plots in the documentation. In
#     practice, you can simply use ``plot_mutag_0.show()`` to view your graph.

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
# The graphs of ``m1_a`` and ``m2_a`` look very similar. In fact,
# it turns out that they are *isomorphic* to each other, which means that the graphs can be made
# identical by permuting their node labels.

##############################################################################
# Creating a feature vector
# -------------------------
#
# Following :cite:`schuld2019quantum`, we can create a *feature vector* to describe each graph.
# These feature vectors contain information about the graphs and can be viewed as a mapping to a
# high-dimensional feature space, a technique often used in machine learning that allows us to
# employ properties of the feature space to separate and classify the vectors.
#
# The feature vector of a graph can be composed in a variety of ways. One approach is to
# associate features with the relative frequencies of certain types of measurements being
# recorded from a GBS device configured to sample from the graph, as we now discuss.
#
# We begin by defining the concept of an *orbit*, which is the set of all GBS samples that are
# equivalent under permutation of the modes. A sample can be converted to its corresponding orbit
# using the :func:`~.sample_to_orbit` function. For example, the first sample of ``m0`` is ``[0,
# 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]`` and has orbit:

print(similarity.sample_to_orbit(m0[0]))

##############################################################################
# Here, ``[1, 1]`` means that two photons were detected, each in a separate mode. Other samples
# can be randomly generated from the ``[1, 1]`` orbit using:

print(similarity.orbit_to_sample([1, 1], modes=m0.modes))

##############################################################################
# Orbits provide a useful way to coarse-grain the samples from GBS into outcomes that are
# statistically more likely to be observed. However, we are interested in coarse-graining further
# into *events*, which correspond to a combination of orbits with the same photon number such
# that the number of photons counted in each mode does not exceed a fixed value
# ``max_count_per_mode``. To understand this, let's look at all of the orbits with a photon
# number of 5:

print(list(similarity.orbits(5)))

##############################################################################
# All 5-photon samples belong to one of the orbits above. A 5-photon event with
# ``max_count_per_mode = 3`` means that we include the orbits: ``[[1, 1, 1, 1, 1], [2, 1, 1, 1],
# [3, 1, 1], [2, 2, 1], [3, 2]]`` and ignore the orbits ``[[4, 1], [5]]``. For example,
# the sample ``[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0]`` is a 5-photon event:

print(similarity.sample_to_event([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0], 3))

##############################################################################
# Samples with more than ``max_count_per_mode`` in any mode are not counted as part of the event:

print(similarity.sample_to_event([0, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3))

##############################################################################
# Now that we have mastered orbits and events, how can we make a feature vector? It was shown in
# :cite:`schuld2019quantum` that one way of making a feature vector of a graph is through the
# frequencies of events. Specifically, for a :math:`k` photon event :math:`E_{k, n_{\max}}`
# with maximum count per mode :math:`n_{\max}` and corresponding probability :math:`p_{k,
# n_{\max}}:=p_{E_{k, n_{\max}}}(G)` with respect to a graph :math:`G`, a feature vector can be
# written as
#
# .. math::
#     f_{\mathbf{k}, n_{\max}} = (p_{k_{1}, n_{\max}}, p_{k_{2}, n_{\max}}, \ldots , p_{k_{K},
#         n_{\max}}),
#
# where :math:`\mathbf{k} := (k_{1}, k_{2}, \ldots , k_{K})` is a list of different total photon
# numbers.
#
# For example, if :math:`\mathbf{k} := (2, 4, 6)` and :math:`n_{\max} = 2`, we have
#
# .. math::
#     f_{(2, 4, 6), 2} = (p_{2, 2}, p_{4, 2}, p_{6, 2}).
#
# In this case, we are interested in the probabilities of events :math:`E_{2, 2}`, :math:`E_{4,
# 2}`, and :math:`E_{6, 2}`. Suppose we are sampling from a four-mode device and have the samples
# ``[0, 3, 0, 1]`` and ``[1, 2, 0, 1]``. These samples are part of the orbits ``[3, 1]`` and
# ``[2, 1, 1]``, respectively. However, ``[3, 1]`` is not part of the :math:`E_{4, 2}` event while
# ``[2, 1, 1]`` is.
#
# Calculating a feature vector
# ----------------------------
#
# We provide two methods for calculating a feature vector of GBS event probabilities in
# Strawberry Fields:
#
# 1. Through sampling.
# 2. Using a Monte Carlo estimate of the probability.
#
# In the first method, all one needs to do is generate some GBS samples from the graph of
# interest and fix the composition of the feature vector. For example, for a feature vector
# :math:`f_{\mathbf{k} = (2, 4, 6), n_{\max}=2}` we use:

print(similarity.feature_vector_sampling(m0, event_photon_numbers=[2, 4, 6], max_count_per_mode=2))

##############################################################################
# For the second method, suppose we want to calculate the event probabilities exactly rather than
# through sampling. To do this, we consider the event probability :math:`p_{k, n_{\max}}` as the
# sum over all sample probabilities in the event. In GBS, each sample probability is determined by
# the hafnian of a relevant sub-adjacency matrix. While this is tough to calculate, what makes
# calculating :math:`p_{k, n_{\max}}` really challenging is the number of samples the corresponding
# event contains! For example, the 6-photon event over 17 modes :math:`E_{k=6, n_{\max}=2}`
# contains the following number of samples :

print(similarity.event_cardinality(6, 2, 17))

##############################################################################
# To avoid calculating a large number of sample probabilities, an alternative is to perform a
# Monte Carlo approximation. Here, samples within an event are selected uniformly at random and
# their resultant probabilities are calculated. If :math:`N` samples :math:`\{S_{1}, S_{2},
# \ldots , S_{N}\}` are generated, then the event probability can be approximated as
#
# .. math::
#     p(E_{k, n_{\max}}) \approx \frac{1}{N}\sum_{i=1}^N p(S_i) |E_{k, n_{\max}}|,
#
# with :math:`|E_{k, n_{\max}}|` denoting the cardinality of the event.
#
# This method can be accessed using the :func:`~.prob_event_mc` function. The 4-photon event is
# approximated as:

print(similarity.prob_event_mc(nx.Graph(m0_a), 4, max_count_per_mode=2, n_mean=6))

##############################################################################
# The feature vector can then be calculated through Monte Carlo sampling using
# :func:`~.feature_vector_mc`.
#
# .. note::
#     The results of :func:`~.prob_event_mc` and :func:`~.feature_vector_mc` are probabilistic and
#     may vary between runs. Increasing the optional ``samples`` parameter will increase accuracy
#     but slow down calculation.
#
# The second method of Monte Carlo approximation is intended for use in scenarios where it is
# computationally intensive to pre-calculate a statistically significant dataset of samples from
# GBS.
#
# Machine learning with GBS graph kernels
# ---------------------------------------
#
# The power of feature vectors that embed graphs in a vector space of real numbers is that we can
# now measure similarities between graphs. This is very useful in machine learning, where similar
# labels are assigned to graphs that are close to each other. GBS feature vectors therefore give
# rise to a similarity measure between graphs!
#
# Let's build this up a bit more. The MUTAG dataset we are considering contains not only graphs
# corresponding to the structure of chemical compounds, but also a *label* of each
# compound based upon its mutagenic effect. The four graphs we consider here have labels:
#
# - MUTAG0: Class 1
# - MUTAG1: Class 0
# - MUTAG2: Class 0
# - MUTAG3: Class 1

classes = [1, 0, 0, 1]

##############################################################################
# We can use GBS feature vectors in a `support vector machine
# <https://en.wikipedia.org/wiki/Support-vector_machine>`__ (SVM) that finds a separating
# hyperplane between classes in the feature space. We start by defining two-dimensional feature
# vectors:

events = [8, 10]
max_count = 2

f1 = similarity.feature_vector_sampling(m0, events, max_count)
f2 = similarity.feature_vector_sampling(m1, events, max_count)
f3 = similarity.feature_vector_sampling(m2, events, max_count)
f4 = similarity.feature_vector_sampling(m3, events, max_count)

import numpy as np

R = np.array([f1, f2, f3, f4])

print(R)

##############################################################################
# There is freedom in the choice of ``events`` composing the feature vectors and we encourage the
# reader to explore different combinations. Note, however, that odd photon-numbered events have
# zero probability because ideal GBS only generates and outputs pairs of photons.
#
# Given our points in the feature space and their target labels, we can use
# scikit-learn's Support Vector Machine `LinearSVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm
# .LinearSVC.html>`__ as our model to train:

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

R_scaled = StandardScaler().fit_transform(R)  # Transform data to zero mean and unit variance

classifier = LinearSVC()
classifier.fit(R_scaled, classes)

##############################################################################
# Here, the term "linear" refers to the *kernel* function used to calculate inner products
# between vectors in the space. We can use a linear SVM because we have already embedded the
# graphs in a feature space based on GBS. We have also rescaled the feature vectors so that they
# zero mean and unit variance using scikit-learn's ``StandardScaler``, a technique
# `often used <https://scikit-learn.org/stable/modules/preprocessing.html>`__ in machine learning.
#
# We can then visualize the trained SVM by plotting the decision boundary with respect to the
# points:

w = classifier.coef_[0]
i = classifier.intercept_[0]

m = -w[0] / w[1]  # finding the values for y = mx + b
b = -i / w[1]

xx = [-1, 1]
yy = [m * x + b for x in xx]

fig = plot.points(R_scaled, classes)
fig.add_trace(plotly.graph_objects.Scatter(x=xx, y=yy, mode="lines"))

plotly.offline.plot(fig, filename="SVM.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/SVM.html
#
# This plot shows the two classes (grey points for class 0 and red points for class 1)
# successfully separated by the linear hyperplane using the GBS feature space. Moreover,
# recall that the two MUTAG1 and MUTAG2 graphs of class 0 are actually isomorphic. Reassuringly,
# their corresponding feature vectors are very similar. In fact, the feature vectors of
# isomorphic graphs should always be identical :cite:`bradler2018graph` - the small discrepancy
# in this plot is due to the statistical approximation from sampling.

# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports,invalid-name
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
#     f_{\mathbf{k}, n_{\max}} = (p_{k_{1}, n_{\max}}, p_{k_{2}, n_{\max}}, \ldots , p_{k_{K},
#         n_{\max}}),
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
# sum over all sample probabilities in the event. In GBS, each sample probability is determined by
# the hafnian of a relevant sub-adjacency matrix. While this is tough to calculate, what makes
# calculating :math:`p_{k, n_{\max}}` really tough is the number of samples the corresponding
# event contains! For example, the 17-photon event :math:`E_{k=6, n_{\max}=2}` contains the
# following number of samples:

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
# such as `support vector machines <https://en.wikipedia.org/wiki/Support-vector_machine>`__ (SVMs).
#
# Let's build this up a bit more. The MUTAG dataset we are considering contains not only graphs
# corresponding to the structure of chemical compounds, but also a *classification* of each
# compound based upon its mutagenic effect. The four graphs we consider here have classifications:
#
# - MUTAG0: Class 1
# - MUTAG1: Class 0
# - MUTAG2: Class 0
# - MUTAG3: Class 1

classes = [1, 0, 0, 1]

##############################################################################
# Can we use GBS feature vectors to classify these graphs? We start by defining two-dimensional
# feature vectors:

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
# Given our points in the feature space and corresponding classifications, we can use scikit learn
# to train an SVM:

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

R_scaled = StandardScaler().fit_transform(R)  # Transform data to zero mean and unit variance

classifier = LinearSVC()
classifier.fit(R_scaled, classes)

##############################################################################
# Here, the term "linear" refers to the *kernel* function used to calculate inner products
# between vectors in the space. We can use a linear SVM because we have already embedded the
# graphs in a feature space based upon GBS. We can then visualize the trained SVM by plotting the
# decision boundary with respect to the points:

w = classifier.coef_[0]
i = classifier.intercept_[0]

m = -w[0] / w[1]  # finding the values for y = mx + b
b = -i / w[1]

xx = [-1, 1]
yy = [m * x + b for x in xx]


from typing import Optional

GREEN = "#3e9651"
RED = "#cc2529"
LIGHT_GREY = "#CDCDCD"
VERY_LIGHT_GREY = "#F2F2F2"


# pylint: disable=redefined-outer-name
def plot_points(
        R: np.ndarray, sample: Optional[list] = None, plot_size: int = 500, point_size: float = 30
):  # pragma: no cover
    """Creates a Plotly plot of two-dimensional points given their input coordinates. Sampled
    points can be optionally highlighted among all points.

    **Example usage:**

    >>> R = np.random.normal(0, 1, (50, 2))
    >>> sample = [1] * 10 + [0] * 40  # select first ten points
    >>> plot_points(R, sample).show()

    .. image:: ../../_static/normal_pp.png
       :width: 40%
       :align: center
       :target: javascript:void(0);

    Args:
        R (np.array): Coordinate matrix. Rows of this array are the coordinates of the points.
        sample (list[int]): optional subset of sampled points to be highlighted
        plot_size (int): size of the plot in pixels, rendered in a square layout
        point_size (int): size of the points, specified by its area

    Returns:
         Figure: figure of points with optionally highlighted sample
    """
    try:
        import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "Plotly required for using plot_points(). Can be installed using pip install "
            "plotly or visiting https://plot.ly/python/getting-started/#installation"
        )

    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=0, l=0, r=0, t=25),
        height=plot_size,
        width=plot_size,
        plot_bgcolor="white",
    )

    points = go.Scatter(
        x=R[:, 0],
        y=R[:, 1],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color=VERY_LIGHT_GREY, size=point_size, line=dict(color="black", width=point_size / 20)
        ),
    )

    points.text = [str(i) for i in range(len(R))]

    if sample:
        s_x = []
        s_y = []
        # pylint: disable=redefined-outer-name
        sampled_points = [i for i in range(len(sample)) if sample[i] > 0]
        for i in sampled_points:
            s_x.append(R[i, 0])
            s_y.append(R[i, 1])

        samp = go.Scatter(
            x=s_x,
            y=s_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                color=RED, size=point_size, line=dict(color="black", width=point_size / 20)
            ),
        )

        samp.text = [str(i) for i in sampled_points]

        f = go.Figure(data=[points, samp], layout=layout)

    else:
        f = go.Figure(data=[points], layout=layout)

    return f


# The above function will be deleted

fig = plot_points(R_scaled, classes)
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
# in this plot is actually due to the statistical approximation from sampling.

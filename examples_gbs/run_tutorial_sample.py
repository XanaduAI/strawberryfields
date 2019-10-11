"""
Sampling Tutorial
=================

This tutorial provides a quick guide to generating samples from GBS. This functionality is
provided in the :mod:`~.gbs.sample` module of the applications layer.

Sampling from GBS
-----------------

A GBS device can be programmed to sample from any symmetric matrix :math:`A`. To sample,
we must specify both the mean number of photons being generated in the device and the form of
detection used at the output: threshold detection or photon-number resolving (PNR) detection.
Threshold detectors are restricted to measuring whether or not one or more photons have arrived
at the detector, whereas PNR detectors are able to count the number of photons.

Let's take a look at both types of sample. We can generate samples from a random 5-dimensional
matrix:
"""

from strawberryfields.gbs import sample
import numpy as np

modes = 5
n_mean = 4
samples = 5

A = np.random.random((modes, modes))
A = A + A.T

s_thresh = sample.sample(A, n_mean, samples, threshold=True)
s_pnr = sample.sample(A, n_mean, samples, threshold=False)

print(s_thresh)
print(s_pnr)

##############################################################################
# In each case, a sample is a sequence of integers of ``len(modes) = 5``. Threshold samples are
# ``0``'s and ``1``'s, corresponding to whether or not photons were detected in a mode. A ``1``
# here is conventionally called a "click". PNR samples are non-negative integers counting the number
# of photons detected in each mode. For example, suppose a PNR sample is ``[2, 1, 1, 0, 0]``,
# meaning that 2 photons were detected in mode 0, 1 photons were detected in modes 1 and 2,
# and 0 photons were detected in modes 3 and 4. If threshold detectors were used instead,
# the sample would be: ``[1, 1, 1, 0, 0]``.
#
# Note that the :func:`~.gbs.sample.seed` function can be used whenever a repeatable output is
# desired, but is not required in general.
#
# Sampling subgraphs
# ------------------
#
# So when would threshold detection or PNR detection be preferred in GBS? Since threshold samples
# can be post-processed from PNR samples, we might expect that PNR detection is always the
# preferred choice. However, *simulating* PNR-based GBS is significantly slower, and it turns out
# that threshold samples can provide enough useful information for a range of applications.
#
# The applications layer of Strawberry Fields focuses primarily on solving graph-based problems.
# In this setting, we typically want to use GBS to sample subgraphs of a graph which are likely
# to be dense due to the probability distribution of GBS :cite:`arrazola2018using`. In this case,
# threshold sampling is enough, since it lets us select nodes of the subgraph. Let's take more of
# a look at this by using a small fixed graph as an example:

from strawberryfields.gbs import plot
import networkx as nx
import plotly

adj = np.array(
    [
        [0, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 0],
    ]
)

graph = nx.Graph(adj)

plotly.offline.plot(plot.plot_graph(graph), filename="random_graph.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/random_graph.html
#
# We can sample this graph from GBS using:

n_mean = 4
samples = 20

s = sample.sample(adj, n_mean, samples)

print(s[:5])

##############################################################################
# Each sample in ``s`` is a list of modes with ``1``'s for nodes that have clicked and ``0``'s
# for nodes that haven't. We want to convert a sample to another representation where the result
# is a list of modes that have clicked. This list of modes can be used to select a subgraph of
# the graph. For example, if ``[0, 1, 0, 1, 1, 0]`` is a sample from GBS then ``[1, 3, 4]`` are
# the selected nodes of the corresponding subgraph.
#
# However, the number of clicks in GBS is a random variable and we are not always guaranteed to
# have enough clicks in a sample for the resultant subgraph to be of interest. We can filter out
# the uninteresting samples using:

min_clicks = 3
max_clicks = 4

s = sample.postselect(s, min_clicks, max_clicks)
print(len(s))


##############################################################################
# As expected, we can see that we have lost some samples. The number of samples that survive this
# postselection is determined by the mean photon number in GBS.
#
# Let's convert our postselected samples to subgraphs:

subgraphs = sample.to_subgraphs(s, graph)

print(subgraphs)

##############################################################################
# We can take a look at one of the sampled subgraphs:

plotly.offline.plot(plot.plot_graph(graph, subgraphs[0]), filename="subgraph.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/subgraph.html
#
# These sampled subgraphs act as the starting point for some of the applications made available
# in Strawberry Fields, including the maximum clique and dense subgraph identification problems.
# Go and check out the other tutorials in the applications layer to see what you can do with GBS!
#
# .. note::
#       Simulating GBS can be computationally intensive when using both threshold and PNR
#       detectors. After all, it is a quantum algorithm! To help users get to grips with the
#       applications layer as quickly as possible, we have provided datasets of pre-calculated
#       GBS samples. These datasets are available in the :mod:`~.gbs.data` module.

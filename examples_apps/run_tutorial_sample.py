# pylint: disable=invalid-name,no-member,wrong-import-position,wrong-import-order,ungrouped-imports
"""
.. _apps-sample-tutorial:

Sampling
========

*Technical details are available in the API documentation:* :doc:`/code/api/strawberryfields.apps.sample`

Quantum computers are probabilistic and a common task is to take samples. Strawberry Fields
can be used to construct quantum photonic circuits and sample from them using a variety of
different measurements.

Gaussian boson sampling (GBS) is a photonic algorithm that can be realized using near-term
devices. Samples from this device can used for :ref:`graph optimization <graphs-intro>`,
:ref:`machine learning <ml-intro>` and :ref:`chemistry calculations <chemistry-intro>`.
Strawberry Fields provides high-level tools for embedding problems into GBS without needing to
worry about designing a quantum circuit.

Sampling from GBS
-----------------

A GBS device can be programmed to sample from any symmetric matrix :math:`A`. To sample,
we must specify the mean number of photons being generated in the device and optionally the form of
detection used at the output: threshold detection or photon-number resolving (PNR) detection.
Threshold detectors are restricted to measuring whether photons have arrived at the detector,
whereas PNR detectors are able to count the number of photons. Photon loss can also be specified
with the ``loss`` argument.

Sampling functionality is provided in the :mod:`~.apps.sample` module.

Let's take a look at both types of sampling methods. We can generate samples from a random
5-dimensional symmetric matrix:
"""

from strawberryfields.apps import sample
import numpy as np

modes = 5
n_mean = 6
samples = 5

A = np.random.normal(0, 1, (modes, modes))
A = A + A.T

s_thresh = sample.sample(A, n_mean, samples, threshold=True)
s_pnr = sample.sample(A, n_mean, samples, threshold=False)

print(s_thresh)
print(s_pnr)

##############################################################################
# In each case, a sample is a sequence of integers of length five, i.e., ``len(modes) = 5``.
# Threshold samples are ``0``'s and ``1``'s, corresponding to whether or not photons were
# detected in a mode. A ``1`` here is conventionally called a "click". PNR samples are
# non-negative integers counting the number of photons detected in each mode. For example,
# suppose a PNR sample is ``[2, 1, 1, 0, 0]``, meaning that 2 photons were detected in mode 0,
# 1 photons were detected in modes 1 and 2, and 0 photons were detected in modes 3 and 4. If
# threshold detectors were used instead, the sample would be: ``[1, 1, 1, 0, 0]``.
#
# A more general :func:`~.apps.sample.gaussian` function allows for sampling from arbitrary pure
# Gaussian states.
#
# Sampling subgraphs
# ------------------
#
# So when would threshold detection or PNR detection be preferred in GBS? Since threshold samples
# can be post-processed from PNR samples, we might expect that PNR detection is always the
# preferred choice. However, in practice *simulating* PNR-based GBS is significantly slower,
# and it turns out that threshold samples can provide enough useful information for a range of
# applications.
#
# Strawberry Fields provides tools for solving graph-based problems. In this setting,
# we typically want to use GBS to sample subgraphs, which are likely to be dense due to the
# probability distribution of GBS :cite:`arrazola2018using`. In this case, threshold sampling
# is enough, since it lets us select nodes of the subgraph. Let's take a look at this by using a
# small fixed graph as an example:

from strawberryfields.apps import plot
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
plot_graph = plot.graph(graph)

plotly.offline.plot(plot_graph, filename="random_graph.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_apps/random_graph.html
#
# .. note::
#     The command ``plotly.offline.plot()`` is used to display plots in the documentation. In
#     practice, you can simply use ``plot_graph.show()`` to view your graph.
#
# This is a 6-node graph with the nodes ``[0, 1, 4, 5]`` fully connected to each other. We expect
# to be able to sample dense subgraphs with high probability.
#
# Samples can be generated from this graph through GBS using the :func:`~.apps.sample.sample`
# function:

n_mean = 4
samples = 20

s = sample.sample(adj, n_mean, samples)

print(s[:5])

##############################################################################
# Each sample in ``s`` is a list of modes with ``1``'s for nodes that have clicked and ``0``'s
# for nodes that haven't. We want to convert a sample to another representation where the result
# is a list of modes that have clicked. This list of modes can be used to select a subgraph.
# For example, if ``[0, 1, 0, 1, 1, 0]`` is a sample from GBS then ``[1, 3, 4]`` are
# the selected nodes of the corresponding subgraph.
#
# However, the number of clicks in GBS is a random variable and we are not always guaranteed to
# have enough clicks in a sample for the resultant subgraph to be of interest. We can filter out
# the uninteresting samples using the :func:`~.apps.sample.postselect` function:

min_clicks = 3
max_clicks = 4

s = sample.postselect(s, min_clicks, max_clicks)

print(len(s))
s.append([0, 1, 0, 1, 1, 0])

##############################################################################
# As expected, we have fewer samples than before. The number of samples that survive this
# postselection is determined by the mean photon number in GBS. We have also added in our example
# sample ``[0, 1, 0, 1, 1, 0]`` to ensure that there is at least one for the following.
#
# Let's convert our postselected samples to subgraphs:

subgraphs = sample.to_subgraphs(s, graph)

print(subgraphs)

##############################################################################
# We can take a look at one of the sampled subgraphs:

plotly.offline.plot(plot.graph(graph, subgraphs[0]), filename="subgraph.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_apps/subgraph.html
#
# These sampled subgraphs act as the starting point for some of the applications made available
# in Strawberry Fields, including the maximum clique and dense subgraph identification problems.
#
# .. note::
#       Simulating GBS can be computationally intensive when using both threshold and PNR
#       detectors. After all, we are using a classical algorithm to simulate a quantum process!
#       To help users get to grips with the applications of Strawberry Fields as quickly as
#       possible, we have provided datasets of pre-calculated GBS samples. These datasets are
#       available in the :mod:`~.apps.data` module.

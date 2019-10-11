"""
Sampling Tutorial
=================

This tutorial provides a quick guide to generating samples from GBS. This functionality is
provided in the `~.gbs.sample` module of the applications layer.

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
n_mean = 2
samples = 5

A = np.random.random((modes, modes))
A = A + A.T
sample.seed(1967)

s_thresh = sample.sample(A, n_mean, samples, threshold=True)
print(s_thresh)

s_pnr = sample.sample(A, n_mean, samples, threshold=False)
print(s_pnr)

##############################################################################
# In each case, a sample is a sequence of integers of ``len(modes) = 5``. Threshold samples are
# ``0``'s and ``1``'s, corresponding to whether or not photons were detected in a mode. PNR
# samples are non-negative integers counting the number of photons detected in each mode. For
# example, the first sample of ``s_pnr`` is ``[2, 1, 1, 0, 0]``, meaning that 2 photons were
# detected in mode 0, 1 photons were detected in modes 1 and 2, and 0 photons were detected in
# modes 3 and 4. If threshold detectors were used instead, the sample would be: ``[1, 1, 1, 0,
# 0]``.
#
# Notice that we used the :func:`~.gbs.sample.seed` function to fix the random seed. This can be
# used whenever a repeatable output is desired, but is not required in general.
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
# a look at this. We can look at the pre-made :class:`~.gbs.data.Planted` graph in the
# :mod:`~.gbs.data` module:

from strawberryfields.gbs import data, plot
import networkx as nx
import plotly

dataset = data.Planted()
adj = dataset.adj
graph = nx.Graph(adj)

plotly.offline.plot(plot.plot_graph(graph), filename="Planted.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/Planted.html
#
# To sample subgraphs of this graph we can do:

n_mean = 2
samples = 5

s = sample.sample(adj, n_mean, samples)
subgraphs = sample.to_subgraphs(s, graph)

print(s)
print(subgraphs)

##############################################################################
#

# pylint: disable=wrong-import-position,wrong-import-order
"""
Graph Similarity Tutorial
=========================

This tutorial looks at how to use GBS to construct a similarity measure between graphs,
what is known as a graph kernel. Kernels can be applied to graph-based data for machine learning
tasks such as classification in machine learning using a support vector machine
:cite:schuld2019quantum.

Graph data
----------

Let's use the MUTAG dataset of graphs for this tutorial
:cite:`debnath1991structure,kriege2012subgraph`. This is a dataset of 188 different graphs that
each correspond to the structure of a chemical compound. Our objective is to use GBS samples from
these graphs to measure their similarity.

The :module:`~.gbs.data` module provides pre-calculated GBS samples from four graphs in the
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
# We can now plot the four graphs using the :module:`~gbs.plot` module. To use this module,
# we need to convert the adjacency matrices into NetworkX Graphs:

import networkx as nx
import plotly

plot_mutag_0 = gbs.plot.plot_subgraph(nx.Graph(m0_a))
plot_mutag_1 = gbs.plot.plot_subgraph(nx.Graph(m1_a))
plot_mutag_2 = gbs.plot.plot_subgraph(nx.Graph(m2_a))
plot_mutag_3 = gbs.plot.plot_subgraph(nx.Graph(m3_a))

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

print(m0[:5])

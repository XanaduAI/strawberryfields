# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
"""
.. _gbs-subgraph-tutorial:

Dense subgraph tutorial
=======================

Graphs can be used to model a wide variety of concepts: social networks, financial markets,
biological networks, and many others. A common problem of interest is to find subgraphs that
contain a large number of connections between their nodes. These subgraphs may correspond to
communities in social networks, correlated assets in a market, or mutually influential proteins
in a biological network. This tutorial studies how GBS can be used to identify dense subgraphs.

Mathematically, this task is known as the `dense subgraph problem
<https://en.wikipedia.org/wiki/Dense_subgraph>`__. The density of a :math:`k`-node subgraph is equal
to the number of its edges divided by the maximum possible number of edges :math:`k(k-1)/2`.
Identifying the densest graph of a given size, known as the densest-:math:`k` subgraph problem,
is `NP-Hard <https://en.wikipedia.org/wiki/NP-hardness>`__.


As shown in :cite:`arrazola2018using`, a defining feature of GBS is that when we encode a graph
into a GBS device, it samples dense subgraphs with high probability. This property can be
used to find dense subgraphs by sampling from a GBS device and postprocessing the outputs.
Let's take a look!

Finding dense subgraphs
-----------------------
As usual, the first step is to import all required modules. We'll need the :mod:`~.gbs.data`
module to load pre-generated samples, the :mod:`~.gbs.sample` module to postselect samples, the
:mod:`~.gbs.subgraph` module to search for dense subgraphs, and the :mod:`~.gbs.plot` module to
visualize the graphs. We'll also use Plotly which is required for the :mod:`~.gbs.plot` module and
NetworkX for graph operations.
"""
from strawberryfields.gbs import data, sample, subgraph, plot
import plotly
import networkx as nx

##############################################################################
# In this tutorial, we'll study a 30-node graph with a planted 10-node graph, as considered in
# :cite:`arrazola2018using`. The graph is generated by joining two Erdős–Rényi random graphs. The
# first graph of 20 nodes is created with edge probability of 0.5. The second planted
# graph is generated with edge probability of 0.875. The planted nodes are the last ten nodes in the
# graph. The two graphs are joined by selecting 8 nodes at random from both graphs and adding an
# edge between them. This graph has the sneaky property that even though the planted subgraph is the
# densest of its size, its nodes have a lower average degree than the nodes in the rest of the
# graph.
#
# The :mod:`~.gbs.data` module has pre-generated GBS samples from this graph. Let's load them,
# postselect on samples with a large number of clicks, and convert them to subgraphs:

planted = data.Planted()
postselected = sample.postselect(planted, 16, 30)
pl_graph = nx.to_networkx_graph(planted.adj)
samples = sample.to_subgraphs(postselected, pl_graph)
print(len(samples))

##############################################################################
# Not bad! We have more than 2000 samples to play with 😎. The planted subgraph is actually easy to
# identify; it even appears clearly from the force-directed Kamada-Kawai algorithm that is used to
# plot graphs in Strawberry Fields:
sub = list(range(20, 30))
plot_graph = plot.plot_graph(pl_graph, sub)
plotly.offline.plot(plot_graph, filename="planted.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/planted.html
#
# .. note::
#     The command ``plotly.offline.plot()`` is used to display plots in the documentation. In
#     practice, you can simply use ``plot_graph.show()`` to view the graph.

##############################################################################
# A more interesting challenge is to find dense subgraphs of different sizes; it is often
# useful to identify many high-density subgraphs, not just the densest ones. This is the purpose of
# the :func:`~.subgraph.search` function in the :mod:`~.gbs.subgraph` module: to identify
# collections of dense subgraphs for a range of sizes. The output of this function is a
# dictionary whose keys correspond to subgraph sizes within the specified range. The values in
# the dictionary are the top subgraphs of that size and their corresponding density.

dense = subgraph.search(samples, pl_graph, 8, 16, max_count=3)  # we look at top 3 densest subgraphs
for k in range(8, 17):
    print(dense[k][0])  # print only the densest subgraph of each size

##############################################################################
# From the results of the search we learn that, depending on their size, the densest subgraphs
# belong to different regions of the graph: dense subgraphs of less than ten nodes are contained
# within the planted subgraph, whereas larger dense subgraphs appear outside of the planted
# subgraph. Smaller dense subgraphs can be cliques, characterized by having
# maximum density of 1, while larger subgraphs are less dense. Let's see what the smallest and
# largest subgraphs look like:

densest_8 = plot.plot_graph(pl_graph, dense[8][0][1])
densest_16 = plot.plot_graph(pl_graph, dense[12][0][1])

plotly.offline.plot(densest_8, filename="densest_8.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/densest_8.html

plotly.offline.plot(densest_16, filename="densest_16.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/densest_16.html

##############################################################################
# In principle there are different methods to postprocess GBS outputs to identify dense
# subgraphs. For example, techniques for finding maximum cliques, included in the
# :mod:`~.gbs.clique` module could help provide initial subgraphs that can be resized to find larger
# dense subgraphs. Such methods are hybrid algorithms combining the ability of GBS to sample dense
# subgraphs with clever classical techniques. Can you think of your own hybrid algorithm? 🤔

# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
"""
Dense Subgraph Tutorial
=========================
This tutorial studies how GBS can be employed to identify dense subgraphs. Graphs can be used to
model a wide variety of objects: social networks, financial markets, biological
networks, and many others. A common problem of interest is to find subgraphs that contain a large
number of connections between its nodes. These may correspond to communities in social networks,
correlated assets in a market, or mutually influential proteins in a biological network.
Mathematically, this task is known as the `dense subgraph problem
<https://en.wikipedia.org/wiki/Dense_subgraph>`_. Identifying the densest graph of a given size,
known as the densest-*k* subgraph problem, is `NP-Hard <https://en.wikipedia.org/wiki/NP-hardness>`_.
Here density denotes the number of edges in the subgraph divided by the maximum possible number
of edges.

As shown in :cite:`arrazola2018using`, a defining feature of GBS is that when we encode a graph
into a GBS device, it samples dense subgraphs with high probability. This feature can be employed to
find dense subgraphs by simply sampling from GBS and postprocessing the outputs. Let's take a
look at how this is done.

"""
from strawberryfields.gbs import data, plot, sample, clique
import numpy as np
import networkx as nx

##############################################################################
# The adjacency matrix of the TACE-AS graph can be loaded from the ``data`` module and the
# graph can be visualized using the :mod:`~gbs.plot` module:

Pl = data.Planted()
A = Pl.adj
Pl_graph = nx.Graph(A)
graph_fig = plot.plot_graph(Pl_graph, list(range(20, 30)))
# graph_fig.show()

samps = sample.to_subgraphs(Pl, Pl_graph)
samples = sample.postselect(samps, 12, 20)
print(len(samples))
shrunk = [clique.shrink(s, Pl_graph) for s in samples[:100]]
cliques = [clique.search(s, Pl_graph, 10) for s in shrunk]

print(cliques)
lens = [len(c) for c in cliques]
print(np.max(lens))
fig = plot.plot_graph(Pl_graph, cliques[np.argmax(lens)])
fig.show()




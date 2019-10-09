# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
"""
Max Clique Tutorial
=========================
This tutorial explores how we can combine GBS with local search algorithms to find large
cliques in graphs.

A clique is a special type of subgraph where all possible connections between
nodes are present; they are densest possible subgraphs of their size. The maximum clique problem,
or max clique for short, asks the following question: given a graph :math:`G`, what is the
largest clique in the graph? This problem is NP-Hard, meaning that finding the actual largest
clique becomes extremely challenging for graphs with many nodes. This is not an impediment to using
clever techniques to find large cliques in graphs, which is precisely what we'll do in this
tutorial.

Graph data
----------
To get started, we'll analyze the 24-node TACE-AS graph used in :cite:`banchi2019molecular`. This
is the *binding interaction graph* representing the spatial compatibility of atom pairs in a
protein-molecule complex. Cliques in this graph correspond to stable docking configurations, which
are of interest in determining how the molecule interacts with the protein.

The first step is to import the gbs module as well as external dependencies:
"""
from strawberryfields import gbs
import numpy as np
import networkx as nx
import plotly

##############################################################################
# The adjacency matrix of the TACE-AS graph can be loaded from the data module and the
# graph can be visualized using the :mod:`~gbs.plot` module:
TA = gbs.data.TaceAs()
A = TA.adj
TA_graph = nx.Graph(A)
fig1 = gbs.plot.plot_graph(TA_graph)
# plotly.offline.plot(fig1, filename="TACE-AS.html")

##############################################################################
# Can you spot any cliques in the graph? It's not so easy using only your eyes.  The TACE-AS graph
# is sufficiently # small that all cliques can be found by performing an exhaustive search over
# all possible subgraphs. For example, below we highlight a small *maximal* clique, i.e., a clique
# not contained inside another clique:

maximal_clique = [4, 11, 12, 18]
fig2 = gbs.plot.plot_graph(TA_graph, maximal_clique)
# plotly.offline.plot(fig2, filename="maximal_clique.html")

##############################################################################
# We'll now use the :mod:`~gbs.clique` module to find larger cliques in the graph. We can make
# use of the pre-generated samples to post-select outputs with any specific number of clicks. For
# this tutorial, we'll look at samples with 8 clicks, of which there 1,984:

def compress(sample):
    out = []
    for i in range(len(sample)):
        if sample[i] > 0:
            out.append(i)

    return out

counts = TA.counts()
samples = []
for i in range(len(TA)):
    if counts[i] == 8:
        samples.append(compress(TA[i]))

print(len(samples))
print(samples[0])

##############################################################################
# GBS produces samples that correspond to subgraphs of large density. For fun, let's confirm this
# is the case by comparing the average subgraph density in the GBS samples to uniformly
# generated samples:

GBS_densities = []
uni_densities = []

for s in samples:
    uni = list(np.random.choice(24, 8))
    GBS_subgraph = TA_graph.subgraph(s)
    uni_subgraph = TA_graph.subgraph(uni)
    GBS_densities.append(nx.density(GBS_subgraph))
    uni_densities.append(nx.density(uni_subgraph))

print('GBS mean density = {:.4f}'.format(np.mean(GBS_densities)))
print('Uniform mean density = {:.4f}'.format(np.mean(uni_densities)))

##############################################################################
# These samples can be shrunk until a clique is formed

##############################################################################
# Look at a few of them, what size cliques have we found?

##############################################################################
# For each such clique, we conduct local search to find larger cliques

##############################################################################
# Plot maximum clique

##############################################################################
# Show full scope with few lines of code for DIMACS graph









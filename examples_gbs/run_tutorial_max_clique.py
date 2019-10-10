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
graph = gbs.plot.plot_graph(TA_graph)
plotly.offline.plot(graph, filename="TACE-AS.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/TACE-AS.html

##############################################################################
# Can you spot any cliques in the graph? It's not so easy using only your eyes.  The TACE-AS graph
# is sufficiently # small that all cliques can be found by performing an exhaustive search over
# all possible subgraphs. For example, below we highlight a small *maximal* clique, i.e., a clique
# not contained inside another clique:

maximal_clique = [4, 11, 12, 18]
clique_0 = gbs.plot.plot_graph(TA_graph, maximal_clique)
plotly.offline.plot(clique_0, filename="maximal_clique.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/maximal_clique.html

##############################################################################
# We'll now use the :mod:`~gbs.clique` module to find larger cliques in the graph. We can make
# use of the pre-generated samples to post-select outputs with any specific number of clicks. For
# this tutorial, we'll look at samples with eight clicks, of which there 1,984:

TA_subgraphs = gbs.sample.to_subgraphs(TA_graph, TA)

counts = TA.counts()
samples = [TA_subgraphs[i] for i in range(len(TA)) if counts[i] == 8]
print(len(samples))

##############################################################################
# GBS produces samples that correspond to subgraphs of large density. For fun, let's confirm this
# by comparing the average subgraph density in the GBS samples to uniformly generated samples:

GBS_dens = []
u_dens = []

for s in samples:
    uniform = list(np.random.choice(24, 8))  # generates uniform samples
    GBS_subgraph = TA_graph.subgraph(s)
    uni_subgraph = TA_graph.subgraph(uniform)
    GBS_dens.append(nx.density(GBS_subgraph))
    u_dens.append(nx.density(uni_subgraph))

print('GBS mean density = {:.4f}'.format(np.mean(GBS_dens)))
print('Uniform mean density = {:.4f}'.format(np.mean(u_dens)))

##############################################################################
# Those look like great GBS samples 💪! To obtain cliques, we shrink the samples by greedily
# removing nodes with low degree until a clique is found. We can also verify that the first sample
# got shrunk to a clique:

shrunk = [gbs.clique.shrink(s, TA_graph) for s in samples]
print(gbs.clique.is_clique(TA_graph.subgraph(shrunk[0])))

##############################################################################
# Let's take a look at some of these cliques. What are the clique sizes in the first ten samples?
# What is the average clique size? How about the largest and smallest clique size?

clique_sizes = [len(s) for s in shrunk]
print('First ten clique sizes = ', clique_sizes[:10])
print('Average clique size = {:.3f}'.format(np.mean(clique_sizes)))
print('Maximum clique size = ', np.max(clique_sizes))
print('Minimum clique size = ', np.min(clique_sizes))

##############################################################################
# Even in the first few samples, we've already identified larger cliques than the maximal clique
# we studied before. Indeed, this simple strategy of shrinking samples gives cliques with average
# size of roughly five.
#
# We can improve this by searching in the neighbourhood of these cliques for larger cliques.
# We'll do this by taking ten iterations of local search and studying the results. Note: this may
# take a few seconds.

searched = [gbs.clique.search(s, TA_graph, 10) for s in shrunk]
clique_sizes = [len(s) for s in searched]
print('First two cliques = ', searched[:2])
print('Average clique size = {:.3f}'.format(np.mean(clique_sizes)))

##############################################################################
# Wow! Local search is very helpful, allowing us to quickly identify larger cliques in
# the vicinity of the smaller cliques. Looking at the cliques we've found, we learn that the largest
# cliques in the graph have size eight, and there are many of them. Let's take a look at the first
# one we found

clique = gbs.plot.plot_graph(TA_graph, searched[0])
plotly.offline.plot(clique, filename="maximum_clique.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/maximum_clique.html

##############################################################################
# The TACE-AS graph is relatively small. A tougher challenge is the 300- node ``p_hat300-1`` random
# graph from the DIMACS maximum clique dataset. In this section of the tutorial, we'll write a short
# program that uses GBS samples in combination with local search to identify large cliques in the
# graph.
#

Phat = gbs.data.PHat()  # Load data
phat_graph = nx.Graph(Phat.adj)  # Obtain graph
phat_subgraphs = gbs.sample.to_subgraphs(phat_graph, Phat)  # Convert samples into subgraphs
counts = Phat.counts()  # Post-select on samples with a minimum click number
samples = [phat_subgraphs[i] for i in range(len(Phat)) if counts[i] >= 16]
shrunk = [gbs.clique.shrink(s, phat_graph) for s in samples]  # Shrink subgraphs to cliques
searched = [gbs.clique.search(s, phat_graph, 10) for s in shrunk]  # Perform local search
clique_sizes = [len(s) for s in searched]  # Show largest clique found
largest_clique = searched[np.argmax(clique_sizes)]
print('Largest clique found is = ', largest_clique)

##############################################################################
# Let's make a plot to take a closer look at the stunning clique we found
largest = gbs.plot.plot_graph(phat_graph, largest_clique)
plotly.offline.plot(largest, filename="largest_clique.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/largest_clique.html

just_largest = gbs.plot.plot_subgraph(phat_graph.subgraph(largest_clique))
plotly.offline.plot(just_largest, filename="just_largest.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/just_largest.html














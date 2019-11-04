# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
"""
.. _gbs-clique-tutorial:

Max Clique Tutorial
=========================
In this tutorial, we'll explore how to combine GBS samples with local search algorithms to
find large cliques in graphs. Let's get started!

A clique is a special type of subgraph where all possible connections between nodes are present;
they are densest possible subgraphs of their size. The maximum clique problem, or max clique for
short, asks the question: given a graph :math:`G`, what is the largest clique in the graph?
Max clique is `NP-Hard <https://en.wikipedia.org/wiki/NP-hardness>`_, so finding the biggest clique
becomes challenging for graphs with many
nodes. This is why we need clever algorithms to identify large cliques!

To get started, we'll analyze the 24-node TACE-AS graph used in :cite:`banchi2019molecular`. This
is the *binding interaction graph* representing the spatial compatibility of atom pairs in a
protein-molecule complex. Cliques in this graph correspond to stable docking configurations, which
are of interest in determining how the molecule interacts with the protein.

The first step is to import the Strawberry Fields GBS module and external dependencies:
"""
from strawberryfields.gbs import data, plot, sample, clique
import numpy as np
import networkx as nx
import plotly

##############################################################################
# The adjacency matrix of the TACE-AS graph can be loaded from the :mod:`~.gbs.data` module and the
# graph can be visualized using the :mod:`~.gbs.plot` module:

TA = data.TaceAs()
A = TA.adj
TA_graph = nx.Graph(A)
plot_graph = plot.graph(TA_graph)
plotly.offline.plot(plot_graph, filename="TACE-AS.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/TACE-AS.html
#
# .. note::
#     The command ``plotly.offline.plot()`` is used to display plots in the documentation. In
#     practice, you can simply use ``plot_graph.show()`` to view your graph.

##############################################################################
# Can you spot any cliques in the graph? It's not so easy using only your eyes! The TACE-AS graph
# is sufficiently small that all cliques can be found by performing an exhaustive search over
# all subgraphs. For example, below we highlight a small *maximal* clique, i.e., a clique
# not contained inside another clique:

maximal_clique = [4, 11, 12, 18]
maximal_fig = plot.graph(TA_graph, maximal_clique)
plotly.offline.plot(maximal_fig, filename="maximal_clique.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/maximal_clique.html

##############################################################################
# We'll now use the :mod:`~.gbs.clique` module to find larger cliques in the graph. We can make
# use of the pre-generated samples from the TACE-AS graph in the :mod:`~.gbs.data` module and post-select samples with a specific number of clicks. For
# this tutorial, we'll look at samples with eight clicks, of which there are a total of 1,984:

postselected = sample.postselect(TA, 8, 8)
samples = sample.to_subgraphs(postselected, TA_graph)
print(len(samples))

##############################################################################
# GBS produces samples that correspond to subgraphs of high density. For fun, let's confirm this
# by comparing the average subgraph density in the GBS samples to uniformly generated samples:

GBS_dens = []
u_dens = []

for s in samples:
    uniform = list(np.random.choice(24, 8, replace=False))  # generates uniform sample
    GBS_dens.append(nx.density(TA_graph.subgraph(s)))
    u_dens.append(nx.density(TA_graph.subgraph(uniform)))

print("GBS mean density = {:.4f}".format(np.mean(GBS_dens)))
print("Uniform mean density = {:.4f}".format(np.mean(u_dens)))

##############################################################################
# Those look like great GBS samples 💪! To obtain cliques, we shrink the samples by greedily
# removing nodes with low degree until a clique is found.

shrunk = [clique.shrink(s, TA_graph) for s in samples]
print(clique.is_clique(TA_graph.subgraph(shrunk[0])))

##############################################################################
# Let's take a look at some of these cliques. What are the clique sizes in the first ten samples?
# What is the average clique size? How about the largest and smallest clique size?

clique_sizes = [len(s) for s in shrunk]
print("First ten clique sizes = ", clique_sizes[:10])
print("Average clique size = {:.3f}".format(np.mean(clique_sizes)))
print("Maximum clique size = ", np.max(clique_sizes))
print("Minimum clique size = ", np.min(clique_sizes))

##############################################################################
# Even in the first few samples, we've already identified larger cliques than the 4-node clique
# we studied before. Awesome! Indeed, this simple shrinking strategy gives cliques with average
# size of roughly five. We can enlarge these cliques by searching for larger cliques in their
# vicinity. We'll do this by taking ten iterations of local search and studying the results.
# Note: this may take a few seconds.

searched = [clique.search(s, TA_graph, 10) for s in shrunk]
clique_sizes = [len(s) for s in searched]
print("First two cliques = ", searched[:2])
print("Average clique size = {:.3f}".format(np.mean(clique_sizes)))

##############################################################################
# Wow! Local search is very helpful, we've found cliques with the maximum size of eight for
# essentially all samples 🤩.  Let's take a look at the first clique we found

clique_fig = plot.graph(TA_graph, searched[0])
plotly.offline.plot(clique_fig, filename="maximum_clique.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/maximum_clique.html

##############################################################################
# The TACE-AS graph is relatively small, so finding large cliques is not particularly difficult. A
# tougher challenge is the 300-node ``p_hat300-1`` random graph from the `DIMACS
# <http://iridia.ulb.ac.be/~fmascia/maximum_clique/DIMACS-benchmark>`_ maximum clique
# dataset. In this section of the tutorial, we'll write a short program that uses GBS samples in
# combination with local search to identify large cliques in this graph.
#

Phat = data.PHat()  # Load data
phat_graph = nx.Graph(Phat.adj)  # Obtain graph
postselected = sample.postselect(Phat, 16, 20)  # Post-select samples
samples = sample.to_subgraphs(postselected, phat_graph)  # Convert samples into subgraphs
shrunk = [clique.shrink(s, phat_graph) for s in samples]  # Shrink subgraphs to cliques
searched = [clique.search(s, phat_graph, 10) for s in shrunk]  # Perform local search
clique_sizes = [len(s) for s in searched]
largest_clique = searched[np.argmax(clique_sizes)]  # Identify largest clique found
print("Largest clique found is = ", largest_clique)

##############################################################################
# Let's make a plot to take a closer look at the largest clique we found
largest_fig = plot.graph(phat_graph, largest_clique)
plotly.offline.plot(largest_fig, filename="largest_clique.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/largest_clique.html

just_largest = plot.subgraph(phat_graph.subgraph(largest_clique))
plotly.offline.plot(just_largest, filename="just_largest.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/just_largest.html

##############################################################################
# The ``p_hat300-1`` graph has several maximum cliques of size eight,
# and we have managed to find them! What other graphs can you analyze using GBS?

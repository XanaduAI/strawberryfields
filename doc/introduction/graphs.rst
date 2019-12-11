Graphs and networking
=====================

.. raw:: html

	<style>
		.sphx-glr-thumbcontainer {
			float: right;
			margin-top: -20px;
			margin-right: 0;
			margin-left: 20px;
		}
	</style>

Graphs can be used to model a wide variety of concepts: social networks, financial markets,
biological networks, and many others. The applications layer of Strawberry Fields provides tools
for solving a range of graph-based problems.

Dense subgraphs
---------------

**Module name:** :mod:`strawberryfields.apps.subgraph`

.. currentmodule:: strawberryfields.apps.subgraph

.. customgalleryitem::
    :tooltip: Dense subgraphs
    :description: :doc:`../tutorials_apps/run_tutorial_dense`
    :figure: figures/dense.png

A common problem of interest is to find subgraphs that contain a large number of connections
between their nodes. These subgraphs may correspond to communities in social networks, correlated
assets in a market, or mutually influential proteins in a biological network.


This module provides tools for users to find dense subgraphs of different sizes.
The search heuristic in this module works by resizing a collection of starting subgraphs and
keeping track of the densest identified. The starting subgraphs can be selected by sampling from
GBS, resulting in candidates that are likely to be dense :cite:`arrazola2018using`.

.. raw:: html

    <div style='clear:both'></div>

Algorithm
^^^^^^^^^

The heuristic algorithm provided proceeds as follows. Each starting subgraph :math:`s` is resized
to a range of sizes :math:`\{k\}_{k_{\min}}^{k_{\max}}`, resulting in the subgraphs
:math:`s_{k}`. For a given size :math:`k`, a collection of the densest :math:`n` subgraphs
identified is recorded, meaning that :math:`s_{k}` is added to the collection only if it has
sufficient density.

.. autosummary::
    search

This algorithm returns a dictionary over the range of sizes specified, with each value being the
collection of densest-:math:`k` subgraphs. This collection is a list of tuple pairs specifying
the subgraph density and nodes.

Subgraph resizing
^^^^^^^^^^^^^^^^^

The key element of the :func:`search` algorithm is the resizing of each subgraph, allowing a
range of subgraph sizes to be tracked. Resizing proceeds by greedily adding or removing nodes to
a subgraph one-at-a-time. Node selection is carried out by picking the node with the greatest
or least degree with respect to to the subgraph, with ties settled uniformly at random.

.. autosummary::
    resize

This function returns a dictionary over the range of sizes specified, with each value being the
corresponding resized subgraph.

Maximum clique
--------------

**Module name:** :mod:`strawberryfields.apps.clique`

.. currentmodule:: strawberryfields.apps.clique

.. customgalleryitem::
    :tooltip: Maximum clique
    :description: :doc:`../tutorials_apps/run_tutorial_max_clique`
    :figure: figures/max_clique.png

A clique is a special type of subgraph where all possible connections between nodes are present;
they are densest possible subgraphs of their size. The maximum clique problem, or max clique for
short, asks the question: given a graph :math:`G`, what is the largest clique in the graph?

This module provides tools for users to identify large cliques in graphs. A clique is a subgraph
where all nodes are connected to each other. The maximum clique problem is to identify the
largest clique in a graph. It has been shown that samples from GBS can
be used to select dense subgraphs as a starting seed for heuristic algorithms
:cite:`banchi2019molecular`.

.. raw:: html

    <div style='clear:both'></div>

Algorithm
^^^^^^^^^

This module provides a variant of the local search heuristics described in
:cite:`pullan2006dynamic` and :cite:`pullan2006phased`. The algorithm proceeds as follows:

#. A small clique in the graph is identified. The initial clique can be a single node, or it can
   be obtained by shrinking a random subgraph, for example obtained from GBS.
#. The clique is grown as much as possible by adding nodes connected to all nodes in the clique.
#. When no more growth is possible, a node in the clique is swapped with a node outside of it.

Steps 2-3 are repeated until a pre-determined number of steps have been completed, or until no
more growth or swaps are possible.

.. autosummary::
    search

Clique growth and swapping
^^^^^^^^^^^^^^^^^^^^^^^^^^

A clique can be grown by evaluating the set :math:`C_0` of nodes in the remainder of the graph that
are connected to all nodes in the clique. A single node from :math:`C_0` is selected and added to
the clique. This process is repeated until :math:`C_0` becomes empty.

.. autosummary::
    grow
    c_0

Searching the local space around a clique can be achieved by swapping a node from the clique with a
node in the remainder of the graph. The first step is to evaluate the set :math:`C_1` of nodes in
the remainder of the graph that are connected to *all but one* of the nodes in the current
clique. A swap is then performed by adding the node into the clique and removing the node in the
clique that is not connected to it.

.. autosummary::
    swap
    c_1

Using GBS to find a starting clique
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Samples from GBS correspond to subgraphs that are likely to be dense.
These subgraphs may not be cliques, which is the required input to the :func:`search` algorithm.
To reconcile this, a subgraph may be shrunk by removing nodes until the remainder forms a
clique. This can be achieved by selecting the node with the lowest degree relative to the rest of
subgraph and removing the node, repeating the process until a clique is found.

.. autosummary::
    shrink
    is_clique

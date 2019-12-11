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
        #right-column.card {
            box-shadow: none!important;
        }
        #right-column.card:hover {
            box-shadow: none!important;
        }
	</style>

Many problems of practical relevance can be stated in terms of a graph or network. For example,
graphs can be used to model financial markets, biological networks, and social networks. Graphs are
composed of a collection of interconnected nodes.

Let's take a financial network as an example. Suppose you want to transfer some money
internationally. Your bank might do this by transferring the currency through a number of
intermediary banks before it gets to its destination. This transaction describes a path on a
payment network, where banks represent nodes on the graph which are connected if a pair of banks
can perform a transfer. One problem a bank may look to solve is to find the shortest path through
the network to minimize transaction costs.

Strawberry Fields provides tools for solving a range of graph-based problems.

Dense subgraphs
---------------

.. customgalleryitem::
    :tooltip: Dense subgraphs
    :description: :doc:`../tutorials_apps/run_tutorial_dense`
    :figure: ../_static/dense.png

.. raw:: html

    <div style='clear:both'></div>

Maximum clique
--------------

.. customgalleryitem::
    :tooltip: Maximum clique
    :description: :doc:`../tutorials_apps/run_tutorial_max_clique`
    :figure: ../_static/max_clique.png

.. raw:: html

    <div style='clear:both'></div>

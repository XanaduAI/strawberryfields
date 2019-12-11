Graphs and networking
=====================

.. raw:: html

	<style>
		.sphx-glr-thumbcontainer {
			float: center;
			margin-top: 20px;
			margin-right: 50px;
            margin-bottom: 20px;
		}
        #right-column.card {
            box-shadow: none!important;
        }
        #right-column.card:hover {
            box-shadow: none!important;
        }
	</style>

Strawberry Fields provides tools for solving graph or network-based problems. Many problems of
practical relevance can be stated in terms of a graph. For example, graphs can be used to model
financial markets, biological networks, and social networks.

Graphs are composed of a collection of interconnected nodes. Let's take a financial network as an
example. Suppose you want to transfer some money internationally. Your bank might do this by
transferring the currency through a number of intermediary banks before it gets to its
destination. This transaction describes a path on a payment network, where banks represent nodes
on the graph which are connected if a pair of banks can perform a transfer. One problem a bank
may look to tackle is to find suspicious payment patterns in the network.

Take a look below to see some of the graph-based problems that photonic quantum devices can solve!

.. customgalleryitem::
    :tooltip: Dense subgraphs
    :description: :doc:`Dense subgraph problems <../tutorials_apps/run_tutorial_dense>`
    :figure: ../_static/dense.png

.. customgalleryitem::
    :tooltip: Maximum clique
    :description: :doc:`Max clique problems <../tutorials_apps/run_tutorial_max_clique>`
    :figure: ../_static/max_clique.png

.. raw:: html

    <div style='clear:both'></div>

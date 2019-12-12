.. _graphs-intro:

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

Strawberry Fields provides tools for solving **graph** or **network-based** problems. Many
problems of practical relevance can be stated in terms of a graph. For example, graphs can be
used to model **financial markets**, **biological networks**, and **social networks**.

Graphs are composed of a collection of interconnected nodes. One task often carried out is to
search for **clusters** in a graph: regions with a high level of connectivity. These subgraphs may
correspond to **communities in social networks**, **correlated assets in a market**, or
**mutually influential proteins** in a biological network. Finding these clusters in large graphs
can be tough, but photonic quantum computers can help.

Take a look below to see some of the graph-based problems that photonic quantum devices can
help solve!

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

Graphs and networking
=====================

Graphs can be used to model a wide variety of concepts: social networks, financial markets,
biological networks, and many others. The applications layer of Strawberry Fields provides tools
for solving a range of graph-based problems.

Dense subgraphs
---------------

A common problem of interest is to find subgraphs that contain a large number of connections
between their nodes. These subgraphs may correspond to communities in social networks, correlated
assets in a market, or mutually influential proteins in a biological network.

.. customgalleryitem::
    :tooltip: Dense subgraphs
    :description: :doc:`../tutorials_apps/run_tutorial_dense`
    :figure: ../_static/dense.png


.. raw:: html

    <div style='clear:both'></div>


Maximum clique
--------------

A clique is a special type of subgraph where all possible connections between nodes are present;
they are densest possible subgraphs of their size. The maximum clique problem, or max clique for
short, asks the question: given a graph :math:`G`, what is the largest clique in the graph?

.. customgalleryitem::
    :tooltip: Maximum clique
    :description: :doc:`../tutorials_apps/run_tutorial_max_clique`
    :figure: ../_static/max_clique.png


.. raw:: html

    <div style='clear:both'></div>

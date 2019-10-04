# pylint: disable=wrong-import-position,wrong-import-order
r"""
Test
====
"""

import networkx as nx
import numpy as np
import plotly

from strawberryfields import gbs

adj = np.array(
    [
        [0, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 0],
    ]
)

graph = nx.Graph(adj)

g_plot = gbs.plot.plot_graph(graph)
plotly.offline.plot(g_plot, filename="file.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/file.html

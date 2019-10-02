# pylint: disable=wrong-import-position,wrong-import-order
r"""
Test
====
"""

import numpy as np
import networkx as nx

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

##############################################################################
# Plotting
# --------
#
# Let's do a quick plot of the graph to compare to the version shown above:

import matplotlib.pyplot as plt

# suppress an annoying warning message from NetworkX plotting
import warnings
import matplotlib

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

pos = nx.spring_layout(graph, seed=1)
nx.draw_networkx_nodes(graph, pos, node_color="#63AC9A")
nx.draw_networkx_edges(graph, pos, width=2, edge_color="#63AC9A")

# resize graph
l, r = plt.xlim()
plt.xlim(l - 0.35, r + 0.35)

plt.axis("off")

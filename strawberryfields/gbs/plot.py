# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Plotting and visualization
==========================

**Module name:** :mod:`strawberryfields.gbs.plot`

.. currentmodule:: strawberryfields.gbs.plot

This module provides functionality for visualizing graphs, subgraphs, and point processes. It
requires the installation of the plotly library, which is not a dependency of Strawberry
Fields. Plotly can be installed using 'pip install plotly' or by visiting their installation
instructions at https://plot.ly/python/getting-started/#installation. Graphs are plotted using
the Kamada-Kawai layout with an aspect ratio of 1:1. The module uses a custom Strawberry Fields
colour scheme. The standard scheme for graphs uses green nodes and grey edges, whereas the scheme
for subgraphs uses red nodes and edges.

.. autosummary::
    plot_graph
    plot_subgraph

Code details
^^^^^^^^^^^^
"""
from typing import Optional, Tuple

import networkx as nx


def _node_coords(graph: nx.Graph, l: dict) -> Tuple:
    """Converts coordinates for the graph nodes for plotting purposes.

    Args:
        graph (nx.Graph): input graph
        l (dict[int, float]): dictionary of nodes and their respective coordinates, can be
            generated using a NetworkX `layout <https://networkx.github.io/documentation/latest/
            reference/drawing.html#module-networkx.drawing.layout>`__

    Returns:
         dict[str, list]: lists of x and y coordinates accessed as keys of a dictionary
    """
    n_x = []
    n_y = []

    for n in graph.nodes():
        n_x.append(l[n][0])
        n_y.append(l[n][1])

    return {"x": n_x, "y": n_y}


def _edge_coords(graph: nx.Graph, l: dict) -> dict:
    """Converts coordinates for the graph edges for plotting purposes.

        Args:
            graph (nx.Graph): input graph
            l (dict): dictionary of nodes and their respective coordinates

        Returns:
             dict: x and y coordinates for beginning and end of each edge. `None` is placed as a
             separator between pairs of nodes/edges.
        """
    e_x = []
    e_y = []

    for e in graph.edges():

        start_x, start_y = l[e[0]]
        end_x, end_y = l[e[1]]

        e_x.append(start_x)
        e_x.append(end_x)

        e_y.append(start_y)
        e_y.append(end_y)

        e_x.append(None)
        e_y.append(None)

    return {"x": e_x, "y": e_y}


GREEN = "#3e9651"
RED = "#cc2529"
LIGHT_GREY = "#CDCDCD"

graph_node_colour = GREEN
graph_edge_colour = LIGHT_GREY
subgraph_node_colour = RED
subgraph_edge_colour = RED

graph_node_size = 14
subgraph_node_size = 16


def plot_graph(graph: nx.Graph, subgraph: Optional[list] = None,
               size: int = 500) -> None:  # pragma: no cover
    """Creates a plotly plot of the input graph.

    This function can plot just the input graph or the graph with a specified subgraph highlighted.

    **Example usage**:

    >>> graph = nx.complete_graph(10)
    >>> fig = plot_graph(graph, [0, 1, 2, 3])
    >>> fig.show()

    .. image:: ../../_static/complete_graph.png
       :width: 40%
       :align: center
       :target: javascript:void(0);

    Args:
        graph (nx.Graph): input graph
        subgraph (list): optional list of nodes comprising the subgraph to highlight
        size (int): size of the plot in pixels, rendered in a square layout

    Returns:
         Figure: Plotly figure for graph and optionally highlighted subgraph
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly required for using plot(). Can be installed using pip install "
                          "plotly or visiting https://plot.ly/python/getting-started/#installation")
    except RuntimeError:
        print("Plotly unable to open display")
        raise

    s = graph.subgraph(subgraph)
    l = nx.kamada_kawai_layout(graph)

    g_nodes = go.Scatter(
        **_node_coords(graph, l),
        mode="markers",
        hoverinfo="text",
        marker=dict(color=graph_node_colour, size=graph_node_size, line_width=2),
    )

    g_edges = go.Scatter(
        **_edge_coords(graph, l),
        line=dict(width=1, color=graph_edge_colour),
        hoverinfo="none",
        mode="lines",
    )

    g_nodes.text = [str(i) for i in graph.nodes()]

    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=0, l=0, r=0, t=25),
        height=size,
        width=size,
        plot_bgcolor="#ffffff",
    )

    if subgraph:

        s_edges = go.Scatter(
            **_edge_coords(s, l),
            line=dict(width=2, color=subgraph_edge_colour),
            hoverinfo="none",
            mode="lines",
        )

        s_nodes = go.Scatter(
            **_node_coords(s, l),
            mode="markers",
            hoverinfo="text",
            marker=dict(color=subgraph_node_colour, size=subgraph_node_size, line_width=2),
        )

        s_nodes.text = [str(i) for i in s.nodes()]

        f = go.Figure(data=[g_edges, s_edges, g_nodes, s_nodes], layout=layout)

    else:
        f = go.Figure(data=[g_edges, g_nodes], layout=layout)

    return f


def plot_subgraph(subgraph: nx.Graph, size: int = 500) -> None:  # pragma: no cover
    """Creates a plotly plot of the input subgraph.

    **Example usage**:

    >>> graph = nx.complete_graph(10)
    >>> subgraph = graph.subgraph([0, 1, 2, 3])
    >>> fig = plot_subgraph(subgraph)
    >>> fig.show()

    .. image:: ../../_static/complete_subgraph.png
       :width: 40%
       :align: center
       :target: javascript:void(0);

    Args:
        subgraph (nx.Graph): input subgraph
        size (int): size of the plot in pixels, rendered in a square layout

    Returns:
         Figure: Plotly figure for subgraph
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly required for using plot_subgraph()")
    except RuntimeError:
        print("Plotly unable to open display")
        raise

    l = nx.kamada_kawai_layout(subgraph)

    g_edges = go.Scatter(
        **_edge_coords(subgraph, l),
        line=dict(width=1.5, color=subgraph_edge_colour),
        hoverinfo="none",
        mode="lines",
    )

    g_nodes = go.Scatter(
        **_node_coords(subgraph, l),
        mode="markers",
        hoverinfo="text",
        marker=dict(color=subgraph_node_colour, size=graph_node_size, line_width=2),
    )

    g_nodes.text = [str(i) for i in subgraph.nodes()]

    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=0, l=0, r=0, t=25),
        height=size,
        width=size,
        plot_bgcolor="#ffffff",
    )

    f = go.Figure(data=[g_edges, g_nodes], layout=layout)

    return f

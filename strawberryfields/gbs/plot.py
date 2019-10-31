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
requires the installation of the Plotly library, which is not a dependency of Strawberry
Fields. Plotly can be installed using 'pip install plotly' or by visiting their installation
instructions at https://plot.ly/python/getting-started/#installation. Graphs are plotted using
the Kamada-Kawai layout with an aspect ratio of 1:1. The module uses a custom Strawberry Fields
colour scheme. The standard scheme for graphs uses green nodes and grey edges, the scheme for
subgraphs uses red nodes and edges, and the scheme for point processes colors points in light
grey, highlighting samples in red.

.. autosummary::
    graph
    subgraph
    points

Code details
^^^^^^^^^^^^
"""
# pylint: disable=import-outside-toplevel
from typing import Optional, Tuple

import networkx as nx
import numpy as np


def _node_coords(g: nx.Graph, l: dict) -> Tuple:
    """Converts coordinates for the graph nodes for plotting purposes.

    Args:
        g (nx.Graph): input graph
        l (dict[int, float]): Dictionary of nodes and their respective coordinates. Can be
            generated using a NetworkX `layout <https://networkx.github.io/documentation/latest/
            reference/drawing.html#module-networkx.drawing.layout>`__

    Returns:
         dict[str, list]: lists of x and y coordinates accessed as keys of a dictionary
    """
    n_x = []
    n_y = []

    for n in g.nodes():
        n_x.append(l[n][0])
        n_y.append(l[n][1])

    return {"x": n_x, "y": n_y}


def _edge_coords(g: nx.Graph, l: dict) -> dict:
    """Converts coordinates for the graph edges for plotting purposes.

        Args:
            g (nx.Graph): input graph
            l (dict[int, float]): Dictionary of nodes and their respective coordinates. Can be
                generated using a NetworkX `layout <https://networkx.github.io/documentation/latest/
                reference/drawing.html#module-networkx.drawing.layout>`__

        Returns:
             dict[str, list]: lists of x and y coordinates for the beginning and end of each edge.
             ``None`` is placed as a separator between pairs of nodes/edges.
        """
    e_x = []
    e_y = []

    for e in g.edges():

        start_x, start_y = l[e[0]]
        end_x, end_y = l[e[1]]

        e_x.append(start_x)
        e_x.append(end_x)

        e_y.append(start_y)
        e_y.append(end_y)

        e_x.append(None)
        e_y.append(None)

    return {"x": e_x, "y": e_y}


plotly_error = (
    "Plotly required for using this function. It can be installed using pip install "
    "plotly or visiting https://plot.ly/python/getting-started/#installation"
)

GREEN = "#3e9651"
RED = "#cc2529"
LIGHT_GREY = "#CDCDCD"
VERY_LIGHT_GREY = "#F2F2F2"

graph_node_colour = GREEN
graph_edge_colour = LIGHT_GREY
subgraph_node_colour = RED
subgraph_edge_colour = RED

graph_node_size = 14
subgraph_node_size = 16


def graph(g: nx.Graph, s: Optional[list] = None, plot_size: int = 500):  # pragma: no cover
    """Creates a plot of the input graph.

    This function can plot the input graph only, or the graph with a specified subgraph highlighted.

    **Example usage:**

    >>> graph = nx.complete_graph(10)
    >>> fig = graph(graph, [0, 1, 2, 3])
    >>> fig.show()

    .. image:: ../../_static/complete_graph.png
       :width: 40%
       :align: center
       :target: javascript:void(0);

    Args:
        g (nx.Graph): input graph
        s (list): optional list of nodes comprising the subgraph to highlight
        plot_size (int): size of the plot in pixels, rendered in a square layout

    Returns:
         Figure: figure for graph and optionally highlighted subgraph
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(plotly_error)

    l = nx.kamada_kawai_layout(g)

    g_nodes = go.Scatter(
        **_node_coords(g, l),
        mode="markers",
        hoverinfo="text",
        marker=dict(color=graph_node_colour, size=graph_node_size, line_width=2),
    )

    g_edges = go.Scatter(
        **_edge_coords(g, l),
        line=dict(width=1, color=graph_edge_colour),
        hoverinfo="none",
        mode="lines",
    )

    g_nodes.text = [str(i) for i in g.nodes()]

    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=0, l=0, r=0, t=25),
        height=plot_size,
        width=plot_size,
        plot_bgcolor="#ffffff",
    )

    if s is not None:
        s = g.subgraph(s)

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


def subgraph(s: nx.Graph, plot_size: int = 500):  # pragma: no cover
    """Creates a plot of the input subgraph.

    **Example usage:**

    >>> graph = nx.complete_graph(10)
    >>> subgraph = graph.subgraph([0, 1, 2, 3])
    >>> fig = subgraph(subgraph)
    >>> fig.show()

    .. image:: ../../_static/complete_subgraph.png
       :width: 40%
       :align: center
       :target: javascript:void(0);

    Args:
        s (nx.Graph): input subgraph
        plot_size (int): size of the plot in pixels, rendered in a square layout

    Returns:
         Figure: figure for subgraph
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(plotly_error)

    l = nx.kamada_kawai_layout(s)

    g_edges = go.Scatter(
        **_edge_coords(s, l),
        line=dict(width=1.5, color=subgraph_edge_colour),
        hoverinfo="none",
        mode="lines",
    )

    g_nodes = go.Scatter(
        **_node_coords(s, l),
        mode="markers",
        hoverinfo="text",
        marker=dict(color=subgraph_node_colour, size=graph_node_size, line_width=2),
    )

    g_nodes.text = [str(i) for i in s.nodes()]

    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=0, l=0, r=0, t=25),
        height=plot_size,
        width=plot_size,
        plot_bgcolor="#ffffff",
    )

    f = go.Figure(data=[g_edges, g_nodes], layout=layout)

    return f


def points(
    R: np.ndarray, sample: Optional[list] = None, plot_size: int = 500, point_size: float = 30
):  # pragma: no cover
    """Creates a plot of two-dimensional points given their input coordinates. Sampled
    points can be optionally highlighted among all points.

    **Example usage:**

    >>> R = np.random.normal(0, 1, (50, 2))
    >>> sample = [1] * 10 + [0] * 40  # select first ten points
    >>> points(R, sample).show()

    .. image:: ../../_static/normal_pp.png
       :width: 40%
       :align: center
       :target: javascript:void(0);

    Args:
        R (np.array): Coordinate matrix. Rows of this array are the coordinates of the points.
        sample (list[int]): optional subset of sampled points to be highlighted
        plot_size (int): size of the plot in pixels, rendered in a square layout
        point_size (int): size of the points, proportional to its radius

    Returns:
         Figure: figure of points with optionally highlighted sample
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(plotly_error)

    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=0, l=0, r=0, t=25),
        height=plot_size,
        width=plot_size,
        plot_bgcolor="white",
    )

    p = go.Scatter(
        x=R[:, 0],
        y=R[:, 1],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color=VERY_LIGHT_GREY, size=point_size, line=dict(color="black", width=point_size / 20)
        ),
    )

    p.text = [str(i) for i in range(len(R))]

    if sample:
        s_x = []
        s_y = []
        sampled_points = [i for i in range(len(sample)) if sample[i] > 0]
        for i in sampled_points:
            s_x.append(R[i, 0])
            s_y.append(R[i, 1])

        samp = go.Scatter(
            x=s_x,
            y=s_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                color=RED, size=point_size, line=dict(color="black", width=point_size / 20)
            ),
        )

        samp.text = [str(i) for i in sampled_points]

        f = go.Figure(data=[p, samp], layout=layout)

    else:
        f = go.Figure(data=[p], layout=layout)

    return f

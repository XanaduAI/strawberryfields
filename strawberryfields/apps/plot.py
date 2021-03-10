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
Tools for visualizing graphs, subgraphs, point processes, and vibronic spectra.

Visualization requires installation of the Plotly library, which is not a dependency of
Strawberry Fields. Plotly can be installed using ``pip install plotly`` or by visiting their
`installation instructions <https://plot.ly/python/getting-started/#installation>`__.
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
GREY = "#737373"
LIGHT_GREY = "#CDCDCD"
VERY_LIGHT_GREY = "#F2F2F2"

graph_node_colour = GREEN
graph_edge_colour = LIGHT_GREY
subgraph_node_colour = RED
subgraph_edge_colour = RED

graph_node_size = 14
subgraph_node_size = 16


def graph(g: nx.Graph, s: Optional[list] = None, plot_size: Tuple = (500, 500)):  # pragma: no cover
    """Creates a plot of the input graph.

    This function can plot the input graph only, or the graph with a specified subgraph highlighted.
    Graphs are plotted using the Kamada-Kawai layout with an aspect ratio of 1:1.

    **Example usage:**

    >>> graph = nx.complete_graph(10)
    >>> fig = plot.graph(graph, [0, 1, 2, 3])
    >>> fig.show()

    .. image:: ../../_static/complete_graph.png
       :width: 40%
       :align: center
       :target: javascript:void(0);

    Args:
        g (nx.Graph): input graph
        s (list): optional list of nodes comprising the subgraph to highlight
        plot_size (int): size of the plot in pixels, given as a pair of integers ``(x_size,
            y_size)``

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
        height=plot_size[1],
        width=plot_size[0],
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


def subgraph(s: nx.Graph, plot_size: Tuple = (500, 500)):  # pragma: no cover
    """Creates a plot of the input subgraph.

    Subgraphs are plotted using the Kamada-Kawai layout with an aspect ratio of 1:1.

    **Example usage:**

    >>> graph = nx.complete_graph(10)
    >>> subgraph = graph.subgraph([0, 1, 2, 3])
    >>> fig = plot.subgraph(subgraph)
    >>> fig.show()

    .. image:: ../../_static/complete_subgraph.png
       :width: 40%
       :align: center
       :target: javascript:void(0);

    Args:
        s (nx.Graph): input subgraph
        plot_size (int): size of the plot in pixels, given as a pair of integers ``(x_size,
            y_size)``

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
        height=plot_size[1],
        width=plot_size[0],
        plot_bgcolor="#ffffff",
    )

    f = go.Figure(data=[g_edges, g_nodes], layout=layout)

    return f


def points(
    R: np.ndarray,
    sample: Optional[list] = None,
    plot_size: Tuple = (500, 500),
    point_size: float = 30,
):  # pragma: no cover
    """Creates a plot of two-dimensional points given their input coordinates. Sampled
    points can be optionally highlighted among all points.

    **Example usage:**

    >>> R = np.random.normal(0, 1, (50, 2))
    >>> sample = [1] * 10 + [0] * 40  # select first ten points
    >>> plot.points(R, sample).show()

    .. image:: ../../_static/normal_pp.png
       :width: 40%
       :align: center
       :target: javascript:void(0);

    Args:
        R (np.array): Coordinate matrix. Rows of this array are the coordinates of the points.
        sample (list[int]): optional subset of sampled points to be highlighted
        plot_size (int): size of the plot in pixels, given as a pair of integers ``(x_size,
            y_size)``
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
        height=plot_size[1],
        width=plot_size[0],
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


def spectrum(
    energies: list, gamma: float = 100.0, xmin: float = None, xmax: float = None
):  # pragma: no cover
    """Plots a vibronic spectrum based on input sampled energies.

    **Example usage:**

    >>> formic = data.Formic()
    >>> e = qchem.vibronic.energies(formic, formic.w, formic.wp)
    >>> full_spectrum = plot.spectrum(e, xmin=-1000, xmax=8000)
    >>> full_spectrum.show()

    .. image:: ../../_static/formic_spectrum.png
       :width: 50%
       :align: center
       :target: javascript:void(0);

    Args:
        energies (list[float]): a list of sampled energies
        gamma (float): parameter specifying the width of the Lorentzian function
        xmin (float): minimum limit of the x axis
        xmax (float): maximum limit of the x axis

    Returns:
         Figure: spectrum in the form of a histogram of energies with a Lorentzian-like curve
    """
    if len(energies) < 2:
        raise ValueError("Number of sampled energies must be at least two")
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(plotly_error)

    emin = min(energies)
    emax = max(energies)
    if xmin is None:
        xmin = emin - 0.1 * (emax - emin)
    if xmax is None:
        xmax = emax + 0.1 * (emax - emin)
    bins = int(emax - emin) // 5
    bar_width = (xmax - xmin) * 0.005
    line_width = 3.0

    h = np.histogram(energies, bins)
    X = np.linspace(xmin, xmax, int(xmax - xmin))
    L = 0
    for e in energies:
        L += (gamma / 2) ** 2 / ((X - e) ** 2 + (gamma / 2) ** 2)

    text_font = dict(color="black", family="Computer Modern")

    axis_style = dict(
        titlefont_size=30,
        tickfont=text_font,
        tickfont_size=20,
        showline=True,
        linecolor="black",
        mirror=True,
    )

    layout = go.Layout(
        yaxis=dict(title={"text": "Counts", "font": text_font}, **axis_style, rangemode="tozero"),
        xaxis=dict(
            title={"text": "Energy (cm<sup>-1</sup>)", "font": text_font},
            **axis_style,
            range=[xmin, xmax],
        ),
        plot_bgcolor="white",
        margin=dict(t=25),
        bargap=0.04,
        showlegend=False,
    )

    bars = go.Bar(x=h[1].tolist(), y=h[0].tolist(), width=bar_width, marker=dict(color=GREY))

    line = go.Scatter(x=X, y=L, mode="lines", line=dict(color=GREEN, width=line_width))

    f = go.Figure([bars, line], layout=layout)

    return f

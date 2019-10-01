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


def plot_points(R, sample, size: int = 500):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly required for using plot(). Can be installed using pip install "
                          "plotly or visiting https://plot.ly/python/getting-started/#installation")
    except RuntimeError:
        print("Plotly unable to open display")
        raise

    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=0, l=0, r=0, t=25),
        height=size,
        width=size,
        plot_bgcolor="white",
    )

    points = go.Scatter(x=R[:, 0], y=R[:, 1],
                        mode="markers",
                        hoverinfo="text",
                        marker=dict(color='white',size=30, line=dict(color='black', width=1)))

    points.text = [str(i) for i in range(len(R))]

    s_x = []
    s_y = []
    s_comp = [i for i in range(len(sample)) if sample[i] > 0]
    for i in s_comp:
        s_x.append(R[i, 0])
        s_y.append(R[i, 1])

    samp = go.Scatter(x=s_x, y=s_y,
                      mode="markers",
                      hoverinfo="text",
                      marker=dict(color='#cc2529', size=30, line=dict(color='black', width=1)))

    samp.text = [str(i) for i in s_comp]

    f = go.Figure(data=[points, samp], layout=layout)

    return f


def twod(n=10, m=10):
    R = []
    for i in range(0, n):
        for j in range(0, m):
            R.append((i, j))

    return np.array(R)


import numpy as np
d = 20
R = np.random.normal(0, 10, (d**2, 2))
# R = twod(d, d)
sample = np.random.randint(0, 2, d**2)
print(sample)
plot_points(R, sample, size=1500).show()




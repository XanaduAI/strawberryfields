# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module provides tools to visualize the state in various interactive
ways using Plot.ly
"""
from copy import deepcopy

import numpy as np
import plotly.io as pio


def plot_wigner(state, renderer="browser", wire=0, dq=0.1):
    """Plot the Wigner function with Plot.ly

    Args:
        state (:class:`.BaseState`): the state used for plotting
        renderer (string): the renderer for plotting with Plot.ly
        wire (int): wire used to calculate the reduced Wigner function
        dq (float): stepsize along the q and p axes
    """
    q = np.arange(-state.cutoff_dim, state.cutoff_dim, dq)
    pio.renderers.default = renderer

    data = state.wigner(wire, q, q)
    new_chart = generate_wigner_chart(data, q)
    pio.show(new_chart)


def generate_wigner_chart(data, qp):
    """Populates a chart dictionary with reduced Wigner function surface plot data.

    Args:
        qp (tuple[float]): minimum and maximum values of the q and p axes

    Returns:
        dict: a Plot.ly JSON-format surface plot
    """
    chart = {
        "data": [{"type": "scatter"}],
        "layout": {
            "scene":{
                "xaxis": {},
                "yaxis": {},
                "zaxis": {}
            }
        }
    }
    chart["data"] = deepcopy(surfaceplotDefault["data"])

    chart["data"][0]["x"] = qp.tolist()
    chart["data"][0]["y"] = qp.tolist()
    chart["data"][0]["z"] = data.tolist()

    chart["data"][0]["cmin"] = -1 / np.pi
    chart["data"][0]["cmax"] = 1 / np.pi

    chart["layout"]["paper_bgcolor"] = "white"
    chart["layout"]["plot_bgcolor"] = "white"
    chart["layout"]["scene"]["bgcolor"] = "white"
    chart["layout"]["font"] = {"color": textcolor}
    chart["layout"]["font"] = {"color": textcolor}

    chart["layout"]["scene"]["xaxis"]["title"] = "x"
    chart["layout"]["scene"]["xaxis"]["color"] = textcolor
    chart["layout"]["scene"]["yaxis"]["title"] = "p"
    chart["layout"]["scene"]["yaxis"]["color"] = textcolor
    chart["layout"]["scene"]["yaxis"]["gridcolor"] = textcolor
    chart["layout"]["scene"]["zaxis"]["title"] = "W(x,p)"

    return chart

textcolor = "#787878"

# Plot.ly default surface plot JSON
surfaceplotDefault = {
    "data": [{
        "cmin": -1 / np.pi,
        "cmax": 1 / np.pi,
        "contours": {"z": {"show": True}},
        "type": "surface",
        "x": [],
        "y": [],
        "z": [],
        "colorscale": [
            [0.0, "rgb(31,144,148)"],
            [0.5, "rgb(255,255,255)"],
            [1.0, "rgb(31,89,154)"]
        ],
    }],
    "layout": {
        "width": 835,
        "height": 500,
        "margin": {
            "l": 5,
            "r": 5,
            "b": 5,
            "t": 5,
            "pad": 4
        },
        "paper_bgcolor": "transparent",
        "plot_bgcolor": "transparent",
        "scene": {
            "bgcolor": "transparent",
            "xaxis": {
                "gridcolor": textcolor,
                "gridwidth": 2,
            },
            "yaxis": {
                "gridcolor": textcolor,
                "gridwidth": 2,
            },
            "zaxis": {
                "autorange": True,
                "gridcolor": textcolor,
                "gridwidth": 2,
            }
        }
    },
    "config": {
        "modeBarButtonsToRemove": ["zoom2d", "lasso2d", "select2d", "toggleSpikelines"],
        "displaylogo": False
    }
}

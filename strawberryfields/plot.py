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
ways using Plot.ly.
"""
import numpy as np

plotly_error = (
    "Plot.ly required for using this function. It can be installed as follows:"
    "pip install plotly."
)


def _get_plotly():
    """Import Plot.ly on demand to avoid errors being raised unnecessarily."""
    try:
        # pylint:disable=import-outside-toplevel
        import plotly.io as pio
    except ImportError as e:
        raise (plotly_error) from e
    return pio


def plot_wigner(state, mode, xvec, pvec, renderer="browser", contours=True):
    """Plot the Wigner function with Plot.ly.

    Args:
        state (:class:`.BaseState`): the state used for plotting
        mode (int): mode used to calculate the reduced Wigner function
        xvec (array): array of discretized :math:`x` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values
        renderer (string): the renderer for plotting with Plot.ly
        contours (bool): whether to show the contour lines in the plot
    """
    pio = _get_plotly()
    pio.renderers.default = renderer

    data = state.wigner(mode, xvec, pvec)
    new_chart = generate_wigner_chart(data, xvec, pvec, contours=contours)
    pio.show(new_chart)


def generate_wigner_chart(data, xvec, pvec, contours=True):
    """Populates a chart dictionary with reduced Wigner function surface plot data.

    Args:
        data (array): 2D array of size [len(xvec), len(pvec)], containing reduced
            Wigner function values for specified x and p values.
        xvec (array): array of discretized :math:`x` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values
        contours (bool): whether to show the contour lines in the plot

    Returns:
        dict: a Plot.ly JSON-format surface plot
    """
    textcolor = "#787878"

    chart = {
        "data": [
            {
                "type": "surface",
                "colorscale": [],
                "x": [],
                "y": [],
                "z": [],
                "contours": {
                    "z": {},
                },
            }
        ],
        "layout": {
            "scene": {
                "xaxis": {},
                "yaxis": {},
                "zaxis": {},
            }
        },
    }

    chart["data"][0]["type"] = "surface"
    chart["data"][0]["colorscale"] = [
        [0.0, "purple"],
        [0.25, "red"],
        [0.5, "yellow"],
        [0.75, "green"],
        [1.0, "blue"],
    ]

    chart["data"][0]["x"] = xvec.tolist()
    chart["data"][0]["y"] = pvec.tolist()
    chart["data"][0]["z"] = data.tolist()

    chart["data"][0]["contours"]["z"]["show"] = contours

    chart["data"][0]["cmin"] = -1 / np.pi
    chart["data"][0]["cmax"] = 1 / np.pi

    chart["layout"]["paper_bgcolor"] = "white"
    chart["layout"]["plot_bgcolor"] = "white"
    chart["layout"]["font"] = {"color": textcolor}
    chart["layout"]["scene"]["bgcolor"] = "white"

    chart["layout"]["scene"]["xaxis"]["title"] = "x"
    chart["layout"]["scene"]["xaxis"]["color"] = textcolor
    chart["layout"]["scene"]["yaxis"]["title"] = "p"
    chart["layout"]["scene"]["yaxis"]["color"] = textcolor
    chart["layout"]["scene"]["yaxis"]["gridcolor"] = textcolor
    chart["layout"]["scene"]["zaxis"]["title"] = "W(x,p)"

    return chart

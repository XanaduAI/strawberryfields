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
from copy import copy, deepcopy
import numpy as np
from strawberryfields.backends.states import BaseGaussianState

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


textcolor = "#787878"


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

    new_chart = generate_wigner_chart(state, mode, xvec, pvec, contours=contours)
    pio.show(new_chart)


def generate_wigner_chart(state, mode, xvec, pvec, contours=True):
    """Populates a chart dictionary with reduced Wigner function surface plot data.

    Args:
        state (:class:`.BaseState`): the state used for plotting
        mode (int): mode used to calculate the reduced Wigner function
        xvec (array): array of discretized :math:`x` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values
        contours (bool): whether to show the contour lines in the plot

    Returns:
        dict: a Plot.ly JSON-format surface plot
    """
    data = state.wigner(mode, xvec, pvec)
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


# Plot.ly default barchart JSON
barchart_default = {
    "data": [{"y": [], "x": [], "type": "bar", "name": "q[0]"}],
    "layout": {
        "width": 835,
        "height": 500,
        "margin": {"l": 100, "r": 100, "b": 100, "t": 100, "pad": 4},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "autosize": True,
        "yaxis": {
            "gridcolor": "#bbb",
            "type": "linear",
            "autorange": True,
            "title": "Probability",
            "fixedrange": True,
        },
        "xaxis": {
            "gridcolor": textcolor,
            "type": "category",
            "autorange": True,
            "title": "q[0]",
            "fixedrange": True,
        },
        "showlegend": False,
        "annotations": [
            {
                "showarrow": False,
                "yanchor": "bottom",
                "xref": "paper",
                "xanchor": "center",
                "yref": "paper",
                "text": "",
                "y": 1,
                "x": 0,
                "font": {"size": 16},
            }
        ],
    },
    "config": {
        "modeBarButtonsToRemove": ["zoom2d", "lasso2d", "select2d", "toggleSpikelines"],
        "displaylogo": False,
    },
}


def plot_fock(state, modes, cutoff=None, renderer="browser"):
    """Plot the (marginal) Fock state probabilities with Plot.ly.

    Args:
        state (:class:`.BaseState`): the state used for plotting
        modes (list): List of modes to generate output for. If more than one mode
            is provided, then the reduced state is computed for each mode, and marginal
            Fock probabilities plotted per mode.
        cutoff (int): The cutoff value determining the maximum Fock state to
            get probabilities for. Only required if state is a
            :class:`~.BaseGaussianState`.
        renderer (string): the renderer for plotting with Plot.ly
    """
    if cutoff is None:
        if isinstance(state, BaseGaussianState):
            raise ValueError(f"No cutoff specified for {state.__class__}.")

        # BaseFockState has cutoff
        cutoff = state.cutoff_dim

    pio = _get_plotly()
    pio.renderers.default = renderer

    new_chart = generate_fock_chart(state, modes, cutoff)
    pio.show(new_chart)


def generate_fock_chart(state, modes, cutoff):
    """Populates a chart dictionary with marginal Fock state probability
    distributions for multiple modes.

    Args:
        state (:class:`.BaseState`): the state used for plotting
        modes (list): list of modes to generate Fock charts for
        cutoff (int): The cutoff value determining the maximum Fock state to
            get probabilities for. Only required if state is a
            :class:`~.BaseGaussianState`.

    Returns:
        dict: a Plot.ly JSON-format bar chart
    """
    num_modes = len(modes)

    # Reduced density matrices
    rho = [state.reduced_dm(n, cutoff=cutoff) for n in range(num_modes)]
    photon_dists = np.array([np.real(np.diagonal(p)) for p in rho])

    n = np.arange(cutoff)
    mean = [np.sum(n * probs).real for probs in photon_dists]

    xlabels = [fr"$|{i}\rangle$" for i in range(0, cutoff, 1)]

    num_modes = len(modes)
    chart = deepcopy(barchart_default)
    chart["data"] = [dict() for i in range(num_modes)]

    for idx, n in enumerate(sorted(modes)):
        chart["data"][idx]["type"] = "bar"
        chart["data"][idx]["marker"] = {"color": "#1f9094"}
        chart["data"][idx]["x"] = xlabels
        chart["data"][idx]["y"] = photon_dists[n].tolist()

        if idx == 0:
            Xax = ("xaxis", "x")
        else:
            chart["layout"]["annotations"].append(copy(chart["layout"]["annotations"][idx - 1]))
            Xax = ("xaxis{}".format(idx + 1), "x{}".format(idx + 1))

        chart["data"][idx]["xaxis"] = Xax[1]
        chart["data"][idx]["yaxis"] = "y"
        chart["data"][idx]["name"] = ""

        dXa = 0.01 if idx != 0 else 0
        dXb = 0.01 if idx != num_modes - 1 else 0

        chart["layout"][Xax[0]] = {
            "type": "category",
            "domain": [idx / num_modes + dXa, (idx + 1) / num_modes - dXb],
            "title": "mode {}".format(n),
            "fixedrange": True,
            "gridcolor": "rgba(0,0,0,0)",
        }
        val_mean_photon = str(np.round(mean[n], 3))
        expr_mean_photon = r"$\langle \hat{n} \rangle=" + val_mean_photon + "$"
        chart["layout"]["annotations"][idx]["text"] = expr_mean_photon
        chart["layout"]["annotations"][idx]["x"] = idx / num_modes + dXa + 0.5 / num_modes

    chart["layout"]["xaxis"]["type"] = "category"
    chart["layout"]["title"] = "Marginal Fock state probabilities"

    chart["layout"]["font"] = {"color": textcolor}

    return chart


# Plot.ly default linechart JSON
linechart_default = {
    "data": [{"type": "scatter", "x": [], "y": []}],
    "layout": {
        "width": 835,
        "height": 500,
        "margin": {"l": 100, "r": 100, "b": 100, "t": 100, "pad": 4},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "xaxis": {
            "gridcolor": textcolor,
            "autorange": True,
            "title": "Quadrature value",
        },
        "yaxis": {
            "gridcolor": textcolor,
            "autorange": True,
            "title": "Probability",
        },
    },
    "config": {
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"],
        "displaylogo": False,
    },
}


def plot_quad(state, modes, xvec, pvec, renderer="browser"):
    """Plot the x and p-quadrature probability distributions with Plot.ly.

    Args:
        state (:class:`.BaseState`): the state used for plotting
        mode (int): mode used to calculate the reduced Wigner function
        xvec (array): array of discretized :math:`x` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values
        renderer (string): the renderer for plotting with Plot.ly
    """
    pio = _get_plotly()
    pio.renderers.default = renderer

    for mode in modes:
        new_chart = generate_quad_chart(state, mode, xvec, pvec)
        pio.show(new_chart)


def generate_quad_chart(state, mode, xvec, pvec):
    """Populates a chart dictionary with x and p reduced quadrature
    probabilities for a single mode.

    Args:
        state (:class:`.BaseState`): the state used for plotting
        mode (int): the mode for which quadrature probabilities are obtained
        xvec (array): array of discretized :math:`x` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values

    Returns:
        dict: a Plot.ly JSON-format line plot
    """
    p_probs = state.p_quad_values(mode, xvec, pvec).tolist()
    x_probs = state.x_quad_values(mode, xvec, pvec).tolist()

    chart = deepcopy(linechart_default)
    data_dict = linechart_default["data"][0]
    chart["data"] = [deepcopy(data_dict), deepcopy(data_dict)]

    chart["data"][0]["x"] = xvec.tolist()
    chart["data"][0]["y"] = x_probs
    chart["data"][0]["mode"] = "lines"
    chart["data"][0]["type"] = "scatter"
    chart["data"][0]["name"] = "x"
    chart["data"][0]["line"] = {"color": "#1f9094"}

    chart["data"][1]["x"] = pvec.tolist()
    chart["data"][1]["y"] = p_probs
    chart["data"][1]["mode"] = "lines"
    chart["data"][1]["type"] = "scatter"
    chart["data"][1]["name"] = "p"
    chart["data"][1]["line"] = {"color": "#1F599A"}

    chart["layout"]["paper_bgcolor"] = "white"
    chart["layout"]["plot_bgcolor"] = "white"
    chart["layout"]["font"] = {"color": textcolor}

    for i in ["xaxis", "yaxis"]:
        chart["layout"][i]["gridcolor"] = "#bbb"
        chart["layout"][i]["color"] = textcolor

    chart["layout"]["title"] = f"Position and momentum quadrature probabilities for mode {mode}"
    return chart

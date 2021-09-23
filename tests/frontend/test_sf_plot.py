# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for strawberryfields.plot
"""
import numpy as np
from copy import deepcopy
import plotly.io as pio
import pytest
from scipy.special import factorial as fac

import strawberryfields as sf
from strawberryfields.ops import BSgate, MeasureFock, Sgate, Vac
from strawberryfields.plot import (
    plot_wigner,
    generate_fock_chart,
    generate_quad_chart,
    barchart_default,
)

pytestmark = pytest.mark.frontend


@pytest.fixture(scope="module")
def prog():
    """Program used for testing"""
    program = sf.Program(2)

    with program.context as q:
        Sgate(0.54, 0) | q[0]
        Sgate(0.54, 0) | q[1]
        BSgate(6.283, 0.6283) | (q[0], q[1])
        MeasureFock() | q

    return program


class TestWignerPlotting:
    """Test the Wigner plotting function"""

    @pytest.mark.parametrize("renderer", ["png", "json", "browser"])
    @pytest.mark.parametrize("mode", [0, 1])
    @pytest.mark.parametrize("contours", [True, False])
    def test_no_errors(self, mode, renderer, contours, prog, monkeypatch):
        """Test that no errors are thrown when calling the `plot_wigner` function"""
        eng = sf.Engine("gaussian")
        results = eng.run(prog)

        xvec = np.arange(-4, 4, 0.1)
        pvec = np.arange(-4, 4, 0.1)
        with monkeypatch.context() as m:
            m.setattr(pio, "show", lambda x: None)
            plot_wigner(results.state, mode, xvec, pvec, renderer=renderer, contours=contours)


def create_example_fock_chart(state, modes, cutoff):
    """Test chart used to plot the Fock state probabilities of a two-mode
    system."""
    all_probs = state.all_fock_probs()
    photon_dists_0 = np.sum(all_probs, 1)
    photon_dists_1 = np.sum(all_probs, 0)
    mean = [state.mean_photon(0)[0], state.mean_photon(1)[0]]

    chart = deepcopy(barchart_default)
    first_x = 0.25
    second_x = 0.76
    domain = [0.0, 0.49]
    domain2 = [0.51, 1.0]
    xlabels = [fr"$|{i}\rangle$" for i in range(0, cutoff, 1)]

    # mean photon as latex
    mean_photon_text1 = "$\\langle \\hat{n} \\rangle=" + f"{mean[0]}" + "$"
    mean_photon_text2 = "$\\langle \\hat{n} \\rangle=" + f"{mean[1]}" + "$"
    data = [
        {
            "type": "bar",
            "marker": {"color": "#1f9094"},
            "x": xlabels,
            "y": photon_dists_0.tolist(),
            "xaxis": "x",
            "yaxis": "y",
            "name": "",
        },
        {
            "type": "bar",
            "marker": {"color": "#1f9094"},
            "x": xlabels,
            "y": photon_dists_1.tolist(),
            "xaxis": "x2",
            "yaxis": "y",
            "name": "",
        },
    ]
    yaxis = {
        "gridcolor": "#bbb",
        "type": "linear",
        "autorange": True,
        "title": "Probability",
        "fixedrange": True,
    }
    xaxis = {
        "type": "category",
        "domain": domain,
        "title": "mode 0",
        "fixedrange": True,
        "gridcolor": "rgba(0,0,0,0)",
    }
    xaxis2 = {
        "type": "category",
        "domain": domain2,
        "title": "mode 1",
        "fixedrange": True,
        "gridcolor": "rgba(0,0,0,0)",
    }
    annotations = [
        {
            "showarrow": False,
            "yanchor": "bottom",
            "xref": "paper",
            "xanchor": "center",
            "yref": "paper",
            "text": mean_photon_text1,
            "y": 1,
            "x": first_x,
            "font": {"size": 16},
        },
        {
            "showarrow": False,
            "yanchor": "bottom",
            "xref": "paper",
            "xanchor": "center",
            "yref": "paper",
            "text": mean_photon_text2,
            "y": 1,
            "x": second_x,
            "font": {"size": 16},
        },
    ]
    layout = {
        "width": 835,
        "height": 500,
        "margin": {"l": 100, "r": 100, "b": 100, "t": 100, "pad": 4},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "autosize": True,
        "yaxis": yaxis,
        "xaxis": xaxis,
        "xaxis2": xaxis2,
        "showlegend": False,
        "annotations": annotations,
        "title": "Marginal Fock state probabilities",
        "font": {"color": "#787878"},
    }
    fock_chart = {
        "data": data,
        "layout": layout,
        "config": {
            "modeBarButtonsToRemove": ["zoom2d", "lasso2d", "select2d", "toggleSpikelines"],
            "displaylogo": False,
        },
    }
    return fock_chart


def create_example_quad_chart(state, mode, xvec, pvec):
    """Test chart used to plot the quadrature probabilities of a one-mode
    system."""
    xlist = xvec.tolist()
    plist = pvec.tolist()
    x_probs = state.x_quad_values(mode, xvec, pvec).tolist()
    p_probs = state.p_quad_values(mode, xvec, pvec).tolist()
    data = [
        {
            "type": "scatter",
            "x": xlist,
            "y": x_probs,
            "mode": "lines",
            "name": "x",
            "line": {"color": "#1f9094"},
        },
        {
            "type": "scatter",
            "x": plist,
            "y": p_probs,
            "mode": "lines",
            "name": "p",
            "line": {"color": "#1F599A"},
        },
    ]
    layout = {
        "width": 835,
        "height": 500,
        "margin": {"l": 100, "r": 100, "b": 100, "t": 100, "pad": 4},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "xaxis": {
            "gridcolor": "#bbb",
            "autorange": True,
            "title": "Quadrature value",
            "color": "#787878",
        },
        "yaxis": {
            "gridcolor": "#bbb",
            "autorange": True,
            "title": "Probability",
            "color": "#787878",
        },
        "font": {"color": "#787878"},
        "title": f"Position and momentum quadrature probabilities for mode {mode}",
    }
    config = {
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"],
        "displaylogo": False,
    }
    quad_chart = {"data": data, "layout": layout, "config": config}
    return quad_chart


def get_example_state(num_subsystems, cutoff):
    """Creates an example state used in tests."""
    prog = sf.Program(num_subsystems)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    backend = eng.backend

    a = 0.3 + 0.1j

    backend.begin_circuit(
        num_subsystems=num_subsystems,
        cutoff_dim=cutoff,
        pure=True,
        batch_size=1,
    )
    backend.prepare_coherent_state(np.abs(a), np.angle(a), 0)
    return backend.state()


class TestFockProbPlotting:
    """Test the Fock state probabilities plotting function."""

    def test_raises_gaussian_no_cutoff(self, monkeypatch):
        """Test that an error is raised if not cutoff value is specified for a
        Gaussian state."""
        prog = sf.Program(1)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            Sgate(2) | q[0]

        state = eng.run(prog).state
        modes = [0]

        with monkeypatch.context() as m:
            # Avoid plotting even if the test failed
            m.setattr(pio, "show", lambda x: None)
            with pytest.raises(ValueError, match="No cutoff specified for"):
                sf.plot_fock(state, modes, renderer="browser")

    def test_expected_args_to_plot_single_mode(self, monkeypatch):
        """Test that given a program the expected values would be plotted with
        plot_fock."""
        modes = [0]
        num_subsystems = len(modes)

        cutoff = 5
        state = get_example_state(num_subsystems, cutoff)

        mean = state.mean_photon(modes[0])[0]
        all_probs = state.all_fock_probs()

        photon_dists = all_probs

        arg_store = []
        with monkeypatch.context() as m:
            m.setattr(pio, "show", lambda x: None)
            m.setattr(sf.plot, "generate_fock_chart", lambda *args: arg_store.append(args))

            sf.plot_fock(state, modes, renderer="browser")
            assert len(arg_store) == 1

            # Extract the stored args
            state_res, modes_res, cutoff_res = arg_store[0]

            # Check the results to be plotted
            mean_res = state_res.mean_photon(modes_res[0])[0]
            photon_dists_res = state_res.all_fock_probs()
            assert np.allclose(mean_res, mean)
            assert np.allclose(photon_dists_res, photon_dists)
            assert modes_res == modes
            assert cutoff_res == cutoff

    def test_expected_args_to_plot_two_modes(self, monkeypatch):
        """Test that given a program the expected values would be plotted for
        two modes."""
        modes = [0, 1]
        num_subsystems = len(modes)

        cutoff = 5
        state = get_example_state(num_subsystems, cutoff)

        mean = [state.mean_photon(mode)[0] for mode in modes]
        all_probs = state.all_fock_probs()

        photon_dists = [np.sum(all_probs, i).tolist() for i in range(num_subsystems - 1, -1, -1)]

        arg_store = []
        with monkeypatch.context() as m:
            m.setattr(pio, "show", lambda x: None)
            m.setattr(sf.plot, "generate_fock_chart", lambda *args: arg_store.append(args))
            sf.plot_fock(state, modes, renderer="browser")

            assert len(arg_store) == 1

            # Extract the stored args
            state_res, modes_res, cutoff_res = arg_store[0]

            mean_res = [state_res.mean_photon(mode)[0] for mode in modes]
            all_probs_res = state_res.all_fock_probs()
            photon_dists_res = [
                np.sum(all_probs_res, mode).tolist() for mode in reversed(modes_res)
            ]

            # Check the results to be plotted
            assert np.allclose(mean_res, mean)
            assert np.allclose(photon_dists_res, photon_dists)
            assert modes_res == modes
            assert cutoff_res == cutoff

    def test_generate_fock_chart(self):
        """Test the chart generated for a two-mode system when plotting the
        Fock state probabilities."""
        prog = sf.Program(2)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})

        with prog.context as q:
            Vac | q[0]
            Vac | q[1]

        state = eng.run(prog).state

        modes = [0, 1]
        res_chart = generate_fock_chart(state, modes, state.cutoff_dim)
        exp_chat = create_example_fock_chart(state, modes, state.cutoff_dim)
        assert res_chart["data"] == exp_chat["data"]
        assert res_chart["layout"] == exp_chat["layout"]
        assert res_chart["config"] == exp_chat["config"]


class TestQuadProbPlotting:
    """Test the quadrature probabilities plotting function."""

    @pytest.mark.parametrize("renderer", ["png", "json", "browser"])
    @pytest.mark.parametrize("modes", [[0], [0, 1]])
    def test_no_errors(self, modes, renderer, prog, monkeypatch):
        """Test that no errors are thrown when calling the `plot_quad`
        function."""
        eng = sf.Engine("gaussian")
        results = eng.run(prog)

        xvec = np.arange(-4, 4, 0.1)
        pvec = np.arange(-4, 4, 0.1)
        with monkeypatch.context() as m:
            m.setattr(pio, "show", lambda x: None)
            sf.plot_quad(results.state, modes, xvec, pvec, renderer=renderer)

    @pytest.mark.parametrize("mode", [0, 1])
    def test_generate_quad_chart(self, mode):
        """Test the chart generated for a single mode of a two-mode system when
        plotting the quadrature probabilities."""
        num_subsystems = 2
        cutoff = 5
        state = get_example_state(num_subsystems, cutoff)
        xvec = np.arange(0, 2, 1)
        pvec = np.arange(0, 2, 1)
        res_chart = generate_quad_chart(state, mode, xvec, pvec)
        exp_chat = create_example_quad_chart(state, mode, xvec, pvec)
        assert res_chart["data"] == exp_chat["data"]
        assert res_chart["layout"] == exp_chat["layout"]
        assert res_chart["config"] == exp_chat["config"]

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
import plotly.io as pio
import pytest
from scipy.special import factorial as fac

import strawberryfields as sf
from strawberryfields.ops import BSgate, MeasureFock, Sgate
from strawberryfields.plot import plot_wigner

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


class TestFockProbPlotting:
    """Test the Fock state probabilities plotting function"""

    def test_raises_gaussian_no_cutoff(self, monkeypatch):
        """Test that an error is raised if not cutoff value is specified for a
        Gaussian state."""
        prog = sf.Program(1)
        eng = sf.Engine('gaussian')

        with prog.context as q:
            Sgate(2) | q[0]

        state = eng.run(prog).state
        modes = [0]

        with monkeypatch.context() as m:
            # Avoid plotting even if the test failed
            m.setattr(pio, "show", lambda x: None)
            with pytest.raises(ValueError, match="No cutoff specified for"):
                sf.plot_fock(state, modes, renderer="browser")

    @pytest.mark.parametrize("modes", [[0], [0,1]])
    def test_expected_args_to_plot(self, monkeypatch, modes):
        """Test that given a program the expected values would be plotted."""
        num_subsystems = len(modes)

        cutoff = 5
        prog = sf.Program(1)
        eng = sf.Engine('fock', backend_options={"cutoff_dim": cutoff})
        backend = eng.backend

        a = 0.3 + 0.1j

        backend.begin_circuit(
            num_subsystems=num_subsystems,
            cutoff_dim=cutoff,
            pure=True,
            batch_size=1,
        )
        backend.prepare_coherent_state(np.abs(a), np.angle(a), 0)

        state = backend.state()

        mean = [state.mean_photon(mode)[0] for mode in modes]
        all_probs = state.all_fock_probs()

        if num_subsystems > 1:
            photon_dists = [np.sum(all_probs, i).tolist() for i in range(num_subsystems-1,-1, -1)]
        else:
            photon_dists = all_probs

        arg_store = []
        with monkeypatch.context() as m:
            # Avoid plotting even if the test failed
            m.setattr(pio, "show", lambda x: None)
            m.setattr(sf.plot, "generate_fock_chart", lambda *args: arg_store.append(args))
            sf.plot_fock(state, modes, cutoff=cutoff, renderer="browser")

            assert len(arg_store) == 1

            # Extract the stored args
            _, modes_res, photon_dists_res, mean_res, xlabels_res = arg_store[0]

            # Check the results to be plotted
            assert np.allclose(mean_res, mean)
            assert np.allclose(photon_dists_res, photon_dists)
            assert modes_res == modes
            assert xlabels_res == ['|0>', '|1>', '|2>', '|3>', '|4>']

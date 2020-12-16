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
import pytest

import numpy as np
import plotly.io as pio

import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, MeasureFock
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

    def test_raises(self, monkeypatch):
        """Test that an error is raised if not cutoff value is specified for a
        Gaussian state."""
        prog = sf.Program(1)
        eng = sf.Engine('gaussian')

        with prog.context as q:
            gamma = 2
            Sgate(gamma) | q[0]

        state = eng.run(prog).state
        modes = [0]

        with monkeypatch.context() as m:
            # Avoid plotting even if the test failed
            m.setattr(pio, "show", lambda x: None)
            with pytest.raises(ValueError, match="No cutoff specified for"):
                sf.plot_fock(state, modes, renderer="browser")

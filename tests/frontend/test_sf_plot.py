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
    @pytest.mark.parametrize("wire", [0, 1])
    def test_no_errors(self, renderer, wire, prog, monkeypatch):
        """Test that no errors are thrown when calling the `plot_wigner` function"""
        eng = sf.Engine("gaussian")
        results = eng.run(prog)

        xvec = np.arange(-4, 4, 0.1)
        pvec = np.arange(-4, 4, 0.1)
        with monkeypatch.context() as m:
            m.setattr(pio, "show", lambda x: None)
            plot_wigner(results.state, xvec, pvec, renderer=renderer, wire=wire)

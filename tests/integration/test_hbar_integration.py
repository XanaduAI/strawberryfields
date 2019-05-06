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
r"""Integration tests to make sure hbar values are set correctly in returned results"""
import pytest

import numpy as np

import strawberryfields as sf
from strawberryfields import ops


N_MEAS = 200
NUM_STDS = 10.0
STD_10 = NUM_STDS / np.sqrt(N_MEAS)

R = 0.3
X = 0.2
P = 0.1
HBAR = [0.5, 1]


@pytest.mark.parametrize("hbar", HBAR, indirect=True)
class TestIntegration:
    """Tests that go through the frontend and backend, and state object"""

    def test_squeeze_variance_frontend(self, setup_eng, hbar, tol):
        """test homodyne measurement of a squeeze state is correct,
        returning a variance of np.exp(-2*r)*h/2, via the frontend"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Sgate(R) | q
            ops.MeasureX | q

        res = np.empty(0)

        for i in range(N_MEAS):
            eng.run(prog)
            res = np.append(res, q[0].val)
            eng.reset()

        assert np.allclose(
            np.var(res), np.exp(-2 * R) * hbar / 2, atol=STD_10 + tol, rtol=0
        )

    @pytest.mark.backends("gaussian")
    def test_x_displacement(self, setup_eng, hbar, tol):
        """test x displacement on the Gaussian backend gives correct displacement"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Xgate(X) | q

        state = eng.run(prog).state
        mu_x = state.means()[0]

        assert state.hbar == hbar
        assert np.allclose(mu_x, X, atol=tol, rtol=0)

    @pytest.mark.backends("gaussian")
    def test_z_displacement(self, setup_eng, hbar, tol):
        """test x displacement on the Gaussian backend gives correct displacement"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Zgate(P) | q

        state = eng.run(prog).state
        mu_z = state.means()[1]

        assert state.hbar == hbar
        assert np.allclose(mu_z, P, atol=tol, rtol=0)


@pytest.mark.parametrize("hbar", HBAR, indirect=True)
class BackendOnly:
    """Tests that ignore the frontend. This test is currently not run."""

    def test_squeeze_variance(self, setup_backend, hbar, pure, monkeypatch, tol):
        """test homodyne measurement of a squeeze state is correct,
        returning a variance of np.exp(-2*r)*h/2"""
        # TODO: this test is a backend test that duplicates
        # the existing `test_squeeze_variance` integration test.
        # It should live in the backend folder, but currently takes too
        # long to run both.
        # We should revisit this test and decide what to do with it.
        backend = setup_backend(1)

        res = np.empty(0)

        for i in range(N_MEAS):
            backend.squeeze(R, mode=0)
            backend.reset(pure=pure)
            x = backend.measure_homodyne(phi=0, mode=0)
            res = np.append(res, x)

        assert np.allclose(
            np.var(res), np.exp(-2 * R) * hbar / 2, atol=STD_10 + tol, rtol=0
        )

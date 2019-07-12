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
r"""Integration tests for frontend measurement"""
import pytest

import numpy as np

from strawberryfields import ops


SHOTS = (1, 4)  # parametrization for shots


def meas_shape(batch_size, shots):
    """Shape of the measurement result array."""
    if batch_size is None:
        return (shots,)
    else:
        return (batch_size, shots)


class TestMeasurement:
    """Test that measurements work correctly from the frontend"""

    @pytest.mark.backends("fock", "tf")
    def test_fock_measurement(self, setup_eng, tol):
        """Test Fock measurements return expected results"""
        eng, prog = setup_eng(2)
        n = [2, 1]

        with prog.context as q:
            for nn, qq in zip(n, q):
                ops.Fock(nn) | qq
            ops.Measure | q

        eng.run(prog)
        for nn, qq in zip(n, q):
            assert np.all(qq.val == nn)

        # Fock measurements put the modes into vacuum state
        assert np.all(eng.backend.is_vacuum(tol))


    @pytest.mark.backends("fock")
    @pytest.mark.parametrize("shots", SHOTS)
    def test_fock_measurement_unordered(self, setup_eng, shots, batch_size, tol):
        """Test a multi-mode Fock measurement with jumbled regrefs."""
        shape = meas_shape(batch_size, shots)
        n = [2, 0, 3]

        eng, prog = setup_eng(3)
        with prog.context as q:
            for nn, qq in zip(n, q):
                ops.Fock(nn) | qq
            ops.Measure | (q[2], q[0], q[1])

        eng.run(prog, run_options={'shots': shots})
        for nn, qq in zip(n, q):
            assert np.all(qq.val == nn * np.ones(shape, dtype=int))

        # Fock measurements put the modes into vacuum state
        assert np.all(eng.backend.is_vacuum(tol))


    @pytest.mark.backends("gaussian")
    def test_heterodyne(self, setup_eng, tol):
        """Test Fock measurements return expected results"""
        eng, prog = setup_eng(2)
        a = [0.43 - 0.12j, 0.02 + 0.2j]

        with prog.context as q:
            ops.Coherent(a[0]) | q[0]
            ops.Coherent(a[1]) | q[1]
            ops.MeasureHD | q[0]
            ops.MeasureHD | q[1]

        eng.run(prog)

        # Heterodyne measurements put the modes into vacuum state
        assert eng.backend.is_vacuum(tol)


class TestPostselection:
    """Measurement tests that include post-selection"""

    def test_homodyne(self, setup_eng, tol):
        """Test that homodyne detection on a TMS state
        returns post-selected value."""
        x = 0.2
        r = 5

        eng, prog = setup_eng(2)

        with prog.context as q:
            ops.S2gate(r) | q
            ops.MeasureHomodyne(0, select=x) | q[0]

        eng.run(prog)
        assert np.allclose(q[0].val, x, atol=tol, rtol=0)

    @pytest.mark.backends("gaussian")
    def test_heterodyne(self, setup_eng, tol):
        """Test that heterodyne detection on a TMS state
        returns post-selected value."""
        alpha = 0.43 - 0.12j
        r = 5

        eng, prog = setup_eng(2)

        with prog.context as q:
            ops.S2gate(r) | q
            ops.MeasureHeterodyne(select=alpha) | q[0]

        eng.run(prog)
        assert np.allclose(q[0].val, alpha, atol=tol, rtol=0)

    @pytest.mark.backends("fock", "tf")
    def test_measure_fock(self, setup_eng, cutoff, batch_size):
        """Test that Fock post-selection on Fock states
        exiting one arm of a beamsplitter results in conservation
        of photon number in the other. """
        total_photons = cutoff - 1
        for n in range(cutoff):
            eng, prog = setup_eng(2)
            with prog.context as q:
                ops.Fock(n) | q[0]
                ops.Fock(total_photons - n) | q[1]
                ops.BSgate() | q
                ops.MeasureFock(select=n // 2) | q[0]
                ops.MeasureFock() | q[1]

            eng.run(prog)  # FIXME measurements above commute, but they should not since the postselection may fail if the other one is performed first!
            photons_out = sum([i.val for i in q])

            if batch_size is not None:
                assert np.all(photons_out == np.tile(total_photons, batch_size))
            else:
                assert np.all(photons_out == total_photons)

    @pytest.mark.backends("fock")
    @pytest.mark.parametrize("shots", SHOTS)
    def test_measure_fock_single_mode_select(self, setup_eng, cutoff, shots, batch_size):
        """Test Fock post-selection on multiple modes."""

        shape = meas_shape(batch_size, shots)
        n = cutoff-1
        expected = n * np.ones(shape, dtype=int)

        # postselect
        eng, prog = setup_eng(2)
        with prog.context as q:
            ops.Dgate(0.1) | q[0]
            ops.Dgate(1.3) | q[1]
            ops.MeasureFock(select=n) | q[0]
        eng.run(prog, run_options={'shots': shots})
        assert np.all(q[0].val == expected)
        assert q[1].val is None

    @pytest.mark.backends("fock")
    @pytest.mark.parametrize("shots", SHOTS)
    def test_measure_fock_multimode_select(self, setup_eng, cutoff, shots, batch_size):
        """Test Fock post-selection on multiple modes."""

        # NOTE for now the Fock backend does not do batching, but the asserts below pass due to broadcasting
        shape = meas_shape(batch_size, shots)
        n = np.array([cutoff-1, 1, 0])
        perm = [1, 2, 0]  # mode permutation for the measurement
        m = len(n)
        expected = np.tensordot(n, np.ones(shape, dtype=int), axes=0)

        # postselect all modes
        eng, prog = setup_eng(m)
        with prog.context as q:
            ops.Dgate(0.1) | q[0]
            ops.Dgate(1.3) | q[1]
            ops.Dgate(0.5) | q[2]
            ops.MeasureFock(select=list(n[perm])) | tuple(q[k] for k in perm)
        eng.run(prog, run_options={'shots': shots})
        for k in range(m):
            assert np.all(q[k].val == expected[k])

        # postselect some modes
        eng, prog = setup_eng(m)
        with prog.context as q:
            ops.Dgate(0.7) | q[0]
            ops.Fock(2)    | q[1]
            ops.Dgate(0.2) | q[2]
            ops.MeasureFock(select=[None, n[2], n[0]]) | tuple(q[k] for k in perm)
        eng.run(prog, run_options={'shots': shots})
        assert np.all(q[0].val == expected[0])
        assert np.all(q[1].val == np.ones(shape, dtype=int) * 2)
        assert np.all(q[2].val == expected[2])



    @pytest.mark.backends("gaussian")
    def test_embed_graph(self, setup_eng, hbar):
        """Test that an embedded graph has the right total mean photon number. """

        eng, prog = setup_eng(2)
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        n_mean_per_mode = 1
        with prog.context as q:
            ops.GraphEmbed(A, n_mean_per_mode) | (q[0], q[1])

        state = eng.run(prog).state
        cov = state.cov()
        n_mean_tot = np.trace(cov / (hbar / 2) - np.identity(4)) / 4
        expected = 2 * n_mean_per_mode
        assert np.allclose(n_mean_tot, expected)

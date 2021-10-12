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


class TestMeasurement:
    """Test that measurements work correctly from the frontend"""

    @pytest.mark.backends("fock", "tf")
    def test_fock_measurement(self, setup_eng, tol):
        """Test Fock measurements return expected results"""
        eng, prog = setup_eng(2)
        n = [2, 1]

        with prog.context as q:
            ops.Fock(n[0]) | q[0]
            ops.Fock(n[1]) | q[1]
            ops.MeasureFock() | q

        eng.run(prog)
        assert np.all(q[0].val == n[0])
        assert np.all(q[1].val == n[1])

        # Fock measurements put the modes into vacuum state
        assert np.all(eng.backend.is_vacuum(tol))

    @pytest.mark.backends("gaussian", "bosonic")
    def test_heterodyne(self, setup_eng, tol):
        """Test heterodyne measurements return expected results"""
        eng, prog = setup_eng(2)
        a0 = 0.44643
        phi0 = -0.2721457
        a1 = 0.200998
        phi1 = 1.47112755

        with prog.context as q:
            ops.Coherent(a0, phi0) | q[0]
            ops.Coherent(a1, phi1) | q[1]
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

    @pytest.mark.backends("gaussian", "bosonic")
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
        of photon number in the other."""
        total_photons = cutoff - 1
        for n in range(cutoff):
            eng, prog = setup_eng(2)
            with prog.context as q:
                ops.Fock(n) | q[0]
                ops.Fock(total_photons - n) | q[1]
                ops.BSgate() | q
                ops.MeasureFock(select=n // 2) | q[0]
                ops.MeasureFock() | q[1]

            eng.run(
                prog
            )  # FIXME measurements above commute, but they should not since the postselection may fail if the other one is performed first!
            photons_out = sum([i.val for i in q])

            if batch_size is not None:
                assert np.all(photons_out == np.tile(total_photons, batch_size))
            else:
                assert np.all(photons_out == total_photons)

    @pytest.mark.backends("gaussian")
    def test_embed_graph(self, setup_eng, hbar):
        """Test that an embedded graph has the right total mean photon number."""

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

    @pytest.mark.backends("gaussian")
    def test_coherent_state_has_photons(self, setup_eng, hbar):
        """Test that a coherent state with a mean photon number of 4 and sampled 100 times will produce photons"""
        shots = 100
        eng, prog = setup_eng(1)
        alpha = 2
        with prog.context as q:
            ops.Coherent(alpha) | q[0]
            ops.MeasureFock() | q[0]
        state = eng.run(prog).state
        samples = np.array(eng.run(prog, shots=shots)).flatten()

        assert not np.all(samples == np.zeros_like(samples, dtype=int))


class TestDarkCounts:
    @pytest.mark.backends("fock")
    @pytest.mark.parametrize("dark_counts", [[4, 2], [3, 0], [0, 6]])
    def test_fock_darkcounts_two_modes(self, dark_counts, setup_eng, monkeypatch):
        """Test Fock measurements return expected results with dark counts on all detectors"""
        with monkeypatch.context() as m:
            # add the number of dark counts instead of sampling from a poisson distribution
            m.setattr(np.random, "poisson", lambda dc, shape: dc * np.ones(shape, dtype=int))

            eng, prog = setup_eng(2)
            n = [2, 1]

            with prog.context as q:
                ops.Fock(n[0]) | q[0]
                ops.Fock(n[1]) | q[1]
                ops.MeasureFock(dark_counts=dark_counts) | q

            eng.run(prog)
            assert q[0].val == n[0] + dark_counts[0]
            assert q[1].val == n[1] + dark_counts[1]

    @pytest.mark.backends("fock")
    @pytest.mark.parametrize("dark_counts", [[4], 6, 0, [0]])
    def test_fock_darkcounts_single_mode(self, dark_counts, setup_eng, monkeypatch):
        """Test Fock measurements return expected results with dark counts on one detector"""
        with monkeypatch.context() as m:
            m.setattr(np.random, "poisson", lambda dc, shape: dc * np.ones(shape, dtype=int))

            eng, prog = setup_eng(2)
            n = [2, 1]

            with prog.context as q:
                ops.Fock(n[0]) | q[0]
                ops.Fock(n[1]) | q[1]
                ops.MeasureFock(dark_counts=dark_counts) | q[0]
                ops.MeasureFock() | q[1]

            eng.run(prog)

            if isinstance(dark_counts, int):
                dark_counts = [dark_counts]

            assert q[0].val == n[0] + dark_counts[0]
            assert q[1].val == n[1]

    @pytest.mark.backends("fock")
    @pytest.mark.parametrize("dark_counts", [[3], 6, [4, 2, 1]])
    def test_fock_darkcounts_errors(self, dark_counts, setup_eng):
        """Test Fock measurements errors when applying dark counts to more/less detectors than measured"""

        eng, prog = setup_eng(2)
        n = [2, 1]

        with prog.context as q:
            ops.Fock(n[0]) | q[0]
            ops.Fock(n[1]) | q[1]
            ops.MeasureFock(dark_counts=dark_counts) | q

        with pytest.raises(
            ValueError,
            match="The number of dark counts must be equal to the number of measured modes",
        ):
            eng.run(prog)

    @pytest.mark.backends("fock")
    def test_fock_darkcounts_with_postselection(self, setup_eng):
        """Test Fock measurements error when using dark counts together with post-selection"""

        eng, prog = setup_eng(2)
        n = [2, 1]

        with prog.context as q:
            ops.Fock(n[0]) | q[0]
            ops.Fock(n[1]) | q[1]
            with pytest.raises(
                NotImplementedError,
                match="Post-selection cannot be used together with dark counts",
            ):
                ops.MeasureFock(select=1, dark_counts=2) | q

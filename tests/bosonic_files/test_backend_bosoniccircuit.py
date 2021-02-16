# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Unit tests for backends.bosonicbackend.bosoniccircuit.py.
"""

import numpy as np
import strawberryfields.backends.bosonicbackend.bosoniccircuit as circuit
import pytest

pytestmark = pytest.mark.bosonic

NUM_WEIGHTS_VALS = [1, 2, 5]
NUM_MODES_VALS = [1, 3, 4]
NUM_NEW_WEIGHTS_VALS = [1, 2, 3]
R_VALS = np.array([-1.0, -1 / 3, 1 / 2, 1.2])
PHI_VALS = np.array([-1.2, -1 / 5, 0.6, 1.3])
NBAR_VALS = np.array([0, 1, 2.5])
NUM_SHOTS_VALS = np.array([1, 2, 10])
R_ANC_VALS = np.array([0.1, 0.8, 1.2, 5])
ETA_ANC_VALS = np.array([0.1, 0.4, 0.9, 1])


class TestBosonicModes:
    r"""Tests all the functions inside backends.bosonicbackend.bosoniccircuit.py"""

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("num_modes", NUM_MODES_VALS)
    def test_circuit_init(self, num_weights, num_modes):
        r"""Checks that the reset method instantiates the correct number of modes
        as vacuum, each with a given number of weights."""
        # Create a state over num_modes, each with num_weights
        example = circuit.BosonicModes(num_modes, num_weights)
        tot_num_weights = num_weights ** num_modes
        num_quad = 2 * num_modes

        # Confirm number of modes
        assert example.nlen == num_modes

        # Confirm normalization
        assert np.isclose(sum(example.get_weights()), 1)

        # Confirm each mode initialized to vacuum
        assert example.is_vacuum(tol=1e-10)
        for i in range(num_modes):
            assert np.isclose(example.fidelity_vacuum([i]), 1)

        # Confirm active modes
        assert example.active == list(range(num_modes))
        # Check mode-wise and quadrature-wise ordering
        to_xp_list = list(range(0, num_quad, 2)) + list(range(1, num_quad + 1, 2))
        from_xp_list = []
        for i in range(num_modes):
            from_xp_list += [i, i + num_modes]
        assert np.allclose(example.to_xp, np.array(to_xp_list))
        assert np.allclose(example.from_xp, np.array(from_xp_list))

        # Confirm number of weights, means and covs
        assert example.get_weights().shape == (tot_num_weights,)
        assert example.get_mean().shape == (tot_num_weights, num_quad)
        assert example.get_covmat().shape == circuit.c_shape(num_modes, num_weights)

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("num_modes", NUM_MODES_VALS)
    def test_reset_circuit(self, num_weights, num_modes):
        r"""Checks that the reset method instantiates the correct number of modes
        as vacuum, each with a given number of weights."""
        # Create a state over 1 mode, 1 weight per mode
        example = circuit.BosonicModes(1, 1)

        # Reset with number of weights and modes and perform the same check
        # as test_circuit_init
        example.reset(num_modes, num_weights)
        tot_num_weights = num_weights ** num_modes
        num_quad = 2 * num_modes
        assert example.nlen == num_modes
        assert np.isclose(sum(example.get_weights()), 1)
        assert example.is_vacuum(tol=1e-10)
        for i in range(num_modes):
            assert np.isclose(example.fidelity_vacuum([i]), 1)
        assert example.active == list(range(num_modes))
        assert example.get_mean().shape == (tot_num_weights, num_quad)
        assert example.get_covmat().shape == circuit.c_shape(num_modes, num_weights)
        to_xp_list = list(range(0, num_quad, 2)) + list(range(1, num_quad + 1, 2))
        from_xp_list = []
        for i in range(num_modes):
            from_xp_list += [i, i + num_modes]
        assert np.allclose(example.to_xp, np.array(to_xp_list))
        assert np.allclose(example.from_xp, np.array(from_xp_list))

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("num_modes", NUM_MODES_VALS)
    def test_del_modes(self, num_weights, num_modes):
        r"""Checks that modes can be deleted."""
        # Create a state with the pre-defined number of weights and modes
        example = circuit.BosonicModes(num_modes, num_weights)
        # Delete first mode and check active modes changed
        example.del_mode(0)
        tot_weights = num_weights ** num_modes
        assert example.get_modes() == list(range(1, num_modes))
        # Check numbers of weights, means and covs
        assert np.isclose(sum(example.get_weights()), 1)
        assert example.get_weights().shape == (tot_weights,)
        assert example.get_mean().shape == (tot_weights, 2 * (num_modes))
        assert example.get_covmat().shape == (tot_weights, 2 * (num_modes), 2 * (num_modes))

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("num_modes", NUM_MODES_VALS)
    @pytest.mark.parametrize("num_weights_new", NUM_NEW_WEIGHTS_VALS)
    def test_add_modes(self, num_weights, num_modes, num_weights_new):
        r"""Checks that modes can be added."""
        # Create a state with the pre-defined number of weights and modes
        example = circuit.BosonicModes(num_modes, num_weights)

        # Add new mode with new number of weights
        example.add_mode([num_weights_new])
        tot_weights = (num_weights ** num_modes) * num_weights_new
        # Check numbers of weights, means and covs
        assert example.get_modes() == list(range(num_modes + 1))
        assert np.isclose(sum(example.get_weights()), 1)
        assert example.get_weights().shape == (tot_weights,)
        assert example.get_mean().shape == (tot_weights, 2 * (num_modes + 1))
        assert example.get_covmat().shape == (tot_weights, 2 * (num_modes + 1), 2 * (num_modes + 1))

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("num_modes", NUM_MODES_VALS)
    @pytest.mark.parametrize("num_weights_new", NUM_NEW_WEIGHTS_VALS)
    def test_add_del_modes_together(self, num_weights, num_modes, num_weights_new):
        r"""Checks that modes can be added and deleted in sequence."""
        # Create a state with the pre-defined number of weights and modes
        example = circuit.BosonicModes(num_modes, num_weights)
        # Delete first mode and check active modes changed
        example.del_mode(0)
        assert example.get_modes() == list(range(1, num_modes))

        # Add new mode with new number of weights
        example.add_mode([num_weights_new])
        tot_weights = (num_weights ** num_modes) * num_weights_new
        # Check numbers of weights, means and covs
        assert example.get_modes() == list(range(1, num_modes + 1))
        assert np.isclose(sum(example.get_weights()), 1)
        assert example.get_weights().shape == (tot_weights,)
        assert example.get_mean().shape == (tot_weights, 2 * (num_modes + 1))
        assert example.get_covmat().shape == (tot_weights, 2 * (num_modes + 1), 2 * (num_modes + 1))

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_displace(self, r, phi):
        r"""Checks the displacement operation."""
        # Ensure zero displacement yields vacuum
        example = circuit.BosonicModes(1, 1)
        example.displace(0, 0, 0)
        assert example.is_vacuum(tol=1e-10)

        # Displace and check it is a coherent state
        example.displace(r, phi, 0)
        assert np.isclose(example.fidelity_coherent(np.array([r * np.exp(1j * phi)])), 1)

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_rotate(self, r, phi):
        r"""Checks the phase shift operation."""
        # Ensure zero rotation yields vacuum
        example = circuit.BosonicModes(1, 1)
        example.phase_shift(0, 0)
        assert example.is_vacuum(tol=1e-10)

        # Displace
        example.displace(r, phi, 0)
        d = example.hbar * (np.array([r * np.exp(1j * phi).real, r * np.exp(1j * phi).imag]))

        # Rotate state and confirm correct mean and covs produced
        example.phase_shift(phi, 0)
        Rmat2 = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        cov_new = Rmat2 @ Rmat2.T
        d_new = Rmat2 @ d
        assert np.allclose(example.get_mean()[0], d_new)
        assert np.allclose(example.get_covmat()[0], cov_new)

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_squeeze(self, r, phi):
        r"""Checks the squeezing operation."""
        # Ensure zero squeezing yields vacuum
        example = circuit.BosonicModes(1, 1)
        example.squeeze(0, 0, 0)
        assert example.is_vacuum(tol=1e-10)

        # Displace
        example.displace(r, phi, 0)

        # Squeeze state and confirm correct mean and covs produced
        example.squeeze(r, phi, 0)
        d = example.hbar * (np.array([r * np.exp(1j * phi).real, r * np.exp(1j * phi).imag]))
        Smat = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        Rmat = np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        cov_new = Rmat @ Smat @ Rmat.T @ Rmat @ Smat.T @ Rmat.T
        d_new = Rmat @ Smat @ Rmat.T @ d
        assert np.allclose(example.get_mean()[0], d_new)
        assert np.allclose(example.get_covmat()[0], cov_new)

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_from_cov_and_mean(self, r, phi):
        r"""Checks Gaussian states can be made from mean and covariance."""
        example = circuit.BosonicModes(1, 1)

        # Create a state with a given mean and cov
        d = example.hbar * (np.array([r * np.exp(1j * phi).real, r * np.exp(1j * phi).imag]))
        Smat = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        Rmat = np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        Rmat2 = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        cov_new = Rmat2 @ Rmat @ Smat @ Rmat.T @ Rmat @ Smat.T @ Rmat.T @ Rmat2.T
        d_new = Rmat2 @ Rmat @ Smat @ Rmat.T @ d

        example.from_mean(d_new)
        example.from_covmat(cov_new)

        # Undo all the operations that would have in principle prepared the same state
        example.phase_shift(-phi, 0)
        example.squeeze(-r, phi, 0)
        example.displace(-r, phi, 0)
        # Check that you are left with vacuum
        assert example.is_vacuum(tol=1e-10)

        # Do again specifying the mode this time
        example.from_mean(d_new, modes=[0])
        example.from_covmat(cov_new, modes=[0])
        example.phase_shift(-phi, 0)
        example.squeeze(-r, phi, 0)
        example.displace(-r, phi, 0)
        # Check that you are left with vacuum
        assert example.is_vacuum(tol=1e-10)

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("num_modes", NUM_MODES_VALS)
    def test_parity(self, num_weights, num_modes):
        r"""Checks parity function for vacuum state."""
        # Create state with number of weights and means
        example = circuit.BosonicModes(num_modes, num_weights)

        # Assert that it has parity of 1 (since it is vacuum)
        assert np.isclose(example.parity_val(), 1)
        for i in range(num_modes):
            assert np.isclose(example.parity_val([i]), 1)

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    @pytest.mark.parametrize("nbar", NBAR_VALS)
    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    def test_losses(self, r, phi, nbar, num_weights):
        r"""Checks loss, thermal_loss, and init_thermal."""
        # Create state with random number of weights and means
        example = circuit.BosonicModes(3, num_weights)

        # Displace, apply total loss to first mode and check it is vacuum
        example.displace(r, phi, 0)
        example.loss(0, 0)
        assert example.is_vacuum(tol=1e-10)

        # Apply thermal loss to second mode and check the covariance is as expected
        example.thermal_loss(0, nbar, 1)
        assert np.allclose(
            np.tile((2 * nbar + 1) * np.eye(2), (num_weights ** 3, 1, 1)),
            example.get_covmat()[:, 2:4, 2:4],
        )

        # Initialize third mode as thermal and check it's the same as second mode
        example.init_thermal(nbar, 2)
        assert np.allclose(example.get_covmat()[:, 4:6, 4:6], example.get_covmat()[:, 2:4, 2:4])

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_gaussian_symplectic(self, r, phi):
        r"""Checks symplectic transformations, and get_covmat_xp, get_mean_xp."""
        example = circuit.BosonicModes(2, 1)

        # Create a symplectic and apply to first mode
        Smat = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        Rmat = np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        Rmat2 = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        symp = Rmat2 @ Rmat @ Smat @ Rmat.T
        symp = example.expandS([0], symp)
        example.apply_channel(symp, np.zeros(symp.shape))

        # Apply the corresponding operations to second mode
        example.squeeze(r, phi, 1)
        example.phase_shift(phi, 1)

        # Check that same modes are in both states
        assert np.allclose(example.get_covmat()[:, 0:2, 0:2], example.get_covmat()[:, 2:4, 2:4])
        assert np.allclose(example.get_mean()[:, 0:2], example.get_mean()[:, 2:4])
        assert np.allclose(
            example.get_covmat_xp()[:, 0::2, 0::2], example.get_covmat_xp()[:, 1::2, 1::2]
        )
        assert np.allclose(example.get_mean_xp()[:, 0::2], example.get_mean_xp()[:, 1::2])

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_beamsplitter(self, r, phi):
        r"""Checks applying a beamsplitter."""
        example = circuit.BosonicModes(2, 1)
        # Send two coherent states through beamspliter
        example.displace(r, 0, 0)
        example.displace(r, 0, 1)
        example.beamsplitter(phi, phi, 0, 1)
        alpha = r * (np.cos(phi) - np.exp(-1j * phi) * np.sin(phi))
        beta = r * (np.cos(phi) + np.exp(1j * phi) * np.sin(phi))
        assert np.isclose(example.fidelity_coherent(np.array([alpha, beta])), 1)

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    @pytest.mark.parametrize("num_shots", NUM_SHOTS_VALS)
    def test_generaldyne(self, num_weights, r, num_shots):
        r"""Checks a generaldyne measurement."""
        # Create two vacuums with multiple weights
        example = circuit.BosonicModes(2, num_weights)
        # Create squeezed covariance matrix
        covmat = np.diag(np.array([np.exp(-r), np.exp(r)]))
        # Send two squeezed states through beamspliter
        example.squeeze(r, 0, 0)
        example.squeeze(-r, 0, 1)
        example.beamsplitter(np.pi / 4, 0, 0, 1)
        # Perform generaldyne on mode 0
        vals = example.measure_dyne(covmat, [0], shots=num_shots)

        # Check output has right shape
        assert vals.shape == (num_shots, 2)
        # Check measured mode was set to vacuum
        assert np.isclose(example.fidelity_vacuum(modes=[0]), 1)

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    def test_homodyne(self, num_weights, r, phi):
        r"""Checks a homodyne measurement."""
        # Create coherent state with multiple weights
        example = circuit.BosonicModes(1, num_weights)
        # Make complex weights

        example.displace(r, phi, 0)
        # Perform homodyne on mode 0
        vals = example.homodyne(0, shots=500)
        mean = vals[:, 0].mean()
        std = vals[:, 0].std()
        assert np.isclose(mean, example.hbar * r * np.cos(phi), rtol=0, atol=10 / np.sqrt(500))
        assert np.isclose(std, 1, rtol=0, atol=10 / np.sqrt(500))

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    def test_heterodyne(self, num_weights, r, phi):
        r"""Checks a heterodyne measurement."""
        # Create coherent state with multiple weights
        example = circuit.BosonicModes(1, num_weights)
        example.displace(r, phi, 0)
        # Perform heterodyne on mode 0
        vals = example.heterodyne(0, shots=500)
        mean = vals[:, 0].mean()
        assert np.isclose(mean, example.hbar * r * np.cos(phi), rtol=0, atol=10 / np.sqrt(500))

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_post_select_homodyne(self, num_weights, r):
        r"""Checks post_select_homodyne."""
        # Create maximally entangled states from highly squeezed states
        example = circuit.BosonicModes(2, num_weights)
        example.squeeze(8, 0, 0)
        example.squeeze(-8, 0, 1)
        example.beamsplitter(np.pi / 4, 0, 0, 1)

        # Post select homodyne on one
        example.post_select_homodyne(0, r, phi=np.pi / 2)
        # Check other mode ended up in correlated state
        mean_compare = np.tile(np.array([0, 0, 0, -r]), (num_weights ** 2, 1))
        assert np.allclose(example.get_mean(), mean_compare)

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_post_select_heterodyne(self, num_weights, r):
        r"""Checks post_select_homodyne."""
        # Create maximally entangled states from highly squeezed states
        example = circuit.BosonicModes(2, num_weights)
        example.squeeze(8, 0, 0)
        example.squeeze(-8, 0, 1)
        example.beamsplitter(np.pi / 4, 0, 0, 1)

        # Post select heterodyne on one
        example.post_select_heterodyne(0, r)
        # Check other mode ended up in correlated state
        mean_compare = np.tile(np.array([0, 0, -r, 0]), (num_weights ** 2, 1))
        cov_compare = np.tile(np.eye(4), (num_weights ** 2, 1, 1))
        assert np.allclose(example.get_mean(), mean_compare)
        assert np.allclose(example.get_covmat(), cov_compare)

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    @pytest.mark.parametrize("r_anc", R_ANC_VALS)
    @pytest.mark.parametrize("eta_anc", ETA_ANC_VALS)
    def test_mb_squeeze_avg(self, r, phi, r_anc, eta_anc):
        r"""Checks measurement-based squeezing average map."""
        example = circuit.BosonicModes(2, 1)

        # Test zero mbsqueezing
        example.mb_squeeze_avg(0, 0, 0, r_anc, eta_anc)
        assert example.is_vacuum(tol=1e-10)

        # Test high-quality mbsqueezing
        example.mb_squeeze_avg(0, r, phi, 9, 1)
        example.squeeze(r, phi, 1)
        assert np.allclose(example.get_covmat()[:, 0:2, 0:2], example.get_covmat()[:, 2:4, 2:4])
        assert np.allclose(example.get_mean()[:, 0:2], example.get_mean()[:, 2:4])

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    @pytest.mark.parametrize("r_anc", R_ANC_VALS)
    @pytest.mark.parametrize("eta_anc", ETA_ANC_VALS)
    def test_mb_squeeze_single_shot(self, r, phi, r_anc, eta_anc):
        r"""Checks measurement-based squeezing."""
        example = circuit.BosonicModes(2, 1)

        # Test zero mbsqueezing
        example.mb_squeeze_single_shot(0, 0, 0, 5, 1)
        assert example.is_vacuum(tol=1e-10)

        # Test high-quality mbsqueezing
        example.mb_squeeze_single_shot(0, r, phi, 9, 1)
        example.squeeze(r, phi, 1)
        assert np.allclose(example.get_covmat()[:, 0:2, 0:2], example.get_covmat()[:, 2:4, 2:4])

    @pytest.mark.parametrize("r", R_VALS)
    def test_complex_weight(self, r):
        r"""Checks that a cat state has correct parity and that the mean from homodyne sampling is 0."""
        # Make a cat state
        example = circuit.BosonicModes(1, 4)

        r = abs(r)

        example.weights[0] = 1.0 + 0j
        example.weights[1] = 1.0 + 0j
        example.weights[2] = np.exp(-2 * abs(r) ** 2)
        example.weights[3] = np.exp(-2 * abs(r) ** 2)
        example.weights /= np.sum(example.weights)

        hbar = example.hbar
        example.means[0, :] = np.sqrt(2 * hbar) * np.array([r.real, r.imag])
        example.means[1, :] = -np.sqrt(2 * hbar) * np.array([r.real, r.imag])
        example.means[2, :] = 1j * np.sqrt(2 * hbar) * np.array([r.imag, -r.real])
        example.means[3, :] = -1j * np.sqrt(2 * hbar) * np.array([r.imag, -r.real])

        assert np.isclose(example.parity_val(), 1)

        # Perform homodyne
        vals = example.homodyne(0, shots=500)
        mean = vals[:, 0].mean()
        assert np.isclose(mean, 0, rtol=0, atol=10 * np.sqrt(abs(r)) / np.sqrt(500))

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_linear_optical_unitary(self, r, phi):
        r"""Checks the linear optical unitary implementation of a phase shift."""
        # Create two displaced states
        example = circuit.BosonicModes(2, 1)
        example.displace(r, 0, 0)
        example.displace(r, 0, 1)
        # Rotate them using different techniques
        example.apply_u(np.diag([np.exp(1j * phi), 1]))
        example.phase_shift(phi, 1)
        # Check they have the same means
        assert np.allclose(example.get_mean()[:, 0:2], example.get_mean()[:, 2:4])

    @pytest.mark.parametrize("num_modes", NUM_MODES_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_expandXY(self, num_modes, r):
        r"""Tests the expandXY function"""
        example = circuit.BosonicModes(num_modes, 1)
        X = r * np.eye(2)
        Y = r * np.eye(2)
        X2, Y2 = example.expandXY([0], X, Y)
        X2_diag = np.ones(2 * num_modes)
        X2_diag[0] = r
        X2_diag[num_modes] = r
        Y2_diag = np.zeros(2 * num_modes)
        Y2_diag[0] = r
        Y2_diag[num_modes] = r
        assert np.allclose(X2, np.diag(X2_diag))
        assert np.allclose(Y2, np.diag(Y2_diag))

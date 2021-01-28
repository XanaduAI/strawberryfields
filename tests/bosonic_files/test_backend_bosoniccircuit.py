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


class TestBosonicCircuit:
    r"""Tests all the functions inside backends.bosonicbackend.bosoniccircuit.py"""

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("num_modes", NUM_MODES_VALS)
    def test_reset_circuit(self, num_weights, num_modes):
        r"""Checks that the reset method instantiates correct number of modes
        as vacuum, each with a given number of weights."""
        # Create a state over 3 modes, 2 weights per mode
        example = circuit.BosonicModes()
        example.reset(3, 2)

        # Confirm number of modes and weights
        assert example.nlen == 3

        # Confirm normalization
        assert np.isclose(sum(example.sweights()), 1)

        # Confirm each mode initialized to vacuum
        assert example.is_vacuum(tol=1e-10)
        for i in range(3):
            assert np.isclose(example.fidelity_vacuum([i]), 1)

        # Confirm active modes
        assert example.active == [0, 1, 2]
        # Check mode-wise and quadrature-wise ordering
        assert np.allclose(example.to_xp, np.array([0, 2, 4, 1, 3, 5]))
        assert np.allclose(example.from_xp, np.array([0, 3, 1, 4, 2, 5]))

        # Confirm number of weights, means and covs
        assert example.sweights().shape == (2 ** 3,)
        assert example.smean().shape == (2 ** 3, 6)
        assert example.scovmat().shape == circuit.c_shape(3, 2)

        # Reset with number of weights and modes and perform the same check
        example.reset(num_modes, num_weights)
        assert example.nlen == num_modes
        assert np.isclose(sum(example.sweights()), 1)
        assert example.is_vacuum(tol=1e-10)
        for i in range(num_modes):
            assert np.isclose(example.fidelity_vacuum([i]), 1)
        assert example.active == [i for i in range(num_modes)]
        assert example.smean().shape == (num_weights ** num_modes, 2 * num_modes)
        assert example.scovmat().shape == (num_weights ** num_modes, 2 * num_modes, 2 * num_modes)

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("num_modes", NUM_MODES_VALS)
    @pytest.mark.parametrize("num_weights_new", NUM_NEW_WEIGHTS_VALS)
    def test_add_del_modes(self, num_weights, num_modes, num_weights_new):
        r"""Checks that modes can be added and deleted."""
        # Create state with number of weights and modes
        example = circuit.BosonicModes()
        example.reset(num_modes, num_weights)
        # Delete first mode and check active modes changed
        example.del_mode(0)
        assert example.get_modes() == [i for i in range(1, num_modes)]

        # Add new mode with new number of weights
        example.add_mode([num_weights_new])
        tot_weights = (num_weights ** num_modes) * num_weights_new
        # Check numbers of weights, means and covs
        assert example.get_modes() == [i for i in range(1, num_modes + 1)]
        assert np.isclose(sum(example.sweights()), 1)
        assert example.sweights().shape == (tot_weights,)
        assert example.smean().shape == (tot_weights, 2 * (num_modes + 1))
        assert example.scovmat().shape == (tot_weights, 2 * (num_modes + 1), 2 * (num_modes + 1))

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_displace_squeeze_rotate(self, r, phi):
        r"""Checks single-mode Gaussian transformations."""
        # Ensure zero displacement, squeezing and rotation all yield vacuum
        example = circuit.BosonicModes()
        example.reset(1, 1)
        example.squeeze(0, 0, 0)
        assert example.is_vacuum(tol=1e-10)
        example.displace(0, 0, 0)
        assert example.is_vacuum(tol=1e-10)
        example.phase_shift(0, 0)
        assert example.is_vacuum(tol=1e-10)

        # Displaceand check it is a coherent state
        example.displace(r, phi, 0)
        assert np.isclose(example.fidelity_coherent(np.array([r * np.exp(1j * phi)])), 1)

        # Squeeze state and confirm correct mean and covs produced
        example.squeeze(r, phi, 0)
        d = example.hbar * (np.array([r * np.exp(1j * phi).real, r * np.exp(1j * phi).imag]))
        Smat = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        Rmat = np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        cov_new = Rmat @ Smat @ Rmat.T @ Rmat @ Smat.T @ Rmat.T
        d_new = Rmat @ Smat @ Rmat.T @ d
        assert np.allclose(example.smean()[0], d_new)
        assert np.allclose(example.scovmat()[0], cov_new)

        # Rotate state and confirm correct mean and covs produced
        example.phase_shift(phi, 0)
        Rmat2 = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        cov_new = Rmat2 @ cov_new @ Rmat2.T
        d_new = Rmat2 @ d_new
        assert np.allclose(example.smean()[0], d_new)
        assert np.allclose(example.scovmat()[0], cov_new)

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_from_cov_and_mean(self, r, phi):
        r"""Checks Gaussian states can be made from mean and covariance."""
        example = circuit.BosonicModes()
        example.reset(1, 1)

        # Create a state with a given mean and cov
        d = example.hbar * (np.array([r * np.exp(1j * phi).real, r * np.exp(1j * phi).imag]))
        Smat = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        Rmat = np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        Rmat2 = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        cov_new = Rmat2 @ Rmat @ Smat @ Rmat.T @ Rmat @ Smat.T @ Rmat.T @ Rmat2.T
        d_new = Rmat2 @ Rmat @ Smat @ Rmat.T @ d

        example.fromsmean(d_new)
        example.fromscovmat(cov_new)

        # Undo all the operations that would have in principle prepared the same state
        example.phase_shift(-phi, 0)
        example.squeeze(-r, phi, 0)
        example.displace(-r, phi, 0)
        # Check that you are left with vacuum
        assert example.is_vacuum(tol=1e-10)

        # Do again specifying the mode this time
        example.fromsmean(d_new, modes=[0])
        example.fromscovmat(cov_new, modes=[0])
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
        example = circuit.BosonicModes()
        example.reset(num_modes, num_weights)

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
        example = circuit.BosonicModes()
        example.reset(3, num_weights)

        # Displace, apply total loss to first mode and check it is vacuum
        example.displace(r, phi, 0)
        example.loss(0, 0)
        assert example.is_vacuum(tol=1e-10)

        # Apply thermal loss to second mode and check the covariance is as expected
        example.thermal_loss(0, nbar, 1)
        assert np.allclose(
            np.tile((2 * nbar + 1) * np.eye(2), (num_weights ** 3, 1, 1)),
            example.scovmat()[:, 2:4, 2:4],
        )

        # Initialize third mode as thermal and check it's the same as second mode
        example.init_thermal(nbar, 2)
        assert np.allclose(example.scovmat()[:, 4:6, 4:6], example.scovmat()[:, 2:4, 2:4])

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_gaussian_symplectic(self, r, phi):
        r"""Checks symplectic transformations, and scovtmatxp, smeanxp."""
        example = circuit.BosonicModes()
        example.reset(2, 1)

        # Create a symplectic and apply to first mode
        Smat = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        Rmat = np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        Rmat2 = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        symp = Rmat2 @ Rmat @ Smat @ Rmat.T
        symp = example.expandS([0], symp)
        example.apply_channel(symp, 0)

        # Apply the corresponding operations to second mode
        example.squeeze(r, phi, 1)
        example.phase_shift(phi, 1)

        # Check that same modes are in both states
        assert np.allclose(example.scovmat()[:, 0:2, 0:2], example.scovmat()[:, 2:4, 2:4])
        assert np.allclose(example.smean()[:, 0:2], example.smean()[:, 2:4])
        assert np.allclose(example.scovmatxp()[:, 0::2, 0::2], example.scovmatxp()[:, 1::2, 1::2])
        assert np.allclose(example.smeanxp()[:, 0::2], example.smeanxp()[:, 1::2])

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_beamsplitter(self, r, phi):
        r"""Checks beamsplitter."""
        example = circuit.BosonicModes()
        example.reset(2, 1)
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
        r"""Checks generaldyne."""
        # Create two vacuums with multiple weights
        example = circuit.BosonicModes()
        example.reset(2, num_weights)
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
        r"""Checks homodyne."""
        # Create coherent state with multiple weights
        example = circuit.BosonicModes()
        example.reset(1, num_weights)
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
        r"""Checks heterodyne."""
        # Create coherent state with multiple weights
        example = circuit.BosonicModes()
        example.reset(1, num_weights)
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
        example = circuit.BosonicModes()
        example.reset(2, num_weights)
        example.squeeze(8, 0, 0)
        example.squeeze(-8, 0, 1)
        example.beamsplitter(np.pi / 4, 0, 0, 1)

        # Post select homodyne on one
        example.post_select_homodyne(0, r, phi=np.pi / 2)
        # Check other mode ended up in correlated state
        mean_compare = np.tile(np.array([0, 0, 0, -r]), (num_weights ** 2, 1))
        assert np.allclose(example.smean(), mean_compare)

    @pytest.mark.parametrize("num_weights", NUM_WEIGHTS_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_post_select_heterodyne(self, num_weights, r):
        r"""Checks post_select_homodyne."""
        # Create maximally entangled states from highly squeezed states
        example = circuit.BosonicModes()
        example.reset(2, num_weights)
        example.squeeze(8, 0, 0)
        example.squeeze(-8, 0, 1)
        example.beamsplitter(np.pi / 4, 0, 0, 1)

        # Post select heterodyne on one
        example.post_select_heterodyne(0, r)
        # Check other mode ended up in correlated state
        mean_compare = np.tile(np.array([0, 0, -r, 0]), (num_weights ** 2, 1))
        cov_compare = np.tile(np.eye(4), (num_weights ** 2, 1, 1))
        assert np.allclose(example.smean(), mean_compare)
        assert np.allclose(example.scovmat(), cov_compare)

    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    @pytest.mark.parametrize("r_anc", R_ANC_VALS)
    @pytest.mark.parametrize("eta_anc", ETA_ANC_VALS)
    def test_mbsqueeze(self, r, phi, r_anc, eta_anc):
        r"""Checks measurement-based squeezing."""
        example = circuit.BosonicModes()
        example.reset(2, 1)

        # Test zero mbsqueezing
        example.mbsqueeze(0, 0, 0, r_anc, eta_anc, True)
        assert example.is_vacuum(tol=1e-10)
        example.mbsqueeze(0, 0, 0, 5, 1, False)
        assert example.is_vacuum(tol=1e-10)

        # Test high-quality mbsqueezing
        example.mbsqueeze(0, r, phi, 9, 1, True)
        example.squeeze(r, phi, 1)
        assert np.allclose(example.scovmat()[:, 0:2, 0:2], example.scovmat()[:, 2:4, 2:4])
        assert np.allclose(example.smean()[:, 0:2], example.smean()[:, 2:4])
        example.mbsqueeze(0, r, phi, 9, 1, False)
        example.squeeze(r, phi, 1)
        assert np.allclose(example.scovmat()[:, 0:2, 0:2], example.scovmat()[:, 2:4, 2:4])

    @pytest.mark.parametrize("r", R_VALS)
    def test_complex_weight(self, r):
        r"""Checks a cat state has correct parity and that mean from homodyne sampling is 0."""
        # Make a cat state
        example = circuit.BosonicModes()
        example.reset(1, 4)

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
        r"""Checks linear optical unitary implementation of a phase shift."""
        # Create two displaced states
        example = circuit.BosonicModes()
        example.reset(2, 1)
        example.displace(r, 0, 0)
        example.displace(r, 0, 1)
        # Rotate them using different techniques
        example.apply_u(np.diag([np.exp(1j * phi), 1]))
        example.phase_shift(phi, 1)
        # Check they have the same means
        assert np.allclose(example.smean()[:, 0:2], example.smean()[:, 2:4])

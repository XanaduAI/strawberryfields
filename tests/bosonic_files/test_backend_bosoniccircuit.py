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
Unit tests for backends.bosonicbackend.ops file.
"""

import numpy as np
import strawberryfields.backends.bosonicbackend.bosoniccircuit as circuit
import pytest


class TestBosonicCircuit:
    def test_reset_circuit(self):
        # Create a state over 3 modes, 2 weights per mode
        example = circuit.BosonicModes()
        example.reset(3, 2)

        # Confirm number of modes and weights
        assert example.nlen == 3
        assert example._trunc == 2

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
        assert example.scovmat().shape == (2 ** 3, 6, 6)

        # Reset with a random number of weights and modes and perform the same check
        num_weights = np.random.randint(1, 5)
        num_modes = np.random.randint(1, 5)

        example.reset(num_modes, num_weights)
        assert example.nlen == num_modes
        assert example._trunc == num_weights
        assert np.isclose(sum(example.sweights()), 1)
        assert example.is_vacuum(tol=1e-10)
        for i in range(num_modes):
            assert np.isclose(example.fidelity_vacuum([i]), 1)
        assert example.active == [i for i in range(num_modes)]
        assert example.smean().shape == (num_weights ** num_modes, 2 * num_modes)
        assert example.scovmat().shape == (num_weights ** num_modes, 2 * num_modes, 2 * num_modes)

    def test_add_del_modes(self):
        # Create state with random number of weights and means
        num_weights = np.random.randint(1, 5)
        num_modes = np.random.randint(2, 5)
        example = circuit.BosonicModes()
        example.reset(num_modes, num_weights)
        # Delete first mode and check active modes changed
        example.del_mode(0)
        assert example.get_modes() == [i for i in range(1, num_modes)]

        # Add new mode with random number of weights
        num_weights_new = np.random.randint(1, 3)
        example.add_mode([num_weights_new])
        tot_weights = (num_weights ** num_modes) * num_weights_new
        # Check numbers of weights, means and covs
        assert example.get_modes() == [i for i in range(1, num_modes + 1)]
        assert np.isclose(sum(example.sweights()), 1)
        assert example.sweights().shape == (tot_weights,)
        assert example.smean().shape == (tot_weights, 2 * (num_modes + 1))
        assert example.scovmat().shape == (tot_weights, 2 * (num_modes + 1), 2 * (num_modes + 1))

    def test_displace_squeeze_rotate(self):
        # Ensure zero displacement, squeezing and rotation all yield vacuum
        example = circuit.BosonicModes()
        example.reset(1, 1)
        example.squeeze(0, 0, 0)
        assert example.is_vacuum(tol=1e-10)
        example.displace(0, 0, 0)
        assert example.is_vacuum(tol=1e-10)
        example.phase_shift(0, 0)
        assert example.is_vacuum(tol=1e-10)

        # Displace by random amount and check it is a coherent state
        r = np.random.rand()
        phi = np.random.rand()
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

    def test_from_cov_and_mean(self):
        example = circuit.BosonicModes()
        example.reset(1, 1)

        # Create a state with a given mean and cov
        r = np.random.rand()
        phi = np.random.rand()

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

    def test_parity(self):
        # Create state with random number of weights and means
        num_weights = np.random.randint(1, 5)
        num_modes = np.random.randint(1, 5)
        example = circuit.BosonicModes()
        example.reset(num_modes, num_weights)

        # Assert that it has parity of 1 (since it is vacuum)
        assert np.isclose(example.parity_val(), 1)
        for i in range(num_modes):
            assert np.isclose(example.parity_val([i]), 1)

    def test_losses(self):
        # Create state with random number of weights and means
        num_weights = np.random.randint(1, 5)
        example = circuit.BosonicModes()
        example.reset(3, num_weights)

        # Displace, apply total loss to first mode and check it is vacuum
        r = np.random.rand()
        phi = np.random.rand()
        example.displace(r, phi, 0)
        example.loss(0, 0)
        assert example.is_vacuum(tol=1e-10)

        # Apply thermal loss to second mode and check the covariance is as expected
        nbar = np.random.rand()
        example.thermal_loss(0, nbar, 1)
        assert np.allclose(
            np.tile((2 * nbar + 1) * np.eye(2), (num_weights ** 3, 1, 1)),
            example.scovmat()[:, 2:4, 2:4],
        )

        # Initialize third mode as thermal and check it's the same as second mode
        example.init_thermal(nbar, 2)
        assert np.allclose(example.scovmat()[:, 4:6, 4:6], example.scovmat()[:, 2:4, 2:4])

    def test_gaussian_symplectic(self):
        example = circuit.BosonicModes()
        example.reset(2, 1)

        # Create a symplectic and apply to first mode
        r = np.random.rand()
        phi = np.random.rand()

        Smat = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        Rmat = np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
        Rmat2 = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        symp = Rmat2 @ Rmat @ Smat @ Rmat.T
        symp = example.expandS([0], symp)
        example.apply_u(symp)

        # Apply the corresponding operations to second mode
        example.squeeze(r, phi, 1)
        example.phase_shift(phi, 1)

        # Check that same modes are in both states
        assert np.allclose(example.scovmat()[:, 0:2, 0:2], example.scovmat()[:, 2:4, 2:4])
        assert np.allclose(example.smean()[:, 0:2], example.smean()[:, 2:4])
        assert np.allclose(example.scovmatxp()[:, 0::2, 0::2], example.scovmatxp()[:, 1::2, 1::2])
        assert np.allclose(example.smeanxp()[:, 0::2], example.smeanxp()[:, 1::2])

    def test_mbsqueeze(self):
        # NOTE: mbsqueezing tests beamsplitter, and homodyne measurements

        example = circuit.BosonicModes()
        example.reset(2, 1)

        # Test zero mbsqueezing
        example.mbsqueeze(0, 0, 0, 5, 1, True)
        assert example.is_vacuum(tol=1e-10)
        example.mbsqueeze(0, 0, 0, 5, 1, False)
        assert example.is_vacuum(tol=1e-10)

        # Test high-quality mbsqueezing
        r = np.random.rand()
        phi = np.random.rand()
        example.mbsqueeze(0, r, phi, 9, 1, True)
        example.squeeze(r, phi, 1)
        assert np.allclose(example.scovmat()[:, 0:2, 0:2], example.scovmat()[:, 2:4, 2:4])
        assert np.allclose(example.smean()[:, 0:2], example.smean()[:, 2:4])
        example.mbsqueeze(0, r, phi, 9, 1, False)
        example.squeeze(r, phi, 1)
        assert np.allclose(example.scovmat()[:, 0:2, 0:2], example.scovmat()[:, 2:4, 2:4])

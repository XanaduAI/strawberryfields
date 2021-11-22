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
r"""Unit tests for the states.py submodule"""
import pytest

import numpy as np
from scipy.special import factorial as fac

from strawberryfields import backends
from strawberryfields import utils


a = 0.3 + 0.1j
r = 0.23
phi = 0.123


class TestBackendStateCreation:
    """Test the backends properly create states"""

    def test_full_state_creation(self, hbar, cutoff, setup_backend):
        """Test backend returns a properly formed state object"""
        backend = setup_backend(3)
        state = backend.state(modes=None)

        assert state.num_modes == 3
        assert state.hbar == hbar
        assert state.mode_names == {0: "q[0]", 1: "q[1]", 2: "q[2]"}
        assert state.mode_indices == {"q[0]": 0, "q[1]": 1, "q[2]": 2}

        if isinstance(backend, backends.BaseFock):
            assert state.cutoff_dim == cutoff

    def test_reduced_state_creation(self, setup_backend):
        """Test backend returns a properly formed reduced state object"""
        backend = setup_backend(3)
        state = backend.state(modes=[0, 2])

        assert state.num_modes == 2
        assert state.mode_names == {0: "q[0]", 1: "q[2]"}
        assert state.mode_indices == {"q[0]": 0, "q[2]": 1}

    def test_reduced_state_fidelity(self, setup_backend, tol):
        """Test backend calculates correct fidelity of reduced coherent state"""
        backend = setup_backend(2)

        backend.prepare_coherent_state(np.abs(a), np.angle(a), 0)
        backend.prepare_squeezed_state(r, phi, 1)

        state = backend.state(modes=[0])
        f = state.fidelity_coherent([a])
        assert np.allclose(f, 1, atol=tol, rtol=0)

    def test_reduced_state_fock_probs(self, cutoff, setup_backend, batch_size, tol):
        """Test backend calculates correct fock prob of reduced coherent state"""
        backend = setup_backend(2)

        backend.prepare_coherent_state(np.abs(a), np.angle(a), 0)
        backend.prepare_squeezed_state(r, phi, 1)

        state = backend.state(modes=[0])
        probs = np.array([state.fock_prob([i]) for i in range(cutoff)]).T

        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(a) ** 2) * a ** n / np.sqrt(fac(n))
        ref_probs = np.abs(ref_state) ** 2

        if batch_size is not None:
            ref_probs = np.tile(ref_probs, batch_size)

        assert np.allclose(probs.flatten(), ref_probs.flatten(), atol=tol, rtol=0)


class TestBaseStateMeanPhotonNumber:
    """Tests for the mean photon number method"""

    def test_mean_photon_coherent(self, setup_backend, tol, batch_size):
        """Test that E(n) = |a|^2 and var(n) = |a|^2 for a coherent state"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(1)

        backend.displacement(np.abs(a), np.angle(a), 0)
        state = backend.state()
        mean_photon, var = state.mean_photon(0)

        assert np.allclose(mean_photon, np.abs(a) ** 2, atol=tol, rtol=0)
        assert np.allclose(var, np.abs(a) ** 2, atol=tol, rtol=0)

    def test_mean_photon_squeezed(self, setup_backend, tol, batch_size):
        """Test that E(n)=sinh^2(r) and var(n)=2(sinh^2(r)+sinh^4(r)) for a squeezed state"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(1)

        r = 0.1
        a = 0.3 + 0.1j

        backend.squeeze(r, phi, 0)
        state = backend.state()
        mean_photon, var = state.mean_photon(0)

        assert np.allclose(mean_photon, np.sinh(r) ** 2, atol=tol, rtol=0)
        assert np.allclose(var, 2 * (np.sinh(r) ** 2 + np.sinh(r) ** 4), atol=tol, rtol=0)

    def test_mean_photon_displaced_squeezed(self, setup_backend, tol, batch_size):
        """Test that E(n) = sinh^2(r)+|a|^2 for a displaced squeezed state"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(1)

        nbar = 0.123
        a = 0.12 - 0.05j
        r = 0.195

        backend.squeeze(r, phi, 0)
        backend.displacement(np.abs(a), np.angle(a), 0)
        state = backend.state()
        mean_photon, var = state.mean_photon(0)

        mag_a = np.abs(a)
        phi_a = np.angle(a)

        magnitude_squared = np.abs(a) ** 2

        mean_ex = magnitude_squared + np.sinh(r) ** 2
        var_ex = (
            -magnitude_squared
            + magnitude_squared ** 2
            + 2 * magnitude_squared * np.cosh(2 * r)
            - np.exp(-1j * phi) * a ** 2 * np.cosh(r) * np.sinh(r)
            - np.exp(1j * phi) * np.conj(a) ** 2 * np.cosh(r) * np.sinh(r)
            + np.sinh(r) ** 4
            - (magnitude_squared + np.conj(np.sinh(r)) * np.sinh(r)) ** 2
            + np.cosh(r) * np.sinh(r) * np.sinh(2 * r)
        )

        assert np.allclose(mean_photon, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)

    def test_mean_photon_displaced_thermal(self, setup_backend, tol, batch_size):
        """Test that E(n)=|a|^2+nbar and var(n)=var_th+|a|^2(1+2nbar)"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(1)

        nbar = 0.123

        backend.prepare_thermal_state(nbar, 0)
        backend.displacement(np.abs(a), np.angle(a), 0)
        state = backend.state()
        mean_photon, var = state.mean_photon(0)

        mean_ex = np.abs(a) ** 2 + nbar
        var_ex = nbar ** 2 + nbar + np.abs(a) ** 2 * (1 + 2 * nbar)
        assert np.allclose(mean_photon, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)


@pytest.mark.backends("fock", "tf", "gaussian")
class TestBaseFockKetDensityMatrix:
    """Tests for the ket, dm, and reduced density matrix function."""

    def test_rdm(self, setup_backend, tol, cutoff, batch_size):
        """Test reduced density matrix of a coherent state is as expected
        This is supported by all backends, since it returns
        the reduced density matrix of a single mode."""
        backend = setup_backend(2)
        backend.prepare_coherent_state(np.abs(a), np.angle(a), 0)
        backend.prepare_coherent_state(0.1, 0, 1)

        state = backend.state()
        rdm = state.reduced_dm(0, cutoff=cutoff)

        n = np.arange(cutoff)
        ket = np.exp(-0.5 * np.abs(a) ** 2) * a ** n / np.sqrt(fac(n))
        rdm_exact = np.outer(ket, ket.conj())

        if batch_size is not None:
            np.tile(rdm_exact, [batch_size, 1])

        assert np.allclose(rdm, rdm_exact, atol=tol, rtol=0)

    def test_ket(self, setup_backend, pure, cutoff, batch_size, tol):
        """Test that the ket of a displaced state matches analytic result"""
        backend = setup_backend(2)
        backend.displacement(np.abs(a), np.angle(a), 0)

        state = backend.state()
        if not pure and backend.short_name != "gaussian":
            assert state.is_pure == False
            pytest.skip("Test only works with pure states.")

        assert state.is_pure == True

        ket = np.sum(state.ket(cutoff=cutoff), axis=-1)

        n = np.arange(cutoff)
        expected = np.exp(-0.5 * np.abs(a) ** 2) * a ** n / np.sqrt(fac(n))

        if batch_size is not None:
            ket = np.tile(ket, [batch_size, 1])

        assert np.allclose(ket, expected, atol=tol, rtol=0)

    def test_density_matrix_thermal_state(self, setup_backend, cutoff, batch_size, tol):
        """Test that a thermal state returns the correct density matrix, using
        both the dm() and reduced_dm() methods."""
        backend = setup_backend(1)
        backend.prepare_thermal_state(r, 0)

        state = backend.state()
        assert not state.is_pure

        rho1 = state.dm(cutoff=cutoff)
        rho2 = state.reduced_dm(0, cutoff=cutoff)

        assert np.allclose(rho1, rho2, atol=tol, rtol=0)

        n = np.arange(cutoff)
        expected = np.diag((r ** n) / ((1 + r) ** (n + 1)))

        if batch_size is not None:
            expected = np.tile(expected, [batch_size, 1]).reshape(-1, cutoff, cutoff)

        assert np.allclose(rho1, expected, atol=tol, rtol=0)


@pytest.mark.backends("gaussian")
class TestBaseGaussianMethods:
    """This tests state methods unique to the BaseGaussian
    class, including is_coherent, displacement, is_squeezed,
    and squeezing."""

    def test_coherent_methods(self, setup_backend, tol):
        """Test that the ket of a displaced state matches analytic result"""
        backend = setup_backend(2)

        a = 1 + 0.5j
        r = 2
        phi = -0.5

        backend.prepare_coherent_state(np.abs(a), np.angle(a), 0)
        backend.prepare_squeezed_state(r, phi, 1)

        state = backend.state()

        coherent_check = []
        for i in range(2):
            coherent_check.append(state.is_coherent(i))

        alpha_list = state.displacement()

        assert np.all(coherent_check == [True, False])
        assert np.allclose(alpha_list, [a, 0.0], atol=tol, rtol=0)

    def test_squeezing_methods(self, setup_backend, tol):
        """Test that the ket of a displaced state matches analytic result"""
        backend = setup_backend(2)

        a = 1 + 0.5j
        r = 2
        phi = -0.5

        backend.prepare_coherent_state(np.abs(a), np.angle(a), 0)
        backend.prepare_squeezed_state(r, phi, 1)

        state = backend.state()

        squeezing_check = []
        for i in range(2):
            squeezing_check.append(state.is_squeezed(i))

        z_list = np.array(state.squeezing())

        assert np.all(squeezing_check == [False, True])
        assert np.allclose(z_list, [[0.0, 0.0], [r, phi]], atol=tol, rtol=0)


class TestQuadratureExpectations:
    """Test quad_expectation methods"""

    def test_vacuum(self, setup_backend, hbar, batch_size, tol):
        """Test vacuum state has zero mean and hbar/2 variance"""
        backend = setup_backend(1)
        state = backend.state()
        res = np.array(state.quad_expectation(0, phi=np.pi / 4)).T

        res_exact = np.array([0, hbar / 2.0])

        if batch_size is not None:
            res_exact = np.tile(res_exact, batch_size)

        assert np.allclose(res.flatten(), res_exact.flatten(), atol=tol, rtol=0)

    def test_squeezed_coherent(self, setup_backend, hbar, batch_size, tol):
        """Test squeezed coherent state has correct mean and variance"""
        # quadrature rotation angle
        backend = setup_backend(1)
        qphi = 0.78

        backend.prepare_displaced_squeezed_state(np.abs(a), np.angle(a), r, phi, 0)

        state = backend.state()
        res = np.array(state.quad_expectation(0, phi=qphi)).T

        xphi_mean = (a.real * np.cos(qphi) + a.imag * np.sin(qphi)) * np.sqrt(2 * hbar)
        xphi_var = (np.cosh(2 * r) - np.cos(phi - 2 * qphi) * np.sinh(2 * r)) * hbar / 2
        res_exact = np.array([xphi_mean, xphi_var])

        if batch_size is not None:
            res_exact = np.tile(res_exact, batch_size)

        assert np.allclose(res.flatten(), res_exact.flatten(), atol=tol, rtol=0)


@pytest.mark.backends("fock", "tf", "gaussian")
class TestNumberExpectation:
    """Multimode photon-number expectation value tests"""

    def test_number_expectation_vacuum(self, setup_backend, tol, batch_size):
        """Tests the expectation value of any photon number in vacuum is zero,
        and the variance is also 0."""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(2)
        state = backend.state()

        expected = (0, 0)
        assert np.allclose(state.number_expectation([0, 1]), expected, atol=tol, rtol=0)
        assert np.allclose(state.number_expectation([0]), expected, atol=tol, rtol=0)
        assert np.allclose(state.number_expectation([1]), expected, atol=tol, rtol=0)

    def test_number_expectation_displaced_squeezed(self, setup_backend, tol, batch_size):
        """Tests the expectation value of photon numbers when there is no correlation"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(2)
        state = backend.state()
        a0 = 0.2 + 0.1 * 1j
        r0 = 0.2
        phi0 = 0.6
        a1 = 0.1 + 0.1 * 1j
        r1 = 0.1
        phi1 = 0.4
        backend.prepare_displaced_squeezed_state(np.abs(a0), np.angle(a0), r0, phi0, 0)
        backend.prepare_displaced_squeezed_state(np.abs(a1), np.angle(a1), r1, phi1, 1)
        state = backend.state()
        n0 = np.sinh(r0) ** 2 + np.abs(a0) ** 2
        n1 = np.sinh(r1) ** 2 + np.abs(a1) ** 2

        def squared_term(a, r, phi):
            magnitude_squared = np.abs(a) ** 2
            squared_term = (
                -magnitude_squared
                + magnitude_squared ** 2
                + 2 * magnitude_squared * np.cosh(2 * r)
                - 2 * np.real(np.exp(-1j * phi) * a ** 2 * np.cosh(r) * np.sinh(r))
                + np.sinh(r) ** 4
                + np.cosh(r) * np.sinh(r) * np.sinh(2 * r)
            )
            return squared_term

        res = state.number_expectation([0, 1])
        var = squared_term(a0, r0, phi0) * squared_term(a1, r1, phi1) - n0 ** 2 * n1 ** 2
        assert np.allclose(res[0], n0 * n1, atol=tol, rtol=0)
        assert np.allclose(res[1], var, atol=tol, rtol=0)

        res = state.number_expectation([0])
        var = squared_term(a0, r0, phi0) - n0 ** 2
        assert np.allclose(res[0], n0, atol=tol, rtol=0)
        assert np.allclose(res[1], var, atol=tol, rtol=0)

        res = state.number_expectation([1])
        var = squared_term(a1, r1, phi1) - n1 ** 2
        assert np.allclose(res[0], n1, atol=tol, rtol=0)
        assert np.allclose(res[1], var, atol=tol, rtol=0)

    def test_number_expectation_repeated_modes(self, setup_backend, tol):
        """Tests that the correct exception is raised for repeated modes"""
        backend = setup_backend(2)
        state = backend.state()
        with pytest.raises(ValueError, match="There can be no duplicates in the modes specified."):
            state.number_expectation([0, 0])

    def test_number_expectation_two_mode_squeezed(self, setup_backend, tol, batch_size):
        """Tests the expectation value of photon numbers when there is correlation"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(3)
        state = backend.state()
        r = 0.2
        phi = 0.0
        backend.prepare_squeezed_state(r, phi, 0)
        backend.prepare_squeezed_state(-r, phi, 2)
        backend.beamsplitter(np.pi / 4, np.pi, 0, 2)
        state = backend.state()
        nbar = np.sinh(r) ** 2

        res = state.number_expectation([2, 0])
        assert np.allclose(res[0], 2 * nbar ** 2 + nbar, atol=tol, rtol=0)

        res = state.number_expectation([0])
        assert np.allclose(res[0], nbar, atol=tol, rtol=0)

        res = state.number_expectation([2])
        assert np.allclose(res[0], nbar, atol=tol, rtol=0)

    def test_number_expectation_photon_displaced_thermal(self, setup_backend, tol, batch_size):
        """Test that E(n)=|a|^2+nbar and var(n)=var_th+|a|^2(1+2nbar)
        for uncorrelated displaced thermal states."""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(2)

        a0 = 0.1
        nbar0 = 0.123

        a1 = 0.05
        nbar1 = 0.21

        backend.prepare_thermal_state(nbar0, 0)
        backend.displacement(a0, 0.0, 0)
        backend.prepare_thermal_state(nbar1, 1)
        backend.displacement(a1, 0.0, 1)
        state = backend.state()

        res = state.number_expectation([0])

        mean0 = np.abs(a0) ** 2 + nbar0
        var0 = nbar0 ** 2 + nbar0 + np.abs(a0) ** 2 * (1 + 2 * nbar0)
        assert np.allclose(res[0], mean0, atol=tol, rtol=0)
        assert np.allclose(res[1], var0, atol=tol, rtol=0.01)

        res = state.number_expectation([1])

        mean1 = np.abs(a1) ** 2 + nbar1
        var1 = nbar1 ** 2 + nbar1 + np.abs(a1) ** 2 * (1 + 2 * nbar1)
        assert np.allclose(res[0], mean1, atol=tol, rtol=0)
        assert np.allclose(res[1], var1, atol=tol, rtol=0.01)

        # test uncorrelated result
        res = state.number_expectation([0, 1])
        expected_mean = mean0 * mean1
        assert np.allclose(res[0], expected_mean, atol=tol, rtol=0)

        expected_var = mean0 ** 2 * var1 + mean1 ** 2 * var0 + var0 * var1
        assert np.allclose(res[1], expected_var, atol=tol, rtol=0.01)

    @pytest.mark.backends("fock", "tf")
    def test_number_expectation_four_modes(self, setup_backend, tol, batch_size):
        """Tests the expectation value of photon numbers when there is correlation"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(4)
        state = backend.state()
        r = 0.2
        phi = 0.0
        backend.prepare_squeezed_state(r, phi, 0)
        backend.prepare_squeezed_state(r, phi + np.pi, 1)
        backend.beamsplitter(np.pi / 4, np.pi, 0, 1)
        backend.prepare_squeezed_state(r, phi, 2)
        backend.prepare_squeezed_state(r, phi + np.pi, 3)
        backend.beamsplitter(np.pi / 4, np.pi, 2, 3)

        state = backend.state()
        nbar = np.sinh(r) ** 2
        assert np.allclose(
            state.number_expectation([0, 1, 2, 3])[0],
            (2 * nbar ** 2 + nbar) ** 2,
            atol=tol,
            rtol=0,
        )
        assert np.allclose(
            state.number_expectation([0, 1, 3])[0],
            nbar * (2 * nbar ** 2 + nbar),
            atol=tol,
            rtol=0,
        )
        assert np.allclose(
            state.number_expectation([3, 1, 2])[0],
            nbar * (2 * nbar ** 2 + nbar),
            atol=tol,
            rtol=0,
        )


class TestParityExpectation:
    @pytest.mark.backends("fock", "tf")
    def test_parity_fock(self, setup_backend, tol, batch_size):
        """Tests the parity operator for an even superposition of the first two number states"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(2)
        state = backend.state()
        n1 = 3
        n2 = 2
        backend.prepare_fock_state(n1, 0)
        backend.prepare_fock_state(n2, 1)
        backend.beamsplitter(np.pi / 4, 0, 0, 1)
        state = backend.state()

        assert np.allclose(state.parity_expectation([0]), 0, atol=tol, rtol=0)

    @pytest.mark.backends("fock", "tf")
    def test_two_mode_fock(self, setup_backend, tol, batch_size):
        """Tests the product of parity operators for two number states"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(2)
        state = backend.state()
        n1 = 3
        n2 = 5
        backend.prepare_fock_state(n1, 0)
        backend.prepare_fock_state(n2, 1)
        state = backend.state()

        assert np.allclose(state.parity_expectation([0, 1]), 1, atol=tol, rtol=0)

    def test_coherent(self, setup_backend, tol, batch_size):
        """Tests the parity operator for a coherent state"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(1)
        state = backend.state()
        r = 0.2
        backend.prepare_coherent_state(r, 0, 0)
        state = backend.state()

        assert np.allclose(
            state.parity_expectation([0]), np.exp(-2 * (np.abs(r) ** 2)), atol=tol, rtol=0
        )

    def test_squeezed(self, setup_backend, tol, batch_size):
        """Tests the parity operator for a squeezed state"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(1)
        state = backend.state()
        r = 0.2
        phi = 0
        backend.prepare_squeezed_state(r, phi, 0)
        state = backend.state()

        assert np.allclose(state.parity_expectation([0]), 1, atol=tol, rtol=0)

    def test_two_mode_squeezed(self, setup_backend, tol, batch_size):
        """Tests the parity operator for a two-mode squeezed state"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(2)
        state = backend.state()
        r = 0.2
        phi = 0
        backend.beamsplitter(np.sqrt(0.5), -np.sqrt(0.5), 0, 1)
        backend.prepare_squeezed_state(r, phi, 0)
        backend.prepare_squeezed_state(-1 * r, phi, 1)
        backend.beamsplitter(np.sqrt(0.5), -np.sqrt(0.5), 0, 1)
        state = backend.state()

        assert np.allclose(state.parity_expectation([0, 1]), 1, atol=tol, rtol=0)

    @pytest.mark.backends("fock", "tf")
    def test_thermal(self, setup_backend, tol, batch_size):
        """Tests the parity operator for a thermal state"""
        if batch_size is not None:
            pytest.skip("Does not support batch mode")
        backend = setup_backend(1)
        state = backend.state()
        m = 0.2
        backend.prepare_thermal_state(m, 0)
        state = backend.state()

        assert np.allclose(state.parity_expectation([0]), (1 / ((2 * m) + 1)), atol=tol, rtol=0)


@pytest.mark.backends("fock", "tf", "gaussian")
class TestFidelities:
    """Fidelity tests."""

    def test_vacuum(self, setup_backend, tol):
        backend = setup_backend(2)
        state = backend.state()
        assert np.allclose(state.fidelity_vacuum(), 1, atol=tol, rtol=0)

    def test_coherent_fidelity(self, setup_backend, cutoff, tol, hbar):
        backend = setup_backend(2)
        backend.prepare_coherent_state(np.abs(a), np.angle(a), 0)
        backend.displacement(np.abs(a), np.angle(a), 1)
        state = backend.state()

        if isinstance(backend, backends.BaseFock):
            in_state = utils.coherent_state(
                np.abs(a), np.angle(a), basis="fock", fock_dim=cutoff, hbar=hbar
            )
        else:
            in_state = utils.coherent_state(np.abs(a), np.angle(a), basis="gaussian", hbar=hbar)

        assert np.allclose(state.fidelity(in_state, 0), 1, atol=tol, rtol=0)
        assert np.allclose(state.fidelity(in_state, 1), 1, atol=tol, rtol=0)
        assert np.allclose(state.fidelity_coherent([a, a]), 1, atol=tol, rtol=0)

    def test_squeezed_fidelity(self, setup_backend, cutoff, tol, hbar):
        backend = setup_backend(2)
        backend.prepare_squeezed_state(r, phi, 0)
        backend.squeeze(r, phi, 1)
        state = backend.state()

        if isinstance(backend, backends.BaseFock):
            in_state = utils.squeezed_state(r, phi, basis="fock", fock_dim=cutoff, hbar=hbar)
        else:
            in_state = utils.squeezed_state(r, phi, basis="gaussian", hbar=hbar)

        assert np.allclose(state.fidelity(in_state, 0), 1, atol=tol, rtol=0)
        assert np.allclose(state.fidelity(in_state, 1), 1, atol=tol, rtol=0)

    def test_squeezed_coherent_fidelity(self, setup_backend, cutoff, tol, hbar):
        backend = setup_backend(2)
        backend.prepare_displaced_squeezed_state(np.abs(a), np.angle(a), r, phi, 0)
        backend.squeeze(r, phi, 1)
        backend.displacement(np.abs(a), np.angle(a), 1)
        state = backend.state()

        if isinstance(backend, backends.BaseFock):
            in_state = utils.displaced_squeezed_state(
                np.abs(a), np.angle(a), r, phi, basis="fock", fock_dim=cutoff, hbar=hbar
            )
        else:
            in_state = utils.displaced_squeezed_state(
                np.abs(a), np.angle(a), r, phi, basis="gaussian", hbar=hbar
            )

        assert np.allclose(state.fidelity(in_state, 0), 1, atol=tol, rtol=0)
        assert np.allclose(state.fidelity(in_state, 1), 1, atol=tol, rtol=0)


@pytest.mark.backends("bosonic")
class TestBosonicState:
    """Tests the bosonic state class."""

    def test_weights(self, setup_backend):
        """Test weights are correct for a Gaussian state represented using the bosonic backend."""
        backend = setup_backend(1)
        backend.prepare_coherent_state(1, 0, 0)
        weights = backend.state().weights()
        assert len(weights) == 1
        assert weights == np.array([1])
        assert sum(weights) == 1

    def test_bosonic_purity(self, setup_backend, tol):
        """Test purities are correct for Gaussian states represented using the bosonic backend."""
        backend = setup_backend(1)
        backend.prepare_coherent_state(r, phi, 0)
        purity1 = backend.state().purity()
        backend.reset()
        backend.prepare_thermal_state(1, 0)
        purity2 = backend.state().purity()
        assert np.allclose(purity1, 1, atol=tol, rtol=0)
        assert purity2 < 1

    def test_displacement(self, setup_backend, tol):
        """Tests average displacement calculation."""
        backend = setup_backend(1)
        backend.prepare_coherent_state(r, phi, 0)
        disp = backend.state().displacement(0)
        assert np.allclose(r * np.cos(phi) + 1j * r * np.sin(phi), disp)

    def test_density_matrix(self, setup_backend, tol):
        """Tests vacuum density matrix."""
        backend = setup_backend(1)
        backend.prepare_vacuum_state(0)
        dm = backend.state().dm()
        dm_compare = np.zeros((10, 10))
        dm_compare[0, 0] = 1
        assert np.allclose(dm, dm_compare)

    def test_is_vacuum(self, setup_backend):
        """Tests fidelity_vacuum method in BaseBosonicState."""
        backend = setup_backend(1)
        backend.prepare_vacuum_state(0)
        assert np.allclose(backend.state().fidelity_vacuum(), 1)
        backend.del_mode(0)
        assert np.allclose(backend.state().fidelity_vacuum(), 1)

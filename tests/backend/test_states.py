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

        backend.prepare_coherent_state(a, 0)
        backend.prepare_squeezed_state(r, phi, 1)

        state = backend.state(modes=[0])
        f = state.fidelity_coherent([a])
        assert np.allclose(f, 1, atol=tol, rtol=0)

    def test_reduced_state_fock_probs(self, cutoff, setup_backend, batch_size, tol):
        """Test backend calculates correct fock prob of reduced coherent state"""
        backend = setup_backend(2)

        backend.prepare_coherent_state(a, 0)
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

    def test_mean_photon_coherent(self, setup_backend, tol):
        """Test that E(n) = |a|^2 and var(n) = |a|^2 for a coherent state"""
        backend = setup_backend(1)

        backend.displacement(a, 0)
        state = backend.state()
        mean_photon, var = state.mean_photon(0)

        assert np.allclose(mean_photon, np.abs(a) ** 2, atol=tol, rtol=0)
        assert np.allclose(var, np.abs(a) ** 2, atol=tol, rtol=0)

    def test_mean_photon_squeezed(self, setup_backend, tol):
        """Test that E(n)=sinh^2(r) and var(n)=2(sinh^2(r)+sinh^4(r)) for a squeezed state"""
        backend = setup_backend(1)

        r = 0.1
        a = 0.3 + 0.1j

        backend.squeeze(r * np.exp(1j * phi), 0)
        state = backend.state()
        mean_photon, var = state.mean_photon(0)

        assert np.allclose(mean_photon, np.sinh(r) ** 2, atol=tol, rtol=0)
        assert np.allclose(
            var, 2 * (np.sinh(r) ** 2 + np.sinh(r) ** 4), atol=tol, rtol=0
        )

    def test_mean_photon_displaced_squeezed(self, setup_backend, tol):
        """Test that E(n) = sinh^2(r)+|a|^2 for a displaced squeezed state"""
        backend = setup_backend(1)

        nbar = 0.123
        a = 0.12 - 0.05j
        r = 0.195

        backend.squeeze(r * np.exp(1j * phi), 0)
        backend.displacement(a, 0)
        state = backend.state()
        mean_photon, var = state.mean_photon(0)

        mag_a = np.abs(a)
        phi_a = np.angle(a)

        mean_ex = np.abs(a) ** 2 + np.sinh(r) ** 2
        assert np.allclose(mean_photon, mean_ex, atol=tol, rtol=0)

    def test_mean_photon_displaced_thermal(self, setup_backend, tol):
        """Test that E(n)=|a|^2+nbar and var(n)=var_th+|a|^2(1+2nbar)"""
        backend = setup_backend(1)

        nbar = 0.123

        backend.prepare_thermal_state(nbar, 0)
        backend.displacement(a, 0)
        state = backend.state()
        mean_photon, var = state.mean_photon(0)

        mean_ex = np.abs(a) ** 2 + nbar
        var_ex = nbar ** 2 + nbar + np.abs(a) ** 2 * (1 + 2 * nbar)
        assert np.allclose(mean_photon, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)


class TestBaseFockKetDensityMatrix:
    """Tests for the ket, dm, and reduced density matrix function."""

    def test_rdm(self, setup_backend, tol, cutoff, batch_size):
        """Test reduced density matrix of a coherent state is as expected
        This is supported by all backends, since it returns
        the reduced density matrix of a single mode."""
        backend = setup_backend(2)
        backend.prepare_coherent_state(a, 0)
        backend.prepare_coherent_state(0.1, 1)

        state = backend.state()
        rdm = state.reduced_dm(0, cutoff=cutoff)

        n = np.arange(cutoff)
        ket = np.exp(-0.5 * np.abs(a) ** 2) * a ** n / np.sqrt(fac(n))
        rdm_exact = np.outer(ket, ket.conj())

        if batch_size is not None:
            np.tile(rdm_exact, [batch_size, 1])

        assert np.allclose(rdm, rdm_exact, atol=tol, rtol=0)

    @pytest.mark.backends("fock", "tf")
    def test_ket(self, setup_backend, pure, cutoff, batch_size, tol):
        """Test that the ket of a displaced state matches analytic result"""
        backend = setup_backend(2)
        backend.displacement(a, 0)

        state = backend.state()
        if not pure:
            assert state.is_pure == False
            pytest.skip("Test only works with pure states.")

        assert state.is_pure == True

        ket = np.sum(state.ket(), axis=-1)

        n = np.arange(cutoff)
        expected = np.exp(-0.5 * np.abs(a) ** 2) * a ** n / np.sqrt(fac(n))

        if batch_size is not None:
            ket = np.tile(ket, [batch_size, 1])

        assert np.allclose(ket, expected, atol=tol, rtol=0)


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

        backend.prepare_coherent_state(a, 0)
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

        backend.prepare_coherent_state(a, 0)
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

        backend.prepare_displaced_squeezed_state(a, r, phi, 0)

        state = backend.state()
        res = np.array(state.quad_expectation(0, phi=qphi)).T

        xphi_mean = (a.real * np.cos(qphi) + a.imag * np.sin(qphi)) * np.sqrt(2 * hbar)
        xphi_var = (np.cosh(2 * r) - np.cos(phi - 2 * qphi) * np.sinh(2 * r)) * hbar / 2
        res_exact = np.array([xphi_mean, xphi_var])

        if batch_size is not None:
            res_exact = np.tile(res_exact, batch_size)

        assert np.allclose(res.flatten(), res_exact.flatten(), atol=tol, rtol=0)


class TestFidelities:
    """Fidelity tests."""

    def test_vacuum(self, setup_backend, tol):
        backend = setup_backend(2)
        state = backend.state()
        assert np.allclose(state.fidelity_vacuum(), 1, atol=tol, rtol=0)

    def test_coherent_fidelity(self, setup_backend, cutoff, tol, hbar):
        backend = setup_backend(2)
        backend.prepare_coherent_state(a, 0)
        backend.displacement(a, 1)
        state = backend.state()

        if isinstance(backend, backends.BaseFock):
            in_state = utils.coherent_state(a, basis="fock", fock_dim=cutoff, hbar=hbar)
        else:
            in_state = utils.coherent_state(a, basis="gaussian", hbar=hbar)

        assert np.allclose(state.fidelity(in_state, 0), 1, atol=tol, rtol=0)
        assert np.allclose(state.fidelity(in_state, 1), 1, atol=tol, rtol=0)
        assert np.allclose(state.fidelity_coherent([a, a]), 1, atol=tol, rtol=0)

    def test_squeezed_fidelity(self, setup_backend, cutoff, tol, hbar):
        backend = setup_backend(2)
        backend.prepare_squeezed_state(r, phi, 0)
        backend.squeeze(r * np.exp(1j * phi), 1)
        state = backend.state()

        if isinstance(backend, backends.BaseFock):
            in_state = utils.squeezed_state(r, phi, basis="fock", fock_dim=cutoff, hbar=hbar)
        else:
            in_state = utils.squeezed_state(r, phi, basis="gaussian", hbar=hbar)

        assert np.allclose(state.fidelity(in_state, 0), 1, atol=tol, rtol=0)
        assert np.allclose(state.fidelity(in_state, 1), 1, atol=tol, rtol=0)

    def test_squeezed_coherent_fidelity(self, setup_backend, cutoff, tol, hbar):
        backend = setup_backend(2)
        backend.prepare_displaced_squeezed_state(a, r, phi, 0)
        backend.squeeze(r * np.exp(1j * phi), 1)
        backend.displacement(a, 1)
        state = backend.state()

        if isinstance(backend, backends.BaseFock):
            in_state = utils.displaced_squeezed_state(
                a, r, phi, basis="fock", fock_dim=cutoff, hbar=hbar
            )
        else:
            in_state = utils.displaced_squeezed_state(a, r, phi, basis="gaussian", hbar=hbar)

        assert np.allclose(state.fidelity(in_state, 0), 1, atol=tol, rtol=0)
        assert np.allclose(state.fidelity(in_state, 1), 1, atol=tol, rtol=0)

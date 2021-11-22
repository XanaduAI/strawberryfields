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
r"""Unit tests for the Strawberry Fields utils.py module"""
import pytest

pytestmark = pytest.mark.frontend

import numpy as np
from numpy.polynomial.hermite import hermval as H
from scipy.special import factorial as fac

import strawberryfields as sf
from strawberryfields.program_utils import RegRef
from strawberryfields.ops import Sgate, BSgate, LossChannel, MeasureX, Squeezed
import strawberryfields.utils as utils


R_D = np.linspace(-2.00562, 1.0198, 4)
PHI_D = np.linspace(-1.64566, 1.3734, 4)
PHI = np.linspace(0, 1.43, 4)
R = np.linspace(0, 0.21, 4)
PHI = np.linspace(0, 1.43, 4)


# ===================================================================================
# Convert function tests
# ===================================================================================


@pytest.fixture
def rr():
    """RegRef fixture."""
    return RegRef(0)


@pytest.fixture
def prog():
    """Program fixture."""
    return sf.Program(2)


# ===================================================================================
# Initial states tests
# ===================================================================================


class TestInitialStates:
    """unit tests for the initial state function utilities"""

    @pytest.mark.parametrize("r, phi", zip(R, PHI))
    def test_squeezed_cov(self, hbar, r, phi, tol):
        """test squeezed covariance utility function returns correct covariance"""
        cov = utils.squeezed_cov(r, phi, hbar=hbar)

        expected = (hbar / 2) * np.array(
            [
                [
                    np.cosh(2 * r) - np.cos(phi) * np.sinh(2 * r),
                    -2 * np.cosh(r) * np.sin(phi) * np.sinh(r),
                ],
                [
                    -2 * np.cosh(r) * np.sin(phi) * np.sinh(r),
                    np.cosh(2 * r) + np.cos(phi) * np.sinh(2 * r),
                ],
            ]
        )

        assert np.allclose(cov, expected, atol=tol, rtol=0)

    def test_vacuum_state_gaussian(self, hbar):
        """test vacuum state returns correct means and covariance"""
        means, cov = utils.vacuum_state(basis="gaussian", hbar=hbar)
        assert np.all(means == np.zeros(2))
        assert np.all(cov == np.identity(2) * hbar / 2)

    def test_vacuum_state_fock(self, cutoff, hbar):
        """test vacuum state returns correct state vector"""
        state = utils.vacuum_state(basis="fock", hbar=hbar, fock_dim=cutoff)
        assert np.all(state == np.eye(1, cutoff, 0))

    @pytest.mark.parametrize("r_d, phi_d", zip(R_D, PHI_D))
    def test_coherent_state_gaussian(self, r_d, phi_d, hbar):
        """test coherent state returns correct means and covariance"""
        means, cov = utils.coherent_state(r_d, phi_d, basis="gaussian", hbar=hbar)
        alpha = r_d * np.exp(1j * phi_d)
        means_expected = np.array([alpha.real, alpha.imag]) * np.sqrt(2 * hbar)
        assert np.all(means == means_expected)
        assert np.all(cov == np.identity(2) * hbar / 2)

    @pytest.mark.parametrize("r_d, phi_d", zip(R_D, PHI_D))
    def test_coherent_state_fock(self, r_d, phi_d, cutoff, hbar, tol):
        """test coherent state returns correct Fock basis state vector"""
        state = utils.coherent_state(r_d, phi_d, basis="fock", fock_dim=cutoff, hbar=hbar)
        n = np.arange(cutoff)
        alpha = r_d * np.exp(1j * phi_d)
        expected = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(fac(n))
        assert np.allclose(state, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("r, phi", zip(R, PHI))
    def test_squeezed_state_gaussian(self, r, phi, hbar, tol):
        """test squeezed state returns correct means and covariance"""
        means, cov = utils.squeezed_state(r, phi, basis="gaussian", hbar=hbar)

        cov_expected = (hbar / 2) * np.array(
            [
                [
                    np.cosh(2 * r) - np.cos(phi) * np.sinh(2 * r),
                    -2 * np.cosh(r) * np.sin(phi) * np.sinh(r),
                ],
                [
                    -2 * np.cosh(r) * np.sin(phi) * np.sinh(r),
                    np.cosh(2 * r) + np.cos(phi) * np.sinh(2 * r),
                ],
            ]
        )

        assert np.all(means == np.zeros([2]))
        assert np.allclose(cov, cov_expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("r, phi", zip(R, PHI))
    def test_squeezed_state_fock(self, r, phi, cutoff, hbar, tol):
        """test squeezed state returns correct Fock basis state vector"""
        state = utils.squeezed_state(r, phi, basis="fock", fock_dim=cutoff, hbar=hbar)

        n = np.arange(cutoff)
        kets = (np.sqrt(fac(2 * (n // 2))) / (2 ** (n // 2) * fac(n // 2))) * (
            -np.exp(1j * phi) * np.tanh(r)
        ) ** (n // 2)
        expected = np.where(n % 2 == 0, np.sqrt(1 / np.cosh(r)) * kets, 0)

        assert np.allclose(state, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("r_d, phi_d, r_s, phi_s", zip(R_D, PHI_D, R, PHI))
    def test_displaced_squeezed_state_gaussian(self, r_d, phi_d, r_s, phi_s, hbar, tol):
        """test displaced squeezed state returns correct means and covariance"""
        means, cov = utils.displaced_squeezed_state(
            r_d, phi_d, r_s, phi_s, basis="gaussian", hbar=hbar
        )

        a = r_d * np.exp(1j * phi_d)
        means_expected = np.array([[a.real, a.imag]]) * np.sqrt(2 * hbar)
        cov_expected = (hbar / 2) * np.array(
            [
                [
                    np.cosh(2 * r_s) - np.cos(phi_s) * np.sinh(2 * r_s),
                    -2 * np.cosh(r_s) * np.sin(phi_s) * np.sinh(r_s),
                ],
                [
                    -2 * np.cosh(r_s) * np.sin(phi_s) * np.sinh(r_s),
                    np.cosh(2 * r_s) + np.cos(phi_s) * np.sinh(2 * r_s),
                ],
            ]
        )

        assert np.allclose(means, means_expected, atol=tol, rtol=0)
        assert np.allclose(cov, cov_expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("r_d, phi_d, r_s, phi_s", zip(R_D, PHI_D, R, PHI))
    def test_displaced_squeezed_state_fock(self, r_d, phi_d, r_s, phi_s, hbar, cutoff, tol):
        """test displaced squeezed state returns correct Fock basis state vector"""
        state = utils.displaced_squeezed_state(
            r_d, phi_d, r_s, phi_s, basis="fock", fock_dim=cutoff, hbar=hbar
        )
        a = r_d * np.exp(1j * phi_d)

        if r_s == 0:
            pytest.skip("test only non-zero squeezing")

        n = np.arange(cutoff)
        gamma = a * np.cosh(r_s) + np.conj(a) * np.exp(1j * phi_s) * np.sinh(r_s)
        coeff = np.diag(
            (0.5 * np.exp(1j * phi_s) * np.tanh(r_s)) ** (n / 2) / np.sqrt(fac(n) * np.cosh(r_s))
        )

        expected = H(gamma / np.sqrt(np.exp(1j * phi_s) * np.sinh(2 * r_s)), coeff)
        expected *= np.exp(
            -0.5 * np.abs(a) ** 2 - 0.5 * np.conj(a) ** 2 * np.exp(1j * phi_s) * np.tanh(r_s)
        )

        assert np.allclose(state, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("r_d, phi_d, phi_s", zip(R_D, PHI_D, PHI))
    def test_displaced_squeezed_fock_no_squeezing(self, r_d, phi_d, phi_s, hbar, cutoff, tol):
        """test displaced squeezed state returns coherent state when there is no squeezing"""
        state = utils.displaced_squeezed_state(
            r_d, phi_d, 0, phi_s, basis="fock", fock_dim=cutoff, hbar=hbar
        )

        a = r_d * np.exp(1j * phi_d)
        n = np.arange(cutoff)
        expected = np.exp(-0.5 * np.abs(a) ** 2) * a ** n / np.sqrt(fac(n))

        assert np.allclose(state, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("r, phi", zip(R, PHI))
    def test_displaced_squeezed_fock_no_displacement(self, r, phi, hbar, cutoff, tol):
        """test displaced squeezed state returns squeezed state when there is no displacement"""
        state = utils.displaced_squeezed_state(
            0, 0, r, phi, basis="fock", fock_dim=cutoff, hbar=hbar
        )

        n = np.arange(cutoff)
        kets = (np.sqrt(fac(2 * (n // 2))) / (2 ** (n // 2) * fac(n // 2))) * (
            -np.exp(1j * phi) * np.tanh(r)
        ) ** (n // 2)
        expected = np.where(n % 2 == 0, np.sqrt(1 / np.cosh(r)) * kets, 0)

        assert np.allclose(state, expected, atol=tol, rtol=0)

    def test_fock_state(self):
        """test correct fock state returned"""
        n = 3
        cutoff = 10
        state = utils.fock_state(n, fock_dim=cutoff)
        assert np.all(state == np.eye(1, cutoff, n))

    @pytest.mark.parametrize("a, cutoff", [(0.212, 10), (4, 50)])
    def test_even_cat_state(self, a, cutoff, tol):
        """test correct even cat state returned"""
        p = 0

        state = utils.cat_state(a, 0, p, fock_dim=cutoff)

        # For the analytic expression, cast the integer parameter to float so
        # that there's no overflow
        a = float(a)
        n = np.arange(cutoff)
        expected = np.exp(-0.5 * np.abs(a) ** 2) * a ** n / np.sqrt(fac(n)) + np.exp(
            -0.5 * np.abs(-a) ** 2
        ) * (-a) ** n / np.sqrt(fac(n))
        expected /= np.linalg.norm(expected)

        assert np.allclose(state, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("a, cutoff", [(0.212, 10), (4, 50)])
    def test_odd_cat_state(self, a, cutoff, tol):
        """test correct odd cat state returned"""
        p = 1

        state = utils.cat_state(a, 0, p, fock_dim=cutoff)

        # For the analytic expression, cast the integer parameter to float so
        # that there's no overflow
        a = float(a)
        n = np.arange(cutoff)
        expected = np.exp(-0.5 * np.abs(a) ** 2) * a ** n / np.sqrt(fac(n)) - np.exp(
            -0.5 * np.abs(-a) ** 2
        ) * (-a) ** n / np.sqrt(fac(n))
        expected /= np.linalg.norm(expected)

        assert np.allclose(state, expected, atol=tol, rtol=0)


# ===================================================================================
# Random matrix tests
# ===================================================================================


class TestRandomMatrices:
    """Unit tests for random matrices"""

    @pytest.fixture
    def modes(self):
        """Number of modes to use when creating matrices"""
        return 3

    @pytest.mark.parametrize("pure_state", [True, False])
    @pytest.mark.parametrize("block_diag", [True, False])
    def test_random_covariance_square(self, modes, hbar, pure_state, block_diag):
        """Test that a random covariance matrix is the right shape"""
        V = utils.random_covariance(modes, hbar=hbar, pure=pure_state, block_diag=block_diag)
        assert np.all(V.shape == np.array([2 * modes, 2 * modes]))

    @pytest.mark.parametrize("pure_state", [True, False])
    @pytest.mark.parametrize("block_diag", [True, False])
    def test_random_covariance_symmetric(self, modes, hbar, pure_state, tol, block_diag):
        """Test that a random covariance matrix is symmetric"""
        V = utils.random_covariance(modes, hbar=hbar, pure=pure_state, block_diag=block_diag)
        assert np.allclose(V.T, V, atol=tol, rtol=0)

    @pytest.mark.parametrize("pure_state", [True, False])
    @pytest.mark.parametrize("block_diag", [True, False])
    def test_random_covariance_valid(self, modes, hbar, pure_state, tol, block_diag):
        """Test that a random covariance matrix satisfies the uncertainty principle V+i hbar O/2 >=0"""
        V = utils.random_covariance(modes, hbar=hbar, pure=pure_state, block_diag=block_diag)

        idm = np.identity(modes)
        omega = np.concatenate(
            (np.concatenate((0 * idm, idm), axis=1), np.concatenate((-idm, 0 * idm), axis=1)),
            axis=0,
        )

        eigs = np.linalg.eigvalsh(V + 1j * (hbar / 2) * omega)
        eigs[np.abs(eigs) < tol] = 0
        assert np.all(eigs >= 0)

    @pytest.mark.parametrize("block_diag", [True, False])
    def test_random_covariance_pure(self, modes, hbar, tol, block_diag):
        """Test that a pure random covariance matrix has correct purity"""
        V = utils.random_covariance(modes, hbar=hbar, pure=True, block_diag=block_diag)
        det = np.linalg.det(V) - (hbar / 2) ** (2 * modes)
        assert np.allclose(det, 0, atol=tol, rtol=0)

    @pytest.mark.parametrize("block_diag", [True, False])
    def test_random_covariance_mixed(self, modes, hbar, tol, block_diag):
        """Test that a mixed random covariance matrix has correct purity"""
        V = utils.random_covariance(modes, hbar=hbar, pure=False, block_diag=block_diag)
        det = np.linalg.det(V) - (hbar / 2) ** (2 * modes)
        assert not np.allclose(det, 0, atol=tol, rtol=0)

    @pytest.mark.parametrize("passive", [True, False])
    @pytest.mark.parametrize("block_diag", [True, False])
    def test_random_symplectic_square(self, modes, hbar, passive, block_diag):
        """Test that a random symplectic matrix on is the right shape"""
        S = utils.random_symplectic(modes, passive=passive, block_diag=block_diag)
        assert np.all(S.shape == np.array([2 * modes, 2 * modes]))

    @pytest.mark.parametrize("passive", [True, False])
    @pytest.mark.parametrize("block_diag", [True, False])
    def test_random_symplectic_symplectic(self, modes, hbar, passive, tol, block_diag):
        """Test that a random symplectic matrix is symplectic"""
        S = utils.random_symplectic(modes, passive=passive, block_diag=block_diag)

        idm = np.identity(modes)
        omega = np.concatenate(
            (np.concatenate((0 * idm, idm), axis=1), np.concatenate((-idm, 0 * idm), axis=1)),
            axis=0,
        )

        assert np.allclose(S @ omega @ S.T, omega, atol=tol, rtol=0)

    @pytest.mark.parametrize("block_diag", [True, False])
    def test_random_symplectic_passive_orthogonal(self, modes, hbar, tol, block_diag):
        """Test that a passive random symplectic matrix is orthogonal"""
        S = utils.random_symplectic(modes, passive=True, block_diag=block_diag)
        assert np.allclose(S @ S.T, np.identity(2 * modes), atol=tol, rtol=0)

    @pytest.mark.parametrize("block_diag", [True, False])
    def test_random_symplectic_active_not_orthogonal(self, modes, hbar, tol, block_diag):
        """Test that an active random symplectic matrix is not orthogonal"""
        S = utils.random_symplectic(modes, passive=False, block_diag=block_diag)
        assert not np.allclose(S @ S.T, np.identity(2 * modes), atol=tol, rtol=0)

    @pytest.mark.parametrize("real", [True, False])
    def test_random_inteferometer_is_square(self, modes, real):
        """Test that a random interferometer is square"""
        U = utils.random_interferometer(modes, real=real)
        assert np.all(U.shape == np.array([modes, modes]))

    @pytest.mark.parametrize("real", [True, False])
    def test_random_inteferometer_is_unitary(self, modes, tol, real):
        """Test that a random interferometer is unitary"""
        U = utils.random_interferometer(modes, real=real)
        assert np.allclose(U @ U.conj().T, np.identity(modes), atol=tol, rtol=0)


# ===================================================================================
# Operation tests
# ===================================================================================


class TestOperation:
    """Tests for the operation decorator"""

    def test_applying_decorator(self):
        """Test the __call__ method of the operation class"""

        def f(x):
            return x ** 2

        op = utils.operation(ns=1)
        op(f)(0, 6, 3)

        assert op.func == f
        assert op.args == (0, 6, 3)

    def test_no_arg_call_function(self, rr):
        """Test exception raised if the wrapped function doesn't accept a register"""

        def dummy_func():
            Sgate(r) | q[0]
            BSgate() | (q[0], q[1])

        op = utils.operation(ns=1)
        op(dummy_func)()

        with pytest.raises(ValueError, match="must receive the qumode register"):
            op._call_function(rr)

    def test_incorrect_arg_num_call_function(self, rr):
        """Test exception raised if the wrapped function is called with wrong number of args"""

        def dummy_func(r, q):
            Sgate(r) | q[0]
            BSgate() | (q[0], q[1])

        op = utils.operation(ns=1)
        op(dummy_func)(1, 2)

        with pytest.raises(ValueError, match="Mismatch in the number of arguments"):
            op._call_function(rr)

    def test_call_function(self, prog):
        """Test _call_function method for operation class"""
        rval = 4.32
        thetaval = -0.654
        phival = 0.543

        def dummy_func(r, theta, phi, q):
            Sgate(r) | q[0]
            BSgate(theta, phi) | (q[0], q[1])

        op = utils.operation(ns=1)
        op(dummy_func)(rval, thetaval, phival)

        with prog.context as q:
            res = op._call_function(q)

        # check register is returned
        assert res == q

        # check eng queue matches the operation
        assert len(prog) == 2

        # check first queue op is Sgate(rval)
        assert isinstance(prog.circuit[0].op, Sgate)
        assert prog.circuit[0].reg == [q[0]]
        assert prog.circuit[0].op.p[0] == rval

        # check second queue op is BSgate(thetaval, phival)
        assert isinstance(prog.circuit[1].op, BSgate)
        assert prog.circuit[1].reg == list(q)
        assert prog.circuit[1].op.p[0] == thetaval
        assert prog.circuit[1].op.p[1] == phival

    def test_multimode_wrong_num_modes_apply_operation(self, prog):
        """Test exceptions raised when applying an operation to
        multiple modes with incorrect num modes"""
        rval = 4.32
        thetaval = -0.654
        phival = 0.543

        @utils.operation(1)
        def dummy_func(r, theta, phi, q):
            Sgate(r) | q[0]
            BSgate(theta, phi) | (q[0], q[1])

        with prog.context as q:
            with pytest.raises(ValueError, match="Wrong number of subsystems"):
                dummy_func(rval, thetaval, phival) | q

    def test_single_mode_no_args_apply_operation(self, prog):
        """Test applying an operation to a single mode"""
        rval = 4.32

        @utils.operation(2)
        def dummy_func(q):
            Sgate(rval) | q[0]

        with prog.context as q:
            dummy_func() | q

        # check eng queue matches the operation
        assert len(prog) == 1

        # check first queue op is Sgate(rval)
        assert isinstance(prog.circuit[0].op, Sgate)
        assert prog.circuit[0].reg == [q[0]]
        assert prog.circuit[0].op.p[0] == rval

    def test_multimode_args_apply_operation(self, prog):
        """Test applying an operation to multiple modes"""
        rval = 4.32
        thetaval = -0.654
        phival = 0.543

        @utils.operation(2)
        def dummy_func(r, theta, phi, q):
            Sgate(r) | q[0]
            BSgate(theta, phi) | (q[0], q[1])

        with prog.context as q:
            with pytest.raises(ValueError, match="Wrong number of subsystems"):
                dummy_func(rval, thetaval, phival) | q[0]

        with prog.context as q:
            dummy_func(rval, thetaval, phival) | q

        # check eng queue matches the operation
        assert len(prog) == 2

        # check first queue op is Sgate(rval)
        assert isinstance(prog.circuit[0].op, Sgate)
        assert prog.circuit[0].reg == [q[0]]
        assert prog.circuit[0].op.p[0] == rval

        # check second queue op is BSgate(thetaval, phival)
        assert isinstance(prog.circuit[1].op, BSgate)
        assert prog.circuit[1].reg == list(q)
        assert prog.circuit[1].op.p[0] == thetaval
        assert prog.circuit[1].op.p[1] == phival


# ===================================================================================
# Engine utility tests
# ===================================================================================


class TestEngineUtilityFunctions:
    """Tests for some engine auxiliary functions"""

    def test_is_unitary_no_channel(self, prog):
        """test that the is_unitary function returns True if no channels are present"""
        assert utils.is_unitary(prog)
        with prog.context as q:
            Sgate(0.4) | q[0]
            BSgate(0.4) | q
        assert utils.is_unitary(prog)

    def test_is_unitary_with_channel(self, prog):
        """test that the is_unitary function returns False if channels are present"""
        with prog.context as q:
            Sgate(0.4) | q[0]
            LossChannel(0.4) | q[0]
            BSgate(0.4) | q
        assert not utils.is_unitary(prog)

    def test_is_channel_no_measurement(self, prog):
        """test that the is_channel function returns True if no measurements are present"""
        assert utils.is_channel(prog)
        with prog.context as q:
            Sgate(0.4) | q[0]
            LossChannel(0.4) | q[0]
            BSgate(0.4) | q
        assert utils.is_channel(prog)

    def test_is_channel_measurement(self, prog):
        """is_channel() returns False if measurements are present"""
        with prog.context as q:
            Sgate(0.4) | q[0]
            BSgate() | q
            MeasureX | q[0]
            Sgate(0.4) | q[1]
        assert not utils.is_channel(prog)

    def test_is_channel_preparation(self, prog):
        """is_channel() returns False if preparations are present"""
        with prog.context as q:
            Sgate(0.4) | q[0]
            BSgate() | q
            Squeezed(0.4) | q[1]
        assert not utils.is_channel(prog)

    def test_vectorize_unvectorize(self, tol):
        """Test vectorize and unvectorize utility function"""
        cutoff = 4

        dm = np.random.rand(*[cutoff] * 8) + 1j * np.random.rand(
            *[cutoff] * 8
        )  # 4^8 -> (4^2)^4 -> 4^8
        dm2 = np.random.rand(*[cutoff] * 4) + 1j * np.random.rand(
            *[cutoff] * 4
        )  # (2^2)^4 -> 2^8 -> (2^2)^4

        assert np.allclose(
            dm,
            utils.program_functions._unvectorize(utils.program_functions._vectorize(dm), 2),
            atol=tol,
            rtol=0,
        )
        assert np.allclose(
            dm2,
            utils.program_functions._vectorize(utils.program_functions._unvectorize(dm2, 2)),
            atol=tol,
            rtol=0,
        )

    def test_interleaved_identities(self, tol):
        """Test interleaved utility function"""

        II = utils.program_functions._interleaved_identities(n=2, cutoff_dim=3)
        assert np.allclose(np.einsum("abab", II), 3 ** 2, atol=tol, rtol=0)

        III = utils.program_functions._interleaved_identities(n=3, cutoff_dim=5)
        assert np.allclose(np.einsum("abcabc", III), 5 ** 3, atol=tol, rtol=0)


# TODO: add unit tests for _engine_with_CJ_cmd_queue
# extract_unitary, extract_channel

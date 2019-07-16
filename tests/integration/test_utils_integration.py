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
r"""Integration tests for the utils.py module"""
import pytest

import numpy as np

try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError) as e:
    import mock
    tf = mock.MagicMock()
    tf.Tensor = int

import strawberryfields as sf
import strawberryfields.ops as ops
import strawberryfields.utils as utils

from strawberryfields.backends.fockbackend.ops import squeezing as sq_U
from strawberryfields.backends.fockbackend.ops import displacement as disp_U
from strawberryfields.backends.fockbackend.ops import beamsplitter as bs_U

ALPHA = np.linspace(-0.15, 0.2, 4) + np.linspace(-0.2, 0.1, 4) * 1j
R = np.linspace(0, 0.21, 4)
PHI = np.linspace(0, 1.43, 4)
np.random.seed(42)


# ===================================================================================
# Initial states integration tests
# ===================================================================================


@pytest.mark.backends("gaussian")
class TestInitialStatesAgreeGaussian:
    """Test that the initial state functions in utils match
    the result returned by the Gaussian backends."""

    def test_vacuum(self, setup_eng, hbar, tol):
        """Test vacuum function matches Gaussian backends"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Vac | q[0]

        state = eng.run(prog).state

        mu, cov = utils.vacuum_state(basis="gaussian", hbar=hbar)
        mu_exp, cov_exp = state.reduced_gaussian(0)
        assert np.allclose(mu, mu_exp, atol=tol, rtol=0)
        assert np.allclose(cov, cov_exp, atol=tol, rtol=0)

    @pytest.mark.parametrize("a", ALPHA)
    def test_coherent(self, a, setup_eng, hbar, tol):
        """Test coherent function matches Gaussian backends"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Dgate(a) | q[0]

        state = eng.run(prog).state

        mu, cov = utils.coherent_state(a, basis="gaussian", hbar=hbar)
        mu_exp, cov_exp = state.reduced_gaussian(0)
        assert np.allclose(mu, mu_exp, atol=tol, rtol=0)
        assert np.allclose(cov, cov_exp, atol=tol, rtol=0)

    @pytest.mark.parametrize("r, phi", zip(R, PHI))
    def test_squeezed(self, r, phi, setup_eng, hbar, tol):
        """Test squeezed function matches Gaussian backends"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Sgate(r, phi) | q[0]

        state = eng.run(prog).state

        mu, cov = utils.squeezed_state(r, phi, basis="gaussian", hbar=hbar)
        mu_exp, cov_exp = state.reduced_gaussian(0)
        assert np.allclose(mu, mu_exp, atol=tol, rtol=0)
        assert np.allclose(cov, cov_exp, atol=tol, rtol=0)

    @pytest.mark.parametrize("a, r, phi", zip(ALPHA, R, PHI))
    def test_displaced_squeezed(self, a, r, phi, setup_eng, hbar, tol):
        """Test displaced squeezed function matches Gaussian backends"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Sgate(r, phi) | q[0]
            ops.Dgate(a) | q[0]

        state = eng.run(prog).state

        mu, cov = utils.displaced_squeezed_state(a, r, phi, basis="gaussian", hbar=hbar)
        mu_exp, cov_exp = state.reduced_gaussian(0)
        assert np.allclose(mu, mu_exp, atol=tol, rtol=0)
        assert np.allclose(cov, cov_exp, atol=tol, rtol=0)


@pytest.fixture
def bsize(batch_size):
    return 1 if batch_size is None else batch_size


@pytest.mark.backends("fock", "tf")
class TestInitialStatesAgreeFock:
    """Test that the initial state functions in utils match
    the result returned by the Fock backends."""

    def test_vacuum(self, setup_eng, hbar, cutoff, bsize, pure, tol):
        """Test vacuum function matches Fock backends"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Vac | q[0]

        state = eng.run(prog).state

        ket = utils.vacuum_state(basis="fock", fock_dim=cutoff, hbar=hbar)

        if not pure:
            expected = state.dm()
            ket = np.tile(np.outer(ket, ket.conj()), (bsize, 1, 1))
        else:
            expected = state.ket()
            ket = np.tile(ket, (bsize, 1))

        assert np.allclose(expected, ket, atol=tol, rtol=0)

    def test_coherent(self, setup_eng, hbar, cutoff, bsize, pure, tol):
        """Test coherent function matches Fock backends"""
        eng, prog = setup_eng(1)
        a = 0.32 + 0.1j
        r = 0.112
        phi = 0.123

        with prog.context as q:
            ops.Dgate(a) | q[0]

        state = eng.run(prog).state
        ket = utils.coherent_state(a, basis="fock", fock_dim=cutoff, hbar=hbar)

        if not pure:
            expected = state.dm()
            ket = np.tile(np.outer(ket, ket.conj()), (bsize, 1, 1))
        else:
            expected = state.ket()
            ket = np.tile(ket, (bsize, 1))

        assert np.allclose(expected, ket, atol=tol, rtol=0)

    def test_squeezed(self, setup_eng, hbar, cutoff, bsize, pure, tol):
        """Test squeezed function matches Fock backends"""
        eng, prog = setup_eng(1)
        r = 0.112
        phi = 0.123

        with prog.context as q:
            ops.Sgate(r, phi) | q[0]

        state = eng.run(prog).state
        ket = utils.squeezed_state(r, phi, basis="fock", fock_dim=cutoff, hbar=hbar)

        if not pure:
            expected = state.dm()
            ket = np.tile(np.outer(ket, ket.conj()), (bsize, 1, 1))
        else:
            expected = state.ket()
            ket = np.tile(ket, (bsize, 1))

        assert np.allclose(expected, ket, atol=tol, rtol=0)

    def test_displaced_squeezed(self, setup_eng, hbar, cutoff, bsize, pure, tol):
        """Test displaced squeezed function matches Fock backends"""
        eng, prog = setup_eng(1)
        a = 0.32 + 0.1j
        r = 0.112
        phi = 0.123

        with prog.context as q:
            ops.Sgate(r, phi) | q[0]
            ops.Dgate(a) | q[0]

        state = eng.run(prog).state
        ket = utils.displaced_squeezed_state(
            a, r, phi, basis="fock", fock_dim=cutoff, hbar=hbar
        )

        if not pure:
            expected = state.dm()
            ket = np.tile(np.outer(ket, ket.conj()), (bsize, 1, 1))
        else:
            expected = state.ket()
            ket = np.tile(ket, (bsize, 1))

        assert np.allclose(expected, ket, atol=tol, rtol=0)

    def test_fock_state(self, setup_eng, hbar, cutoff, bsize, pure, tol):
        """Test fock state function matches Fock backends"""
        eng, prog = setup_eng(1)
        n = 2

        with prog.context as q:
            ops.Fock(n) | q[0]

        state = eng.run(prog).state
        ket = utils.fock_state(n, fock_dim=cutoff)
        if not pure:
            expected = state.dm()
            ket = np.tile(np.outer(ket, ket.conj()), (bsize, 1, 1))
        else:
            expected = state.ket()
            ket = np.tile(ket, (bsize, 1))

        assert np.allclose(expected, ket, atol=tol, rtol=0)

    def test_cat_state(self, setup_eng, hbar, cutoff, bsize, pure, tol):
        """Test cat state function matches Fock backends"""
        eng, prog = setup_eng(1)
        a = 0.32 + 0.1j
        p = 0.43

        with prog.context as q:
            ops.Catstate(a, p) | q[0]

        state = eng.run(prog).state
        ket = utils.cat_state(a, p, fock_dim=cutoff)

        if not pure:
            expected = state.dm()
            ket = np.tile(np.outer(ket, ket.conj()), (bsize, 1, 1))
        else:
            expected = state.ket()
            ket = np.tile(ket, (bsize, 1))

        assert np.allclose(expected, ket, atol=tol, rtol=0)


# ===================================================================================
# Operation integration tests
# ===================================================================================


@utils.operation(1)
def prepare_state(v1, q):
    ops.Dgate(v1) | q


@utils.operation(2)
def entangle_states(q):
    ops.Sgate(-2) | q[0]
    ops.Sgate(2) | q[1]
    ops.BSgate(np.pi / 4, 0) | (q[0], q[1])


class TestTeleportationOperationTest:
    """Run a teleportation algorithm but split the circuit into operations.
    Operations can be also methods of a class"""

    @pytest.mark.parametrize('cutoff', [10], indirect=True)  # override default cutoff fixture
    def test_teleportation_fidelity(self, setup_eng):
        """Test teleportation algorithm gives correct fid when using operations"""
        eng, prog = setup_eng(3)
        tol = 0.1
        with prog.context as q:
            prepare_state(0.5 + 0.2j) | q[0]
            entangle_states() | (q[1], q[2])
            ops.BSgate(np.pi / 4, 0) | (q[0], q[1])
            ops.MeasureHomodyne(0, select=0) | q[0]
            ops.MeasureHomodyne(np.pi / 2, select=0) | q[1]

        state = eng.run(prog).state
        fidelity = state.fidelity_coherent([0, 0, 0.5 + 0.2j])
        assert np.allclose(fidelity, 1, atol=0.1, rtol=0)

    def test_validate_argument(self):
        """Test correct exceptions are raised for wrong arguments"""

        with pytest.raises(ValueError):
            prepare_state(1, 2, 3) | (1, 2)

        with pytest.raises(ValueError):
            prepare_state(1) | (1, 2)

        with pytest.raises(ValueError):
            prepare_state(1) | ()

        with pytest.raises(ValueError):
            entangle_states() | (1, 2, 3)

        with pytest.raises(ValueError):
            entangle_states() | (1)

        with pytest.raises(ValueError):
            entangle_states() | 1


# ===================================================================================
# Engine unitary and channel extraction tests
# ===================================================================================

@pytest.fixture
def backend_name(setup_backend_pars):
    return setup_backend_pars[0]


@pytest.mark.backends("fock", "tf")
class TestExtractUnitary:
    """Test extraction of unitaries"""

    def test_extract_kerr(self, backend_name, cutoff, tol):
        """test that the Kerr gate is correctly extracted"""

        prog = sf.Program(1)
        kappa = 0.432
        with prog.context as q:
            ops.Kgate(kappa) | q

        U = utils.extract_unitary(prog, cutoff_dim=cutoff, backend=backend_name)
        expected = np.diag(np.exp(1j * kappa * np.arange(cutoff) ** 2))

        if isinstance(U, tf.Tensor):
            U = tf.Session().run(U)

        assert np.allclose(U, expected, atol=tol, rtol=0)

    def test_extract_squeezing(self, backend_name, cutoff, tol):
        """test that the squeezing gate is correctly extracted"""
        prog = sf.Program(1)

        r = 0.432
        phi = -0.96543

        with prog.context as q:
            ops.Sgate(r, phi) | q

        U = utils.extract_unitary(prog, cutoff_dim=cutoff, backend=backend_name)
        expected = sq_U(r, phi, cutoff)

        if isinstance(U, tf.Tensor):
            U = tf.Session().run(U)

        assert np.allclose(U, expected, atol=tol, rtol=0)

    def test_extract_displacement(self, backend_name, cutoff, tol):
        """test that the displacement gate is correctly extracted"""
        prog = sf.Program(1)
        alpha = 0.432 - 0.8543j
        with prog.context as q:
            ops.Dgate(alpha) | q

        U = utils.extract_unitary(prog, cutoff_dim=cutoff, backend=backend_name)
        expected = disp_U(alpha, cutoff)

        if isinstance(U, tf.Tensor):
            U = tf.Session().run(U)

        assert np.allclose(U, expected, atol=tol, rtol=0)

    def test_extract_beamsplitter(self, backend_name, cutoff, tol):
        """test that the beamsplitter gate is correctly extracted"""
        prog = sf.Program(2)
        theta = 0.432
        phi = 0.765
        with prog.context as q:
            ops.BSgate(theta, phi) | q

        U = utils.extract_unitary(prog, cutoff_dim=cutoff, backend=backend_name)
        expected = bs_U(np.cos(theta), np.sin(theta), phi, cutoff)

        if isinstance(U, tf.Tensor):
            U = tf.Session().run(U)

        assert np.allclose(U, expected, atol=tol, rtol=0)

    def test_extract_arbitrary_unitary_one_mode(self, setup_eng, cutoff, tol):
        """Test that arbitrary unitary extraction works for 1 mode"""
        S = ops.Sgate(0.4, -1.2)
        D = ops.Dgate(2, 0.9)
        K = ops.Kgate(-1.5)

        # not a state but it doesn't matter
        initial_state = np.random.rand(cutoff) + 1j * np.random.rand(cutoff)

        eng_ref, p0 = setup_eng(1)
        with p0.context as q:
            ops.Ket(initial_state) | q

        prog = sf.Program(p0)
        with prog.context as q:
            S | q
            D | q
            K | q

        U = utils.extract_unitary(prog, cutoff_dim=cutoff, backend=eng_ref.backend_name)

        if isinstance(U, tf.Tensor):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                in_state = tf.constant(initial_state.reshape([-1]), dtype=tf.complex64)
                final_state = sess.run(tf.einsum("ab,b", U, in_state))
        else:
            final_state = U @ initial_state

        expected_state = eng_ref.run([p0, prog]).state.ket()
        assert np.allclose(final_state, expected_state, atol=tol, rtol=0)

    def test_extract_arbitrary_unitary_two_modes_vectorized(
        self, setup_eng, cutoff, batch_size, tol
    ):
        """Test that arbitrary unitary extraction works for 2 mode when vectorized"""
        bsize = 1
        if batch_size is not None:
            bsize = batch_size

        S = ops.Sgate(0.4, -1.2)
        B = ops.BSgate(2.234, -1.165)

        # not a state but it doesn't matter
        initial_state = np.complex64(
            np.random.rand(cutoff, cutoff) + 1j * np.random.rand(cutoff, cutoff)
        )

        eng_ref, p0 = setup_eng(2)
        with p0.context as q:
            ops.Ket(initial_state) | q

        prog = sf.Program(p0)
        with prog.context as q:
            S | q[0]
            B | q
            S | q[1]
            B | q

        U = utils.extract_unitary(
            prog,
            cutoff_dim=cutoff,
            vectorize_modes=True,
            backend=eng_ref.backend_name,
        )

        if isinstance(U, tf.Tensor):
            final_state = tf.Session().run(
                tf.einsum("ab,b", U, tf.constant(initial_state.reshape([-1])))
            )
            final_state = np.tile(final_state, bsize)
        else:
            final_state = U @ initial_state.reshape([-1])

        expected_state = eng_ref.run([p0, prog]).state.ket().reshape([-1])
        assert np.allclose(final_state, expected_state, atol=tol, rtol=0)

    def test_extract_arbitrary_unitary_two_modes_not_vectorized(
        self, setup_eng, cutoff, tol
    ):
        """Test that arbitrary unitary extraction works for 2 mode"""
        S = ops.Sgate(0.4, -1.2)
        B = ops.BSgate(2.234, -1.165)

        # not a state but it doesn't matter
        initial_state = np.complex64(
            np.random.rand(cutoff, cutoff) + 1j * np.random.rand(cutoff, cutoff)
        )

        eng_ref, p0 = setup_eng(2)
        if eng_ref.backend.short_name == "tf":
            pytest.skip("Un vectorized mode only supports Fock backend for now")

        with p0.context as q:
            ops.Ket(initial_state) | q

        prog = sf.Program(p0)
        with prog.context as q:
            S | q[0]
            B | q
            S | q[1]
            B | q

        U = utils.extract_unitary(
            prog,
            cutoff_dim=cutoff,
            vectorize_modes=False,
            backend=eng_ref.backend_name,
        )
        final_state = np.einsum("abcd,bd->ac", U, initial_state)
        expected_state = eng_ref.run([p0, prog]).state.ket()

        assert np.allclose(final_state, expected_state, atol=tol, rtol=0)


@pytest.mark.backends("fock")
class TestExtractChannelOneMode:
    """Test extraction of unitaries"""

    @pytest.fixture
    def setup_one_mode_circuit(self, setup_eng, cutoff):
        """Create the circuit for following tests"""
        eng_ref, p0 = setup_eng(1)

        S = ops.Sgate(1.1, -1.4)
        L = ops.LossChannel(0.45)

        initial_state = np.random.rand(cutoff, cutoff)

        with p0.context as q:
            ops.DensityMatrix(initial_state) | q

        prog = sf.Program(p0)
        with prog.context as q:
            S | q
            L | q

        rho = eng_ref.run([p0, prog]).state.dm()
        return prog, rho, initial_state

    def test_extract_choi_channel(self, setup_one_mode_circuit, cutoff, tol):
        """Test that Choi channel extraction works for 1 mode"""
        prog, rho, initial_state = setup_one_mode_circuit

        choi = utils.extract_channel(prog, cutoff_dim=cutoff, representation="choi")
        final_rho = np.einsum("abcd,ab -> cd", choi, initial_state)

        assert np.allclose(final_rho, rho, atol=tol, rtol=0)

    def test_extract_liouville_channel(self, setup_one_mode_circuit, cutoff, tol):
        """Test that Liouville channel extraction works for 1 mode"""
        prog, rho, initial_state = setup_one_mode_circuit

        liouville = utils.extract_channel(
            prog, cutoff_dim=cutoff, representation="liouville"
        )
        final_rho = np.einsum("abcd,db -> ca", liouville, initial_state)

        assert np.allclose(final_rho, rho, atol=tol, rtol=0)

    def test_extract_kraus_channel(self, setup_one_mode_circuit, cutoff, tol):
        """Test that Kraus channel extraction works for 1 mode"""
        prog, rho, initial_state = setup_one_mode_circuit

        kraus = utils.extract_channel(prog, cutoff_dim=cutoff, representation="kraus")
        final_rho = np.einsum("abc,cd,aed -> be", kraus, initial_state, np.conj(kraus))

        assert np.allclose(final_rho, rho, atol=tol, rtol=0)


@pytest.mark.backends("fock")
class TestExtractChannelTwoMode:
    """Test extraction of unitaries"""

    @pytest.fixture
    def setup_two_mode_circuit(self, setup_eng, cutoff):
        """Create the circuit for following tests"""
        eng_ref, p0 = setup_eng(2)

        S = ops.Sgate(2)
        B = ops.BSgate(2.234, -1.165)

        initial_state = np.complex64(
            np.random.rand(*[cutoff] * 4) + 1j * np.random.rand(*[cutoff] * 4)
        )

        with p0.context as q:
            ops.DensityMatrix(initial_state) | q

        prog = sf.Program(p0)
        with prog.context as q:
            S | q[0]
            B | q
            S | q[1]
            B | q

        rho = eng_ref.run([p0, prog]).state.dm()
        return prog, rho, initial_state

    def test_extract_choi_channel(self, setup_two_mode_circuit, cutoff, tol):
        """Test that Choi channel extraction works for 2 mode"""
        prog, rho, initial_state = setup_two_mode_circuit

        choi = utils.extract_channel(
            prog, cutoff_dim=cutoff, vectorize_modes=False, representation="choi"
        )
        final_rho = np.einsum("abcdefgh,abcd -> efgh", choi, initial_state)

        assert np.allclose(final_rho, rho, atol=tol, rtol=0)

    def test_extract_liouville_channel(self, setup_two_mode_circuit, cutoff, tol):
        """Test that Liouville channel extraction works for 2 mode"""
        prog, rho, initial_state = setup_two_mode_circuit

        liouville = utils.extract_channel(
            prog, cutoff_dim=cutoff, vectorize_modes=False, representation="liouville"
        )
        final_rho = np.einsum("abcdefgh,fbhd -> eagc", liouville, initial_state)

        assert np.allclose(final_rho, rho, atol=tol, rtol=0)

    def test_extract_kraus_channel(self, setup_two_mode_circuit, cutoff, tol):
        """Test that Kraus channel extraction works for 2 mode"""
        prog, rho, initial_state = setup_two_mode_circuit

        kraus = utils.extract_channel(
            prog, cutoff_dim=cutoff, vectorize_modes=False, representation="kraus"
        )
        final_rho = np.einsum(
            "abcde,cfeg,ahfig -> bhdi", kraus, initial_state, np.conj(kraus)
        )

        assert np.allclose(final_rho, rho, atol=tol, rtol=0)

    def test_extract_choi_channel_vectorize(self, setup_two_mode_circuit, cutoff, tol):
        """Test that Choi channel extraction works for 2 mode vectorized"""
        prog, rho, initial_state = setup_two_mode_circuit
        rho = np.einsum("abcd->acbd", rho).reshape(cutoff ** 2, cutoff ** 2)
        initial_state = np.einsum("abcd->acbd", initial_state).reshape(
            cutoff ** 2, cutoff ** 2
        )

        choi = utils.extract_channel(
            prog, cutoff_dim=cutoff, vectorize_modes=True, representation="choi"
        )
        final_rho = np.einsum("abcd,ab -> cd", choi, initial_state)

        assert np.allclose(final_rho, rho, atol=tol, rtol=0)

    def test_extract_liouville_channel_vectorize(
        self, setup_two_mode_circuit, cutoff, tol
    ):
        """Test that Liouville channel extraction works for 2 mode vectorized"""
        prog, rho, initial_state = setup_two_mode_circuit
        rho = np.einsum("abcd->acbd", rho).reshape(cutoff ** 2, cutoff ** 2)
        initial_state = np.einsum("abcd->acbd", initial_state).reshape(
            cutoff ** 2, cutoff ** 2
        )

        liouville = utils.extract_channel(
            prog, cutoff_dim=cutoff, vectorize_modes=True, representation="liouville"
        )
        final_rho = np.einsum("abcd,db -> ca", liouville, initial_state)

        assert np.allclose(final_rho, rho, atol=tol, rtol=0)

    def test_extract_kraus_channel_vectorize(self, setup_two_mode_circuit, cutoff, tol):
        """Test that Kraus channel extraction works for 2 mode vectorized"""
        prog, rho, initial_state = setup_two_mode_circuit
        rho = np.einsum("abcd->acbd", rho).reshape(cutoff ** 2, cutoff ** 2)
        initial_state = np.einsum("abcd->acbd", initial_state).reshape(
            cutoff ** 2, cutoff ** 2
        )

        kraus = utils.extract_channel(
            prog, cutoff_dim=cutoff, vectorize_modes=True, representation="kraus"
        )
        final_rho = np.einsum("abc,cd,aed -> be", kraus, initial_state, np.conj(kraus))

        assert np.allclose(final_rho, rho, atol=tol, rtol=0)

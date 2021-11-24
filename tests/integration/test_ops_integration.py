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
r"""Integration tests for frontend operations applied to the backend"""
import pytest
import numpy as np
from thewalrus.random import random_symplectic
from scipy.stats import unitary_group

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.backends import BaseGaussian
from strawberryfields.backends.states import BaseFockState, BaseGaussianState

try:
    import tensorflow as tf
except:
    backends = ["fock", "tf"]
else:
    backends = ["fock"]

# make test deterministic
np.random.seed(42)
A = 0.1234
B = -0.543


@pytest.mark.parametrize("gate", ops.gates)
class TestGateApplication:
    """tests that involve gate application"""

    @pytest.fixture
    def G(self, gate):
        """Initialize each gate"""
        if gate in ops.zero_args_gates:
            return gate()

        if gate in ops.one_args_gates:
            return gate(A)

        if gate in ops.two_args_gates:
            return gate(A, B)

    def test_gate_dagger_vacuum(self, G, setup_eng, tol):
        """Test applying gate inverses after the gate cancels out"""
        eng, prog = setup_eng(2)

        if isinstance(G, (ops.Vgate, ops.Kgate, ops.CKgate)) and isinstance(
            eng.backend, BaseGaussian
        ):
            pytest.skip("Non-Gaussian gates cannot be applied to the Gaussian backend")

        with prog.context as q:
            if G.ns == 1:
                G | q[0]
                G.H | q[0]
            elif G.ns == 2:
                G | (q[0], q[1])
                G.H | (q[0], q[1])

        eng.run(prog)

        # we must end up back in vacuum since G and G.H cancel each other
        assert np.all(eng.backend.is_vacuum(tol))


class TestChannelApplication:
    """tests that involve channel application"""

    def test_loss_channel(self, setup_eng, tol):
        """Test loss channel with no transmission produces vacuum"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Dgate(np.abs(A), np.angle(A)) | q[0]
            ops.LossChannel(0) | q[0]

        eng.run(prog)
        assert np.all(eng.backend.is_vacuum(tol))

    @pytest.mark.backends("gaussian", "bosonic")
    def test_thermal_loss_channel(self, setup_eng, tol):
        """Test thermal loss channel with no transmission produces thermal state"""
        eng, prog = setup_eng(1)
        nbar = 0.43

        with prog.context as q:
            ops.Dgate(np.abs(A), np.angle(A)) | q[0]
            ops.ThermalLossChannel(0, nbar) | q[0]

        state = eng.run(prog).state
        mean_photon, var = state.mean_photon(0)
        assert np.allclose(mean_photon, nbar, atol=tol, rtol=0)
        assert np.allclose(var, nbar ** 2 + nbar, atol=tol, rtol=0)

    @pytest.mark.backends("gaussian")
    @pytest.mark.parametrize("M", range(1, 5))
    def test_passive_channel_vacuum(self, M, setup_eng, tol):
        """test that you get vacuum on all modes if you apply a channel with all zero"""
        eng, prog = setup_eng(M)

        with prog.context as q:
            for i in range(M):
                ops.Dgate(abs(A), np.angle(A)) | q[i]
            ops.PassiveChannel(np.zeros((M, M))) | q

        eng.run(prog)
        assert np.all(eng.backend.is_vacuum(tol))

    @pytest.mark.backends("gaussian")
    @pytest.mark.parametrize("M", range(2, 7))
    def test_passive_channel(self, M, setup_eng, tol):
        """check that passive channel is consistent with unitary methods"""
        U = unitary_group.rvs(M)

        loss_in = np.random.random(M)
        loss_out = np.random.random(M)

        T = (np.sqrt(loss_in) * U) * np.sqrt(loss_out[np.newaxis].T)

        eng, prog = setup_eng(M)
        with prog.context as q:
            for i in range(M):
                ops.Sgate(1) | q[i]
                ops.Dgate(A) | q[i]
            ops.PassiveChannel(T) | q

        state = eng.run(prog).state
        cov1 = state.cov()
        mu1 = state.means()

        eng, prog = setup_eng(M)
        with prog.context as q:
            for i in range(M):
                ops.Sgate(1) | q[i]
                ops.Dgate(A) | q[i]
                ops.LossChannel(loss_in[i]) | q[i]
            ops.Interferometer(U) | q
            for i in range(M):
                ops.LossChannel(loss_out[i]) | q[i]

        state = eng.run(prog).state
        cov2 = state.cov()
        mu2 = state.means()

        assert np.allclose(cov1, cov2, atol=tol, rtol=0)
        assert np.allclose(mu1, mu2, atol=tol, rtol=0)

        u, s, v = np.linalg.svd(T)

        eng, prog = setup_eng(M)
        with prog.context as q:
            for i in range(M):
                ops.Sgate(1) | q[i]
                ops.Dgate(A) | q[i]
            ops.Interferometer(v) | q
            for i in range(M):
                ops.LossChannel(s[i] ** 2) | q[i]
            ops.Interferometer(u) | q

        state = eng.run(prog).state
        cov3 = state.cov()
        mu3 = state.means()

        assert np.allclose(cov1, cov3, atol=tol, rtol=0)
        assert np.allclose(mu1, mu3, atol=tol, rtol=0)

        T1 = u * s
        eng, prog = setup_eng(M)
        with prog.context as q:
            for i in range(M):
                ops.Sgate(1) | q[i]
                ops.Dgate(A) | q[i]
            ops.PassiveChannel(v) | q
            ops.PassiveChannel(T1) | q

        state = eng.run(prog).state
        cov4 = state.cov()
        mu4 = state.means()

        assert np.allclose(cov1, cov4, atol=tol, rtol=0)
        assert np.allclose(mu1, mu4, atol=tol, rtol=0)


class TestPreparationApplication:
    """Tests that involve state preparation application"""

    @pytest.mark.backends(*backends)
    def test_ket_state_object(self, setup_eng, pure):
        """Test loading a ket from a prior state object"""
        if not pure:
            pytest.skip("Test only runs on pure states")

        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Dgate(0.2, 0.0) | q[0]

        state1 = eng.run(prog).state

        # create new engine
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Ket(state1) | q[0]

        state2 = eng.run(prog).state

        # verify it is the same state
        assert state1 == state2

    @pytest.mark.backends(*backends)
    def test_ket_gaussian_state_object(self, setup_eng):
        """Test exception if loading a ket from a Gaussian state object"""
        eng = sf.Engine("gaussian")
        prog = sf.Program(1)
        state = eng.run(prog).state

        # create new engine
        eng, prog = setup_eng(1)

        with prog.context as q:
            with pytest.raises(ValueError, match="Gaussian states are not supported"):
                ops.Ket(state) | q[0]

    @pytest.mark.backends(*backends)
    def test_ket_mixed_state_object(self, setup_eng, pure):
        """Test exception if loading a ket from a prior mixed state object"""
        if pure:
            pytest.skip("Test only runs on mixed states")

        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Dgate(0.2, 0.0) | q[0]

        state1 = eng.run(prog).state

        # create new engine
        eng, prog = setup_eng(1)

        with prog.context as q:
            with pytest.raises(ValueError, match="Fock state is not pure"):
                ops.Ket(state1) | q[0]

    @pytest.mark.backends(*backends)
    def test_dm_state_object(self, setup_eng, tol):
        """Test loading a density matrix from a prior state object"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Dgate(0.2, 0.0) | q[0]

        state1 = eng.run(prog).state

        # create new engine
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.DensityMatrix(state1) | q[0]

        state2 = eng.run(prog).state

        # verify it is the same state
        assert np.allclose(state1.dm(), state2.dm(), atol=tol, rtol=0)

    @pytest.mark.backends(*backends)
    def test_dm_gaussian_state_object(self, setup_eng):
        """Test exception if loading a ket from a Gaussian state object"""
        eng = sf.Engine("gaussian")
        prog = sf.Program(1)
        state = eng.run(prog).state

        # create new engine
        eng, prog = setup_eng(1)

        with prog.context as q:
            with pytest.raises(ValueError, match="Gaussian states are not supported"):
                ops.DensityMatrix(state) | q[0]


@pytest.mark.backends(*backends)
class TestKetDensityMatrixIntegration:
    """Tests for the frontend Fock multi-mode state preparations"""

    def test_ket_input_validation(self, setup_eng, hbar, cutoff):
        """Test exceptions"""
        mu = np.array([0.0, 0.0])
        cov = np.identity(2)
        state1 = BaseGaussianState((mu, cov), 1)
        state2 = BaseFockState(np.zeros(cutoff), 1, False, cutoff)

        eng, prog = setup_eng(2)

        with prog.context as q:
            with pytest.raises(ValueError, match="Gaussian states are not supported"):
                ops.Ket(state1) | q[0]
            with pytest.raises(ValueError, match="not pure"):
                ops.Ket(state2) | q[0]

    def test_ket_one_mode(self, setup_eng, hbar, cutoff, tol):
        """Tests single mode ket preparation"""
        eng, prog = setup_eng(2)
        ket0 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(-1, 1, cutoff)
        ket0 = ket0 / np.linalg.norm(ket0)
        with prog.context as q:
            ops.Ket(ket0) | q[0]
        state = eng.run(prog, **{"modes": [0]}).state
        assert np.allclose(state.dm(), np.outer(ket0, ket0.conj()), atol=tol, rtol=0)

        eng.reset()

        prog = sf.Program(2)
        state1 = BaseFockState(ket0, 1, True, cutoff)
        with prog.context as q:
            ops.Ket(state1) | q[0]
        state2 = eng.run(prog, **{"modes": [0]}).state
        assert np.allclose(state1.dm(), state2.dm(), atol=tol, rtol=0)

    def test_ket_two_mode(self, setup_eng, hbar, cutoff, tol):
        """Tests multimode ket preparation"""
        eng, prog = setup_eng(2)
        ket0 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(-1, 1, cutoff)
        ket0 = ket0 / np.linalg.norm(ket0)
        ket1 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(-1, 1, cutoff)
        ket1 = ket1 / np.linalg.norm(ket1)

        ket = np.outer(ket0, ket1)
        with prog.context as q:
            ops.Ket(ket) | q
        state = eng.run(prog).state
        assert np.allclose(state.dm(), np.einsum("ij,kl->ikjl", ket, ket.conj()), atol=tol, rtol=0)

        eng.reset()

        prog = sf.Program(2)
        state1 = BaseFockState(ket, 2, True, cutoff)
        with prog.context as q:
            ops.Ket(state1) | q
        state2 = eng.run(prog).state
        assert np.allclose(state1.dm(), state2.dm(), atol=tol, rtol=0)

    def test_dm_input_validation(self, setup_eng, hbar, cutoff, tol):
        """Test exceptions"""
        mu = np.array([0.0, 0.0])
        cov = np.identity(2)
        state = BaseGaussianState((mu, cov), 1)

        eng, prog = setup_eng(2)

        with prog.context as q:
            with pytest.raises(ValueError, match="Gaussian states are not supported"):
                ops.DensityMatrix(state) | q[0]

    def test_dm_one_mode(self, setup_eng, hbar, cutoff, tol):
        """Tests single mode DM preparation"""
        eng, prog = setup_eng(2)

        ket = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(-1, 1, cutoff)
        ket = ket / np.linalg.norm(ket)
        rho = np.outer(ket, ket.conj())
        with prog.context as q:
            ops.DensityMatrix(rho) | q[0]
        state = eng.run(prog, **{"modes": [0]}).state
        assert np.allclose(state.dm(), rho, atol=tol, rtol=0)

        eng.reset()

        prog = sf.Program(2)
        state1 = BaseFockState(rho, 1, False, cutoff)
        with prog.context as q:
            ops.DensityMatrix(state1) | q[0]
        state2 = eng.run(prog, **{"modes": [0]}).state
        assert np.allclose(state1.dm(), state2.dm(), atol=tol, rtol=0)

    def test_dm_two_mode(self, setup_eng, hbar, cutoff, tol):
        """Tests multimode dm preparation"""
        eng, prog = setup_eng(2)

        ket0 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(-1, 1, cutoff)
        ket0 = ket0 / np.linalg.norm(ket0)
        ket1 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(-1, 1, cutoff)
        ket1 = ket1 / np.linalg.norm(ket1)

        ket = np.outer(ket0, ket1)
        rho = np.einsum("ij,kl->ikjl", ket, ket.conj())
        with prog.context as q:
            ops.DensityMatrix(rho) | q
        state = eng.run(prog).state
        assert np.allclose(state.dm(), rho, atol=tol, rtol=0)

        eng.reset()

        prog = sf.Program(2)
        state1 = BaseFockState(rho, 2, False, cutoff)
        with prog.context as q:
            ops.DensityMatrix(state1) | q
        state2 = eng.run(prog).state
        assert np.allclose(state1.dm(), state2.dm(), atol=tol, rtol=0)


@pytest.mark.skipif("tf" not in backends, reason="Tests require TF")
@pytest.mark.backends(*backends)
class TestGaussianGateApplication:
    def test_multimode_gaussian_gate(self, setup_backend, pure):
        """Test applying gaussian gate on multiple modes"""
        num_mode = 1
        eng = sf.Engine("tf", backend_options={"cutoff_dim": 5})
        prog = sf.Program(num_mode)
        S = tf.Variable(random_symplectic(num_mode), dtype=tf.complex128)
        d = tf.Variable(np.random.random(2 * num_mode), dtype=tf.complex128)
        with prog.context as q:
            ops.Ggate(S, d) | q
        # tests that no exceptions are raised
        eng.run(prog).state.ket()

    def test_gradient_gaussian_gate(self, setup_backend, pure):
        if not pure:
            pytest.skip("Test only runs on pure states")
        num_mode = 2
        eng = sf.Engine("tf", backend_options={"cutoff_dim": 5})
        prog = sf.Program(num_mode)
        S = tf.Variable(random_symplectic(num_mode), dtype=tf.complex128)
        d = tf.Variable(np.random.random(2 * num_mode), dtype=tf.complex128)
        with prog.context as q:
            sf.ops.Ggate(S, d) | q
        with tf.GradientTape() as tape:
            if pure:
                state = eng.run(prog).state.ket()
            else:
                state = eng.run(prog).state.dm()
        # tests that no exceptions are raised
        tape.gradient(state, [S, d])

    def test_Ggate_optimization(self, setup_backend, pure):
        if not pure:
            pytest.skip("Test only runs on pure states")
        num_mode = 2
        eng = sf.Engine("tf", backend_options={"cutoff_dim": 5})
        prog = sf.Program(num_mode)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        S = tf.Variable(random_symplectic(num_mode), dtype=tf.complex128)
        d = tf.Variable(np.random.random(2 * num_mode), dtype=tf.complex128)

        prog = sf.Program(num_mode)
        with prog.context as q:
            ops.Ggate(S, d) | q

        loss_vals = []
        for _ in range(11):
            with tf.GradientTape() as tape:
                state_out = eng.run(prog).state.ket()
                loss_val = tf.abs(state_out[1, 1] - 0.25) ** 2
            eng.reset()
            grad_S, gradients_d = tape.gradient(loss_val, [S, d])
            optimizer.apply_gradients([(gradients_d, d)])
            sf.backends.tfbackend.ops.update_symplectic(S, grad_S, lr=0.05)
            loss_vals.append(loss_val)
            print(loss_val)
        assert all([bool(l1 > l2) for l1, l2 in zip(loss_vals, loss_vals[1:])])

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
r"""
Tests for the various Tensorflow-specific symbolic options of the frontend/backend.
"""
# pylint: disable=expression-not-assigned,too-many-public-methods,pointless-statement,no-self-use

import pytest
import numpy as np
from scipy.special import factorial

tf = pytest.importorskip("tensorflow", minversion="2.0")

from strawberryfields.ops import Dgate, Sgate, MeasureX, Thermal, BSgate, MZgate, Ket, S2gate, Fock


# this test file is only supported by the TF backend
pytestmark = pytest.mark.backends("tf")


ALPHA = 0.5


def coherent_state(alpha, cutoff):
    """Returns the Fock representation of the coherent state |alpha> up to dimension given by cutoff"""
    n = np.arange(cutoff)
    return np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n))


def _vac_ket(cutoff):
    """Returns the ket of the vacuum state up to dimension given by cutoff"""
    vac = np.zeros(cutoff)
    vac[0] = 1
    return vac


def _vac_dm(cutoff):
    """Returns the density matrix of the vacuum state up to dimension given by cutoff"""
    vac = _vac_ket(cutoff)
    return np.outer(vac, np.conj(vac))


class TestOneMode:
    """Tests for workflows on one mode."""

    def test_eng_float_parameter_returns_tensor(self, setup_eng):
        """Tests whether eng.run for programs with non-tensor parameters
        successfully returns a Tensor."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        state_data = state.data

        assert isinstance(state_data, tf.Tensor)

    def test_eng_tensor_parameter_returns_tensor(self, setup_eng):
        """Tests whether eng.run for programs with tensor parameters
        successfully returns a Tensor."""
        eng, prog = setup_eng(1)

        alpha = prog.params("alpha")

        with prog.context as q:
            Dgate(alpha) | q

        state = eng.run(prog, args={"alpha": tf.Variable(0.5)}).state
        state_data = state.data

        assert isinstance(state_data, tf.Tensor)

    def test_eng_run_measurements_are_tensors(self, setup_eng):
        """Tests whether eng.run for programs with measurements
        successfully returns a tensors for samples."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q
            MeasureX | q

        res = eng.run(prog)
        assert isinstance(res.samples, tf.Tensor)

    def test_eng_run_state_ket(self, setup_eng, cutoff, pure, tol):
        """Tests whether the ket of the returned state is a
        Tensor object when executed with `eng.run`."""
        if not pure:
            pytest.skip("Tested only for pure states")

        eng, prog = setup_eng(1)
        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        ket = state.ket()
        assert isinstance(ket, tf.Tensor)

        coh_ket = coherent_state(ALPHA, cutoff)
        assert np.allclose(ket, coh_ket, atol=tol, rtol=0.0)

    def test_eng_run_state_dm(self, pure, cutoff, setup_eng, tol):
        """Tests whether the density matrix of the returned state is an
        Tensor object when executed with `eng.run`."""
        if not pure:
            pytest.skip("Tested only for pure states")

        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        dm = state.dm()
        assert isinstance(dm, tf.Tensor)

        ket = coherent_state(ALPHA, cutoff)
        coh_dm = np.einsum("i,j->ij", ket, ket.conj())
        assert np.allclose(dm, coh_dm, atol=tol, rtol=0.0)

    def test_eng_run_state_trace(self, setup_eng, tol):
        """Tests whether the trace of the returned state is an
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        tr = state.trace()
        assert isinstance(tr, tf.Tensor)
        assert np.allclose(tr, 1, atol=tol, rtol=0.0)

    def test_eng_run_state_reduced_dm(self, setup_eng, cutoff, tol):
        """Tests whether the reduced_density matrix of the returned state
        is a Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        rho = state.reduced_dm([0])
        assert isinstance(rho, tf.Tensor)

        ket = coherent_state(ALPHA, cutoff)
        coh_dm = np.einsum("i,j->ij", ket, ket.conj())
        assert np.allclose(rho, coh_dm, atol=tol, rtol=0.0)

    def test_eng_run_state_fidelity_vacuum(self, setup_eng, tol):
        """Tests whether the fidelity_vacuum method of the state returns an
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0) | q

        state = eng.run(prog).state
        fidel_vac = state.fidelity_vacuum()
        assert isinstance(fidel_vac, tf.Tensor)
        assert np.allclose(fidel_vac, 1.0, atol=tol, rtol=0.0)

    def test_eng_run_state_is_vacuum(self, setup_eng):
        """Tests whether the is_vacuum method of the state returns an
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0) | q

        state = eng.run(prog).state
        is_vac = state.is_vacuum()
        assert isinstance(is_vac, tf.Tensor)
        assert np.all(is_vac)

    def test_eng_run_state_fidelity_coherent(self, setup_eng, tol):
        """Tests whether the fidelity of the state with respect to coherent states is
        a Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        fidel_coh = state.fidelity_coherent([ALPHA])
        assert isinstance(fidel_coh, tf.Tensor)
        assert np.allclose(fidel_coh, 1, atol=tol, rtol=0.0)

    def test_eng_run_state_fidelity(self, setup_eng, cutoff, tol):
        """Tests whether the fidelity of the state with respect to a local state is an
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        fidel_coh = state.fidelity(coherent_state(ALPHA, cutoff), 0)
        assert isinstance(fidel_coh, tf.Tensor)
        assert np.allclose(fidel_coh, 1, atol=tol, rtol=0.0)

    def test_eng_run_state_quad_expectation(self, setup_eng, tol, hbar):
        """Tests whether the local quadrature expectation of the state is
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        e, v = state.quad_expectation(0, 0)
        assert isinstance(e, tf.Tensor)
        assert isinstance(v, tf.Tensor)

        true_exp = np.sqrt(hbar / 2.0) * (ALPHA + np.conj(ALPHA))
        true_var = hbar / 2.0
        assert np.allclose(e, true_exp, atol=tol, rtol=0.0)
        assert np.allclose(v, true_var, atol=tol, rtol=0.0)

    def test_eng_run_state_mean_photon(self, setup_eng, tol):
        """Tests whether the local mean photon number of the state is
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        nbar, var = state.mean_photon(0)
        assert isinstance(nbar, tf.Tensor)
        assert isinstance(var, tf.Tensor)

        ref_nbar = np.abs(ALPHA) ** 2
        ref_var = np.abs(ALPHA) ** 2
        assert np.allclose(nbar, ref_nbar, atol=tol, rtol=0.0)
        assert np.allclose(var, ref_var, atol=tol, rtol=0.0)


class TestTwoModes:
    """Tests for workflows on two modes."""

    def test_eval_true_state_all_fock_probs(self, setup_eng, cutoff, batch_size, tol):
        """Tests whether the Fock-basis probabilities of the state return
        a Tensor with the correct value."""
        eng, prog = setup_eng(2)

        with prog.context as q:
            Dgate(ALPHA) | q[0]
            Dgate(-ALPHA) | q[1]

        state = eng.run(prog).state
        probs = state.all_fock_probs()

        assert isinstance(probs, tf.Tensor)

        probs = probs.numpy().flatten()
        ref_probs = (
            np.abs(
                np.outer(coherent_state(ALPHA, cutoff), coherent_state(-ALPHA, cutoff))
            ).flatten()
            ** 2
        )

        if batch_size is not None:
            ref_probs = np.tile(ref_probs, batch_size)

        assert np.allclose(probs, ref_probs, atol=tol, rtol=0.0)

    def test_eval_true_state_fock_prob(self, setup_eng, cutoff, tol):
        """Tests whether the probability of a Fock measurement outcome on the state returns
        a tensor with the correct value."""
        n1 = cutoff // 2
        n2 = cutoff // 3

        eng, prog = setup_eng(2)

        with prog.context as q:
            Dgate(ALPHA) | q[0]
            Dgate(-ALPHA) | q[1]

        state = eng.run(prog).state
        prob = state.fock_prob([n1, n2])
        assert isinstance(prob, tf.Tensor)

        ref_prob = np.abs(
            np.outer(coherent_state(ALPHA, cutoff), coherent_state(-ALPHA, cutoff)) ** 2
        )[n1, n2]
        assert np.allclose(prob, ref_prob, atol=tol, rtol=0.0)


class TestThreeModes:
    """Tests for workflows on three modes."""

    def test_three_mode_fock_state_mean_photon(self, setup_eng, tol):
        """Tests that the mean photon numbers of a three-mode system,
        where one mode is prepared in a Fock state, are correct."""

        eng, prog = setup_eng(3)

        with prog.context as q:
            Fock(1) | q[1]

        result = eng.run(prog)
        state = result.state
        nbar0, var_n0 = state.mean_photon(0)
        nbar1, var_n1 = state.mean_photon(1)
        nbar2, var_n2 = state.mean_photon(2)

        assert np.allclose(nbar0, 0.0, atol=tol, rtol=0)
        assert np.allclose(nbar1, 1.0, atol=tol, rtol=0)
        assert np.allclose(nbar2, 0.0, atol=tol, rtol=0)

        assert np.allclose(var_n0, 0.0, atol=tol, rtol=0)
        assert np.allclose(var_n1, 0.0, atol=tol, rtol=0)
        assert np.allclose(var_n2, 0.0, atol=tol, rtol=0)


class TestGradient:
    """Integration tests for the gradient computation"""

    def test_displacement_mean_photon_gradient(self, setup_eng, tol, batch_size):
        """Tests whether the gradient for the mean photon variance
        on a displaced state is correct."""
        if batch_size is not None:
            pytest.skip(
                "Cannot calculate gradient in batch mode, as tape.gradient "
                "cannot differentiate non-scalar output."
            )

        eng, prog = setup_eng(1)

        alpha = prog.params("alpha")

        with prog.context as q:
            Dgate(alpha) | q

        a = tf.Variable(ALPHA)

        with tf.GradientTape() as tape:
            state = eng.run(prog, args={"alpha": a}).state
            mean, var = state.mean_photon(0)

        # test the mean and variance of the photon number is correct
        assert np.allclose(mean, ALPHA ** 2, atol=tol, rtol=0)
        assert np.allclose(var, ALPHA ** 2, atol=tol, rtol=0)

        # test the gradient of the variance is correct
        grad = tape.gradient(var, [a])
        assert np.allclose(grad, 2 * ALPHA, atol=tol, rtol=0)

    def test_displaced_thermal_mean_photon_gradient(self, setup_eng, tol, batch_size):
        """Tests whether the gradient for the mean photon variance
        on a displaced thermal state is correct:

        E(n)=|a|^2+nbar and var(n)=var_th+|a|^2(1+2nbar)
        """
        if batch_size is not None:
            pytest.skip(
                "Cannot calculate gradient in batch mode, as tape.gradient "
                "cannot differentiate non-scalar output."
            )

        eng, prog = setup_eng(1)

        alpha = prog.params("alpha")
        nbar = prog.params("nbar")

        with prog.context as q:
            Thermal(nbar) | q
            Dgate(alpha) | q

        a = tf.Variable(0.2)
        n = tf.Variable(0.052)

        with tf.GradientTape() as tape:
            state = eng.run(prog, args={"nbar": n, "alpha": a}).state
            mean, var = state.mean_photon(0)

        # test the mean and variance of the photon number is correct
        mean_ex = a ** 2 + n
        var_ex = n ** 2 + n + a ** 2 * (1 + 2 * n)
        assert np.allclose(mean, mean_ex, atol=tol, rtol=0)
        assert np.allclose(var, var_ex, atol=tol, rtol=0)

        # test the gradient of the variance is correct
        grad = tape.gradient(var, [a, n])
        grad_ex = [2 * a * (1 + 2 * n), 2 * n + 1 + 2 * a ** 2]
        assert np.allclose(grad, grad_ex, atol=tol, rtol=0)

    def test_coherent_ket_gradient(self, setup_eng, cutoff, tol, pure, batch_size):
        """Test whether the gradient of the third element (|2>) of the coherent
        state vector is correct."""
        if not pure:
            pytest.skip("Test skipped in mixed state mode")

        if batch_size is not None:
            pytest.skip(
                "Cannot calculate gradient in batch mode, as tape.gradient "
                "cannot differentiate non-scalar output."
            )

        eng, prog = setup_eng(1)

        alpha = prog.params("alpha")

        with prog.context as q:
            Dgate(alpha) | q

        a = tf.Variable(ALPHA)

        with tf.GradientTape() as tape:
            state = eng.run(prog, args={"alpha": a}).state
            res = tf.cast(state.ket()[2], dtype=tf.float64)

        res_ex = np.exp(-0.5 * a ** 2) * a ** 2 / np.sqrt(2)
        assert np.allclose(res, res_ex, atol=tol, rtol=0)

        grad = tape.gradient(res, [a])
        grad_ex = -a * (a ** 2 - 2) * np.exp(-(a ** 2) / 2) / np.sqrt(2)
        assert np.allclose(grad, grad_ex, atol=tol, rtol=0)

    def test_coherent_dm_gradient(self, setup_eng, cutoff, tol, batch_size):
        """Test whether the gradient of the 3, 3 element of the coherent
        density matrix is correct."""
        if batch_size is not None:
            pytest.skip(
                "Cannot calculate gradient in batch mode, as tape.gradient "
                "cannot differentiate non-scalar output."
            )

        eng, prog = setup_eng(1)

        alpha = prog.params("alpha")

        with prog.context as q:
            Dgate(alpha) | q

        a = tf.Variable(ALPHA)

        with tf.GradientTape() as tape:
            state = eng.run(prog, args={"alpha": a}).state
            res = tf.cast(state.dm()[2, 2], dtype=tf.float64)

        res_ex = (np.exp(-0.5 * a ** 2) * a ** 2 / np.sqrt(2)) ** 2
        assert np.allclose(res, res_ex, atol=tol, rtol=0)

        grad = tape.gradient(res, [a])
        grad_ex = -(a ** 3) * (a ** 2 - 2) * np.exp(-(a ** 2))
        assert np.allclose(grad, grad_ex, atol=tol, rtol=0)

    def test_displaced_squeezed_mean_photon_gradient(self, setup_eng, cutoff, tol, batch_size):
        """Test whether the gradient of the mean photon number of a displaced squeezed
        state is correct.

        .. note::

            As this test contains multiple gates being applied to the program,
            this test will fail in TensorFlow 2.1 due to the bug discussed in
            https://github.com/tensorflow/tensorflow/issues/37307, if `tf.einsum` is being used
            in ``tfbackend/ops.py`` rather than _einsum_v1.
        """
        if batch_size is not None:
            pytest.skip(
                "Cannot calculate gradient in batch mode, as tape.gradient "
                "cannot differentiate non-scalar output."
            )

        eng, prog = setup_eng(1)

        with prog.context as q:
            Sgate(prog.params("r"), prog.params("phi")) | q
            Dgate(prog.params("a")) | q

        a = tf.Variable(ALPHA)
        r = tf.Variable(0.105)
        phi = tf.Variable(0.123)

        with tf.GradientTape() as tape:
            state = eng.run(prog, args={"a": a, "r": r, "phi": phi}).state
            mean, _ = state.mean_photon(0)

        # test the mean and variance of the photon number is correct
        mean_ex = a ** 2 + tf.sinh(r) ** 2
        assert np.allclose(mean, mean_ex, atol=tol, rtol=0)

        # test the gradient of the mean is correct
        grad = tape.gradient(mean, [a, r, phi])
        grad_ex = [2 * a, 2 * tf.sinh(r) * tf.cosh(r), 0]
        assert np.allclose(grad, grad_ex, atol=tol, rtol=0)

    def test_BS_state_gradients(self, setup_eng, cutoff, tol, batch_size):
        """Tests whether the gradient for the state created by interfering two single photons at
        a beam splitter is correct."""
        if batch_size is not None:
            pytest.skip(
                "Cannot calculate gradient in batch mode, as tape.gradient "
                "cannot differentiate non-scalar output."
            )

        THETA = 0.3
        PHI = 0.2

        eng, prog = setup_eng(2)
        _theta, _phi = prog.params("theta", "phi")
        photon_11 = np.zeros((cutoff, cutoff), dtype=np.complex64)
        photon_11[1, 1] = 1.0 + 0.0j

        with prog.context as q:
            Ket(photon_11) | (q[0], q[1])
            BSgate(_theta, _phi) | (q[0], q[1])

        theta = tf.Variable(THETA)
        phi = tf.Variable(PHI)

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"theta": theta, "phi": phi}).state
            prob11 = tf.abs(state.ket()[1, 1]) ** 2
            prob02 = tf.abs(state.ket()[0, 2]) ** 2

        theta_grad, phi_grad = tape.gradient(prob11, [theta, phi])
        assert np.allclose(theta_grad, -4 * np.sin(2 * THETA) * np.cos(2 * THETA), atol=tol, rtol=0)
        assert np.allclose(phi_grad, 0.0, atol=tol, rtol=0)

        theta_grad, phi_grad = tape.gradient(prob02, [theta, phi])
        assert np.allclose(theta_grad, np.sin(4 * THETA), atol=tol, rtol=0)
        assert np.allclose(phi_grad, 0.0, atol=tol, rtol=0)

    def test_MZ_state_gradients(self, setup_eng, cutoff, tol, batch_size):
        """Tests whether the gradient for the state created by interfering two single photons at
        a beam splitter is correct."""
        if batch_size is not None:
            pytest.skip(
                "Cannot calculate gradient in batch mode, as tape.gradient "
                "cannot differentiate non-scalar output."
            )

        PHI_IN = 0.3
        PHI_EX = 0.2

        eng, prog = setup_eng(2)
        _phi_in, _phi_ex = prog.params("phi_in", "phi_ex")
        photon_11 = np.zeros((cutoff, cutoff), dtype=np.complex64)
        photon_11[1, 1] = 1.0 + 0.0j

        with prog.context as q:
            Ket(photon_11) | (q[0], q[1])
            MZgate(_phi_in, _phi_ex) | (q[0], q[1])

        phi_in = tf.Variable(PHI_IN)
        phi_ex = tf.Variable(PHI_EX)

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"phi_in": phi_in, "phi_ex": phi_ex}).state
            prob11 = tf.abs(state.ket()[1, 1]) ** 2
            prob02 = tf.abs(state.ket()[0, 2]) ** 2

        phi_in_grad, phi_ex_grad = tape.gradient(prob11, [phi_in, phi_ex])
        assert np.allclose(phi_in_grad, -np.sin(2 * PHI_IN), atol=tol, rtol=0)
        assert np.allclose(phi_ex_grad, 0, atol=tol, rtol=0)

        phi_in_grad, phi_ex_grad = tape.gradient(prob02, [phi_in, phi_ex])
        assert np.allclose(phi_in_grad, np.sin(PHI_IN) * np.cos(PHI_IN), atol=tol, rtol=0)
        assert np.allclose(phi_ex_grad, 0, atol=tol, rtol=0)

    def test_2mode_squeezed_vacuum_gradients(self, setup_eng, cutoff, tol, batch_size):
        """Tests whether the gradient for the probability of the states |0,0> and |1,1>
        created by an S2gate is correct."""
        if batch_size is not None:
            pytest.skip(
                "Cannot calculate gradient in batch mode, as tape.gradient "
                "cannot differentiate non-scalar output."
            )

        R = 0.3
        PHI = 0.2

        eng, prog = setup_eng(2)
        _r, _phi = prog.params("r", "phi")
        vacuum = np.zeros((cutoff, cutoff), dtype=np.complex64)
        vacuum[0, 0] = 1.0 + 0.0j

        with prog.context as q:
            Ket(vacuum) | (q[0], q[1])
            S2gate(_r, _phi) | (q[0], q[1])

        r = tf.Variable(R)
        phi = tf.Variable(PHI)

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"r": r, "phi": phi}).state
            prob00 = tf.abs(state.ket()[0, 0]) ** 2
            prob11 = tf.abs(state.ket()[1, 1]) ** 2

        r_grad, phi_grad = tape.gradient(prob00, [r, phi])
        assert np.allclose(r_grad, -2 * np.tanh(R) / np.cosh(R) ** 2, atol=tol, rtol=0)
        assert np.allclose(phi_grad, 0.0, atol=tol, rtol=0)

        r_grad, phi_grad = tape.gradient(prob11, [r, phi])
        assert np.allclose(
            r_grad, 2 * (np.sinh(R) - np.sinh(R) ** 3) / np.cosh(R) ** 5, atol=tol, rtol=0
        )
        assert np.allclose(phi_grad, 0.0, atol=tol, rtol=0)


@pytest.mark.xfail(
    reason="If this test passes, then the _einsum_v1 patch is no longer needed.",
    strict=True,
    raises=AssertionError,
)
def test_einsum_complex_gradients(tol):
    """Integration test to check the complex gradient
    when using einsum in TensorFlow version 2.1+.

    With TF 2.1+, the legacy tf.einsum was renamed to _einsum_v1, while
    the replacement tf.einsum introduced a bug; the computed einsum
    value is correct when applied to complex tensors, but the returned
    gradient is incorrect. For more details, see
    https://github.com/tensorflow/tensorflow/issues/37307.

    This test is expected to fail, confirming that the complex einsum
    gradient bug is still occuring. If this test passes, it means that
    the bug has been fixed.
    """
    import sys

    del sys.modules["tensorflow"]
    tf = pytest.importorskip("tensorflow", minversion="2.1")

    # import the legacy einsum implementation
    from tensorflow.python.ops.special_math_ops import _einsum_v1

    def f0(h):
        """Sum reduction of complex matrix h@h performed using matmul"""
        return tf.abs(tf.reduce_sum(tf.matmul(h, h)))

    def f1(h):
        """Sum reduction of complex matrix h@h performed using tf.einsum"""
        return tf.abs(tf.reduce_sum(tf.einsum("ab,bc->ac", h, h)))

    def f2(h):
        """Sum reduction of complex matrix h@h performed using _einsum_v1"""
        return tf.abs(tf.reduce_sum(_einsum_v1("ab,bc->ac", h, h)))

    # Create a real 2x2 variable A; this is the variable we will be differentiating
    # the cost function with respect to.
    A = tf.Variable([[0.16513085, 0.9014813], [0.6309742, 0.4345461]], dtype=tf.float32)

    # constant complex tensor
    B = tf.constant([[0.51010704, 0.44353175], [0.4085331, 0.9924923]], dtype=tf.float32)

    grads = []

    for f in (f0, f1, f2):
        with tf.GradientTape() as tape:
            # Create a complex tensor C = A + B*1j
            C = tf.cast(A, dtype=tf.complex64) + 1j * tf.cast(B, dtype=tf.complex64)
            loss = f(C)

        # compute the gradient
        grads.append(tape.gradient(loss, A))

    # gradient of f0 and f2 should agree
    assert np.allclose(grads[0], grads[2], atol=tol, rtol=0)

    # gradient of f0 and f1 should fail
    assert np.allclose(grads[0], grads[1], atol=tol, rtol=0)

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
# pylint: disable=expression-not-assigned,too-many-public-methods,pointless-statement

import pytest

# this test file is only supported by the TF backend
pytestmark = pytest.mark.backends("tf")

import numpy as np
from scipy.special import factorial

try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError):
    pytestmark = pytest.mark.skip("TensorFlow not installed")
else:
    if tf.__version__[:3] != "1.3":
        pytestmark = pytest.mark.skip("Test only runs with TensorFlow 1.3")

from strawberryfields.ops import Dgate, MeasureX


ALPHA = 0.5
evalf = {'eval': False}


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


class TestOneModeSymbolic:
    """Tests for symbolic workflows on one mode."""

    #########################################
    # tests basic eval behaviour of eng.run and states class.

    def test_eng_run_eval_false_returns_tensor(self, setup_eng):
        """Tests whether the eval=False option to the `eng.run` command
        successfully returns an unevaluated Tensor."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog, state_options=evalf).state
        state_data = state.data

        assert isinstance(state_data, tf.Tensor)

    def test_eng_run_eval_false_measurements_are_tensors(self, setup_eng):
        """Tests whether the eval=False option to the `eng.run` command
        successfully returns a unevaluated Tensors for measurment results."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q
            MeasureX | q

        eng.run(prog, eval=False)
        val = q[0].val
        assert isinstance(val, tf.Tensor)

    def test_eng_run_with_session_and_feed_dict(self, setup_eng, batch_size, cutoff, tol):
        """Tests whether passing a tf Session and feed_dict
        through `eng.run` leads to proper numerical simulation."""
        a = tf.Variable(0.5)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf_params = {'session': sess, 'feed_dict': {a: 0.0}}
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(a) | q

        state = eng.run(prog, state_options=tf_params).state

        if state.is_pure:
            k = state.ket()

            if batch_size is not None:
                dm = np.einsum("bi,bj->bij", k, np.conj(k))
            else:
                dm = np.outer(k, np.conj(k))
        else:
            dm = state.dm()

        vac_dm = _vac_dm(cutoff)
        assert np.allclose(dm, vac_dm, atol=tol, rtol=0.0)

    #########################################
    # tests of eval behaviour of ket method

    def test_eng_run_eval_false_state_ket(self, setup_eng, pure):
        """Tests whether the ket of the returned state is an unevaluated
        Tensor object when eval=False is passed to `eng.run`."""
        if not pure:
            pytest.skip("Tested only for pure states")

        eng, prog = setup_eng(1)
        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog, state_options=evalf).state
        ket = state.ket()
        assert isinstance(ket, tf.Tensor)

    def test_eval_false_state_ket(self, setup_eng, pure):
        """Tests whether the ket of the returned state is an unevaluated
        Tensor object when eval=False is passed to the ket method of a state."""
        if not pure:
            pytest.skip("Tested only for pure states")

        eng, prog = setup_eng(1)
        with prog.context as q:
            Dgate(0.5) | q
        state = eng.run(prog).state
        ket = state.ket(eval=False)
        assert isinstance(ket, tf.Tensor)

    def test_eval_true_state_ket(self, setup_eng, pure, cutoff, tol):
        """Tests whether the ket of the returned state is equal to the
        correct value when eval=True is passed to the ket method of a state."""
        if not pure:
            pytest.skip("Tested only for pure states")

        eng, prog = setup_eng(1)
        state = eng.run(prog).state
        ket = state.ket(eval=True)
        vac = _vac_ket(cutoff)
        assert np.allclose(ket, vac, atol=tol, rtol=0.0)

    #########################################
    # tests of eval behaviour of dm method

    def test_eng_run_eval_false_state_dm(self, pure, setup_eng):
        """Tests whether the density matrix of the returned state is an
        unevaluated Tensor object when eval=False is passed to `eng.run`."""
        if not pure:
            pytest.skip("Tested only for pure states")

        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog, state_options=evalf).state
        dm = state.dm()
        assert isinstance(dm, tf.Tensor)

    def test_eval_false_state_dm(self, pure, setup_eng):
        """Tests whether the density matrix of the returned state is an
        unevaluated Tensor object when eval=False is passed to the ket method of a state."""
        if not pure:
            pytest.skip("Tested only for pure states")

        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog).state
        dm = state.dm(eval=False)
        assert isinstance(dm, tf.Tensor)

    def test_eval_true_state_dm(self, setup_eng, pure, cutoff, tol):
        """Tests whether the density matrix of the returned state is equal
        to the correct value when eval=True is passed to the ket method of a state."""
        if not pure:
            pytest.skip("Tested only for pure states")

        eng, prog = setup_eng(1)
        state = eng.run(prog).state
        dm = state.dm(eval=True)
        vac_dm = _vac_dm(cutoff)
        assert np.allclose(dm, vac_dm, atol=tol, rtol=0.0)

    #########################################
    # tests of eval behaviour of trace method

    def test_eng_run_eval_false_state_trace(self, setup_eng):
        """Tests whether the trace of the returned state is an
        unevaluated Tensor object when eval=False is passed to `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog, state_options=evalf).state
        tr = state.trace()
        assert isinstance(tr, tf.Tensor)

    def test_eval_false_state_trace(self, setup_eng):
        """Tests whether the trace of the returned state is an
        unevaluated Tensor object when eval=False is passed to the trace method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog).state
        tr = state.trace(eval=False)
        assert isinstance(tr, tf.Tensor)

    def test_eval_true_state_trace(self, setup_eng, tol):
        """Tests whether the trace of the returned state is equal
        to the correct value when eval=True is passed to the trace method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q
        state = eng.run(prog).state

        tr = state.trace(eval=True)
        assert np.allclose(tr, 1, atol=tol, rtol=0.0)

    #########################################
    # tests of eval behaviour of reduced_dm method

    def test_eng_run_eval_false_state_reduced_dm(self, setup_eng):
        """Tests whether the reduced_density matrix of the returned state
        is an unevaluated Tensor object when eval=False is passed to `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog, state_options=evalf).state
        rho = state.reduced_dm([0])
        assert isinstance(rho, tf.Tensor)

    def test_eval_false_state_reduced_dm(self, setup_eng):
        """Tests whether the reduced density matrix of the returned state is an
        unevaluated Tensor object when eval=False is passed to the reduced_dm method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog).state
        rho = state.reduced_dm([0], eval=False)
        assert isinstance(rho, tf.Tensor)

    def test_eval_true_state_reduced_dm(self, setup_eng, cutoff, tol):
        """Tests whether the reduced density matrix of the returned state is
        equal to the correct value when eval=True is passed to the reduced_dm method of a state."""
        eng, prog = setup_eng(1)
        state = eng.run(prog).state
        rho = state.reduced_dm([0], eval=True)
        vac_dm = _vac_dm(cutoff)
        assert np.allclose(rho, vac_dm, atol=tol, rtol=0.0)

    #########################################
    # tests of eval behaviour of fidelity_vacuum method

    def test_eng_run_eval_false_state_fidelity_vacuum(self, setup_eng):
        """Tests whether the fidelity_vacuum method of the state returns an
        unevaluated Tensor object when eval=False is passed to `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog, state_options=evalf).state
        fidel_vac = state.fidelity_vacuum()
        assert isinstance(fidel_vac, tf.Tensor)

    def test_eval_false_state_fidelity_vacuum(self, setup_eng):
        """Tests whether the vacuum fidelity of the returned state is an
        unevaluated Tensor object when eval=False is passed to the fidelity_vacuum method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog).state
        fidel_vac = state.fidelity_vacuum(eval=False)
        assert isinstance(fidel_vac, tf.Tensor)

    def test_eval_true_state_fidelity_vacuum(self, setup_eng, tol):
        """Tests whether the vacuum fidelity of the returned state is equal
        to the correct value when eval=True is passed to the fidelity_vacuum method of a state."""
        eng, prog = setup_eng(1)
        state = eng.run(prog).state
        fidel_vac = state.fidelity_vacuum(eval=True)
        assert np.allclose(fidel_vac, 1.0, atol=tol, rtol=0.0)

    #########################################
    # tests of eval behaviour of is_vacuum method

    def test_eng_run_eval_false_state_is_vacuum(self, setup_eng):
        """Tests whether the is_vacuum method of the state returns an
        unevaluated Tensor object when eval=False is passed to `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog, state_options=evalf).state
        is_vac = state.is_vacuum()
        assert isinstance(is_vac, tf.Tensor)

    def test_eval_false_state_is_vacuum(self, setup_eng):
        """Tests whether the is_vacuum method of the state returns an
        unevaluated Tensor object when eval=False is passed to the is_vacuum method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0.5) | q

        state = eng.run(prog).state
        is_vac = state.is_vacuum(eval=False)
        assert isinstance(is_vac, tf.Tensor)

    def test_eval_true_state_is_vacuum(self, setup_eng):
        """Tests whether the is_vacuum method of the state returns
        the correct value when eval=True is passed to the is_vacuum method of a state."""
        eng, prog = setup_eng(1)
        state = eng.run(prog).state
        is_vac = state.is_vacuum(eval=True)
        assert np.all(is_vac)

    #########################################
    # tests of eval behaviour of fidelity_coherent method

    def test_eng_run_eval_false_state_fidelity_coherent(self, setup_eng):
        """Tests whether the fidelity of the state with respect to coherent states is
        an unevaluated Tensor object when eval=False is passed to `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog, state_options=evalf).state
        fidel_coh = state.fidelity_coherent([ALPHA])
        assert isinstance(fidel_coh, tf.Tensor)

    def test_eval_false_state_fidelity_coherent(self, setup_eng):
        """Tests whether the fidelity of the state with respect to coherent states
        is an unevaluated Tensor object when eval=False is passed to the fidelity_coherent method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        fidel_coh = state.fidelity_coherent([ALPHA], eval=False)
        assert isinstance(fidel_coh, tf.Tensor)

    def test_eval_true_state_fidelity_coherent(self, setup_eng, tol):
        """Tests whether the fidelity of the state with respect to coherent states returns
        the correct value when eval=True is passed to the fidelity_coherent method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        fidel_coh = state.fidelity_coherent([ALPHA], eval=True)
        assert np.allclose(fidel_coh, 1, atol=tol, rtol=0.0)

    #########################################
    # tests of eval behaviour of fidelity method

    def test_eng_run_eval_false_state_fidelity(self, setup_eng, cutoff):
        """Tests whether the fidelity of the state with respect to a local state is an
        unevaluated Tensor object when eval=False is passed to `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog, state_options=evalf).state
        fidel = state.fidelity(coherent_state(ALPHA, cutoff), 0)
        assert isinstance(fidel, tf.Tensor)

    def test_eval_false_state_fidelity(self, setup_eng, cutoff):
        """Tests whether the fidelity of the state with respect to a local state is
        an unevaluated Tensor object when eval=False is passed to the fidelity method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        fidel = state.fidelity(coherent_state(ALPHA, cutoff), 0, eval=False)
        assert isinstance(fidel, tf.Tensor)

    def test_eval_true_state_fidelity(self, setup_eng, cutoff, tol):
        """Tests whether the fidelity of the state with respect to a local state
        returns the correct value when eval=True is passed to the fidelity method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        fidel_coh = state.fidelity(coherent_state(ALPHA, cutoff), 0, eval=True)
        assert np.allclose(fidel_coh, 1, atol=tol, rtol=0.0)

    #########################################
    # tests of eval behaviour of quad_expectation method

    def test_eng_run_eval_false_state_quad_expectation(self, setup_eng):
        """Tests whether the local quadrature expectation of the state is
        unevaluated Tensor object when eval=False is passed to `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog, state_options=evalf).state
        e, v = state.quad_expectation(0, 0)
        assert isinstance(e, tf.Tensor)
        assert isinstance(v, tf.Tensor)

    def test_eval_false_state_quad_expectation(self, setup_eng):
        """Tests whether the local quadrature expectation value of the state is
        an unevaluated Tensor object when eval=False is passed to the quad_expectation method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q
        state = eng.run(prog).state

        e, v = state.quad_expectation(0, 0, eval=False)
        assert isinstance(e, tf.Tensor)
        assert isinstance(v, tf.Tensor)

    def test_eval_true_state_quad_expectation(self, setup_eng, tol, hbar):
        """Tests whether the local quadrature expectation value of the state returns
        the correct value when eval=True is passed to the quad_expectation method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        e, v = state.quad_expectation(0, 0, eval=True)
        true_exp = np.sqrt(hbar / 2.0) * (ALPHA + np.conj(ALPHA))
        true_var = hbar / 2.0
        assert np.allclose(e, true_exp, atol=tol, rtol=0.0)
        assert np.allclose(v, true_var, atol=tol, rtol=0.0)

    #########################################
    # tests of eval behaviour of mean_photon method

    def test_eng_run_eval_false_state_mean_photon(self, setup_eng):
        """Tests whether the local mean photon number of the state is
        unevaluated Tensor object when eval=False is passed to `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog, state_options=evalf).state
        nbar, var = state.mean_photon(0)
        assert isinstance(nbar, tf.Tensor)
        assert isinstance(var, tf.Tensor)

    def test_eval_false_state_mean_photon(self, setup_eng):
        """Tests whether the local mean photon number of the state is
        an unevaluated Tensor object when eval=False is passed to the mean_photon_number method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        nbar, var = state.mean_photon(0, eval=False)
        assert isinstance(nbar, tf.Tensor)
        assert isinstance(var, tf.Tensor)

    def test_eval_true_state_mean_photon(self, setup_eng, tol):
        """Tests whether the local mean photon number of the state returns
        the correct value when eval=True is passed to the mean_photon method of a state."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        nbar, var = state.mean_photon(0, eval=True)
        ref_nbar = np.abs(ALPHA) ** 2
        ref_var = np.abs(ALPHA) ** 2
        assert np.allclose(nbar, ref_nbar, atol=tol, rtol=0.0)
        assert np.allclose(var, ref_var, atol=tol, rtol=0.0)


class TestTwoModeSymbolic:
    """Tests for symbolic workflows on two modes."""

    #########################################
    # tests of eval behaviour of all_fock_probs method

    def test_eng_run_eval_false_state_all_fock_probs(self, setup_eng):
        """Tests whether the Fock-basis probabilities of the state are an
        unevaluated Tensor object when eval=False is passed to `eng.run`."""
        eng, prog = setup_eng(2)

        with prog.context as q:
            Dgate(ALPHA) | q[0]
            Dgate(-ALPHA) | q[1]

        state = eng.run(prog, state_options=evalf).state
        probs = state.all_fock_probs()
        assert isinstance(probs, tf.Tensor)

    def test_eval_false_state_all_fock_probs(self, setup_eng):
        """Tests whether the Fock-basis probabilities of the state are an
        unevaluated Tensor object when eval=False is passed to the all_fock_probs method of a state."""
        eng, prog = setup_eng(2)

        with prog.context as q:
            Dgate(ALPHA) | q[0]
            Dgate(-ALPHA) | q[1]

        state = eng.run(prog).state
        probs = state.all_fock_probs(eval=False)
        assert isinstance(probs, tf.Tensor)

    def test_eval_true_state_all_fock_probs(self, setup_eng, cutoff, batch_size, tol):
        """Tests whether the Fock-basis probabilities of the state return
        the correct value when eval=True is passed to the all_fock_probs method of a state."""
        eng, prog = setup_eng(2)

        with prog.context as q:
            Dgate(ALPHA) | q[0]
            Dgate(-ALPHA) | q[1]

        state = eng.run(prog).state
        probs = state.all_fock_probs(eval=True).flatten()
        ref_probs = np.abs(np.outer(coherent_state(ALPHA, cutoff),\
                                    coherent_state(-ALPHA, cutoff))).flatten() ** 2

        if batch_size is not None:
            ref_probs = np.tile(ref_probs, batch_size)

        assert np.allclose(probs, ref_probs, atol=tol, rtol=0.0)

    #########################################
    # tests of eval behaviour of fock_prob method

    def test_eng_run_eval_false_state_fock_prob(self, setup_eng, cutoff):
        """Tests whether the probability of a Fock measurement outcome on the state is an
        unevaluated Tensor object when eval=False is passed to `eng.run`."""
        eng, prog = setup_eng(2)

        with prog.context as q:
            Dgate(ALPHA) | q[0]
            Dgate(-ALPHA) | q[1]

        state = eng.run(prog, state_options=evalf).state
        prob = state.fock_prob([cutoff // 2, cutoff // 2])
        assert isinstance(prob, tf.Tensor)

    def test_eval_false_state_fock_prob(self, setup_eng, cutoff):
        """Tests whether the probability of a Fock measurement outcome on the state is an
        unevaluated Tensor object when eval=False is passed to the fock_prob method of a state."""
        eng, prog = setup_eng(2)

        with prog.context as q:
            Dgate(ALPHA) | q[0]
            Dgate(-ALPHA) | q[1]

        state = eng.run(prog).state
        prob = state.fock_prob([cutoff // 2, cutoff // 2], eval=False)
        assert isinstance(prob, tf.Tensor)

    def test_eval_true_state_fock_prob(self, setup_eng, cutoff, tol):
        """Tests whether the probability of a Fock measurement outcome on the state returns
         the correct value when eval=True is passed to the fock_prob method of a state."""
        n1 = cutoff // 2
        n2 = cutoff // 3

        eng, prog = setup_eng(2)

        with prog.context as q:
            Dgate(ALPHA) | q[0]
            Dgate(-ALPHA) | q[1]

        state = eng.run(prog).state
        prob = state.fock_prob([n1, n2], eval=True)
        ref_prob = np.abs(
            np.outer(coherent_state(ALPHA, cutoff), coherent_state(-ALPHA, cutoff)) ** 2
        )[n1, n2]
        assert np.allclose(prob, ref_prob, atol=tol, rtol=0.0)

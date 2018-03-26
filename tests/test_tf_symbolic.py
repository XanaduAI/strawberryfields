 ##############################################################################
#
# Tests for the various Tensorflow-specific symbolic options of the frontend/backend.
# These tests are specific to backends which are written in Tensorflow
#
##############################################################################

import unittest
import os, sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.special import factorial
import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
from defaults import SymbolicBaseTest


###################################################################

class OneModeSymbolicTests(SymbolicBaseTest):
  """Basic tests for symbolic workflows."""
  num_subsystems = 1
  def setUp(self):
    super().setUp()
    self.vac = np.zeros(self.D)
    self.vac[0] = 1
    self.vac_dm = np.outer(self.vac, np.conj(self.vac))
    self.alpha = 0.5
    self.coh = np.array([np.exp(- 0.5 * np.abs(self.alpha) ** 2) * self.alpha ** k / np.sqrt(factorial(k)) for k in range(self.D)])

  #########################################
  # tests basic eval behaviour of eng.run and states class

  def test_eng_run_eval_false_returns_tensor(self):
    """Tests whether the eval=False option to the `eng.run` command
    successfully returns an unevaluated Tensor."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    state_data = state.data
    self.assertTrue(isinstance(state_data, tf.Tensor))

  def test_eng_run_eval_false_measurements_are_tensors(self):
    """Tests whether the eval=False option to the `eng.run` command
    successfully returns a unevaluated Tensors for measurment results."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
      MeasureX   | q
    eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    val = q[0].val
    self.assertTrue(isinstance(val, tf.Tensor))

  def test_eng_run_with_session_and_feed_dict(self):
    """Tests whether passing a tf Session and feed_dict
    through `eng.run` leads to proper numerical simulation."""
    a = tf.Variable(0.5)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(a) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D,
                    session=sess, feed_dict={a: 0.0})
    if state.is_pure:
      k = state.ket()
      if self.batched:
        dm = np.einsum('bi,bj->bij', k, np.conj(k))
      else:
        dm = np.outer(k, np.conj(k))
    else:
      dm = state.dm()
    self.assertAllEqual(dm, self.vac_dm)

  #########################################
  # tests of eval behaviour of ket method

  def test_eng_run_eval_false_state_ket(self):
    """Tests whether the ket of the returned state is an unevaluated
    Tensor object when eval=False is passed to `eng.run`."""
    if self.args.mixed:
      return
    else:
      eng, q = sf.Engine(self.num_subsystems)
      with eng:
        Dgate(0.5) | q
      state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
      ket = state.ket()
      self.assertTrue(isinstance(ket, tf.Tensor))

  def test_eval_false_state_ket(self):
    """Tests whether the ket of the returned state is an unevaluated
    Tensor object when eval=False is passed to the ket method of a state."""
    if self.args.mixed:
      return
    else:
      eng, q = sf.Engine(self.num_subsystems)
      with eng:
        Dgate(0.5) | q
      state = eng.run(backend=self.backend, cutoff_dim=self.D)
      ket = state.ket(eval=False)
      self.assertTrue(isinstance(ket, tf.Tensor))

  def test_eval_true_state_ket(self):
    """Tests whether the ket of the returned state is equal to the
    correct value when eval=True is passed to the ket method of a state."""
    if self.args.mixed:
      return
    else:
      eng, q = sf.Engine(self.num_subsystems)
      state = eng.run(backend=self.backend, cutoff_dim=self.D)
      ket = state.ket(eval=True)
      self.assertAllAlmostEqual(ket, self.vac, delta=self.tol)

  #########################################
  # tests of eval behaviour of dm method

  def test_eng_run_eval_false_state_dm(self):
    """Tests whether the density matrix of the returned state is an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    if self.args.mixed:
      return
    else:
      eng, q = sf.Engine(self.num_subsystems)
      with eng:
        Dgate(0.5) | q
      state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
      dm = state.dm()
      self.assertTrue(isinstance(dm, tf.Tensor))

  def test_eval_false_state_dm(self):
    """Tests whether the density matrix of the returned state is an
    unevaluated Tensor object when eval=False is passed to the ket method of a state."""
    if self.args.mixed:
      return
    else:
      eng, q = sf.Engine(self.num_subsystems)
      with eng:
        Dgate(0.5) | q
      state = eng.run(backend=self.backend, cutoff_dim=self.D)
      dm = state.dm(eval=False)
      self.assertTrue(isinstance(dm, tf.Tensor))

  def test_eval_true_state_dm(self):
    """Tests whether the density matrix of the returned state is equal
    to the correct value when eval=True is passed to the ket method of a state."""
    if self.args.mixed:
      return
    else:
      eng, q = sf.Engine(self.num_subsystems)
      state = eng.run(backend=self.backend, cutoff_dim=self.D)
      dm = state.dm(eval=True)
      self.assertAllAlmostEqual(dm, self.vac_dm, delta=self.tol)

  #########################################
  # tests of eval behaviour of trace method

  def test_eng_run_eval_false_state_trace(self):
    """Tests whether the trace of the returned state is an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    tr = state.trace()
    self.assertTrue(isinstance(tr, tf.Tensor))

  def test_eval_false_state_trace(self):
    """Tests whether the trace of the returned state is an
    unevaluated Tensor object when eval=False is passed to the trace method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    tr = state.trace(eval=False)
    self.assertTrue(isinstance(tr, tf.Tensor))

  def test_eval_true_state_trace(self):
    """Tests whether the trace of the returned state is equal
    to the correct value when eval=True is passed to the trace method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    tr = state.trace(eval=True)
    self.assertAllAlmostEqual(tr, 1, delta=self.tol)

  #########################################
  # tests of eval behaviour of reduced_dm method

  def test_eng_run_eval_false_state_reduced_dm(self):
    """Tests whether the reduced_density matrix of the returned state
    is an unevaluated Tensor object when eval=False is passed to `eng.run`."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    rho = state.reduced_dm([0])
    self.assertTrue(isinstance(rho, tf.Tensor))

  def test_eval_false_state_reduced_dm(self):
    """Tests whether the reduced density matrix of the returned state is an
    unevaluated Tensor object when eval=False is passed to the reduced_dm method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    rho = state.reduced_dm([0], eval=False)
    self.assertTrue(isinstance(rho, tf.Tensor))

  def test_eval_true_state_reduced_dm(self):
    """Tests whether the reduced density matrix of the returned state is
    equal to the correct value when eval=True is passed to the reduced_dm method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    rho = state.reduced_dm([0], eval=True)
    self.assertAllAlmostEqual(rho, self.vac_dm, delta=self.tol)

  #########################################
  # tests of eval behaviour of fidelity_vacuum method

  def test_eng_run_eval_false_state_fidelity_vacuum(self):
    """Tests whether the fidelity_vacuum method of the state returns an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    fidel_vac = state.fidelity_vacuum()
    self.assertTrue(isinstance(fidel_vac, tf.Tensor))

  def test_eval_false_state_fidelity_vacuum(self):
    """Tests whether the vacuum fidelity of the returned state is an
    unevaluated Tensor object when eval=False is passed to the fidelity_vacuum method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    fidel_vac = state.fidelity_vacuum(eval=False)
    self.assertTrue(isinstance(fidel_vac, tf.Tensor))

  def test_eval_true_state_fidelity_vacuum(self):
    """Tests whether the vacuum fidelity of the returned state is equal
    to the correct value when eval=True is passed to the fidelity_vacuum method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    fidel_vac = state.fidelity_vacuum(eval=True)
    self.assertAllAlmostEqual(fidel_vac, 1., delta=self.tol)

  #########################################
  # tests of eval behaviour of is_vacuum method

  def test_eng_run_eval_false_state_is_vacuum(self):
    """Tests whether the is_vacuum method of the state returns an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    is_vac = state.is_vacuum()
    self.assertTrue(isinstance(is_vac, tf.Tensor))

  def test_eval_false_state_is_vacuum(self):
    """Tests whether the is_vacuum method of the state returns an
    unevaluated Tensor object when eval=False is passed to the is_vacuum method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(0.5) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    is_vac = state.is_vacuum(eval=False)
    self.assertTrue(isinstance(is_vac, tf.Tensor))

  def test_eval_true_state_is_vacuum(self):
    """Tests whether the is_vacuum method of the state returns
    the correct value when eval=True is passed to the is_vacuum method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    is_vac = state.is_vacuum(eval=True)
    self.assertAllTrue(is_vac)

  #########################################
  # tests of eval behaviour of fidelity_coherent method

  def test_eng_run_eval_false_state_fidelity_coherent(self):
    """Tests whether the fidelity of the state with respect to coherent states is
    an unevaluated Tensor object when eval=False is passed to `eng.run`."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    fidel_coh = state.fidelity_coherent([self.alpha])
    self.assertTrue(isinstance(fidel_coh, tf.Tensor))

  def test_eval_false_state_fidelity_coherent(self):
    """Tests whether the fidelity of the state with respect to coherent states
    is an unevaluated Tensor object when eval=False is passed to the fidelity_coherent method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    fidel_coh = state.fidelity_coherent([self.alpha], eval=False)
    self.assertTrue(isinstance(fidel_coh, tf.Tensor))

  def test_eval_true_state_fidelity_coherent(self):
    """Tests whether the fidelity of the state with respect to coherent states returns
    the correct value when eval=True is passed to the fidelity_coherent method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    fidel_coh = state.fidelity_coherent([self.alpha], eval=True)
    self.assertAllAlmostEqual(fidel_coh, 1, delta=self.tol)

  #########################################
  # tests of eval behaviour of fidelity method

  def test_eng_run_eval_false_state_fidelity(self):
    """Tests whether the fidelity of the state with respect to a local state is an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    fidel = state.fidelity(self.coh, 0)
    self.assertTrue(isinstance(fidel, tf.Tensor))

  def test_eval_false_state_fidelity(self):
    """Tests whether the fidelity of the state with respect to a local state is
    an unevaluated Tensor object when eval=False is passed to the fidelity method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    fidel = state.fidelity(self.coh, 0, eval=False)
    self.assertTrue(isinstance(fidel, tf.Tensor))

  def test_eval_true_state_fidelity(self):
    """Tests whether the fidelity of the state with respect to a local state
    returns the correct value when eval=True is passed to the fidelity method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    fidel_coh = state.fidelity(self.coh, 0, eval=True)
    self.assertAllAlmostEqual(fidel_coh, 1, delta=self.tol)

  #########################################
  # tests of eval behaviour of quad_expectation method

  def test_eng_run_eval_false_state_quad_expectation(self):
    """Tests whether the local quadrature expectation of the state is
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    e, v = state.quad_expectation(0, 0)
    self.assertTrue(isinstance(e, tf.Tensor))
    self.assertTrue(isinstance(v, tf.Tensor))

  def test_eval_false_state_quad_expectation(self):
    """Tests whether the local quadrature expectation value of the state is
    an unevaluated Tensor object when eval=False is passed to the quad_expectation method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    e, v = state.quad_expectation(0, 0, eval=False)
    self.assertTrue(isinstance(e, tf.Tensor))
    self.assertTrue(isinstance(v, tf.Tensor))

  def test_eval_true_state_quad_expectation(self):
    """Tests whether the local quadrature expectation value of the state returns
    the correct value when eval=True is passed to the quad_expectation method of a state."""
    hbar = 2.
    eng, q = sf.Engine(self.num_subsystems, hbar=hbar)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    e, v = state.quad_expectation(0, 0, eval=True)
    true_exp = np.sqrt(hbar / 2.) * (self.alpha + np.conj(self.alpha))
    true_var = hbar / 2.
    self.assertAllAlmostEqual(e, true_exp, delta=self.tol)
    self.assertAllAlmostEqual(v, true_var, delta=self.tol)

  #########################################
  # tests of eval behaviour of mean_photon method

  def test_eng_run_eval_false_state_mean_photon(self):
    """Tests whether the local mean photon number of the state is
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    nbar = state.mean_photon(0)
    self.assertTrue(isinstance(nbar, tf.Tensor))

  def test_eval_false_state_mean_photon(self):
    """Tests whether the local mean photon number of the state is
    an unevaluated Tensor object when eval=False is passed to the mean_photon_number method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    nbar = state.mean_photon(0, eval=False)
    self.assertTrue(isinstance(nbar, tf.Tensor))

  def test_eval_true_state_mean_photon(self):
    """Tests whether the local mean photon number of the state returns
    the correct value when eval=True is passed to the mean_photon method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    nbar = state.mean_photon(0, eval=True)
    ref_nbar = np.abs(self.alpha) ** 2
    self.assertAllAlmostEqual(nbar, ref_nbar, delta=self.tol)

  #########################################


class TwoModeSymbolicTests(SymbolicBaseTest):
  """Basic tests for symbolic workflows."""
  num_subsystems = 2
  def setUp(self):
    super().setUp()
    self.alpha = 0.5
    self.coh = np.array([np.exp(- 0.5 * np.abs(self.alpha) ** 2) * self.alpha ** k / np.sqrt(factorial(k)) for k in range(self.D)])
    self.neg_coh = np.array([np.exp(- 0.5 * np.abs(-self.alpha) ** 2) * (-self.alpha) ** k / np.sqrt(factorial(k)) for k in range(self.D)])

  #########################################
  # tests of eval behaviour of all_fock_probs method

  def test_eng_run_eval_false_state_all_fock_probs(self):
    """Tests whether the Fock-basis probabilities of the state are an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    probs = state.all_fock_probs()
    self.assertTrue(isinstance(probs, tf.Tensor))

  def test_eval_false_state_all_fock_probs(self):
    """Tests whether the Fock-basis probabilities of the state are an
    unevaluated Tensor object when eval=False is passed to the all_fock_probs method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    probs = state.all_fock_probs(eval=False)
    self.assertTrue(isinstance(probs, tf.Tensor))

  def test_eval_true_state_all_fock_probs(self):
    """Tests whether the Fock-basis probabilities of the state return
    the correct value when eval=True is passed to the all_fock_probs method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    probs = state.all_fock_probs(eval=True).flatten()
    ref_probs = np.tile(np.abs(np.outer(self.coh, self.neg_coh)).flatten() ** 2, self.bsize)
    self.assertAllAlmostEqual(probs, ref_probs, delta=self.tol)

  #########################################
  # tests of eval behaviour of fock_prob method

  def test_eng_run_eval_false_state_fock_prob(self):
    """Tests whether the probability of a Fock measurement outcome on the state is an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = eng.run(backend=self.backend, cutoff_dim=self.D, eval=False)
    prob = state.fock_prob([self.D // 2, self.D // 2])
    self.assertTrue(isinstance(prob, tf.Tensor))

  def test_eval_false_state_fock_prob(self):
    """Tests whether the probability of a Fock measurement outcome on the state is an
    unevaluated Tensor object when eval=False is passed to the fock_prob method of a state."""
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    prob = state.fock_prob([self.D // 2, self.D // 2], eval=False)
    self.assertTrue(isinstance(prob, tf.Tensor))

  def test_eval_false_state_fock_prob(self):
    """Tests whether the probability of a Fock measurement outcome on the state returns
     the correct value when eval=True is passed to the fock_prob method of a state."""
    n1 = self.D // 2
    n2 = self.D // 3
    eng, q = sf.Engine(self.num_subsystems)
    with eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = eng.run(backend=self.backend, cutoff_dim=self.D)
    prob = state.fock_prob([n1, n2], eval=True)
    ref_prob = (np.abs(np.outer(self.coh, self.neg_coh)) ** 2)[n1, n2]
    self.assertAllAlmostEqual(prob, ref_prob, delta=self.tol)

if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (TwoModeSymbolicTests,OneModeSymbolicTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)

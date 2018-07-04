 ##############################################################################
#
# Tests for the various Tensorflow-specific symbolic options of the frontend/backend.
# These tests are specific to backends which are written in Tensorflow
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial
import tensorflow as tf

from defaults import SymbolicBaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.engine import Engine


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
    self.eng = Engine(self.num_subsystems)
    # attach the backend (NOTE: self.backend is shared between the tests, make sure it is reset before use!)
    self.eng.backend = self.backend
    self.backend.reset()

  #########################################
  # tests basic eval behaviour of eng.run and states class

  def test_eng_run_eval_false_returns_tensor(self):
    """Tests whether the eval=False option to the `eng.run` command
    successfully returns an unevaluated Tensor."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
    state = self.eng.run(eval=False)
    state_data = state.data
    self.assertTrue(isinstance(state_data, tf.Tensor))

  def test_eng_run_eval_false_measurements_are_tensors(self):
    """Tests whether the eval=False option to the `eng.run` command
    successfully returns a unevaluated Tensors for measurment results."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
      MeasureX   | q
    self.eng.run(eval=False)
    val = q[0].val
    self.assertTrue(isinstance(val, tf.Tensor))

  def test_eng_run_with_session_and_feed_dict(self):
    """Tests whether passing a tf Session and feed_dict
    through `eng.run` leads to proper numerical simulation."""
    self.logTestName()
    a = tf.Variable(0.5)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    q = self.eng.register
    with self.eng:
      Dgate(a) | q
    state = self.eng.run(session=sess, feed_dict={a: 0.0})

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
    self.logTestName()
    if self.args.mixed:
      return
    else:
      q = self.eng.register
      with self.eng:
        Dgate(0.5) | q
      state = self.eng.run(eval=False)
      ket = state.ket()
      self.assertTrue(isinstance(ket, tf.Tensor))

  def test_eval_false_state_ket(self):
    """Tests whether the ket of the returned state is an unevaluated
    Tensor object when eval=False is passed to the ket method of a state."""
    self.logTestName()
    if self.args.mixed:
      return
    else:
      q = self.eng.register
      with self.eng:
        Dgate(0.5) | q
      state = self.eng.run()
      ket = state.ket(eval=False)
      self.assertTrue(isinstance(ket, tf.Tensor))

  def test_eval_true_state_ket(self):
    """Tests whether the ket of the returned state is equal to the
    correct value when eval=True is passed to the ket method of a state."""
    self.logTestName()
    if self.args.mixed:
      return
    else:
      q = self.eng.register
      state = self.eng.run()
      ket = state.ket(eval=True)
      self.assertAllAlmostEqual(ket, self.vac, delta=self.tol)

  #########################################
  # tests of eval behaviour of dm method

  def test_eng_run_eval_false_state_dm(self):
    """Tests whether the density matrix of the returned state is an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    self.logTestName()
    if self.args.mixed:
      return
    else:
      q = self.eng.register
      with self.eng:
        Dgate(0.5) | q
      state = self.eng.run(eval=False)
      dm = state.dm()
      self.assertTrue(isinstance(dm, tf.Tensor))

  def test_eval_false_state_dm(self):
    """Tests whether the density matrix of the returned state is an
    unevaluated Tensor object when eval=False is passed to the ket method of a state."""
    self.logTestName()
    if self.args.mixed:
      return
    else:
      q = self.eng.register
      with self.eng:
        Dgate(0.5) | q
      state = self.eng.run()
      dm = state.dm(eval=False)
      self.assertTrue(isinstance(dm, tf.Tensor))

  def test_eval_true_state_dm(self):
    """Tests whether the density matrix of the returned state is equal
    to the correct value when eval=True is passed to the ket method of a state."""
    self.logTestName()
    if self.args.mixed:
      return
    else:
      q = self.eng.register
      state = self.eng.run()
      dm = state.dm(eval=True)
      self.assertAllAlmostEqual(dm, self.vac_dm, delta=self.tol)

  #########################################
  # tests of eval behaviour of trace method

  def test_eng_run_eval_false_state_trace(self):
    """Tests whether the trace of the returned state is an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
    state = self.eng.run(eval=False)
    tr = state.trace()
    self.assertTrue(isinstance(tr, tf.Tensor))

  def test_eval_false_state_trace(self):
    """Tests whether the trace of the returned state is an
    unevaluated Tensor object when eval=False is passed to the trace method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
    state = self.eng.run()
    tr = state.trace(eval=False)
    self.assertTrue(isinstance(tr, tf.Tensor))

  def test_eval_true_state_trace(self):
    """Tests whether the trace of the returned state is equal
    to the correct value when eval=True is passed to the trace method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
    state = self.eng.run()
    tr = state.trace(eval=True)
    self.assertAllAlmostEqual(tr, 1, delta=self.tol)

  #########################################
  # tests of eval behaviour of reduced_dm method

  def test_eng_run_eval_false_state_reduced_dm(self):
    """Tests whether the reduced_density matrix of the returned state
    is an unevaluated Tensor object when eval=False is passed to `eng.run`."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
    state = self.eng.run(eval=False)
    rho = state.reduced_dm([0])
    self.assertTrue(isinstance(rho, tf.Tensor))

  def test_eval_false_state_reduced_dm(self):
    """Tests whether the reduced density matrix of the returned state is an
    unevaluated Tensor object when eval=False is passed to the reduced_dm method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
    state = self.eng.run()
    rho = state.reduced_dm([0], eval=False)
    self.assertTrue(isinstance(rho, tf.Tensor))

  def test_eval_true_state_reduced_dm(self):
    """Tests whether the reduced density matrix of the returned state is
    equal to the correct value when eval=True is passed to the reduced_dm method of a state."""
    self.logTestName()
    q = self.eng.register
    state = self.eng.run()
    rho = state.reduced_dm([0], eval=True)
    self.assertAllAlmostEqual(rho, self.vac_dm, delta=self.tol)

  #########################################
  # tests of eval behaviour of fidelity_vacuum method

  def test_eng_run_eval_false_state_fidelity_vacuum(self):
    """Tests whether the fidelity_vacuum method of the state returns an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
    state = self.eng.run(eval=False)
    fidel_vac = state.fidelity_vacuum()
    self.assertTrue(isinstance(fidel_vac, tf.Tensor))

  def test_eval_false_state_fidelity_vacuum(self):
    """Tests whether the vacuum fidelity of the returned state is an
    unevaluated Tensor object when eval=False is passed to the fidelity_vacuum method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
    state = self.eng.run()
    fidel_vac = state.fidelity_vacuum(eval=False)
    self.assertTrue(isinstance(fidel_vac, tf.Tensor))

  def test_eval_true_state_fidelity_vacuum(self):
    """Tests whether the vacuum fidelity of the returned state is equal
    to the correct value when eval=True is passed to the fidelity_vacuum method of a state."""
    self.logTestName()
    q = self.eng.register
    state = self.eng.run()
    fidel_vac = state.fidelity_vacuum(eval=True)
    self.assertAllAlmostEqual(fidel_vac, 1., delta=self.tol)

  #########################################
  # tests of eval behaviour of is_vacuum method

  def test_eng_run_eval_false_state_is_vacuum(self):
    """Tests whether the is_vacuum method of the state returns an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
    state = self.eng.run(eval=False)
    is_vac = state.is_vacuum()
    self.assertTrue(isinstance(is_vac, tf.Tensor))

  def test_eval_false_state_is_vacuum(self):
    """Tests whether the is_vacuum method of the state returns an
    unevaluated Tensor object when eval=False is passed to the is_vacuum method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(0.5) | q
    state = self.eng.run()
    is_vac = state.is_vacuum(eval=False)
    self.assertTrue(isinstance(is_vac, tf.Tensor))

  def test_eval_true_state_is_vacuum(self):
    """Tests whether the is_vacuum method of the state returns
    the correct value when eval=True is passed to the is_vacuum method of a state."""
    self.logTestName()
    q = self.eng.register
    state = self.eng.run()
    is_vac = state.is_vacuum(eval=True)
    self.assertAllTrue(is_vac)

  #########################################
  # tests of eval behaviour of fidelity_coherent method

  def test_eng_run_eval_false_state_fidelity_coherent(self):
    """Tests whether the fidelity of the state with respect to coherent states is
    an unevaluated Tensor object when eval=False is passed to `eng.run`."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run(eval=False)
    fidel_coh = state.fidelity_coherent([self.alpha])
    self.assertTrue(isinstance(fidel_coh, tf.Tensor))

  def test_eval_false_state_fidelity_coherent(self):
    """Tests whether the fidelity of the state with respect to coherent states
    is an unevaluated Tensor object when eval=False is passed to the fidelity_coherent method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run()
    fidel_coh = state.fidelity_coherent([self.alpha], eval=False)
    self.assertTrue(isinstance(fidel_coh, tf.Tensor))

  def test_eval_true_state_fidelity_coherent(self):
    """Tests whether the fidelity of the state with respect to coherent states returns
    the correct value when eval=True is passed to the fidelity_coherent method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run()
    fidel_coh = state.fidelity_coherent([self.alpha], eval=True)
    self.assertAllAlmostEqual(fidel_coh, 1, delta=self.tol)

  #########################################
  # tests of eval behaviour of fidelity method

  def test_eng_run_eval_false_state_fidelity(self):
    """Tests whether the fidelity of the state with respect to a local state is an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run(eval=False)
    fidel = state.fidelity(self.coh, 0)
    self.assertTrue(isinstance(fidel, tf.Tensor))

  def test_eval_false_state_fidelity(self):
    """Tests whether the fidelity of the state with respect to a local state is
    an unevaluated Tensor object when eval=False is passed to the fidelity method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run()
    fidel = state.fidelity(self.coh, 0, eval=False)
    self.assertTrue(isinstance(fidel, tf.Tensor))

  def test_eval_true_state_fidelity(self):
    """Tests whether the fidelity of the state with respect to a local state
    returns the correct value when eval=True is passed to the fidelity method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run()
    fidel_coh = state.fidelity(self.coh, 0, eval=True)
    self.assertAllAlmostEqual(fidel_coh, 1, delta=self.tol)

  #########################################
  # tests of eval behaviour of quad_expectation method

  def test_eng_run_eval_false_state_quad_expectation(self):
    """Tests whether the local quadrature expectation of the state is
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run(eval=False)
    e, v = state.quad_expectation(0, 0)
    self.assertTrue(isinstance(e, tf.Tensor))
    self.assertTrue(isinstance(v, tf.Tensor))

  def test_eval_false_state_quad_expectation(self):
    """Tests whether the local quadrature expectation value of the state is
    an unevaluated Tensor object when eval=False is passed to the quad_expectation method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run()
    e, v = state.quad_expectation(0, 0, eval=False)
    self.assertTrue(isinstance(e, tf.Tensor))
    self.assertTrue(isinstance(v, tf.Tensor))

  def test_eval_true_state_quad_expectation(self):
    """Tests whether the local quadrature expectation value of the state returns
    the correct value when eval=True is passed to the quad_expectation method of a state."""
    self.logTestName()
    hbar = 2.
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run()
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
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run(eval=False)
    nbar = state.mean_photon(0)
    self.assertTrue(isinstance(nbar, tf.Tensor))

  def test_eval_false_state_mean_photon(self):
    """Tests whether the local mean photon number of the state is
    an unevaluated Tensor object when eval=False is passed to the mean_photon_number method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run()
    nbar = state.mean_photon(0, eval=False)
    self.assertTrue(isinstance(nbar, tf.Tensor))

  def test_eval_true_state_mean_photon(self):
    """Tests whether the local mean photon number of the state returns
    the correct value when eval=True is passed to the mean_photon method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q
    state = self.eng.run()
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
    self.eng = Engine(self.num_subsystems)
    # attach the backend (NOTE: self.backend is shared between the tests, make sure it is reset before use!)
    self.eng.backend = self.backend
    self.backend.reset()


  #########################################
  # tests of eval behaviour of all_fock_probs method

  def test_eng_run_eval_false_state_all_fock_probs(self):
    """Tests whether the Fock-basis probabilities of the state are an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = self.eng.run(eval=False)
    probs = state.all_fock_probs()
    self.assertTrue(isinstance(probs, tf.Tensor))

  def test_eval_false_state_all_fock_probs(self):
    """Tests whether the Fock-basis probabilities of the state are an
    unevaluated Tensor object when eval=False is passed to the all_fock_probs method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = self.eng.run()
    probs = state.all_fock_probs(eval=False)
    self.assertTrue(isinstance(probs, tf.Tensor))

  def test_eval_true_state_all_fock_probs(self):
    """Tests whether the Fock-basis probabilities of the state return
    the correct value when eval=True is passed to the all_fock_probs method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = self.eng.run()
    probs = state.all_fock_probs(eval=True).flatten()
    ref_probs = np.tile(np.abs(np.outer(self.coh, self.neg_coh)).flatten() ** 2, self.bsize)
    self.assertAllAlmostEqual(probs, ref_probs, delta=self.tol)

  #########################################
  # tests of eval behaviour of fock_prob method

  def test_eng_run_eval_false_state_fock_prob(self):
    """Tests whether the probability of a Fock measurement outcome on the state is an
    unevaluated Tensor object when eval=False is passed to `eng.run`."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = self.eng.run(eval=False)
    prob = state.fock_prob([self.D // 2, self.D // 2])
    self.assertTrue(isinstance(prob, tf.Tensor))

  def test_eval_false_state_fock_prob(self):
    """Tests whether the probability of a Fock measurement outcome on the state is an
    unevaluated Tensor object when eval=False is passed to the fock_prob method of a state."""
    self.logTestName()
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = self.eng.run()
    prob = state.fock_prob([self.D // 2, self.D // 2], eval=False)
    self.assertTrue(isinstance(prob, tf.Tensor))

  def test_eval_false_state_fock_prob(self):
    """Tests whether the probability of a Fock measurement outcome on the state returns
     the correct value when eval=True is passed to the fock_prob method of a state."""
    self.logTestName()
    n1 = self.D // 2
    n2 = self.D // 3
    q = self.eng.register
    with self.eng:
      Dgate(self.alpha) | q[0]
      Dgate(-self.alpha) | q[1]
    state = self.eng.run()
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

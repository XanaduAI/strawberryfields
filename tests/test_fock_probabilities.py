##############################################################################
#
# Unit tests for measurements in the Fock basis
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial

from defaults import BaseTest, FockBaseTest

mag_alphas = np.linspace(0, .8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)

###################################################################

class FockStateProbabilitiesTest(BaseTest):
  num_subsystems = 1
  def test_prob_fock_state_gaussian(self):
    """Tests that probabilities of particular Fock states |n> are correct for a gaussian state."""
    self.logTestName()
    for mag_alpha in mag_alphas:
      for phase_alpha in phase_alphas:
        self.circuit.reset(pure=self.kwargs['pure'])
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        ref_state = np.array([np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n)) for n in range(self.D)])
        ref_probs = np.abs(ref_state) ** 2
        self.circuit.prepare_coherent_state(alpha, 0)
        state = self.circuit.state()
        for n in range(self.D):
          prob_n = state.fock_prob([n])
          self.assertAllAlmostEqual(prob_n, ref_probs[n], delta=self.tol)


class AllFockStateProbabilitiesTestPure(FockBaseTest):
  num_subsystems = 1

  def test_all_fock_probs_pure(self):
    """Tests that the numeric probabilities in the full Fock basis are correct for a one-mode pure state."""
    self.logTestName()
    for mag_alpha in mag_alphas:
      for phase_alpha in phase_alphas:
        self.circuit.reset(pure=True)
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        ref_state = np.array([np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n)) for n in range(self.D)])
        ref_probs = np.abs(ref_state) ** 2
        self.circuit.prepare_coherent_state(alpha, 0)
        state = self.circuit.state()
        probs = state.all_fock_probs().flatten()
        ref_probs = np.tile(ref_probs, self.bsize)
        self.assertAllAlmostEqual(probs, ref_probs, delta=self.tol)


class AllFockProbabilitiesTests(FockBaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 2

  def test_prob_fock_state_nongaussian(self):
    """Tests that probabilities of particular Fock states |n> are correct for a nongaussian state."""
    self.logTestName()
    for mag_alpha in mag_alphas:
      for phase_alpha in phase_alphas:
        self.circuit.reset(pure=self.kwargs['pure'])
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        ref_state = np.array([np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n)) for n in range(self.D)])
        ref_probs = np.abs(ref_state) ** 2
        self.circuit.prepare_coherent_state(alpha, 0)
        self.circuit.prepare_fock_state(self.D // 2, 1)
        state = self.circuit.state()
        for n in range(self.D):
          prob_n = state.fock_prob([n, self.D // 2])
          self.assertAllAlmostEqual(prob_n, ref_probs[n], delta=self.tol)

  def test_all_fock_state_probs(self):
    """Tests that the numeric probabilities in the full Fock basis are correct for a two-mode gaussian state."""
    self.logTestName()
    for mag_alpha in mag_alphas:
      for phase_alpha in phase_alphas:
        self.circuit.reset(pure=self.kwargs['pure'])
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        ref_state1 = np.array([np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n)) for n in range(self.D)])
        ref_state2 = np.array([np.exp(-0.5 * np.abs(-alpha) ** 2) * (-alpha) ** n / np.sqrt(factorial(n)) for n in range(self.D)])
        ref_state = np.outer(ref_state1, ref_state2)
        ref_probs = np.abs(np.reshape(ref_state ** 2, -1))
        ref_probs = np.tile(ref_probs, self.bsize)
        self.circuit.prepare_coherent_state(alpha, 0)
        self.circuit.prepare_coherent_state(-alpha, 1)
        state = self.circuit.state()
        for n in range(self.D):
          for m in range(self.D):
            probs = state.all_fock_probs().flatten()
            self.assertAllAlmostEqual(probs, ref_probs, delta=self.tol)
        if mag_alpha == 0.:
          pass


if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (FockStateProbabilitiesTest, AllFockStateProbabilitiesTestPure, AllFockProbabilitiesTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)



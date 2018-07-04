##############################################################################
#
# Unit tests for displacement operations
# Convention: The displacement unitary is fixed to be
# U(\alpha) = \exp(alpha \hat{a}^\dagger - \alpha^* \hat{a})
# where \hat{a}^\dagger is the photon creation operator.
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial

from defaults import BaseTest, FockBaseTest


mag_alphas = np.linspace(0, .8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)

###################################################################

class BasicTests(BaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 1

  def test_no_displacement(self):
    """Tests displacement operation in some limiting cases where the result should be a vacuum state."""
    self.logTestName()
    mag_alpha = 0.
    for phase_alpha in phase_alphas:
      alpha = mag_alpha * np.exp(1j * phase_alpha)
      self.circuit.reset(pure=self.kwargs['pure'])
      self.circuit.displacement(alpha, 0)
      self.assertAllTrue(self.circuit.is_vacuum(self.tol))
        
  def test_fidelity_coherent(self):
    """Tests if a range of alpha-displaced states have the correct fidelity with the corresponding coherent state."""
    self.logTestName()
    for mag_alpha in mag_alphas:
      for phase_alpha in phase_alphas:
        self.circuit.reset(pure=self.kwargs['pure'])
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        self.circuit.displacement(alpha, 0)
        state = self.circuit.state()
        fidel = state.fidelity_coherent([alpha])
        self.assertAllAlmostEqual(state.fidelity_coherent([alpha]), 1, delta=self.tol)

  
class FockBasisTests(FockBaseTest):
  """Tests for simulators that use Fock basis."""
  num_subsystems = 1

  def test_normalized_displaced_state(self):
    """Tests if a range of displaced states are normalized."""
    self.logTestName()
    for mag_alpha in mag_alphas:
      for phase_alpha in phase_alphas:
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.displacement(alpha, 0)
        state = self.circuit.state()
        tr = state.trace()
        self.assertAllAlmostEqual(tr, 1, delta=self.tol)
        if alpha==0.: break

  def test_coherent_state_fock_elements(self):
    r"""Tests if a range of alpha-displaced states have the correct Fock basis elements
       |\alpha> = exp(-0.5 |\alpha|^2) \sum_n \alpha^n / \sqrt{n!} |n>"""
    self.logTestName()
    for mag_alpha in mag_alphas:
        for phase_alpha in phase_alphas:
          alpha = mag_alpha * np.exp(1j * phase_alpha)
          self.circuit.reset(pure=self.kwargs['pure'])
          self.circuit.displacement(alpha, 0)
          state = self.circuit.state()
          if state.is_pure:
            numer_state = state.ket()
          else:
            numer_state = state.dm()
          ref_state = np.array([np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(factorial(n)) for n in range(self.D)])
          if not self.kwargs['pure']:
            ref_state = np.outer(ref_state, np.conj(ref_state))
          self.assertAllAlmostEqual(numer_state, ref_state, delta=self.tol)

  

if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BasicTests, FockBasisTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)

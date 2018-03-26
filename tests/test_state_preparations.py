##############################################################################
#
# Unit tests for various state preparation operations
#
##############################################################################

import unittest
import os, sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.special import factorial
from defaults import BaseTest, FockBaseTest

mag_alphas = np.linspace(0, .8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)
nbars = np.linspace(0, 5)

###################################################################

class BasicTests(BaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 1

  def test_prepare_vac(self):
    """Tests the ability to prepare vacuum states."""
    self.circuit.prepare_vacuum_state(0)
    self.assertAllTrue(self.circuit.is_vacuum(self.tol))

  def test_fidelity_coherent(self):
    """Tests if a range of coherent states have the correct fidelity."""
    for mag_alpha in mag_alphas:
      for phase_alpha in phase_alphas:
        self.circuit.reset(pure=self.kwargs['pure'])
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        self.circuit.prepare_coherent_state(alpha, 0)
        state = self.circuit.state()
        self.assertAllAlmostEqual(state.fidelity_coherent([alpha]), 1, delta=self.tol)

  def test_prepare_thermal_state(self):
    for nbar in nbars:
      self.circuit.reset(pure=self.kwargs['pure'])
      self.circuit.prepare_thermal_state(nbar, 0)
      ref_probs = np.array([nbar ** n / (nbar + 1) ** (n + 1) for n in range(self.D)])
      state = self.circuit.state()
      state_probs = np.array([state.fock_prob([n]) for n in range(self.D)]).T # transpose needed for array broadcasting to work in batch mode
      self.assertAllAlmostEqual(ref_probs, state_probs, delta=self.tol)

class FockBasisTests(FockBaseTest):
  """Tests for simulators that use Fock basis."""
  num_subsystems = 1

  def test_normalized_prepare_vac(self):
    """Tests the ability to prepare vacuum states."""
    self.circuit.prepare_vacuum_state(0)
    state = self.circuit.state()
    tr = state.trace()
    self.assertAllAlmostEqual(tr, 1, delta=self.tol)

  def test_normalized_coherent_state(self):
    """Tests if a range of coherent states are normalized."""
    for mag_alpha in mag_alphas:
      for phase_alpha in phase_alphas:
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_coherent_state(alpha, 0)
        state = self.circuit.state()
        tr = state.trace()
        self.assertAllAlmostEqual(tr, 1, delta=self.tol)
        if alpha == 0.: break

  def test_prepare_ket_state(self):
    """Tests if a ket state with arbitrary parameters is correctly prepared."""
    for _ in range(10):
      random_ket = np.random.uniform(-1,1,self.D)
      random_ket = random_ket / np.linalg.norm(random_ket)
      self.circuit.reset(pure=self.kwargs['pure'])
      self.circuit.prepare_ket_state(random_ket, 0)
      state = self.circuit.state()
      self.assertAllAlmostEqual(state.fidelity(random_ket, 0), 1, delta=self.tol)

if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BasicTests,FockBasisTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)

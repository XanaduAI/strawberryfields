##############################################################################
#
# Unit tests for phase-shifter operations
# Convention: The phase-shift unitary is fixed to be
# U(\theta) = \exp(i * \theta \hat{n})
# where \hat{n} is the number operator. For positive \theta, this corresponds
# to a clockwise rotation of the state in phase space.
#
##############################################################################

import unittest

import numpy as np

from defaults import BaseTest, FockBaseTest


shift_thetas = np.linspace(0, 2 * np.pi, 7, endpoint=False)

###################################################################

class BasicTests(BaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 1

  def test_rotated_vacuum(self):
    """Tests phase shift operation in some limiting cases where the result should be a vacuum state."""
    self.logTestName()
    for theta in shift_thetas:
      self.circuit.reset(pure=self.kwargs['pure'])
      self.circuit.rotation(theta, 0)
      self.assertAllTrue(self.circuit.is_vacuum(self.tol))

class FockBasisTests(FockBaseTest):
  """Tests for simulators that use Fock basis."""
  num_subsystems = 1

  def test_normalized_rotated_coherent_states(self):
    """Tests if a range of phase-shifted coherent states are normalized."""
    self.logTestName()
    alpha = 1.
    self.circuit.reset(pure=self.kwargs['pure'])
    self.circuit.prepare_coherent_state(alpha, 0)
    for theta in shift_thetas:
      self.circuit.rotation(theta, 0)
      state = self.circuit.state()
      tr = state.trace()
      self.assertAllAlmostEqual(tr, 1, delta=self.tol)

  def test_rotated_fock_states(self):
    """Tests if a range of phase-shifted fock states |n> are equal to the form of
    exp(i * theta * n)|n>"""
    self.logTestName()
    for theta in shift_thetas:
        for n in range(self.D):
          self.circuit.reset(pure=self.kwargs['pure'])
          self.circuit.prepare_fock_state(n, 0)
          self.circuit.rotation(theta, 0)
          s = self.circuit.state()
          if s.is_pure:
            numer_state = s.ket()
          else:
            numer_state = s.dm()
          ref_state = np.array([np.exp(1j * theta * k) if k == n else 0.0 for k in range(self.D)])
          if not self.kwargs['pure']:
            ref_state = np.outer(ref_state, np.conj(ref_state))
          self.assertAllAlmostEqual(numer_state, ref_state, delta=self.tol)

  def test_rotated_superposition_states(self):
    r"""Tests if a range of phase-shifted superposition states are equal to the form of
    \sum_n exp(i * theta * n)|n>"""
    self.logTestName()
    for theta in shift_thetas:
        for n in range(self.D):
          self.circuit.reset(pure=self.kwargs['pure'])
          ref_state = np.array([np.exp(1j * theta * k) for k in range(self.D)]) / np.sqrt(self.D)
          if not self.kwargs['pure']:
            ref_state = np.outer(ref_state, np.conj(ref_state))
          self.circuit.prepare_ket_state(ref_state, 0)
          s = self.circuit.state()
          if s.is_pure:
            numer_state = s.ket()
          else:
            numer_state = s.dm()
          self.assertAllAlmostEqual(numer_state, ref_state, delta=self.tol)

if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BasicTests, FockBasisTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)


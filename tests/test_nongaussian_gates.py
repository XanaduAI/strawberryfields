##############################################################################
#
# Unit tests for non-Gaussian gates
#
##############################################################################

import unittest

import numpy as np

from defaults import BaseTest, FockBaseTest


kappas = np.linspace(0, 2 * np.pi, 7)

###################################################################

class FockBasisKerrTests(FockBaseTest):
  """Tests for simulators that use Fock basis."""
  num_subsystems = 1

  def test_kerr_interaction(self):
    """Tests if the Kerr interaction has the right effect on states in the Fock basis"""
    self.logTestName()
    for kappa in kappas:
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_ket_state(np.array([1.0 for n in range(self.D)]) / self.D, 0)
        self.circuit.kerr_interaction(kappa, 0)
        s = self.circuit.state()
        if s.is_pure:
          numer_state = s.ket()
        else:
          numer_state = s.dm()
        ref_state = np.array([np.exp(1j * kappa * n ** 2) for n in range(self.D)]) / self.D
        self.assertAllAlmostEqual(numer_state, ref_state, delta=self.tol)


class FockBasisCrossKerrTests(FockBaseTest):
  """Tests for simulators that use Fock basis."""
  num_subsystems = 2

  def test_cross_kerr_interaction(self):
    """Tests if the Kerr interaction has the right effect on states in the Fock basis"""
    self.logTestName()
    for kappa in kappas:
        self.circuit.reset(pure=self.kwargs['pure'])

        ket1 = np.array([1.0 for n in range(self.D)])
        ket1 /= np.linalg.norm(ket1)
        ket = np.outer(ket1, ket1)

        self.circuit.prepare_ket_state(ket, modes=[0, 1])
        self.circuit.cross_kerr_interaction(kappa, 0, 1)
        s = self.circuit.state()

        if s.is_pure:
          numer_state = s.ket()
        else:
          numer_state = s.dm()

        ref_state = np.array([np.exp(1j*kappa*n1*n2) for n1 in range(self.D) for n2 in range(self.D)])/self.D
        ref_state = np.reshape(ref_state, [self.D]*2)
        self.assertAllAlmostEqual(numer_state, ref_state, delta=self.tol)


if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (FockBasisKerrTests, FockBasisCrossKerrTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)


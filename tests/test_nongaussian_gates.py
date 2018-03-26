##############################################################################
#
# Unit tests for non-Gaussian gates
#
##############################################################################

import unittest
import os, sys
sys.path.append(os.getcwd())
import numpy as np
from defaults import BaseTest, FockBaseTest

kappas = np.linspace(0, 2 * np.pi, 7)

###################################################################

class FockBasisTests(FockBaseTest):
  """Tests for simulators that use Fock basis."""
  num_subsystems = 1

  def test_kerr_interaction(self):
    """Tests if the Kerr interaction has the right effect on states in the Fock basis"""
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


if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (FockBasisTests,):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)


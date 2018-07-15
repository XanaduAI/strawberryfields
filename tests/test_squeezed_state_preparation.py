##############################################################################
#
# Unit tests for operations that prepare squeezed states
# Convention: The squeezing unitary is fixed to be
# U(z) = \exp(0.5 (z^* \hat{a}^2 - z (\hat{a^\dagger}^2)))
# where \hat{a} is the photon annihilation operator.
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial

from defaults import BaseTest, FockBaseTest


#sqz_scale = np.sqrt(np.arcsinh(CUTOFF_N / 9))
sqz_r = np.linspace(0.0, 0.1, 5)
sqz_theta = np.linspace(0,2 * np.pi, 3, endpoint=False)

###################################################################

class BasicTests(BaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 1

  def test_no_squeezing(self):
    """Tests squeezing operation in some limiting cases where the result should be a vacuum state."""
    self.logTestName()
    self.circuit.prepare_vacuum_state(0)
    for theta in sqz_theta:
      self.circuit.reset(pure=self.kwargs['pure'])
      self.circuit.prepare_squeezed_state(0, theta, 0)
      self.assertAllTrue(self.circuit.is_vacuum(self.tol))

class FockBasisTests(FockBaseTest):
  """Tests for simulators that use Fock basis."""
  num_subsystems = 1

  def test_normalized_squeezed_state(self):
    """Tests if a range of squeezed vacuum states are normalized."""
    self.logTestName()
    for r in sqz_r:
      for theta in sqz_theta:
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_squeezed_state(r, theta, 0)
        state = self.circuit.state()
        tr = state.trace()
        self.assertAllAlmostEqual(tr, 1, delta=self.tol)

  def test_no_odd_fock(self):
    """Tests if a range of squeezed vacuum states have
    only nonzero entries for even Fock states."""
    self.logTestName()
    for r in sqz_r:
        for theta in sqz_theta:
          self.circuit.reset(pure=self.kwargs['pure'])
          self.circuit.prepare_squeezed_state(r, theta, 0)
          s = self.circuit.state()
          if s.is_pure:
            num_state = s.ket()
          else:
            num_state = s.dm()
          if self.args.batched:
            odd_entries = num_state[:,1::2]
          else:
            odd_entries = num_state[1::2]
          self.assertAllTrue(np.all(odd_entries == 0))

  def test_reference_squeezed_states(self):
    """Tests if a range of squeezed vacuum states are equal to the form of Eq. (5.5.6) in Loudon."""
    self.logTestName()
    def sech(x):
      return 1 / np.cosh(x)

    for r in sqz_r:
        for theta in sqz_theta:
          self.circuit.reset(pure=self.kwargs['pure'])
          self.circuit.prepare_squeezed_state(r, theta, 0)
          s = self.circuit.state()
          if s.is_pure:
            num_state = s.ket()
          else:
            num_state = s.dm()
          even_refs = np.array([np.sqrt(sech(r)) * np.sqrt(factorial(k)) / factorial(k / 2) * (-0.5 * np.exp(1j * theta) * np.tanh(r)) ** (k / 2) for k in range(0, self.D, 2)])
          if self.args.batched:
            if self.kwargs["pure"]:
              even_entries = num_state[:,::2]
            else:
              even_entries = num_state[:,::2,::2]
              even_refs = np.outer(even_refs, np.conj(even_refs))
          else:
            if self.kwargs["pure"]:
              even_entries = num_state[::2]
            else:
              even_entries = num_state[::2,::2]
              even_refs = np.outer(even_refs, np.conj(even_refs))
          self.assertAllAlmostEqual(even_entries, even_refs, delta=self.tol)

if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BasicTests, FockBasisTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)

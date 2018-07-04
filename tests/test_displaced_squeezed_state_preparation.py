##############################################################################
#
# Unit tests for operations that prepare squeezed coherent states
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial

from defaults import BaseTest, FockBaseTest


#sqz_scale = np.sqrt(np.arcsinh(CUTOFF_N / 9))
mag_alphas = np.linspace(0.1, .5, 2)
phase_alphas = np.linspace(np.pi/6, 2 * np.pi, 2, endpoint=False)
sqz_r = np.linspace(0.01, 0.1, 2)
sqz_phi = np.linspace(np.pi / 3, 2 * np.pi, 2, endpoint=False)

###################################################################

class BasicTests(BaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 1

  def test_no_squeezing_no_displacement(self):
    """Tests squeezing operation in the limiting case where the result should be a vacuum state."""
    self.logTestName()
    alpha = 0
    r = 0
    for phi in sqz_phi:
      self.circuit.reset(pure=self.kwargs['pure'])
      self.circuit.prepare_displaced_squeezed_state(alpha, r, phi, 0)
      self.assertAllTrue(self.circuit.is_vacuum(self.tol))

  def test_displaced_squeezed_with_no_squeezing(self):
    """Tests if a squeezed coherent state with no squeezing is equal to a coherent state."""
    self.logTestName()
    r = phi = 0
    for mag_alpha in mag_alphas:
      for phase_alpha in phase_alphas:
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(alpha, r, phi, 0)
        state = self.circuit.state()
        fidel = state.fidelity_coherent([alpha])
        self.assertAllAlmostEqual(fidel, 1, delta=self.tol, msg="alpha={}".format(alpha))

class FockBasisTests(FockBaseTest):
  """Tests for simulators that use Fock basis."""
  num_subsystems = 1

  def test_normalized_displaced_squeezed_state(self):
    """Tests if a range of squeezed vacuum states are normalized."""
    self.logTestName()
    for mag_alpha in mag_alphas:
      for phase_alpha in phase_alphas:
        for r in sqz_r:
          for phi in sqz_phi:
            alpha = mag_alpha * np.exp(1j * phase_alpha)
            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.prepare_displaced_squeezed_state(alpha, r, phi, 0)
            state = self.circuit.state()
            tr = state.trace()
            self.assertAllAlmostEqual(tr, 1, delta=self.tol)

  def test_displaced_squeezed_with_no_displacement(self):
    """Tests if a squeezed coherent state with no displacement is equal to a squeezed state (Eq. (5.5.6) in Loudon)."""
    self.logTestName()
    def sech(x):
      return 1 / np.cosh(x)
    alpha = 0
    for r in sqz_r:
      for phi in sqz_phi:
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_displaced_squeezed_state(alpha, r, phi, 0)
        state = self.circuit.state()
        if state.is_pure:
          num_state = state.ket()
        else:
          num_state = state.dm()
        even_refs = np.array([np.sqrt(sech(r)) * np.sqrt(factorial(k)) / factorial(k / 2) * (-0.5 * np.exp(1j * phi) * np.tanh(r)) ** (k / 2) for k in range(0, self.D, 2)])
        if self.args.batched:
          if self.kwargs["pure"]:
            even_entries = num_state[:, ::2]
          else:
            even_entries = num_state[:, ::2, ::2]
            even_refs = np.outer(even_refs, np.conj(even_refs))
        else:
          if self.kwargs["pure"]:
            even_entries = num_state[::2]
          else:
            even_entries = num_state[::2, ::2]
            even_refs = np.outer(even_refs, np.conj(even_refs))
        self.assertAllAlmostEqual(even_entries, even_refs, delta=self.tol)


if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BasicTests, FockBasisTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)

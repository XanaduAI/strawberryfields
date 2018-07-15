##############################################################################
#
# Unit tests for squeezing operation
# Convention: The squeezing unitary is fixed to be
# U(z) = \exp(0.5 (z^* \hat{a}^2 - z (\hat{a^\dagger}^2)))
# where \hat{a} is the photon annihilation operator.
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial, erfinv
from scipy.special import gammaln as lg

from defaults import BaseTest, FockBaseTest


# sqz_r = np.linspace(0.0, 0.1, 5)
sqz_theta = np.linspace(0, 2 * np.pi, 3, endpoint=False)

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
      r = 0
      z = r * np.exp(1j * theta)
      self.circuit.squeeze(z, 0)
      self.assertAllTrue(self.circuit.is_vacuum(self.tol))


class FockBasisTests(FockBaseTest):
  """Tests for simulators that use Fock basis."""
  num_subsystems = 1

  def setUp(self):
    super().setUp()

    eps = 0.01
    sqz_max = np.log( np.sqrt(self.D-1)/(erfinv(1-eps)*np.sqrt(2)) )

    self.sqz_r = np.linspace(0.0, sqz_max, 5)

  def test_normalized_squeezed_state(self):
    """Tests if a range of squeezed vacuum states are normalized."""
    self.logTestName()
    for r in self.sqz_r:
      for theta in sqz_theta:
        z = r * np.exp(1j * theta)
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.squeeze(z, 0)
        state = self.circuit.state()
        tr = state.trace()
        self.assertAllAlmostEqual(tr, 1, delta=self.tol)

  def test_no_odd_fock(self):
    """Tests if a range of squeezed vacuum states have
    only nonzero entries for even Fock states."""
    self.logTestName()
    for r in self.sqz_r:
        for theta in sqz_theta:
          z = r * np.exp(1j * theta)
          self.circuit.reset(pure=self.kwargs['pure'])
          self.circuit.squeeze(z, 0)
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

  def test_reference_squeezed_vacuum(self):
    """Tests if a range of squeezed vacuum states are equal to the form of Eq. (5.5.6) in Loudon."""
    self.logTestName()
    def sech(x):
      return 1 / np.cosh(x)

    for r in self.sqz_r:
        for theta in sqz_theta:
          z = r * np.exp(1j * theta)
          self.circuit.reset(pure=self.kwargs['pure'])
          self.circuit.squeeze(z, 0)
          s = self.circuit.state()
          if s.is_pure:
            num_state = s.ket()
          else:
            num_state = s.dm()
          even_refs = np.array([np.sqrt(sech(r)) * np.sqrt(factorial(k)) / factorial(k / 2) * (-0.5 * np.exp(1j * theta) * np.tanh(r)) ** (k / 2) for k in range(0, self.D, 2)])
          if self.kwargs['pure']:
            if self.args.batched:
              even_entries = num_state[:,::2]
            else:
              even_entries = num_state[::2]
          else:
            even_refs = np.outer(even_refs, np.conj(even_refs))
            if self.args.batched:
              even_entries = num_state[:,::2, ::2]
            else:
              even_entries = num_state[::2,::2]
          self.assertAllAlmostEqual(even_entries, even_refs, delta=self.tol)

  def test_reference_squeezed_fock(self):
    """Tests if a range of squeezed fock states are equal to the form of Eq. (20)
    in 'On the Squeezed Number States and their Phase Space Representations'
    (https://arxiv.org/abs/quant-ph/0108024)."""
    self.logTestName()
    def matrix_elem(n,r,m):
        eps = 1e-10
        if n % 2 != m % 2:
          return 0.0
        elif r == 0.:
          return np.complex(n==m) # delta function
        else:
          k = np.arange(m % 2, min([m, n]) + 1, 2)
          res = np.sum(
                (-1)**((n-k)/2)
                  * np.exp((lg(m+1) + lg(n+1))/2 - lg(k+1) - lg((m-k)/2+1) - lg((n-k)/2+1))
                  * (np.sinh(r)/2+eps)**((n+m-2*k)/2) / (np.cosh(r)**((n+m+1)/2))
                )
          return res

    for m in range(self.D):
        for r in self.sqz_r:
          self.circuit.reset(pure=self.kwargs['pure'])
          self.circuit.prepare_fock_state(m, 0)
          self.circuit.squeeze(r, 0)
          s = self.circuit.state()
          if s.is_pure:
            num_state = s.ket()
          else:
            num_state = s.dm()
          ref_state = np.array([matrix_elem(n,r,m) for n in range(self.D)])
          if not self.kwargs['pure']:
            ref_state = np.outer(ref_state, np.conj(ref_state))
          self.assertAllAlmostEqual(num_state, ref_state, delta=self.tol)

if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BasicTests, FockBasisTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)

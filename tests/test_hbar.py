##############################################################################
#
# Unit tests for hbar convention
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial

from defaults import BaseTest, FockBaseTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields.engine import Engine
from strawberryfields.ops import *

n_meas = 200
num_stds = 10.
std_10 = num_stds / np.sqrt(n_meas)

r = 0.3
x = 0.2
p = 0.1
hbar = [0.5, 1, 2]

###################################################################

class BaseFrontendHbar(BaseTest):
  """Fock representation postselection tests for the frontend"""
  num_subsystems = 1

  def setUp(self):
    super().setUp()

  def test_squeeze_variance(self):
    self.logTestName()
    if self.batched:
      raise unittest.SkipTest('Test is only relevant for non-batched mode.')

    for h in hbar:
      self.eng = Engine(num_subsystems=self.num_subsystems, hbar=h)
      self.eng.backend = self.backend
      q = self.eng.register

      with self.eng:
        Sgate(r) | q[0]
        MeasureX | q[0]

      res = np.empty(0)
      for i in range(n_meas):
        self.eng.reset(keep_history=True)
        self.eng.run()
        res = np.append(res, q[0].val)

      self.assertAllAlmostEqual(np.var(res), np.exp(-2*r)*h/2, delta = std_10 + self.tol)


class GaussianFrontendHbar(GaussianBaseTest):
  """Gaussian representation postselection tests for the frontend"""
  num_subsystems = 1

  def setUp(self):
    super().setUp()

  def test_x_displacement(self):
    self.logTestName()
    for h in hbar:
      self.eng = Engine(num_subsystems=self.num_subsystems, hbar=h)
      #self.eng.backend = self.backend
      q = self.eng.register

      with self.eng:
        Xgate(x) | q[0]

      state = self.eng.run(backend='gaussian')
      mu_x = state.means()[0]
      self.assertAllAlmostEqual(mu_x, x, delta = self.tol)

  def test_p_displacement(self):
    self.logTestName()
    for h in hbar:
      self.eng = Engine(num_subsystems=self.num_subsystems, hbar=h)
      #self.eng.backend = self.backend
      q = self.eng.register

      with self.eng:
        Zgate(p) | q[0]

      state = self.eng.run(backend='gaussian')
      mu_p = state.means()[1]
      self.assertAllAlmostEqual(mu_p, p, delta = self.tol)



if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BaseFrontendHbar, GaussianFrontendHbar):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)


##############################################################################
#
# Unit tests for hbar convention
#
##############################################################################

import unittest
import os, sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.special import factorial
from defaults import BaseTest, FockBaseTest, GaussianBaseTest

import strawberryfields as sf
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
    for h in hbar:
      self.eng = Engine(num_subsystems=self.num_subsystems, hbar=h)
      q = self.eng.register

      with self.eng:
        Sgate(r) | q[0]
        MeasureX | q[0]

      res = np.empty(0)
      for i in range(n_meas):
        self.eng.run(self.backend_name, cutoff_dim=self.D, reset_backend=True)
        res = np.append(res, q[0].val)

      self.assertAllAlmostEqual(np.var(res), np.exp(-2*r)*h/2, delta = std_10 + self.tol)


class GaussianFrontendHbar(GaussianBaseTest):
  """Gaussian representation postselection tests for the frontend"""
  num_subsystems = 1

  def setUp(self):
    super().setUp()

  def test_x_displacement(self):
    for h in hbar:
      self.eng = Engine(num_subsystems=self.num_subsystems, hbar=h)
      q = self.eng.register

      with self.eng:
        Xgate(x) | q[0]

      state = self.eng.run(backend='gaussian', reset_backend=True)
      mu_x = state.means()[0]
      self.assertAllAlmostEqual(mu_x, x, delta = self.tol)

  def test_p_displacement(self):
    for h in hbar:
      self.eng = Engine(num_subsystems=self.num_subsystems, hbar=h)
      q = self.eng.register

      with self.eng:
        Zgate(p) | q[0]

      state = self.eng.run(backend='gaussian', reset_backend=True)
      mu_p = state.means()[1]
      self.assertAllAlmostEqual(mu_p, p, delta = self.tol)



if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BaseFrontendHbar, GaussianFrontendHbar):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)


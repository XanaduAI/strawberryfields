##############################################################################
#
# Unit tests for postselection
#
##############################################################################

import unittest
import os, sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.special import factorial
from defaults import BaseTest, FockBaseTest, GaussianBaseTest

from strawberryfields.engine import Engine
from strawberryfields.ops import *

n_meas = 200
num_stds = 10.
std_10 = num_stds / np.sqrt(n_meas)

###################################################################


class GaussianFrontendPostselection(GaussianBaseTest):
  """Gaussian representation postselection tests for the frontend"""
  num_subsystems = 2

  def setUp(self):
    super().setUp()
    # construct a compiler engine
    self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)

  def test_homodyne_on_two_mode_squeezed_state(self):
    self.eng.reset()
    q = self.eng.register

    with self.eng:
      S2gate(5) | (q[0], q[1])
      MeasureHomodyne(0, select=0.2) | q[0]

    self.eng.run(backend=self.backend)
    x_mean = self.eng.backend.state([1]).means()[0]
    self.assertAllAlmostEqual(x_mean, 0.2, delta = self.tol)

  def test_heterodyne(self):
    self.eng.reset()
    q = self.eng.register

    with self.eng:
      S2gate(5) | (q[0], q[1])
      MeasureHeterodyne(select=0.0) | q[0]

    state = self.eng.run(backend=self.backend)
    dist = state.fidelity_vacuum()
    self.assertAllAlmostEqual(dist, 1.0, delta = self.tol)


  def test_heterodyne_return(self):
    self.eng.reset()
    q = self.eng.register
    alpha=10*(np.random.rand()+1j*np.random.rand())
    with self.eng:
      S2gate(5) | (q[0], q[1])
      MeasureHeterodyne(select=alpha) | q[0]

    self.eng.run(backend=self.backend)
    self.assertAllAlmostEqual(q[0].val, alpha, delta = self.tol)


class FockFrontendPostselection(FockBaseTest):
  """Fock representation postselection tests for the frontend"""
  num_subsystems = 2

  def setUp(self):
    super().setUp()
    # construct a compiler engine
    self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)

  def test_homodyne_on_two_mode_squeezed_state(self):
    self.eng.reset()
    q = self.eng.register

    with self.eng:
      S2gate(1) | (q[0], q[1])
      MeasureHomodyne(0, select=0.2) | q[0]
      MeasureHomodyne(0) | q[1]

    x = np.empty(0)
    for i in range(n_meas):
      self.eng.run(backend=self.backend, reset_backend=True)
      x = np.append(x, q[1].val)

    self.assertAllAlmostEqual(x.mean(), 0.2, delta = std_10 + self.tol)

  def onhold_test_fock_elements_on_two_mode_squeezed_state(self):
    self.eng.reset()
    for n in range(self.D - 1):
      q = self.eng.register

      with self.eng:
        S2gate(1) | (q[0], q[1])
        MeasureFock(select=n) | q[0]
        MeasureFock() | q[1]

      self.eng.run(backend=self.backend, reset_backend=True)
      self.assertAllEqual(q[1].val, n)


  def test_convervation_of_photon_number_in_beamsplitter(self):
    self.eng.reset()
    for n in range(self.D - 1):
      q = self.eng.register

      with self.eng:
        Fock(n) | q[0]
        Fock(self.D - 1 - n) | q[1]
        BSgate() | (q[0], q[1])
        MeasureFock(select = self.D // 2) | q[0]
        MeasureFock() | q[1]

      self.eng.run(backend=self.backend, reset_backend=True)
      photons_out = sum([i.val for i in q])
      self.assertAllEqual(photons_out, self.D - 1)


if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (FockFrontendPostselection, GaussianFrontendPostselection):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)


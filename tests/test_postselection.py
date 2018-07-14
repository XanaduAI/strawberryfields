##############################################################################
#
# Unit tests for postselection
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial

from defaults import BaseTest, FockBaseTest, GaussianBaseTest, strawberryfields
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
    # attach the backend (NOTE: self.backend is shared between the tests, make sure it is reset before use!)
    self.eng.backend = self.backend
    self.backend.reset()

  def test_homodyne_on_two_mode_squeezed_state(self):
    self.logTestName()
    q = self.eng.register
    with self.eng:
      S2gate(5) | (q[0], q[1])
      MeasureHomodyne(0, select=0.2) | q[0]

    self.eng.run()
    x_mean = self.eng.backend.state([1]).means()[0]
    self.assertAllAlmostEqual(x_mean, 0.2, delta = self.tol)

  def test_heterodyne(self):
    self.logTestName()
    q = self.eng.register
    with self.eng:
      S2gate(5) | (q[0], q[1])
      MeasureHeterodyne(select=0.0) | q[0]

    state = self.eng.run()
    dist = state.fidelity_vacuum()
    self.assertAllAlmostEqual(dist, 1.0, delta = self.tol)


  def test_heterodyne_return(self):
    self.logTestName()
    q = self.eng.register
    alpha=10*(np.random.rand()+1j*np.random.rand())
    with self.eng:
      S2gate(5) | (q[0], q[1])
      MeasureHeterodyne(select=alpha) | q[0]

    self.eng.run()
    self.assertAllAlmostEqual(q[0].val, alpha, delta = self.tol)


class FockFrontendPostselection(FockBaseTest):
  """Fock representation postselection tests for the frontend"""
  num_subsystems = 2

  def setUp(self):
    super().setUp()
    # construct a compiler engine
    self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)
    # attach the backend (NOTE: self.backend is shared between the tests, make sure it is reset before use!)
    self.eng.backend = self.backend
    self.backend.reset()

  def test_homodyne_on_two_mode_squeezed_state(self):
    self.logTestName()
    q = self.eng.register
    with self.eng:
      S2gate(1) | (q[0], q[1])
      MeasureHomodyne(0, select=0.2) | q[0]
      MeasureHomodyne(0) | q[1]

    x = np.empty(0)
    for i in range(n_meas):
      self.eng.run()
      x = np.append(x, q[1].val)
      self.eng.reset(keep_history=True)

    self.assertAllAlmostEqual(x.mean(), 0.2, delta = std_10 + self.tol)

  def onhold_test_fock_elements_on_two_mode_squeezed_state(self):
    self.logTestName()
    q = self.eng.register
    for n in range(self.D - 1):
      with self.eng:
        S2gate(1) | (q[0], q[1])
        MeasureFock(select=n) | q[0]
        MeasureFock() | q[1]

      self.eng.run()
      self.assertAllEqual(q[1].val, n)
      self.eng.reset()

  def test_conservation_of_photon_number_in_beamsplitter(self):
    self.logTestName()
    q = self.eng.register
    for n in range(self.D - 1):
      with self.eng:
        Fock(n) | q[0]
        Fock(self.D - 1 - n) | q[1]
        BSgate() | (q[0], q[1])
        MeasureFock(select = self.D // 2) | q[0]
        MeasureFock() | q[1]

      self.eng.run()
      photons_out = sum([i.val for i in q])
      self.assertAllEqual(photons_out, self.D - 1)
      self.eng.reset()


if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (FockFrontendPostselection, GaussianFrontendPostselection):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)

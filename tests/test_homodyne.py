##############################################################################
#
# Unit tests for homodyne measurements
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial

from defaults import BaseTest, FockBaseTest


mag_alphas = np.linspace(0, .8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)
squeeze_val = np.arcsinh(1.0)
n_meas = 300
disp_val = 1.0+1j*1.0
num_stds = 10.
std_10 = num_stds / np.sqrt(n_meas)

###################################################################

class BasicTests(BaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 1
  def test_mode_reset_vacuum(self):
    self.logTestName()
    self.circuit.squeeze(squeeze_val,0)
    self.circuit.displacement(mag_alphas[-1],0)
    self.circuit.measure_homodyne(0,0)
    self.assertAllTrue(self.circuit.is_vacuum(self.tol))
    
  def test_mean_and_std_vacuum(self):
    self.logTestName()
    x = np.empty(0)
    for i in range(n_meas):
      self.circuit.reset(pure=self.kwargs['pure'])
      meas_result = self.circuit.measure_homodyne(0,0)
      x = np.append(x, meas_result)
    self.assertAllAlmostEqual(x.mean(), 0., delta = std_10 + self.tol)
    self.assertAllAlmostEqual(x.std(), 1., delta = std_10 + self.tol)

  def test_mean_coherent(self):
    self.logTestName()
    x = np.empty(0)
    for i in range(n_meas):
      self.circuit.reset(pure=self.kwargs['pure'])
      self.circuit.prepare_coherent_state(disp_val,0)
      meas_result = self.circuit.measure_homodyne(0,0)
      x = np.append(x, meas_result)
    self.assertAllAlmostEqual(x.mean(), 2 * disp_val.real, delta = std_10 + self.tol)


if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BasicTests,):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)


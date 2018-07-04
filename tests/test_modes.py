##############################################################################
#
# Unit tests for add_mode and del_mode functions
#
##############################################################################

import unittest
from itertools import combinations

import numpy as np

from defaults import BaseTest, FockBaseTest


num_repeats = 10

###################################################################

class BasicTests(BaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 1

  def test_add_mode_vacuum(self):
    """Tests if added modes are initialized to the vacuum state."""
    self.logTestName()
    self.circuit.begin_circuit(1, **self.kwargs)
    for n in range(4):
      self.circuit.add_mode(1)
      self.assertAllTrue(self.circuit.is_vacuum(self.tol))

  def test_del_mode_vacuum(self):
    """Tests if reduced density matrix is in vacuum after deleting some modes."""
    self.logTestName()
    self.circuit.begin_circuit(4, **self.kwargs)
    for n in range(4):
      self.circuit.del_mode([n])
      self.assertAllTrue(self.circuit.is_vacuum(self.tol))

  def test_get_modes(self):
    """Tests that get modes returns the correct result after deleting modes from the circuit"""
    self.logTestName()
    self.circuit.begin_circuit(4, **self.kwargs)
    self.circuit.squeeze(0.1, 0)
    self.circuit.del_mode([0,2])
    res = self.circuit.get_modes()
    self.assertAllEqual(res, [1,3])


class FockBasisTests(FockBaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 1

  def test_normalized_add_mode(self):
    """Tests if a state is normalized after adding modes."""
    self.logTestName()
    self.circuit.begin_circuit(1, **self.kwargs)
    for num_subsystems in range(3):
      self.circuit.add_mode(num_subsystems)
      state = self.circuit.state()
      tr = state.trace()
      self.assertAllAlmostEqual(tr, 1, delta=self.tol)

  def test_normalized_del_mode(self):
    """Tests if a state is normalized after deleting modes."""
    self.logTestName()
    self.circuit.begin_circuit(4, **self.kwargs)
    for n in range(4):
      self.circuit.del_mode(n)
      state = self.circuit.state()
      tr = state.trace()
      self.assertAllAlmostEqual(tr, 1, delta=self.tol)

  def test_fock_measurements_after_add_mode(self):
    """Tests Fock measurements on a system after adding vacuum modes."""
    self.logTestName()
    for m in range(3):
      meas_results = []
      for _ in range(num_repeats):
        self.circuit.begin_circuit(1, **self.kwargs)
        self.circuit.prepare_fock_state(self.D - 1, 0)
        self.circuit.add_mode(m)
        meas_result = self.circuit.measure_fock([0])
        meas_results.append(meas_result)
      self.assertAllTrue(np.all(np.array(meas_results) == self.D - 1))

  def test_fock_measurements_after_del_mode(self):
    """Tests Fock measurements on a system after tracing out an unentagled mode."""
    self.logTestName()
    for m in range(1,4):
      meas_results = []
      for _ in range(num_repeats):
        self.circuit.begin_circuit(4, **self.kwargs)
        self.circuit.prepare_fock_state(self.D - 1, 0)
        self.circuit.del_mode(m)
        meas_result = self.circuit.measure_fock([0])
        meas_results.append(meas_result)
      self.assertAllTrue(np.all(np.array(meas_results) == self.D - 1))


if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BasicTests, FockBasisTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)

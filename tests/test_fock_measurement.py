##############################################################################
#
# Unit tests for measurements in the Fock basis
#
##############################################################################

import unittest
from itertools import combinations

import numpy as np

from defaults import BaseTest, FockBaseTest


num_repeats = 50

###################################################################

class FockMeasurementTests(FockBaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 3

  def test_vacuum_measurements(self):
    """Tests Fock measurement on the vacuum state."""
    self.logTestName()
    for _ in range(num_repeats):
      self.circuit.reset(pure=self.kwargs['pure'])
      meas = self.circuit.measure_fock([0,1,2])[0]
      self.assertAllTrue(np.all(np.array(meas)==0))

  def test_normalized_conditional_states(self):
    """Tests if the conditional states resulting from Fock measurements in a subset of modes are normalized."""
    self.logTestName()
    state_preps = [n for n in range(self.D)] + [self.D - n for n in range(self.D)]
    for idx in range(num_repeats):
      self.circuit.reset(pure=self.kwargs['pure'])
      self.circuit.prepare_fock_state(state_preps[idx % self.D], 0)
      self.circuit.prepare_fock_state(state_preps[(idx + 1) % self.D], 1)
      self.circuit.prepare_fock_state(state_preps[(idx + 2) % self.D], 2)
      state = self.circuit.state()
      tr = state.trace()
      self.assertAllAlmostEqual(tr, 1, delta=self.tol)

  def test_fock_measurements(self):
    """Tests if Fock measurements results on a variety of multi-mode Fock states are correct."""
    self.logTestName()
    state_preps = [n for n in range(self.D)] + [self.D - n for n in range(self.D)]
    mode_choices = [p for p in combinations(range(3), 1)] + \
                   [p for p in combinations(range(3), 2)] + \
                   [p for p in combinations(range(3), 3)]
    for idx in range(num_repeats):
      n = [state_preps[idx % self.D],
           state_preps[(idx + 1) % self.D],
           state_preps[(idx + 2) % self.D]]
      n = np.array(n)
      meas_modes = np.array(mode_choices[idx % len(mode_choices)])
      self.circuit.reset(pure=self.kwargs['pure'])
      self.circuit.prepare_fock_state(n[0], 0)
      self.circuit.prepare_fock_state(n[1], 1)
      self.circuit.prepare_fock_state(n[2], 2)
      meas_result = self.circuit.measure_fock(meas_modes)
      ref_result = n[meas_modes]
      self.assertAllEqual(meas_result, ref_result)


if __name__=="__main__":
  runner = unittest.TextTestRunner()
  fock_tests = unittest.TestLoader().loadTestsFromTestCase(FockMeasurementTests)
  runner.run(fock_tests)


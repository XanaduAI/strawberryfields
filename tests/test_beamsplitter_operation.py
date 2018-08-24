##############################################################################
#
# Unit tests for beamsplitter operations
# Convention: The beamsplitter operation transforms
# \hat{a} -> t \hat{a} + r \hat{b}
# \hat{b} -> - r^* \hat{a} + t^* \hat{b}
# where \hat{a}, \hat{b} are the photon creation operators of the two modes
# Equivalently, we have t:=\cos(\theta) (t assumed real) and r:=\exp{i\phi}\sin(\theta)
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial

from defaults import BaseTest, FockBaseTest

phase_alphas = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 13
t_values = np.linspace(-0.2, 1., 3)
phase_r = np.linspace(0, 2 * np.pi, 3, endpoint=False)

###################################################################


class BasicTests(BaseTest):
  """Basic implementation-independent tests."""
  num_subsystems = 2
  def setUp(self):
    super().setUp()
    self.mag_alphas = np.linspace(0., self.args.alpha, 3)

  def test_complex_t(self):
    """Test exception raised if t is complex"""
    self.logTestName()
    t = 0.1 + 0.5j
    r = np.exp(1j * 0.2) * np.sqrt(1. - np.abs(t) ** 2)
    with self.assertRaisesRegex(ValueError, "must be a float"):
      self.circuit.reset(pure=self.kwargs['pure'])
      self.circuit.beamsplitter(t, r, 0, 1)

  def test_vacuum_beamsplitter(self):
    """Tests beamsplitter operation in some limiting cases where the output
    should be the vacuum in both modes."""
    self.logTestName()
    for t in t_values:
      for r_phi in phase_r:
        r = np.exp(1j * r_phi) * np.sqrt(1. - np.abs(t) ** 2)
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.beamsplitter(t, r, 0, 1)
        self.assertAllTrue(self.circuit.is_vacuum(self.tol),msg="Test failed for t={}, r_phi={}.".format(t, r))

  def test_coherent_vacuum_interfered(self):
    r"""Tests if a range of beamsplitter output states (formed from a coherent state interfering with vacuum)
       have the correct fidelity with the expected coherent states outputs.
       |\psi_in> = |\alpha>|0> --> |t \alpha>|r \alpha> = |\psi_out>
       and for each output mode,
       |\gamma> = exp(-0.5 |\gamma|^2) \sum_n \gamma^n / \sqrt{n!} |n>"""
    self.logTestName()
    phase_alpha = np.pi / 5
    for mag_alpha in self.mag_alphas[1:]:
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        for t in t_values:
          for r_phi in phase_r:
            r = np.exp(1j * r_phi) * np.sqrt(1. - np.abs(t) ** 2)
            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.displacement(alpha, 0)
            self.circuit.beamsplitter(t, r, 0, 1)
            alpha_outA = t * alpha
            alpha_outB = r * alpha
            state = self.circuit.state()
            fidel = state.fidelity_coherent([alpha_outA, alpha_outB])
            self.assertAllAlmostEqual(fidel, 1, delta=self.tol, msg="Test failed for t={}, r_phi={}.".format(t, r))


class FockBasisTests(FockBaseTest):
  """Tests for simulators that use Fock basis."""
  num_subsystems = 2
  def setUp(self):
    super().setUp()
    self.mag_alphas = np.linspace(0., self.args.alpha, 3)

  def test_normalized_beamsplitter_output(self):
    """Tests if a range of beamsplitter outputs states are normalized."""
    self.logTestName()
    alpha = self.args.alpha * np.exp(1j * np.pi / 3)
    for t in t_values:
      for r_phi in phase_r:
        r = np.exp(1j * r_phi) * np.sqrt(1. - np.abs(t) ** 2)
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.displacement(alpha, 1)
        self.circuit.beamsplitter(t, r, 0, 1)
        state = self.circuit.state()
        tr = state.trace()
        self.assertAllAlmostEqual(tr, 1, delta=self.tol, msg="Test failed for t={}, r_phi={}.".format(t, r))

  def test_coherent_vacuum_interfered_fock_elements(self):
    r"""Tests if a range of beamsplitter output states (formed from a coherent state interfering with vacuum)
       have the correct Fock basis elements.
       |\psi_in> = |\alpha>|0> --> |t \alpha>|r \alpha> = |\psi_out>
       and for each output mode,
       |\gamma> = exp(-0.5 |\gamma|^2) \sum_n \gamma^n / \sqrt{n!} |n>"""
    self.logTestName()
    phase_alpha = np.pi / 5
    for mag_alpha in self.mag_alphas[1:]:
        alpha = mag_alpha * np.exp(1j * phase_alpha)
        for t in t_values:
          for r_phi in phase_r:
            r = np.exp(1j * r_phi) * np.sqrt(1. - np.abs(t) ** 2)
            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.displacement(alpha, 0)
            self.circuit.beamsplitter(t, r, 0, 1)
            state = self.circuit.state()
            if state.is_pure:
              numer_state = state.ket()
            else:
              numer_state = state.dm()
            alpha_outA = t * alpha
            alpha_outB = r * alpha
            ref_stateA = np.array([np.exp(-0.5 * np.abs(alpha_outA) ** 2) * alpha_outA ** n / np.sqrt(factorial(n)) for n in range(self.D)])
            ref_stateB = np.array([np.exp(-0.5 * np.abs(alpha_outB) ** 2) * alpha_outB ** n / np.sqrt(factorial(n)) for n in range(self.D)])
            ref_state = np.einsum('i,j->ij',ref_stateA, ref_stateB)
            if not self.kwargs['pure']:
              ref_state = np.einsum('i,j,k,l->ijkl', ref_stateA, np.conj(ref_stateA), ref_stateB, np.conj(ref_stateB))
            self.assertAllAlmostEqual(numer_state, ref_state, delta=self.tol, msg="Test failed for t={}, r_phi={}.".format(t, r))


if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (BasicTests, FockBasisTests):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)

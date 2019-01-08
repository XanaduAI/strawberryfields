"""
Unit tests for :mod:`strawberryfields.ops`

The tests in this module do not require the quantum program to be run, and thus do not use a backend.
"""

import logging
logging.getLogger()

import unittest

from defaults import BaseTest, strawberryfields as sf
from strawberryfields.ops import *

from draw_circuit_test_utils import *


class CircuitDrawerTests(BaseTest):

    num_subsystems = 3
    cutoff = 10

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)

    def test_x_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == x_test_0_output, failure_message(result, x_test_0_output))

    def test_x_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == x_test_1_output, failure_message(result, x_test_1_output))

    def test_xx_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[1])
            Xgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == xx_test_1_output, failure_message(result, xx_test_1_output))

    def test_x_z_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[0])
            Zgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == x_z_test_0_output, failure_message(result, x_z_test_0_output))

    def test_x_0_z_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[0])
            Zgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == x_0_z_1_test_output, failure_message(result, x_0_z_1_test_output))

    def test_z_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Zgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == z_test_0_output, failure_message(result, z_test_0_output))

    def test_z_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Zgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == z_test_1_output, failure_message(result, z_test_1_output))

    def test_zz_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Zgate(1) | (q[1])
            Zgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == zz_test_1_output, failure_message(result, zz_test_1_output))

    def test_cx(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            CXgate(1) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == cx_test_output, failure_message(result, cx_test_output))

    def test_cz(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            CZgate(1) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == cz_test_output, failure_message(result, cz_test_output))

    def test_bs(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            BSgate(1) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == bs_test_output, failure_message(result, bs_test_output))

    def test_s2(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            S2gate(1) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == s2_test_output, failure_message(result, s2_test_output))

    def test_ck(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            CKgate(1) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == ck_test_output, failure_message(result, ck_test_output))
        
    def test_k_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Kgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == k_test_0_output, failure_message(result, k_test_0_output))

    def test_k_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Kgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == k_test_1_output, failure_message(result, k_test_1_output))
        
    def test_v_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Vgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == v_test_0_output, failure_message(result, v_test_0_output))

    def test_v_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Vgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == v_test_1_output, failure_message(result, v_test_1_output))
        
    def test_p_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Pgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == p_test_0_output, failure_message(result, p_test_0_output))

    def test_p_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Pgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == p_test_1_output, failure_message(result, p_test_1_output))
        
    def test_r_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Rgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == r_test_0_output, failure_message(result, r_test_0_output))

    def test_r_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Rgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == r_test_1_output, failure_message(result, r_test_1_output))

    def test_s_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Sgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == s_test_0_output, failure_message(result, s_test_0_output))

    def test_s_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Sgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == s_test_1_output, failure_message(result, s_test_1_output))

    def test_d_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Dgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == d_test_0_output, failure_message(result, d_test_0_output))

    def test_d_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Dgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True, compile_pdf=False)
        self.assertTrue(result == d_test_1_output, failure_message(result, d_test_1_output))


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', circuit drawer module.')
    # run the tests in this file
    suite = unittest.TestSuite()
    ttt = unittest.TestLoader().loadTestsFromTestCase(CircuitDrawerTests)
    suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

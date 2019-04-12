"""
Unit tests for :mod:`strawberryfields.ops`

The tests in this module do not require the quantum program to be run, and thus do not use a backend.
"""

import logging
logging.getLogger()

import unittest
import datetime
from pathlib import Path

from defaults import BaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.circuitdrawer import Circuit as CircuitDrawer

from draw_circuit_test_utils import *


class CircuitDrawerTests(BaseTest):

    num_subsystems = 3
    cutoff = 10

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)
        self.drawer = CircuitDrawer(CircuitDrawerTests.num_subsystems)

    def test_end_wire(self):
        self.logTestName()
        self.drawer._end_wire()
        self.assertTrue(self.drawer._document.endswith(WIRE_TERMINATOR))

    def test_end_circuit(self):
        self.logTestName()
        self.drawer._end_circuit()
        self.assertTrue(self.drawer._document.endswith(CIRCUIT_BODY_TERMINATOR))

    def test_begin_circuit(self):
        self.logTestName()
        self.drawer._begin_circuit()
        self.assertTrue(self.drawer._document.endswith(CIRCUIT_BODY_START))\

    def test_init_document(self):
        self.logTestName()
        self.drawer._init_document()
        self.assertTrue(self.drawer._document.endswith(INIT_DOCUMENT))

    def test_set_row_spacing(self):
        self.logTestName()
        self.drawer._set_row_spacing('1em')
        self.assertTrue(self.drawer._row_spacing == '1em')

    def test_set_column_spacing(self):
        self.logTestName()
        self.drawer._set_column_spacing('2em')
        self.assertTrue(self.drawer._column_spacing == '2em')

    def test_apply_spacing(self):
        self.logTestName()
        self.drawer._set_row_spacing('1em')
        self.drawer._set_column_spacing('2em')
        self.drawer._apply_spacing()
        self.assertTrue(self.drawer._pad_with_spaces(COLUMN_SPACING.format('2em')) in self.drawer._document)
        self.assertTrue(self.drawer._pad_with_spaces(ROW_SPACING.format('1em')) in self.drawer._document)

    def test_one_mode_gates_from_operators(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[0])
            Zgate(1) | (q[0])
            Kgate(1) | (q[0])
            Vgate(1) | (q[0])
            Pgate(1) | (q[0])
            Rgate(1) | (q[0])
            Sgate(1) | (q[0])
            Dgate(1) | (q[0])

        for op in self.eng.cmd_queue:
            method, mode = self.drawer._gate_from_operator(op)
            self.assertTrue((callable(method) and hasattr(self.drawer, method.__name__)))
            self.assertTrue(mode == 1)

    def test_two_mode_gates_from_operators(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            CXgate(1) | (q[0], q[1])
            CZgate(1) | (q[0], q[1])
            BSgate(1) | (q[0], q[1])
            S2gate(1) | (q[0], q[1])
            CKgate(1) | (q[0], q[1])

        for op in self.eng.cmd_queue:
            method, mode = self.drawer._gate_from_operator(op)
            self.assertTrue((callable(method) and hasattr(self.drawer, method.__name__)))
            self.assertTrue(mode == 2)

    def test_op_applications(self):
        self.drawer._x(0)
        self.drawer._z(0)
        self.drawer._cx(0, 1)
        self.drawer._cz(0, 1)
        self.drawer._bs(0, 1)
        self.drawer._s2(0, 1)
        self.drawer._ck(0, 1)
        self.drawer._k(0)
        self.drawer._v(0)
        self.drawer._p(0)
        self.drawer._r(0)
        self.drawer._s(0)
        self.drawer._d(0)

        self.assertTrue(self.drawer._circuit_matrix == expected_circuit_matrix)

    def test_parse_op(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[0])
            Zgate(1) | (q[0])
            CXgate(1) | (q[0], q[1])
            CZgate(1) | (q[0], q[1])
            BSgate(0, 1) | (q[0], q[1])
            S2gate(0, 1) | (q[0], q[1])
            CKgate(1) | (q[0], q[1])
            Kgate(1) | (q[0])
            Vgate(1) | (q[0])
            Pgate(1) | (q[0])
            Rgate(1) | (q[0])
            Sgate(1) | (q[0])
            Dgate(1) | (q[0])

        for op in self.eng.cmd_queue:
            self.drawer.parse_op(op)

        self.assertTrue(self.drawer._circuit_matrix == expected_circuit_matrix)

    def test_add_column(self):
        self.drawer._add_column()
        for wire in self.drawer._circuit_matrix:
            self.assertTrue(wire[-1] == QUANTUM_WIRE)

    def test_on_empty_column(self):
        self.logTestName()

        self.drawer._add_column()
        self.assertTrue(self.drawer._on_empty_column())

    def test_fourier(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Fourier | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == fourier_output, failure_message(result, fourier_output))

    def test_x_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == x_test_0_output, failure_message(result, x_test_0_output))

    def test_x_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == x_test_1_output, failure_message(result, x_test_1_output))

    def test_xx_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[1])
            Xgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == xx_test_1_output, failure_message(result, xx_test_1_output))

    def test_x_z_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[0])
            Zgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == x_z_test_0_output, failure_message(result, x_z_test_0_output))

    def test_x_0_z_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Xgate(1) | (q[0])
            Zgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == x_0_z_1_test_output, failure_message(result, x_0_z_1_test_output))

    def test_z_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Zgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == z_test_0_output, failure_message(result, z_test_0_output))

    def test_z_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Zgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == z_test_1_output, failure_message(result, z_test_1_output))

    def test_zz_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Zgate(1) | (q[1])
            Zgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == zz_test_1_output, failure_message(result, zz_test_1_output))

    def test_cx(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            CXgate(1) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == cx_test_output, failure_message(result, cx_test_output))

    def test_cz(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            CZgate(1) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == cz_test_output, failure_message(result, cz_test_output))

    def test_bs(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            BSgate(0, 1) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == bs_test_output, failure_message(result, bs_test_output))

    def test_s2(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            S2gate(1) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == s2_test_output, failure_message(result, s2_test_output))

    def test_ck(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            CKgate(1) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == ck_test_output, failure_message(result, ck_test_output))
        
    def test_k_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Kgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == k_test_0_output, failure_message(result, k_test_0_output))

    def test_k_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Kgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == k_test_1_output, failure_message(result, k_test_1_output))
        
    def test_v_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Vgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == v_test_0_output, failure_message(result, v_test_0_output))

    def test_v_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Vgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == v_test_1_output, failure_message(result, v_test_1_output))
        
    def test_p_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Pgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == p_test_0_output, failure_message(result, p_test_0_output))

    def test_p_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Pgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == p_test_1_output, failure_message(result, p_test_1_output))
        
    def test_r_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Rgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == r_test_0_output, failure_message(result, r_test_0_output))

    def test_r_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Rgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == r_test_1_output, failure_message(result, r_test_1_output))

    def test_s_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Sgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == s_test_0_output, failure_message(result, s_test_0_output))

    def test_s_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Sgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == s_test_1_output, failure_message(result, s_test_1_output))

    def test_d_0(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Dgate(1) | (q[0])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == d_test_0_output, failure_message(result, d_test_0_output))

    def test_d_1(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Dgate(1) | (q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)[1]
        self.assertTrue(result == d_test_1_output, failure_message(result, d_test_1_output))

    def test_not_drawable(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            BSgate(0, 2) | (q[0], q[2])

        with self.assertRaises(Exception):
            self.eng.draw_circuit(print_queued_ops=True)

    def test_compile(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Dgate(1) | (q[1])
            Rgate(1) | (q[1])
            S2gate(1) | (q[0], q[1])

        document = self.eng.draw_circuit(print_queued_ops=True)[0]

        file_name = "output_{0}.tex".format(datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p"))
        self.assertTrue(document.split('/')[-1] == file_name)

        output_file = Path(document)
        self.assertTrue(output_file.is_file())


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', circuit drawer module.')
    # run the tests in this file
    suite = unittest.TestSuite()
    ttt = unittest.TestLoader().loadTestsFromTestCase(CircuitDrawerTests)
    suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

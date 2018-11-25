"""
Unit tests for :mod:`strawberryfields.ops`

The tests in this module do not require the quantum program to be run, and thus do not use a backend.
"""

import logging
logging.getLogger()

import unittest

from defaults import BaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import operation


@operation(1)
def prepare_state(v1, q):
    Coherent(v1) | q


@operation(2)
def entangle_states(q):
    Squeezed(-2) | q[0]
    Squeezed(2) | q[1]
    BSgate(pi / 4, 0) | (q[0], q[1])


class CircuitDrawerTests(BaseTest):

    num_subsystems = 3
    cutoff = 10

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)

    def test_queue_draw(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            prepare_state(0.5+0.2j) | q[0]
            entangle_states() | (q[1], q[2])
            BSgate(pi / 4, 0) | (q[0], q[1])

        result = self.eng.draw_circuit(print_queued_ops=True)
        self.assertTrue(result != -1)


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', circuit drawer module.')
    # run the tests in this file
    suite = unittest.TestSuite()
    ttt = unittest.TestLoader().loadTestsFromTestCase(CircuitDrawerTests)
    suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

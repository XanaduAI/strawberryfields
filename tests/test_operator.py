"""
Unit tests for the :mod:`strawberryfields` operator decorator.
"""

import unittest

from defaults import BaseTest, FockBaseTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import operator

float_regex = r'(-)?(\d+\.\d+)'


class TeleportationOperator(BaseTest):
    """
    Run a teleportation algorithm but split the circuit into operators.
    Operators can be also methods of a class
    """
    num_subsystems = 3
    delta = 0.1
    cutoff = 10

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems)
        # attach the backend (NOTE: self.backend is shared between the tests, make sure it is reset before use!)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.cutoff)

    @operator(1)
    def prepare_state(self, v1, q):
        Coherent(v1) | q

    @operator(2)
    def entangle_states(self, q):
        Squeezed(-2) | q[0]
        Squeezed(2) | q[1]
        BSgate(pi / 4, 0) | (q[0], q[1])

    def test_teleportation_fidelity(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            self.prepare_state(0.5+0.2j) | q[0]

            self.entangle_states() | (q[1], q[2])

            BSgate(pi / 4, 0) | (q[0], q[1])
            MeasureHomodyne(0, select=0) | q[0]
            MeasureHomodyne(pi/2, select=0) | q[1]

        state = self.eng.run()
        fidelity = state.fidelity_coherent([0,0,0.5+0.2j])
        self.assertAllAlmostEqual(fidelity, 1, delta=self.delta)

    def test_validate_argument(self):
        self.logTestName()

        with self.assertRaises(ValueError):
            self.prepare_state(1, 2, 3) | (1, 2)

        with self.assertRaises(ValueError):
            self.prepare_state(1) | (1, 2)

        with self.assertRaises(ValueError):
            self.prepare_state(1) | ()

        with self.assertRaises(ValueError):
            self.entangle_states() | (1, 2, 3)

        with self.assertRaises(ValueError):
            self.entangle_states() | (1)

        with self.assertRaises(ValueError):
            self.entangle_states() | 1

if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', examples.')

    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [
        TeleportationOperator,
    ]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

"""
Unit tests for utilities in :class:`strawberryfields.utils`.
"""

import os
import unittest

import numpy as np
from numpy import pi

from defaults import FockBaseTest, strawberryfields
import strawberryfields.backends.shared_ops as so


bs_4_val = np.array([ 1.00000000+0.j,  1.00000000+0.j,  1.00000000+0.j,  1.00000000+0.j,
        1.00000000+0.j,  1.41421356+0.j,  1.73205081+0.j,  1.00000000+0.j,
        1.73205081+0.j,  1.00000000+0.j, -1.00000000+0.j, -1.41421356+0.j,
       -1.73205081+0.j,  1.00000000+0.j, -1.00000000+0.j,  1.00000000+0.j,
       -2.00000000+0.j,  1.00000000+0.j, -3.00000000+0.j,  1.00000000+0.j,
        1.41421356+0.j, -1.00000000+0.j,  2.00000000+0.j, -2.44948974+0.j,
        2.44948974+0.j,  1.73205081+0.j, -1.00000000+0.j,  3.00000000+0.j,
        1.00000000+0.j,  1.73205081+0.j, -1.41421356+0.j,  1.00000000+0.j,
       -2.00000000+0.j,  2.44948974+0.j, -2.44948974+0.j,  1.00000000+0.j,
       -2.00000000+0.j,  1.00000000+0.j,  1.00000000+0.j, -4.00000000+0.j,
        1.00000000+0.j,  3.00000000+0.j, -6.00000000+0.j,  1.00000000+0.j,
        1.73205081+0.j, -2.44948974+0.j,  2.44948974+0.j,  1.00000000+0.j,
       -6.00000000+0.j,  3.00000000+0.j, -1.00000000+0.j,  1.73205081+0.j,
       -1.00000000+0.j,  3.00000000+0.j, -1.73205081+0.j,  2.44948974+0.j,
       -2.44948974+0.j, -1.00000000+0.j,  6.00000000+0.j, -3.00000000+0.j,
        1.00000000+0.j, -3.00000000+0.j,  1.00000000+0.j,  3.00000000+0.j,
       -6.00000000+0.j,  1.00000000+0.j, -1.00000000+0.j,  9.00000000+0.j,
       -9.00000000+0.j,  1.00000000+0.j])

bs_4_idx = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3],
       [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2,
        2, 2, 2, 3, 3, 3, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3],
       [0, 1, 2, 3, 1, 2, 3, 2, 3, 3, 0, 1, 2, 0, 1, 1, 2, 2, 3, 3, 1, 2,
        2, 3, 3, 2, 3, 3, 0, 1, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2, 3, 3, 3,
        1, 2, 2, 3, 3, 3, 0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2,
        3, 3, 3, 3],
       [0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0, 1,
        1, 2, 2, 0, 1, 1, 2, 3, 1, 2, 2, 3, 3, 0, 1, 1, 2, 2, 2, 3, 3, 3,
        0, 1, 1, 2, 2, 2, 3, 2, 3, 3, 1, 2, 2, 3, 3, 3, 0, 1, 1, 2, 2, 2,
        3, 3, 3, 3],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0,
        1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2,
        2, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 1, 2, 0, 1, 2, 3, 2, 3, 1, 2, 3,
        0, 1, 2, 3]])

squeeze_parity_8 = np.array(
        [[ 1,  0, -1,  0,  1,  0, -1,  0],
        [ 0,  1,  0, -1,  0,  1,  0, -1],
        [-1,  0,  1,  0, -1,  0,  1,  0],
        [ 0, -1,  0,  1,  0, -1,  0,  1],
        [ 1,  0, -1,  0,  1,  0, -1,  0],
        [ 0,  1,  0, -1,  0,  1,  0, -1],
        [-1,  0,  1,  0, -1,  0,  1,  0],
        [ 0, -1,  0,  1,  0, -1,  0,  1]]
    )

squeeze_factor_4 = np.array(
    [[[ 1.        ,  0.        , -0.        ,  0.        ],
        [ 0.        ,  0.        , -0.        ,  0.        ],
        [ 1.41421356,  0.        , -0.        ,  0.        ],
        [ 0.        ,  0.        , -0.        ,  0.        ]],

       [[ 0.        ,  0.        ,  0.        , -0.        ],
        [ 0.        ,  1.        ,  0.        , -0.        ],
        [ 0.        ,  0.        ,  0.        , -0.        ],
        [ 0.        ,  2.44948974,  0.        , -0.        ]],

       [[-1.41421356,  0.        ,  0.        ,  0.        ],
        [-0.        ,  0.        ,  0.        ,  0.        ],
        [-2.        ,  0.        ,  1.        ,  0.        ],
        [-0.        ,  0.        ,  0.        ,  0.        ]],

       [[ 0.        , -0.        ,  0.        ,  0.        ],
        [ 0.        , -2.44948974,  0.        ,  0.        ],
        [ 0.        , -0.        ,  0.        ,  0.        ],
        [ 0.        , -6.        ,  0.        ,  1.        ]]]
    )

class BeamsplitterFactors(FockBaseTest):
    num_subsystems = 1

    def test_generate_bs_factors(self):
        self.logTestName()
        factors = so.generate_bs_factors(4)
        factors_val = factors[factors!=0.]
        factors_idx = np.array(np.nonzero(factors))

        self.assertAllAlmostEqual(factors_val, bs_4_val, delta=self.tol)
        self.assertAllAlmostEqual(factors_idx, bs_4_idx, delta=self.tol)

    def test_save_load_bs_factors(self):
        self.logTestName()
        factors = so.generate_bs_factors(4)
        so.save_bs_factors(factors, directory="./")

        factors = so.load_bs_factors(4, directory="./")
        factors_val = factors[factors!=0.]
        factors_idx = np.array(np.nonzero(factors))

        self.assertAllAlmostEqual(factors_val, bs_4_val, delta=self.tol)
        self.assertAllAlmostEqual(factors_idx, bs_4_idx, delta=self.tol)

    def tearDown(self):
        self.logTestName()
        try:
            os.remove("fock_beamsplitter_factors_4.npz")
        except OSError:
            pass


class SqueezingFactors(FockBaseTest):
    num_subsystems = 1

    def test_squeeze_parity(self):
        self.logTestName()
        parity = so.squeeze_parity(8)
        self.assertAllAlmostEqual(parity, squeeze_parity_8, delta=self.tol)

    def test_generate_squeeze_factors(self):
        self.logTestName()
        factors = so.generate_squeeze_factors(4)
        self.assertAllAlmostEqual(factors, squeeze_factor_4, delta=self.tol)

    def test_save_load_squeeze_factors(self):
        self.logTestName()
        factors_in = so.generate_squeeze_factors(4)
        so.save_squeeze_factors(factors_in, directory="./")
        factors_out = so.load_squeeze_factors(4, directory="./")

        self.assertAllAlmostEqual(factors_out, squeeze_factor_4, delta=self.tol)

    def tearDown(self):
        self.logTestName()
        try:
            os.remove("fock_squeeze_factors_4.npz")
        except OSError:
            pass


if __name__ == '__main__':
    print('Testing Strawberry Fields shared_ops.py')

    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [BeamsplitterFactors, SqueezingFactors]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

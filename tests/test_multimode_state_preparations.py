##############################################################################
#
# Unit tests for various multi mode state preparation operations
#
##############################################################################

import unittest
import os, sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.special import factorial
from defaults import BaseTest, FockBaseTest

mag_alphas = np.linspace(0, .8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)
nbars = np.linspace(0, 5)

###################################################################

class BasicTests(BaseTest):
    """Basic implementation-independent tests."""
    num_subsystems = 2


class FockBasisTests(FockBaseTest):
    """Tests for simulators that use Fock basis."""
    num_subsystems = 2
    D = 2

    def test_prepare_multimode_random_dm_state(self):
        """Tests if a random multi mode dm state is correctly prepared."""
        random_rho = np.random.normal(size=[self.D]*2*self.num_subsystems) + 1j*np.random.normal(size=[self.D]*2*self.num_subsystems)
        random_rho = np.dot(random_rho.conj().T, random_rho)
        random_rho = random_rho/random_rho.trace()

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_multimode_dm_state(random_rho, range(self.num_subsystems))
        state = self.circuit.state()
        rho_probs = np.array([state.fock_prob([n]) for n in range(self.D)])


if __name__=="__main__":
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTests, FockBasisTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

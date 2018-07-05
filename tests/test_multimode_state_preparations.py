##############################################################################
#
# Unit tests for various multi mode state preparation operations
#
##############################################################################

import unittest
import os, sys
sys.path.append(os.getcwd())
import numpy as np
import itertools as it
from scipy.special import factorial
from defaults import BaseTest, FockBaseTest

mag_alphas = np.linspace(0, .8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)
nbars = np.linspace(0, 5)

###################################################################

class FockBasisMultimodeTests(FockBaseTest):
    """Tests for simulators that use Fock basis."""
    num_subsystems = 4

    def test_compare_single_mode_and_multimode_preparation(self):
        random_rho0 = np.random.normal(size=[self.D]*2) + 1j*np.random.normal(size=[self.D]*2)
        random_rho0 = np.dot(random_rho0.conj().T, random_rho0)
        random_rho0 = random_rho0/random_rho0.trace()

        random_rho1 = np.random.normal(size=[self.D]*2) + 1j*np.random.normal(size=[self.D]*2)
        random_rho1 = np.dot(random_rho1.conj().T, random_rho1)
        random_rho1 = random_rho1/random_rho1.trace()
        random_dm = np.outer(random_rho0, random_rho1)
        random_dm = random_dm.reshape([self.D]*4)

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_dm_state(random_rho0, 0)
        self.circuit.prepare_dm_state(random_rho1, 1)
        single_mode_preparation_dm = self.circuit.state([0, 1]).dm()

        #first we do a preparation from random_dm, with shape [self.D]*4
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_multimode_dm_state(random_dm, [0, 1])
        multi_mode_preparation_dm = self.circuit.state(modes=[0, 1]).dm()

        #second we do a preparation from the corresponding matrix with shape [self.D**2]*2
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_multimode_dm_state(random_dm.reshape([self.D**2]*2), [0, 1])
        multi_mode_preparation_from_matrix_dm = self.circuit.state(modes=[0, 1]).dm()

        self.assertAllEqual(random_dm.shape, single_mode_preparation_dm.shape)
        self.assertAllEqual(random_dm.shape, multi_mode_preparation_dm.shape)
        self.assertAllEqual(random_dm.shape, multi_mode_preparation_from_matrix_dm.shape)
        self.assertAllAlmostEqual(multi_mode_preparation_dm, random_dm, delta=self.tol)
        self.assertAllAlmostEqual(multi_mode_preparation_dm, single_mode_preparation_dm, delta=self.tol)
        self.assertAllAlmostEqual(multi_mode_preparation_from_matrix_dm, single_mode_preparation_dm, delta=self.tol)


#     def test_prepare_multimode_random_product_dm_state_on_different_modes(self):
#         """Tests if a random multi mode dm state is correctly prepared on differnt modes."""
#         random_rho = np.random.normal(size=[self.D**2]*2) + 1j*np.random.normal(size=[self.D**2]*2)
#         random_rho = np.dot(random_rho.conj().T, random_rho)
#         random_rho = random_rho/random_rho.trace()
#         self.circuit.reset(pure=self.kwargs['pure'])
#         self.circuit.prepare_multimode_dm_state(random_rho, [0, 1])
#         reference_dm = self.circuit.state([0, 1]).dm()

#         for subsystems in list(it.permutations(range(self.num_subsystems), 2)):
#             print(subsystems)
#             self.circuit.reset(pure=self.kwargs['pure'])
#             self.circuit.prepare_multimode_dm_state(random_rho, subsystems)
#             #dm = self.circuit.state(modes=sorted(subsystems)).dm()
#             dm = self.circuit.state(modes=[0]).dm()
#             print(dm)
# #            self.assertAllAlmostEqual(reference_dm, dm, delta=self.tol)


#     def test_prepare_multimode_random_dm_state(self):
#         """Tests if a random multi mode dm state is correctly prepared."""
#         random_rho = np.random.normal(size=[self.D**self.num_subsystems]*2) + 1j*np.random.normal(size=[self.D**self.num_subsystems]*2)
#         random_rho = np.dot(random_rho.conj().T, random_rho)
#         random_rho = random_rho/random_rho.trace()

#         self.circuit.reset(pure=self.kwargs['pure'])
# #        self.circuit.prepare_multimode_dm_state(random_rho, range(self.num_subsystems))
#         self.circuit.prepare_multimode_dm_state(random_rho, [1,0])
#         state = self.circuit.state()
#         rho_probs = np.array([state.fock_prob([n,self.D-n-1]) for n in range(self.D)])


if __name__=="__main__":
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (FockBasisMultimodeTests,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

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

    def test_compare_single_mode_and_multimode_ket_preparation(self):
        random_ket0 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
        random_ket0 = random_ket0 / np.linalg.norm(random_ket0)

        random_ket1 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
        random_ket1 = random_ket1 / np.linalg.norm(random_ket1)

        random_ket = np.outer(random_ket0, random_ket1)

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_ket_state(random_ket0, 0)
        self.circuit.prepare_ket_state(random_ket1, 1)
        state = self.circuit.state([0, 1])
        single_mode_preparation_dm = state.dm()
        single_mode_preparation_probs = np.array(state.all_fock_probs())
        if self.batched:
            single_mode_preparation_dm = single_mode_preparation_dm[0]
            single_mode_preparation_probs = single_mode_preparation_probs[0]


        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_multimode_ket_state(random_ket, [0, 1])
        state = self.circuit.state([0, 1])
        multi_mode_preparation_dm = state.dm()
        multi_mode_preparation_probs = np.array(state.all_fock_probs())
        if self.batched:
            multi_mode_preparation_dm = multi_mode_preparation_dm[0]
            multi_mode_preparation_probs = multi_mode_preparation_probs[0]

        self.assertAllEqual(single_mode_preparation_dm.shape, multi_mode_preparation_dm.shape)
        self.assertAllAlmostEqual(single_mode_preparation_dm, multi_mode_preparation_dm, delta=self.tol)
        self.assertAllAlmostEqual(single_mode_preparation_probs, multi_mode_preparation_probs, delta=self.tol)


    def test_compare_single_mode_and_multimode_dm_preparation(self):
        """Compare the results of a successive single mode preparations and a multi mode preparation of a product state."""
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
        state = self.circuit.state([0, 1])
        single_mode_preparation_dm = state.dm()
        single_mode_preparation_probs = np.array(state.all_fock_probs())
        if self.batched:
            single_mode_preparation_dm = single_mode_preparation_dm[0]
            single_mode_preparation_probs = single_mode_preparation_probs[0]

        # first we do a preparation from random_dm, with shape [self.D]*4
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_multimode_dm_state(random_dm, [0, 1])
        state = self.circuit.state(modes=[0, 1])
        multi_mode_preparation_dm = state.dm()
        multi_mode_preparation_probs = np.array(state.all_fock_probs())
        if self.batched:
            multi_mode_preparation_dm = multi_mode_preparation_dm[0]
            multi_mode_preparation_probs = multi_mode_preparation_probs[0]

        # second we do a preparation from the corresponding matrix with shape [self.D**2]*2
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_multimode_dm_state(random_dm.reshape([self.D**2]*2), [0, 1])
        state = self.circuit.state(modes=[0, 1])
        multi_mode_preparation_from_matrix_dm = state.dm()
        multi_mode_preparation_from_matrix_probs = np.array(state.all_fock_probs())
        if self.batched:
            multi_mode_preparation_from_matrix_dm = multi_mode_preparation_from_matrix_dm[0]
            multi_mode_preparation_from_matrix_probs = multi_mode_preparation_from_matrix_probs[0]

        # third we do a preparation from random_dm on modes 3 and 1 (in that order!) and test if the states end up in the correct modes
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_multimode_dm_state(random_dm, [3, 1])

        multi_mode_preparation_31_mode_0 = self.circuit.state(modes=0).dm()
        multi_mode_preparation_31_mode_1 = self.circuit.state(modes=1).dm()
        multi_mode_preparation_31_mode_2 = self.circuit.state(modes=2).dm()
        multi_mode_preparation_31_mode_3 = self.circuit.state(modes=3).dm()
        multi_mode_preparation_31_probs = np.array(self.circuit.state(modes=[3,1]).all_fock_probs())
        if self.batched:
            multi_mode_preparation_31_mode_0 = multi_mode_preparation_31_mode_0[0]
            multi_mode_preparation_31_mode_1 = multi_mode_preparation_31_mode_1[0]
            multi_mode_preparation_31_mode_2 = multi_mode_preparation_31_mode_2[0]
            multi_mode_preparation_31_mode_3 = multi_mode_preparation_31_mode_3[0]
            multi_mode_preparation_31_probs = multi_mode_preparation_31_probs[0]

        single_mode_vac = np.zeros(multi_mode_preparation_31_mode_0.shape, dtype=np.complex128)
        single_mode_vac.itemset(0, 1.0 + 0.0j)

        self.assertAllEqual(random_dm.shape, single_mode_preparation_dm.shape)
        self.assertAllEqual(random_dm.shape, multi_mode_preparation_dm.shape)
        self.assertAllEqual(random_dm.shape, multi_mode_preparation_from_matrix_dm.shape)
        self.assertAllAlmostEqual(random_dm, single_mode_preparation_dm, delta=self.tol)
        self.assertAllAlmostEqual(multi_mode_preparation_dm, single_mode_preparation_dm, delta=self.tol)
        self.assertAllAlmostEqual(multi_mode_preparation_from_matrix_dm, single_mode_preparation_dm, delta=self.tol)

        self.assertAllAlmostEqual(multi_mode_preparation_31_mode_0, single_mode_vac, delta=self.tol)
        self.assertAllAlmostEqual(multi_mode_preparation_31_mode_1, random_rho1, delta=self.tol)
        self.assertAllAlmostEqual(multi_mode_preparation_31_mode_2, single_mode_vac, delta=self.tol)
        self.assertAllAlmostEqual(multi_mode_preparation_31_mode_3, random_rho0, delta=self.tol)

        #also check the fock probabilities to catch errors in both the preparation and state() the would cancel each other out
        self.assertAllAlmostEqual(single_mode_preparation_probs, multi_mode_preparation_probs, delta=self.tol)
        self.assertAllAlmostEqual(single_mode_preparation_probs, multi_mode_preparation_from_matrix_probs, delta=self.tol)
        self.assertAllAlmostEqual(single_mode_preparation_probs, multi_mode_preparation_31_probs, delta=self.tol)

    def test_prepare_multimode_random_product_dm_state_on_different_modes(self):
        """Tests if a random multi mode dm state is correctly prepared on differnt modes."""
        random_rho = np.random.normal(size=[self.D**2]*2) + 1j*np.random.normal(size=[self.D**2]*2) # two mode random state
        random_rho = np.dot(random_rho.conj().T, random_rho)
        random_rho = random_rho/random_rho.trace()
        random_rho = random_rho.reshape([self.D]*4) #reshape for easier comparison later

        # test the state preparation on the first two modes
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_multimode_dm_state(random_rho, [0, 1])
        multi_mode_preparation_with_modes_ordered = self.circuit.state([0, 1]).dm()
        self.assertAllAlmostEqual(multi_mode_preparation_with_modes_ordered, random_rho, delta=self.tol)

        # test the state preparation on two other modes that are not in order
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_multimode_dm_state(random_rho, [1, 2])
        multi_mode_preparation_with_modes_inverted = self.circuit.state([1, 2]).dm()
        self.assertAllAlmostEqual(multi_mode_preparation_with_modes_inverted, random_rho, delta=self.tol)

        # test various subsets of subsystems in various orders
        for subsystems in list(it.permutations(range(self.num_subsystems), 2)):
            subsystems = list(subsystems)
            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.prepare_multimode_dm_state(random_rho, subsystems)
            dm = self.circuit.state(modes=subsystems).dm()
            if self.batched:
                dm = dm[0]

            self.assertAllEqual(random_rho.shape, dm.shape)
            self.assertAllAlmostEqual(random_rho, dm, delta=self.tol)

    def test_fast_state_prep_on_all_modes(self):
        """Tests if a random multi mode ket state is correctly prepared with the shortcut method on all modes."""
        random_ket = np.random.normal(size=[self.D]*self.num_subsystems) + 1j*np.random.normal(size=[self.D]*self.num_subsystems)
        random_ket = random_ket / np.linalg.norm(random_ket)

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_multimode_dm_state(random_ket)
        all_mode_preparation_ket = self.circuit.state().ket() # Returns None if the state if mixed
        if self.batched:
            all_mode_preparation_ket = all_mode_preparation_ket[0]

        self.assertAllEqual(all_mode_preparation_ket.shape, random_ket.shape)
        self.assertAllAlmostEqual(all_mode_preparation_ket, random_ket, delta=self.tol)

if __name__=="__main__":
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (FockBasisMultimodeTests,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

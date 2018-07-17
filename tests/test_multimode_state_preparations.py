##############################################################################
#
# Tests for various multi mode state preparation operations
#
##############################################################################

import logging
logging.getLogger()

import unittest
import os, sys
import itertools as it

import numpy as np
from scipy.special import factorial

from defaults import BaseTest, FockBaseTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.backends.gaussianbackend.states import GaussianState
from strawberryfields.utils import random_covariance, displaced_squeezed_state

mag_alphas = np.linspace(0, .8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)
nbars = np.linspace(0, 5)

###################################################################

class FockBasisMultimodeTests(FockBaseTest):
    """Tests for simulators that use Fock basis."""
    num_subsystems = 4

    def test_multimode_ket_mode_permutations(self):
        """Test multimode ket preparation when modes are permuted"""
        self.logTestName()
        random_ket0 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
        random_ket0 = random_ket0 / np.linalg.norm(random_ket0)

        random_ket1 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
        random_ket1 = random_ket1 / np.linalg.norm(random_ket1)

        random_ket = np.outer(random_ket0, random_ket1)
        rho = np.einsum('ij,kl->ikjl', random_ket, random_ket.conj())

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_ket_state(random_ket, modes=[3, 1])
        state = self.circuit.state([3, 1])
        multi_mode_preparation_dm = state.dm()

        self.assertAllAlmostEqual(multi_mode_preparation_dm, rho, delta=self.tol)

    def test_compare_single_mode_and_multimode_ket_preparation(self):
        """Test single and multimode ket preparation"""
        self.logTestName()
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

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_ket_state(random_ket, [0, 1])
        state = self.circuit.state([0, 1])
        multi_mode_preparation_dm = state.dm()
        multi_mode_preparation_probs = np.array(state.all_fock_probs())

        self.assertAllAlmostEqual(single_mode_preparation_dm, multi_mode_preparation_dm, delta=self.tol)
        self.assertAllAlmostEqual(single_mode_preparation_probs, multi_mode_preparation_probs, delta=self.tol)
        if self.batched:
            single_mode_preparation_dm = single_mode_preparation_dm[0]
            multi_mode_preparation_dm = multi_mode_preparation_dm[0]
        self.assertAllEqual(single_mode_preparation_dm.shape, multi_mode_preparation_dm.shape)


    def test_compare_single_mode_and_multimode_dm_preparation(self):
        """Compare the results of a successive single mode preparations
        and a multi mode preparation of a product state."""
        self.logTestName()
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

        # first we do a preparation from random_dm, with shape [self.D]*4
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_dm_state(random_dm, [0, 1])
        state = self.circuit.state(modes=[0, 1])
        multi_mode_preparation_dm = state.dm()
        multi_mode_preparation_probs = np.array(state.all_fock_probs())

        # second we do a preparation from the corresponding matrix with shape [self.D**2]*2
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_dm_state(random_dm.reshape([self.D**2]*2), [0, 1])
        state = self.circuit.state(modes=[0, 1])
        multi_mode_preparation_from_matrix_dm = state.dm()
        multi_mode_preparation_from_matrix_probs = np.array(state.all_fock_probs())

        # third we do a preparation from random_dm on modes 3 and 1 (in that order!) and test if the states end up in the correct modes
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_dm_state(random_dm, [3, 1])

        multi_mode_preparation_31_mode_0 = self.circuit.state(modes=0).dm()
        multi_mode_preparation_31_mode_1 = self.circuit.state(modes=1).dm()
        multi_mode_preparation_31_mode_2 = self.circuit.state(modes=2).dm()
        multi_mode_preparation_31_mode_3 = self.circuit.state(modes=3).dm()
        multi_mode_preparation_31_probs = np.array(self.circuit.state(modes=[3,1]).all_fock_probs())

        single_mode_vac = np.zeros((self.D, self.D), dtype=np.complex128)
        single_mode_vac.itemset(0, 1.0 + 0.0j)

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

        if self.batched:
            single_mode_preparation_dm = single_mode_preparation_dm[0]
            multi_mode_preparation_dm = multi_mode_preparation_dm[0]
            multi_mode_preparation_from_matrix_dm = multi_mode_preparation_from_matrix_dm[0]
        self.assertAllEqual(random_dm.shape, single_mode_preparation_dm.shape)
        self.assertAllEqual(random_dm.shape, multi_mode_preparation_dm.shape)
        self.assertAllEqual(random_dm.shape, multi_mode_preparation_from_matrix_dm.shape)


    def test_prepare_multimode_random_product_dm_state_on_different_modes(self):
        """Tests if a random multi mode dm state is correctly prepared on different modes."""
        self.logTestName()
        random_rho = np.random.normal(size=[self.D**2]*2) + 1j*np.random.normal(size=[self.D**2]*2) # two mode random state
        random_rho = np.dot(random_rho.conj().T, random_rho)
        random_rho = random_rho/random_rho.trace()
        random_rho = random_rho.reshape([self.D]*4) #reshape for easier comparison later

        # test the state preparation on the first two modes
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_dm_state(random_rho, [0, 1])
        multi_mode_preparation_with_modes_ordered = self.circuit.state([0, 1]).dm()
        self.assertAllAlmostEqual(multi_mode_preparation_with_modes_ordered, random_rho, delta=self.tol)

        # test the state preparation on two other modes that are not in order
        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_dm_state(random_rho, [1, 2])
        multi_mode_preparation_with_modes_inverted = self.circuit.state([1, 2]).dm()
        self.assertAllAlmostEqual(multi_mode_preparation_with_modes_inverted, random_rho, delta=self.tol)

        # test various subsets of subsystems in various orders
        for subsystems in list(it.permutations(range(self.num_subsystems), 2)):
            subsystems = list(subsystems)
            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.prepare_dm_state(random_rho, subsystems)
            dm = self.circuit.state(modes=subsystems).dm()

            self.assertAllAlmostEqual(random_rho, dm, delta=self.tol)
            if self.batched:
                dm = dm[0]
            self.assertAllEqual(random_rho.shape, dm.shape)

    def test_fast_state_prep_on_all_modes(self):
        """Tests if a random multi mode ket state is correctly prepared with
        the shortcut method on all modes."""
        self.logTestName()
        random_ket = np.random.normal(size=[self.D]*self.num_subsystems) + 1j*np.random.normal(size=[self.D]*self.num_subsystems)
        random_ket = random_ket / np.linalg.norm(random_ket)

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_dm_state(random_ket, modes=range(self.num_subsystems))
        all_mode_preparation_ket = self.circuit.state().ket() # Returns None if the state if mixed

        self.assertAllAlmostEqual(all_mode_preparation_ket, random_ket, delta=self.tol)
        if self.batched:
            all_mode_preparation_ket = all_mode_preparation_ket[0]
        self.assertAllEqual(all_mode_preparation_ket.shape, random_ket.shape)


class GaussianMultimodeTests(GaussianBaseTest):
    """Tests for simulators that use the Gaussian representation."""
    num_subsystems = 4

    def test_singlemode_gaussian_state(self):
        """Test single mode Gaussian state preparation"""
        self.logTestName()
        means = 2*np.random.random(size=[2])-1
        cov = random_covariance(1, pure=self.kwargs['pure'])

        a = 0.2+0.4j
        r = 1
        phi = 0

        self.circuit.reset(pure=self.kwargs['pure'])

        # circuit is initially in displaced squeezed state
        for i in range(self.num_subsystems):
            self.circuit.prepare_displaced_squeezed_state(a, r, phi, mode=i)

        # prepare Gaussian state in mode 1
        self.circuit.prepare_gaussian_state(means, cov, modes=1)

        # test Gaussian state is correct
        state = self.circuit.state([1])
        self.assertAllAlmostEqual(state.means(), means, delta=self.tol)
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)

        # test that displaced squeezed states are unchanged
        ex_means, ex_V = displaced_squeezed_state(a, r, phi, basis='gaussian')
        for i in [0, 2, 3]:
            state = self.circuit.state([i])
            self.assertAllAlmostEqual(state.means(), ex_means, delta=self.tol)
            self.assertAllAlmostEqual(state.cov(), ex_V, delta=self.tol)

    def test_multimode_gaussian_state(self):
        """Test multimode Gaussian state preparation"""
        self.logTestName()
        cov = np.diag(np.exp(2*np.array([-1, -1, 1, 1])))
        means = np.zeros([4])

        self.circuit.reset(pure=self.kwargs['pure'])

        # prepare displaced squeezed states in all modes
        a = 0.2+0.4j
        r = 0.5
        phi = 0.12
        for i in range(self.num_subsystems):
            self.circuit.prepare_displaced_squeezed_state(a, r, phi, i)

        # prepare new squeezed displaced state in mode 1 and 3
        self.circuit.prepare_gaussian_state(means, cov, modes=[1, 3])
        state = self.circuit.state([1, 3])

        # test Gaussian state is correct
        state = self.circuit.state([1, 3])
        self.assertAllAlmostEqual(state.means(), means, delta=self.tol)
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)

        # test that displaced squeezed states are unchanged
        ex_means, ex_V = displaced_squeezed_state(a, r, phi, basis='gaussian')
        for i in [0, 2]:
            state = self.circuit.state([i])
            self.assertAllAlmostEqual(state.means(), ex_means, delta=self.tol)
            self.assertAllAlmostEqual(state.cov(), ex_V, delta=self.tol)

    def test_full_mode_squeezed_state(self):
        """Test full register Gaussian state preparation"""
        self.logTestName()
        cov = np.diag(np.exp(2*np.array([-1, -1, -1, -1, 1, 1, 1, 1])))
        means = np.zeros([8])

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_gaussian_state(means, cov, modes=range(self.num_subsystems))
        state = self.circuit.state()

        # test Gaussian state is correct
        state = self.circuit.state()
        self.assertAllAlmostEqual(state.means(), means, delta=self.tol)
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)

    def test_multimode_gaussian_random_state(self):
        """Test multimode Gaussian state preparation on a random state"""
        self.logTestName()
        means = 2*np.random.random(size=[2*self.num_subsystems])-1
        cov = random_covariance(self.num_subsystems, pure=self.kwargs['pure'])

        self.circuit.reset(pure=self.kwargs['pure'])

        # circuit is initially in a random state
        self.circuit.prepare_gaussian_state(means, cov, modes=range(self.num_subsystems))

        # test Gaussian state is correct
        state = self.circuit.state()
        self.assertAllAlmostEqual(state.means(), means, delta=self.tol)
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)

        # prepare Gaussian state in mode 2 and 1
        means2 = 2*np.random.random(size=[4])-1
        cov2 = random_covariance(2, pure=self.kwargs['pure'])
        self.circuit.prepare_gaussian_state(means2, cov2, modes=[2, 1])

        # test resulting Gaussian state is correct
        state = self.circuit.state()

        # in the new means vector, the modes 0 and 3 remain unchanged
        # Modes 1 and 2, however, now have values given from elements
        # means2[1] and means2[0].
        ex_means = np.array([means[0], means2[1], means2[0], means[3],  # position
                             means[4], means2[3], means2[2], means[7]]) # momentum

        ex_cov = np.zeros([8, 8])

        # in the new covariance matrix, modes 0 and 3 remain unchanged
        idx = np.array([0, 3, 4, 7])
        rows = idx.reshape(-1, 1)
        cols = idx.reshape(1, -1)
        ex_cov[rows, cols] = cov[rows, cols]

        # in the new covariance matrix, modes 1 and 2 have values given by
        # rows 1 and 0 respectively from cov2
        idx = np.array([1, 2, 5, 6])
        rows = idx.reshape(-1, 1)
        cols = idx.reshape(1, -1)

        idx = np.array([1, 0, 3, 2])
        rows2 = idx.reshape(-1, 1)
        cols2 = idx.reshape(1, -1)

        ex_cov[rows, cols] = cov2[rows2, cols2]

        self.assertAllAlmostEqual(state.means(), ex_means, delta=self.tol)
        self.assertAllAlmostEqual(state.cov(), ex_cov, delta=self.tol)


class FrontendFockTests(FockBaseTest):
    """Tests for the frontend Fock state preparations"""
    num_subsystems = 2

    def setUp(self):
        """Create the engine"""
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend

    def test_ket_input_validation(self):
        """Test exceptions"""
        mu = np.array([0., 0.])
        cov = np.identity(2)
        state1 = GaussianState((mu, cov), 1, True, self.hbar)
        state2 = BaseFockState(np.zeros(self.D), 1, False, self.D, self.hbar)

        q = self.eng.register

        with self.eng:
            with self.assertRaisesRegex(ValueError, "Gaussian states are not supported"):
                Ket(state1) | q[0]
            with self.assertRaisesRegex(ValueError, "not pure"):
                Ket(state2) | q[0]

    def test_ket_one_mode(self):
        """Tests single mode ket preparation"""
        q = self.eng.register
        ket0 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
        ket0 = ket0 / np.linalg.norm(ket0)

        with self.eng:
            Ket(ket0) | q[0]

        state = self.eng.run(modes=[0])
        self.assertAllAlmostEqual(state.dm(), np.outer(ket0, ket0.conj()), delta=self.tol)

        self.eng.reset()
        state1 = BaseFockState(ket0, 1, True, self.D, self.hbar)

        with self.eng:
            Ket(state1) | q[0]

        state2 = self.eng.run(modes=[0])
        self.assertAllAlmostEqual(state1.dm(), state2.dm(), delta=self.tol)

    def test_ket_two_mode(self):
        """Tests multimode ket preparation"""
        q = self.eng.register
        ket0 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
        ket0 = ket0 / np.linalg.norm(ket0)
        ket1 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
        ket1 = ket1 / np.linalg.norm(ket1)

        ket = np.outer(ket0, ket1)

        with self.eng:
            Ket(ket) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.dm(), np.einsum('ij,kl->ikjl', ket, ket.conj()), delta=self.tol)

        self.eng.reset()
        state1 = BaseFockState(ket, 2, True, self.D, self.hbar)

        with self.eng:
            Ket(state1) | q

        state2 = self.eng.run()
        self.assertAllAlmostEqual(state1.dm(), state2.dm(), delta=self.tol)

    def test_dm_input_validation(self):
        """Test exceptions"""
        mu = np.array([0., 0.])
        cov = np.identity(2)
        state = GaussianState((mu, cov), 1, True, self.hbar)

        q = self.eng.register

        with self.eng:
            with self.assertRaisesRegex(ValueError, "Gaussian states are not supported"):
                DensityMatrix(state) | q[0]

    def test_dm_one_mode(self):
        """Tests single mode DM preparation"""
        q = self.eng.register
        ket = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
        ket = ket / np.linalg.norm(ket)
        rho = np.outer(ket, ket.conj())

        with self.eng:
            DensityMatrix(rho) | q[0]

        state = self.eng.run(modes=[0])
        self.assertAllAlmostEqual(state.dm(), rho, delta=self.tol)

        self.eng.reset()
        state1 = BaseFockState(rho, 1, False, self.D, self.hbar)

        with self.eng:
            DensityMatrix(state1) | q[0]

        state2 = self.eng.run(modes=[0])
        self.assertAllAlmostEqual(state1.dm(), state2.dm(), delta=self.tol)

    def test_dm_two_mode(self):
        """Tests multimode dm preparation"""
        q = self.eng.register
        ket0 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
        ket0 = ket0 / np.linalg.norm(ket0)
        ket1 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
        ket1 = ket1 / np.linalg.norm(ket1)

        ket = np.outer(ket0, ket1)
        rho = np.einsum('ij,kl->ikjl', ket, ket.conj())

        with self.eng:
            DensityMatrix(rho) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.dm(), rho, delta=self.tol)

        self.eng.reset()
        state1 = BaseFockState(rho, 2, False, self.D, self.hbar)

        with self.eng:
            DensityMatrix(state1) | q

        state2 = self.eng.run()
        self.assertAllAlmostEqual(state1.dm(), state2.dm(), delta=self.tol)


if __name__=="__main__":
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (FockBasisMultimodeTests, GaussianMultimodeTests, FrontendFockTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

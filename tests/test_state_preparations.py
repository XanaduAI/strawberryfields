##############################################################################
#
# Unit tests for various state preparation operations
#
##############################################################################

import unittest

import numpy as np
from scipy.special import factorial

from defaults import BaseTest, FockBaseTest


mag_alphas = np.linspace(0, .8, 4)
phase_alphas = np.linspace(0, 2 * np.pi, 7, endpoint=False)
nbars = np.linspace(0, 5)

###################################################################

class BasicTests(BaseTest):
    """Basic implementation-independent tests."""
    num_subsystems = 1

    def test_prepare_vac(self):
        """Tests the ability to prepare vacuum states."""
        self.logTestName()
        self.circuit.prepare_vacuum_state(0)
        self.assertAllTrue(self.circuit.is_vacuum(self.tol))

    def test_fidelity_coherent(self):
        """Tests if a range of coherent states have the correct fidelity."""
        self.logTestName()
        for mag_alpha in mag_alphas:
            for phase_alpha in phase_alphas:
                self.circuit.reset(pure=self.kwargs['pure'])
                alpha = mag_alpha * np.exp(1j * phase_alpha)
                self.circuit.prepare_coherent_state(alpha, 0)
                state = self.circuit.state()
                self.assertAllAlmostEqual(state.fidelity_coherent([alpha]), 1, delta=self.tol)

    def test_prepare_thermal_state(self):
        self.logTestName()
        for nbar in nbars:
            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.prepare_thermal_state(nbar, 0)
            ref_probs = np.array([nbar ** n / (nbar + 1) ** (n + 1) for n in range(self.D)])
            state = self.circuit.state()
            state_probs = np.array([state.fock_prob([n]) for n in range(self.D)]).T # transpose needed for array broadcasting to work in batch mode
            self.assertAllAlmostEqual(ref_probs, state_probs, delta=self.tol)

class FockBasisTests(FockBaseTest):
    """Tests for simulators that use Fock basis."""
    num_subsystems = 1

    def test_normalized_prepare_vac(self):
        """Tests the ability to prepare vacuum states."""
        self.logTestName()
        self.circuit.prepare_vacuum_state(0)
        state = self.circuit.state()
        tr = state.trace()
        self.assertAllAlmostEqual(tr, 1, delta=self.tol)

    def test_normalized_coherent_state(self):
        """Tests if a range of coherent states are normalized."""
        self.logTestName()
        for mag_alpha in mag_alphas:
            for phase_alpha in phase_alphas:
                alpha = mag_alpha * np.exp(1j * phase_alpha)
                self.circuit.reset(pure=self.kwargs['pure'])
                self.circuit.prepare_coherent_state(alpha, 0)
                state = self.circuit.state()
                tr = state.trace()
                self.assertAllAlmostEqual(tr, 1, delta=self.tol)
                if alpha == 0.: break

    def test_prepare_ket_state(self):
        """Tests if a ket state with arbitrary parameters is correctly prepared."""
        self.logTestName()
        for _ in range(10):
            random_ket = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
            random_ket = random_ket / np.linalg.norm(random_ket)
            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.prepare_ket_state(random_ket, 0)
            state = self.circuit.state()
            self.assertAllAlmostEqual(state.fidelity(random_ket, 0), 1, delta=self.tol)

    def test_prepare_batched_ket_state(self):
        """Tests if a batch of ket states with arbitrary parameters is correctly
        prepared by comparing the fock probabilities of the batched case with
        individual runs with non batched input states."""
        self.logTestName()

        if not self.args.batched:
            return

        random_kets = np.array([(lambda ket: ket / np.linalg.norm(ket))(np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)) for _ in range(self.bsize)])

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_ket_state(random_kets, 0)
        state = self.circuit.state()
        batched_probs = np.array(state.all_fock_probs())

        individual_probs = []
        for random_ket in random_kets:
            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.prepare_ket_state(random_ket, 0)
            state = self.circuit.state()
            probs_for_this_ket = np.array(state.all_fock_probs())
            individual_probs.append(probs_for_this_ket[0])

        individual_probs = np.array(individual_probs)

        self.assertAllAlmostEqual(batched_probs, individual_probs, delta=self.tol)

    def test_prepare_rank_two_dm_state(self):
        """Tests if rank two dm states with arbitrary parameters are correctly prepared."""
        self.logTestName()
        for _ in range(10):
            random_ket1 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
            random_ket1 = random_ket1 / np.linalg.norm(random_ket1)
            random_ket2 = np.random.uniform(-1, 1, self.D) + 1j*np.random.uniform(-1, 1, self.D)
            random_ket2 = random_ket2 / np.linalg.norm(random_ket2)

            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.prepare_ket_state(random_ket1, 0)
            state = self.circuit.state()
            ket_probs1 = np.array([state.fock_prob([n]) for n in range(self.D)])

            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.prepare_ket_state(random_ket2, 0)
            state = self.circuit.state()
            ket_probs2 = np.array([state.fock_prob([n]) for n in range(self.D)])

            ket_probs = 0.2*ket_probs1 + 0.8*ket_probs2

            random_rho = 0.2*np.outer(np.conj(random_ket1), random_ket1) + 0.8*np.outer(np.conj(random_ket2), random_ket2)

            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.prepare_dm_state(random_rho, 0)
            state = self.circuit.state()
            rho_probs = np.array([state.fock_prob([n]) for n in range(self.D)])

            self.assertAllAlmostEqual(state.trace(), 1, delta=self.tol)
            self.assertAllAlmostEqual(rho_probs, ket_probs, delta=self.tol)

    def test_prepare_random_dm_state(self):
        """Tests if a random dm state is correctly prepared."""
        self.logTestName()
        random_rho = np.random.normal(size=[self.D, self.D]) + 1j*np.random.normal(size=[self.D, self.D])
        random_rho = np.dot(random_rho.conj().T, random_rho)
        random_rho = random_rho/random_rho.trace()

        self.circuit.reset(pure=self.kwargs['pure'])
        self.circuit.prepare_dm_state(random_rho, 0)
        state = self.circuit.state()
        rho_probs = np.array(state.all_fock_probs())

        es, vs = np.linalg.eig(random_rho)
        if self.args.batched:
            kets_mixed_probs = np.zeros([self.bsize, len(es)], dtype=complex)
        else:
            kets_mixed_probs = np.zeros([len(es)], dtype=complex)
        for e, v in zip(es, vs.T.conj()):
            self.circuit.reset(pure=self.kwargs['pure'])
            self.circuit.prepare_ket_state(v, 0)
            state = self.circuit.state()
            probs_for_this_v = np.array(state.all_fock_probs())
            kets_mixed_probs += e*probs_for_this_v

        self.assertAllAlmostEqual(rho_probs, kets_mixed_probs, delta=self.tol)


if __name__=="__main__":
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTests, FockBasisTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

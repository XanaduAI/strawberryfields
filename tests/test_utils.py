"""
Unit tests for utilities in :class:`strawberryfields.utils`.
"""

import unittest

import numpy as np
from numpy import pi

import defaults
from defaults import BaseTest, FockBaseTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import *
from strawberryfields.utils import _interleaved_identities, _vectorize_dm, _unvectorize_dm
from strawberryfields.backends.shared_ops import sympmat

from strawberryfields.backends.fockbackend.ops import squeezing as sq_U
from strawberryfields.backends.fockbackend.ops import displacement as disp_U
from strawberryfields.backends.fockbackend.ops import beamsplitter as bs_U


a = 0.32+0.1j
r = 0.112
phi = 0.123
n = 2


class InitialStates(BaseTest):
    num_subsystems = 1

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.D)

    def test_vacuum_state(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Vac | q[0]

        state = self.eng.run()

        if self.backend_name == 'gaussian':
            psi = state.reduced_gaussian(0)
            psi_exact = vacuum_state(basis='gaussian', hbar=self.hbar)
        else:
            psi = state.ket()
            psi_exact = vacuum_state(basis='fock', fock_dim=self.D, hbar=self.hbar)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)

    def test_coherent_state(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Dgate(a) | q[0]

        state = self.eng.run()

        if self.backend_name == 'gaussian':
            psi = state.reduced_gaussian(0)
            psi_exact = coherent_state(a, basis='gaussian', hbar=self.hbar)
        else:
            psi = state.ket()
            psi_exact = coherent_state(a, basis='fock', fock_dim=self.D, hbar=self.hbar)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)

    def test_squeezed_state(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Sgate(r, phi) | q[0]

        state = self.eng.run()

        if self.backend_name == 'gaussian':
            psi = state.reduced_gaussian(0)
            psi_exact = squeezed_state(r, phi, basis='gaussian', hbar=self.hbar)
        else:
            psi = state.ket()
            psi_exact = squeezed_state(r, phi, basis='fock', fock_dim=self.D, hbar=self.hbar)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)

    def test_displaced_squeezed_state(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Sgate(r, phi) | q[0]
            Dgate(a) | q[0]

        state = self.eng.run()

        if self.backend_name == 'gaussian':
            psi = state.reduced_gaussian(0)
            psi_exact = displaced_squeezed_state(a, r, phi, basis='gaussian', hbar=self.hbar)
        else:
            psi = state.ket()
            psi_exact = displaced_squeezed_state(a, r, phi, basis='fock', fock_dim=self.D, hbar=self.hbar)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)


class FockInitialStates(FockBaseTest):
    num_subsystems = 1

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend
        self.backend.reset(cutoff_dim=self.D)

    def test_fock_state(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Fock(n) | q[0]

        state = self.eng.run()

        psi = state.ket().real
        psi_exact = fock_state(n, fock_dim=self.D)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)

    def test_cat_state(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Catstate(a, 0) | q[0]

        state = self.eng.run()

        psi = state.ket()
        psi_exact = cat_state(a, 0, fock_dim=self.D)

        self.assertAllAlmostEqual(psi, psi_exact, delta=self.tol)


class ConvertFunctions(BaseTest):
    num_subsystems = 1

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)

    def test_neg(self):
        self.logTestName()
        a = np.random.random()
        q = self.eng.register
        rrt = neg(q[0])

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(a), -a)

    def test_mag(self):
        self.logTestName()
        x = np.random.random()
        y = np.random.random()
        q = self.eng.register
        rrt = mag(q[0])

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x+1j*y), np.abs(x+1j*y))

    def test_phase(self):
        self.logTestName()
        x = np.random.random()
        y = np.random.random()
        q = self.eng.register
        rrt = phase(q[0])

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x+1j*y), np.angle(x+1j*y))

    def test_scale(self):
        self.logTestName()
        a = np.random.random()
        x = np.random.random()
        q = self.eng.register
        rrt = scale(q[0], a)

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x), a*x)

    def test_shift(self):
        self.logTestName()
        a = np.random.random()
        x = np.random.random()
        q = self.eng.register
        rrt = shift(q[0], a)

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x), a+x)

    def test_scale_shift(self):
        self.logTestName()
        a = np.random.random()
        b = np.random.random()
        x = np.random.random()
        q = self.eng.register
        rrt = scale_shift(q[0], a, b)

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x), a*x+b)

    def test_power_positive_frac(self):
        self.logTestName()
        a = np.random.random()
        x = np.random.random()
        q = self.eng.register
        rrt = power(q[0], a)

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x), x**a)

    def test_power_negative_int(self):
        self.logTestName()
        a = -3
        x = np.random.random()
        q = self.eng.register
        rrt = power(q[0], a)

        self.assertEqual(type(rrt), sf.engine.RegRefTransform)
        self.assertEqual(rrt.func(x), x**a)


class RandomStates(BaseTest):
    num_subsystems = 1
    M = 3
    Om = sympmat(3)

    def test_random_covariance_pure(self):
        self.logTestName()
        V = random_covariance(self.M, hbar=self.hbar, pure=True)
        det = np.linalg.det(V) - (self.hbar/2)**(2*self.M)
        eigs = np.linalg.eigvalsh(V)

        # check square and even number of rows/columns
        self.assertAllEqual(V.shape, np.array([2*self.M]*2))
        # check symmetric
        self.assertAllAlmostEqual(V, V.T, delta=self.tol)
        # check positive definite
        self.assertTrue(np.all(eigs > 0))
        # check pure
        self.assertAlmostEqual(det, 0, delta=self.tol)

    def test_random_covariance_mixed(self):
        self.logTestName()
        V = random_covariance(self.M, hbar=self.hbar, pure=False)
        det = np.linalg.det(V) - (self.hbar/2)**(2*self.M)
        eigs = np.linalg.eigvalsh(V)

        # check square and even number of rows/columns
        self.assertAllEqual(V.shape, np.array([2*self.M]*2))
        # check symmetric
        self.assertAllAlmostEqual(V, V.T, delta=self.tol)
        # check positive definite
        self.assertTrue(np.all(eigs > 0))
        # check not pure
        self.assertNotAlmostEqual(det, 0, delta=self.tol)

    def test_random_symplectic_passive(self):
        self.logTestName()
        S = random_symplectic(self.M, passive=True)
        # check square and even number of rows/columns
        self.assertAllEqual(S.shape, np.array([2*self.M]*2))
        # check symplectic
        self.assertAllAlmostEqual(S @ self.Om @ S.T, self.Om, delta=self.tol)
        # check orthogonal
        self.assertAllAlmostEqual(S @ S.T, np.identity(2*self.M), delta=self.tol)

    def test_random_symplectic_active(self):
        self.logTestName()
        S = random_symplectic(self.M, passive=False)
        # check square and even number of rows/columns
        self.assertAllEqual(S.shape, np.array([2*self.M]*2))
        # check symplectic
        self.assertAllAlmostEqual(S @ self.Om @ S.T, self.Om, delta=self.tol)
        # check not orthogonal
        self.assertTrue(np.all(S @ S.T != np.identity(2*self.M)))

    def test_random_interferometer(self):
        self.logTestName()
        U = random_interferometer(self.M)
        # check unitary
        self.assertAllAlmostEqual(U @ U.conj().T, np.identity(self.M), delta=self.tol)


class ExtractOneMode(FockBaseTest):
    """Test the channel and unitary extraction functions on a single mode circuit"""
    num_subsystems = 1

    def setUp(self):
        super().setUp()
        self.eng_sf, self.q_sf = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng_sf.backend = self.backend
        self.backend.reset(cutoff_dim=self.D)

    def test_extract_kerr(self):
        """test that kerr gate is correctly extracted"""
        self.logTestName()

        kappa = 0.432

        with self.eng_sf:
            Kgate(kappa) | self.q_sf

        U = extract_unitary(self.eng_sf, cutoff_dim=self.D)
        expected = np.diag(np.exp(1j*kappa*np.arange(self.D)**2))

        self.assertAllAlmostEqual(U, expected, delta=self.tol)

    def test_extract_squeezing(self):
        """test that squeezing gate is correctly extracted"""
        self.logTestName()

        r = 0.432
        phi = -0.96543

        with self.eng_sf:
            Sgate(r, phi) | self.q_sf

        U = extract_unitary(self.eng_sf, cutoff_dim=self.D)
        expected = sq_U(r, phi, self.D)

        self.assertAllAlmostEqual(U, expected, delta=self.tol)

    def test_extract_displacement(self):
        """test that displacement gate is correctly extracted"""
        self.logTestName()

        alpha = 0.432-0.8543j

        with self.eng_sf:
            Dgate(alpha) | self.q_sf

        U = extract_unitary(self.eng_sf, cutoff_dim=self.D)
        expected = disp_U(alpha, self.D)

        self.assertAllAlmostEqual(U, expected, delta=self.tol)

    def test_extract_unitary_1_mode_fock(self):
        """Test that unitary extraction works for 1 mode using the fock backend"""
        self.logTestName()
        eng_extract, q_extract = sf.Engine(num_subsystems=1)

        S = sf.ops.Sgate(0.4, -1.2)
        D = sf.ops.Dgate(2, 0.9)
        K = sf.ops.Kgate(-1.5)

        initial_state = np.random.rand(self.D) + 1j*np.random.rand(self.D) # not a state but it doesn't matter

        with self.eng_sf:
            sf.ops.Ket(initial_state) | self.q_sf
            S | self.q_sf
            D | self.q_sf
            K | self.q_sf

        with eng_extract:
            S | q_extract
            D | q_extract
            K | q_extract

        matrix_extract = extract_unitary(eng_extract, cutoff_dim=self.D)
        final_state_extract = matrix_extract.dot(initial_state)
        final_state_sf = self.eng_sf.run().ket()

        self.assertAllAlmostEqual(final_state_extract, final_state_sf, delta=self.tol)

    def test_extract_unitary_1_mode_tf(self):
        """Test that unitary extraction works for 1 mode using the tf backend"""
        self.logTestName()
        eng_extract, q_extract = sf.Engine(num_subsystems=1)

        S = sf.ops.Sgate(0.4, -1.2)
        D = sf.ops.Dgate(2, 0.9)
        K = sf.ops.Kgate(-1.5)

        initial_state = np.random.rand(self.D) + 1j*np.random.rand(self.D) # not a state but it doesn't matter

        with self.eng_sf:
            sf.ops.Ket(initial_state) | self.q_sf
            S | self.q_sf
            D | self.q_sf
            K | self.q_sf

        with eng_extract:
            S | q_extract
            D | q_extract
            K | q_extract

        matrix_extract_tf_backend = extract_unitary(eng_extract, cutoff_dim=self.D, backend='tf')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            in_state = tf.constant(initial_state.reshape([-1]), dtype=tf.complex64)
            final_state_tf_backend = sess.run(tf.einsum('ab,b', matrix_extract_tf_backend, in_state))

        final_state_sf = self.eng_sf.run().ket()

        self.assertAllAlmostEqual(final_state_tf_backend, final_state_sf, delta=self.tol)

    def test_extract_channel_1_mode(self):
        """Test that channel extraction works for 1 mode"""
        self.logTestName()
        eng_extract, q_extract = sf.Engine(num_subsystems=1)

        S = sf.ops.Sgate(1.1, -1.4)
        L = sf.ops.LossChannel(0.45)

        initial_state = np.random.rand(self.D, self.D)

        with self.eng_sf:
            sf.ops.DensityMatrix(initial_state) | self.q_sf
            S | self.q_sf
            L | self.q_sf

        with eng_extract:
            S | q_extract
            L | q_extract

        choi = extract_channel(eng_extract, cutoff_dim=self.D, vectorize_modes=True, representation='choi')
        liouville = extract_channel(eng_extract, cutoff_dim=self.D, vectorize_modes=True, representation='liouville')
        kraus = extract_channel(eng_extract, cutoff_dim=self.D, vectorize_modes=True, representation='kraus')

        final_rho_sf = self.eng_sf.run().dm()

        final_rho_choi = np.einsum('abcd,ab -> cd', choi, initial_state)
        final_rho_liouville = np.einsum('abcd,db -> ca', liouville, initial_state)
        final_rho_kraus = np.einsum('abc,cd,aed -> be', kraus, initial_state, np.conj(kraus))

        self.assertAllAlmostEqual(final_rho_choi, final_rho_sf, delta=self.tol)
        self.assertAllAlmostEqual(final_rho_liouville, final_rho_sf, delta=self.tol)
        self.assertAllAlmostEqual(final_rho_kraus, final_rho_sf, delta=self.tol)

    def test_vectorize_unvectorize(self):
        """Test vectorize utility function"""
        self.logTestName()

        cutoff = 4

        dm = np.random.rand(*[cutoff]*8) + 1j*np.random.rand(*[cutoff]*8) # 4^8 -> (4^2)^4 -> 4^8
        dm2 = np.random.rand(*[cutoff]*4) + 1j*np.random.rand(*[cutoff]*4) # (2^2)^4 -> 2^8 -> (2^2)^4
        self.assertAllAlmostEqual(dm, _unvectorize_dm(_vectorize_dm(dm), 2), delta=self.tol)
        self.assertAllAlmostEqual(dm2, _vectorize_dm(_unvectorize_dm(dm2, 2)), delta=self.tol)

    def test_interleaved_identities(self):
        """Test interleaved utility function"""
        self.logTestName()

        II = _interleaved_identities(num_subsystems=2, cutoff_dim=3)
        self.assertAllAlmostEqual(np.einsum('abab', II), 3**2, delta=self.tol)

        III = _interleaved_identities(num_subsystems=3, cutoff_dim=5)
        self.assertAllAlmostEqual(np.einsum('abcabc', III), 5**3, delta=self.tol)


class ExtractTwoModes(FockBaseTest):
    """Test the channel and unitary extraction functions on a two mode circuit"""
    num_subsystems = 2

    def setUp(self):
        super().setUp()
        self.eng_sf, self.q_sf = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng_sf.backend = self.backend
        self.backend.reset(cutoff_dim=self.D)

    def test_extract_beamsplitter(self):
        """test that BSgate is correctly extracted"""
        self.logTestName()

        theta = 0.432
        phi = 0.765

        with self.eng_sf:
            BSgate(theta, phi) | self.q_sf

        U = extract_unitary(self.eng_sf, cutoff_dim=self.D)
        expected = bs_U(np.cos(theta), np.sin(theta), phi, self.D)

        self.assertAllAlmostEqual(U, expected, delta=self.tol)

    def test_extract_unitary_2_modes_fock(self):
        """Test that unitary extraction works for 2 modes using the fock backend"""
        self.logTestName()

        eng_extract, q_extract = sf.Engine(num_subsystems=2)

        S = sf.ops.Sgate(2)
        B = sf.ops.BSgate(2.234, -1.165)

        initial_state = np.complex64(np.random.rand(self.D, self.D) + 1j*np.random.rand(self.D, self.D))

        with self.eng_sf:
            sf.ops.Ket(initial_state) | (self.q_sf[0], self.q_sf[1])
            S | self.q_sf[0]
            B | self.q_sf
            S | self.q_sf[1]
            B | self.q_sf

        with eng_extract:
            S | q_extract[0]
            B | q_extract
            S | q_extract[1]
            B | q_extract

        # calculate final states for vectorize_mode= True
        matrix_extract = extract_unitary(eng_extract, cutoff_dim=self.D, vectorize_modes=True)
        final_state_extract = matrix_extract.dot(initial_state.reshape([-1]))
        final_state_sf = self.eng_sf.run().ket().reshape([-1])

        # calculate final states for vectorize_mode=False
        array_extract = extract_unitary(eng_extract, cutoff_dim=self.D, vectorize_modes=False)
        final_array_extract = np.einsum("abcd,bd->ac", array_extract, initial_state)
        final_array_sf = self.eng_sf.run().ket()

        self.assertAllAlmostEqual(final_state_extract, final_state_sf, delta=self.tol)
        self.assertAllAlmostEqual(final_array_extract, final_array_sf, delta=self.tol)

    def test_extract_unitary_2_modes_tf(self):
        """Test that unitary extraction works for 2 modes using the tf backend"""
        self.logTestName()

        eng_extract, q_extract = sf.Engine(num_subsystems=2)

        S = sf.ops.Sgate(2)
        B = sf.ops.BSgate(2.234, -1.165)

        initial_state = np.complex64(np.random.rand(self.D, self.D) + 1j*np.random.rand(self.D, self.D))

        with self.eng_sf:
            sf.ops.Ket(initial_state) | (self.q_sf[0], self.q_sf[1])
            S | self.q_sf[0]
            B | self.q_sf
            S | self.q_sf[1]
            B | self.q_sf

        with eng_extract:
            S | q_extract[0]
            B | q_extract
            S | q_extract[1]
            B | q_extract

        # calculate final states for vectorize_mode= True
        matrix_extract_tf_backend = extract_unitary(eng_extract, cutoff_dim=self.D, vectorize_modes=True, backend='tf')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            final_state_tf_backend = sess.run(tf.einsum('ab,b', matrix_extract_tf_backend, tf.constant(initial_state.reshape([-1]))))

        final_state_sf = self.eng_sf.run().ket().reshape([-1])

        self.assertAllAlmostEqual(final_state_tf_backend, final_state_sf, delta=self.tol)

    def test_extract_channel_2_modes(self):
        """Test that channel extraction works for 2 modes"""
        self.logTestName()

        eng_extract, q_extract = sf.Engine(num_subsystems=2)

        S = sf.ops.Sgate(2)
        B = sf.ops.BSgate(2.234, -1.165)

        initial_state = np.random.rand(self.D, self.D, self.D, self.D) + 1j*np.random.rand(self.D, self.D, self.D, self.D)

        with self.eng_sf:
            sf.ops.DensityMatrix(initial_state) | (self.q_sf[0], self.q_sf[1])
            S | self.q_sf[0]
            B | self.q_sf
            S | self.q_sf[1]
            B | self.q_sf

        with eng_extract:
            S | q_extract[0]
            B | q_extract
            S | q_extract[1]
            B | q_extract

        #vectorize = false
        choi = extract_channel(eng_extract, cutoff_dim=self.D, vectorize_modes=False, representation='choi')
        liouville = extract_channel(eng_extract, cutoff_dim=self.D, vectorize_modes=False, representation='liouville')
        kraus = extract_channel(eng_extract, cutoff_dim=self.D, vectorize_modes=False, representation='kraus')

        final_rho_sf = self.eng_sf.run().dm()

        final_rho_choi = np.einsum('abcdefgh,abcd -> efgh', choi, initial_state)
        final_rho_liouville = np.einsum('abcdefgh,fbhd -> eagc', liouville, initial_state)
        final_rho_kraus = np.einsum('abcde,cfeg,ahfig -> bhdi', kraus, initial_state, np.conj(kraus))


        self.assertAllAlmostEqual(final_rho_choi, final_rho_sf, delta=self.tol)
        self.assertAllAlmostEqual(final_rho_liouville, final_rho_sf, delta=self.tol)
        self.assertAllAlmostEqual(final_rho_kraus, final_rho_sf, delta=self.tol)

    def test_extract_channel_2_modes_vectorize(self):
        """Test that channel extraction works for 2 modes vectorized"""
        self.logTestName()

        eng_extract, q_extract = sf.Engine(num_subsystems=2)

        S = sf.ops.Sgate(2)
        B = sf.ops.BSgate(2.234, -1.165)

        initial_state = np.random.rand(self.D, self.D, self.D, self.D) + 1j*np.random.rand(self.D, self.D, self.D, self.D)

        with self.eng_sf:
            sf.ops.DensityMatrix(initial_state) | (self.q_sf[0], self.q_sf[1])
            S | self.q_sf[0]
            B | self.q_sf
            S | self.q_sf[1]
            B | self.q_sf

        with eng_extract:
            S | q_extract[0]
            B | q_extract
            S | q_extract[1]
            B | q_extract

        #vectorize = true

        choi = extract_channel(eng_extract, cutoff_dim=self.D, vectorize_modes=True, representation='choi')
        liouville = extract_channel(eng_extract, cutoff_dim=self.D, vectorize_modes=True, representation='liouville')
        kraus = extract_channel(eng_extract, cutoff_dim=self.D, vectorize_modes=True, representation='kraus')

        final_rho_sf = self.eng_sf.run().dm()
        final_rho_sf = np.einsum('abcd->acbd', final_rho_sf).reshape(self.D**2, self.D**2)
        initial_state = np.einsum('abcd->acbd', initial_state).reshape(self.D**2, self.D**2)

        final_rho_choi = np.einsum('abcd,ab -> cd', choi, initial_state)
        final_rho_liouville = np.einsum('abcd,db -> ca', liouville, initial_state)
        final_rho_kraus = np.einsum('abc,cd,aed -> be', kraus, initial_state, np.conj(kraus))

        self.assertAllAlmostEqual(final_rho_choi, final_rho_sf, delta=self.tol)
        self.assertAllAlmostEqual(final_rho_liouville, final_rho_sf, delta=self.tol)
        self.assertAllAlmostEqual(final_rho_kraus, final_rho_sf, delta=self.tol)


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', Utils.')

    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [ExtractOneMode, ExtractTwoModes, InitialStates, FockInitialStates, ConvertFunctions, RandomStates]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

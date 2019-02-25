"""
Unit tests for utilities in :class:`strawberryfields.utils`.
"""

import unittest

import numpy as np
from numpy import pi

import defaults
from defaults import BaseTest, FockBaseTest, ExtractChannelTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import *
from strawberryfields.backends.shared_ops import sympmat


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




class ExtractChannel(ExtractChannelTest):
    num_subsystems = 3

    def setUp(self):
        super().setUp()

    def test_extract_unitary_1_mode(self):
        num_subsystems = 1

        eng_sf, q_sf = sf.Engine(num_subsystems=num_subsystems)
        eng_extract, q_extract = sf.Engine(num_subsystems=num_subsystems)

        cutoff_dim = 10

        S = sf.ops.Sgate(0.4,-1.2)
        D = sf.ops.Dgate(2, 0.9)
        K = sf.ops.Kgate(-1.5)

        initial_state = np.complex64(np.random.rand(cutoff_dim) + 1j*np.random.rand(cutoff_dim)) # not a state but it doesn't matter

        with eng_sf:
            sf.ops.Ket(initial_state) | q_sf
            S | q_sf
            D | q_sf
            K | q_sf

        with eng_extract:
            S | q_extract
            D | q_extract
            K | q_extract

        matrix_extract = extract_unitary(eng_extract, cutoff_dim=cutoff_dim)
        matrix_extract_tf_backend = extract_unitary(eng_extract, cutoff_dim=cutoff_dim, backend='tf')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            final_state_tf_backend = sess.run(tf.einsum('ab,b', matrix_extract_tf_backend, tf.constant(initial_state.reshape([-1]))))
        final_state_sf = eng_sf.run("fock", cutoff_dim=cutoff_dim).ket().reshape([-1])
        final_state_extract = matrix_extract.dot(initial_state)
        final_state_sf = eng_sf.run("fock", cutoff_dim=cutoff_dim).ket()

        self.assertAllAlmostEqual(final_state_extract, final_state_sf, delta=defaults.TOLERANCE)
        self.assertAllAlmostEqual(final_state_tf_backend, final_state_sf, delta=defaults.TOLERANCE)

    def test_extract_unitary_2_modes(self):
        num_subsystems = 2

        eng_sf, q_sf = sf.Engine(num_subsystems=num_subsystems)
        eng_extract, q_extract = sf.Engine(num_subsystems=num_subsystems)

        cutoff_dim = 4

        S = sf.ops.Sgate(2)
        B = sf.ops.BSgate(2.234, -1.165)

        initial_state = np.complex64(np.random.rand(cutoff_dim, cutoff_dim) + 1j*np.random.rand(cutoff_dim, cutoff_dim))

        with eng_sf:
            sf.ops.Ket(initial_state) | (q_sf[0], q_sf[1])
            S | q_sf[0]
            B | q_sf
            S | q_sf[1]
            B | q_sf

        with eng_extract:
            S | q_extract[0]
            B | q_extract
            S | q_extract[1]
            B | q_extract

        # calculate final states for vectorize_mode= True
        matrix_extract = extract_unitary(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=True)
        matrix_extract_tf_backend = extract_unitary(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=True, backend='tf')
        final_state_extract = matrix_extract.dot(initial_state.reshape([-1]))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            final_state_tf_backend = sess.run(tf.einsum('ab,b', matrix_extract_tf_backend, tf.constant(initial_state.reshape([-1]))))
        final_state_sf = eng_sf.run("fock", cutoff_dim=cutoff_dim).ket().reshape([-1])

        # calculate final states for vectorize_mode= False
        array_extract = extract_unitary(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=False)
        final_array_extract = np.einsum("abcd,bd->ac", array_extract, initial_state)
        final_array_sf = eng_sf.run("fock", cutoff_dim=cutoff_dim).ket()

        self.assertAllAlmostEqual(final_state_extract, final_state_sf, delta = defaults.TOLERANCE)
        self.assertAllAlmostEqual(final_state_tf_backend, final_state_sf, delta = defaults.TOLERANCE)
        self.assertAllAlmostEqual(final_array_extract, final_array_sf, delta = defaults.TOLERANCE)

    def test_extract_channel_1_mode(self):
        num_subsystems = 1
        eng_sf, q_sf = sf.Engine(num_subsystems=num_subsystems)
        eng_extract, q_extract = sf.Engine(num_subsystems=num_subsystems)

        cutoff_dim = 3

        S = sf.ops.Sgate(1.1, -1.4)
        L = sf.ops.LossChannel(0.45)

        initial_state = np.random.rand(cutoff_dim, cutoff_dim)

        with eng_sf:
            sf.ops.DensityMatrix(initial_state) | q_sf
            S | q_sf
            L | q_sf

        with eng_extract:
            S | q_extract
            L | q_extract

        choi = extract_channel(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=True, representation='choi')
        liouville = extract_channel(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=True, representation='liouville')
        kraus = extract_channel(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=True, representation='kraus')

        final_rho_sf = eng_sf.run("fock", cutoff_dim=cutoff_dim).dm()
        
        final_rho_choi = np.einsum('abcd,ab -> cd', choi, initial_state)
        final_rho_liouville = np.einsum('abcd,db -> ca', liouville, initial_state)
        final_rho_kraus = np.einsum('abc,cd,aed -> be', kraus, initial_state, np.conj(kraus))

        self.assertAllAlmostEqual(final_rho_choi, final_rho_sf, delta=defaults.TOLERANCE)
        self.assertAllAlmostEqual(final_rho_liouville, final_rho_sf, delta=defaults.TOLERANCE)
        self.assertAllAlmostEqual(final_rho_kraus, final_rho_sf, delta=defaults.TOLERANCE)

    def test_extract_channel_2_modes(self):
        num_subsystems = 2

        eng_sf, q_sf = sf.Engine(num_subsystems=num_subsystems)
        eng_extract, q_extract = sf.Engine(num_subsystems=num_subsystems)

        cutoff_dim = 2

        S = sf.ops.Sgate(2) 
        B = sf.ops.BSgate(2.234, -1.165)

        initial_state = np.random.rand(cutoff_dim, cutoff_dim, cutoff_dim, cutoff_dim) + 1j*np.random.rand(cutoff_dim, cutoff_dim, cutoff_dim, cutoff_dim)


        with eng_sf:
            sf.ops.DensityMatrix(initial_state) | (q_sf[0], q_sf[1])
            S | q_sf[0]
            B | q_sf
            S | q_sf[1]
            B | q_sf

        with eng_extract:
            S | q_extract[0]
            B | q_extract
            S | q_extract[1]
            B | q_extract

        #vectorize = false
        choi = extract_channel(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=False, representation='choi')
        liouville = extract_channel(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=False, representation='liouville')
        kraus = extract_channel(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=False, representation='kraus')

        final_rho_sf = eng_sf.run("fock", cutoff_dim=cutoff_dim).dm()

        final_rho_choi = np.einsum('abcdefgh,abcd -> efgh', choi, initial_state)
        final_rho_liouville = np.einsum('abcdefgh,fbhd -> eagc', liouville, initial_state)
        final_rho_kraus = np.einsum('abcde,cfeg,ahfig -> bhdi', kraus, initial_state, np.conj(kraus))


        self.assertAllAlmostEqual(final_rho_choi, final_rho_sf, delta=defaults.TOLERANCE)
        self.assertAllAlmostEqual(final_rho_liouville, final_rho_sf, delta=defaults.TOLERANCE)
        self.assertAllAlmostEqual(final_rho_kraus, final_rho_sf, delta=defaults.TOLERANCE)


        #vectorize = true
        choi = extract_channel(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=True, representation='choi')
        liouville = extract_channel(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=True, representation='liouville')
        kraus = extract_channel(eng_extract, cutoff_dim=cutoff_dim, vectorize_modes=True, representation='kraus')

        final_rho_sf = np.einsum('abcd->acbd', final_rho_sf).reshape(cutoff_dim**2,cutoff_dim**2)
        initial_state = np.einsum('abcd->acbd', initial_state).reshape(cutoff_dim**2,cutoff_dim**2)

        final_rho_choi = np.einsum('abcd,ab -> cd', choi, initial_state)
        final_rho_liouville = np.einsum('abcd,db -> ca', liouville, initial_state)
        final_rho_kraus = np.einsum('abc,cd,aed -> be', kraus, initial_state, np.conj(kraus))

        self.assertAllAlmostEqual(final_rho_choi, final_rho_sf, delta=defaults.TOLERANCE)
        self.assertAllAlmostEqual(final_rho_liouville, final_rho_sf, delta=defaults.TOLERANCE)
        self.assertAllAlmostEqual(final_rho_kraus, final_rho_sf, delta=defaults.TOLERANCE)

        
    def test_vectorize_unvectorize(self):
        cutoff_dim = 4
        dm = np.random.rand(*[cutoff_dim]*8) + 1j*np.random.rand(*[cutoff_dim]*8) # 4^8 -> (4^2)^4 -> 4^8
        dm2 = np.random.rand(*[cutoff_dim]*4) + 1j*np.random.rand(*[cutoff_dim]*4) # (2^2)^4 -> 2^8 -> (2^2)^4
        self.assertAllAlmostEqual(dm, unvectorize_dm(vectorize_dm(dm), 2), delta=defaults.TOLERANCE)
        self.assertAllAlmostEqual(dm2, vectorize_dm(unvectorize_dm(dm2, 2)), delta=defaults.TOLERANCE)










if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', Utils.')

    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [ExtractChannel, InitialStates, FockInitialStates, ConvertFunctions, RandomStates]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

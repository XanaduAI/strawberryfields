"""
Unit tests for decompositions in :class:`strawberryfields.backends.decompositions`,
as well as the frontend decompositions in :class:`strawberryfields.ops`.
"""

import unittest

import numpy as np
from numpy import pi
from scipy.linalg import qr, block_diag

# NOTE: strawberryfields must be imported from defaults
from defaults import BaseTest, FockBaseTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import *
from strawberryfields import decompositions as dec
from strawberryfields.backends.gaussianbackend import gaussiancircuit
from strawberryfields.backends.shared_ops import haar_measure, changebasis, rotation_matrix as rot



nsamples=10


def random_degenerate_symmetric():
    iis=[1+np.random.randint(2),1+np.random.randint(3),1+np.random.randint(3),1]
    vv=[[i]*iis[i] for i in range(len(iis))]
    dd=np.array(sum(vv, []))
    n=len(dd)
    U=haar_measure(n)
    symmat=U @ np.diag(dd) @ np.transpose(U)
    return symmat


class DecompositionsModule(BaseTest):
    num_subsystems = 1

    def test_takagi_random_symm(self):
        self.logTestName()
        error=np.empty(nsamples)
        for i in range(nsamples):
            X=random_degenerate_symmetric()
            rl, U = dec.takagi(X)
            Xr= U @ np.diag(rl) @ np.transpose(U)
            diff= np.linalg.norm(Xr-X)

            error[i]=diff

        self.assertAlmostEqual(error.mean() , 0)

    def test_clements_identity(self):
        self.logTestName()
        n=20
        U=np.identity(n)
        (tilist,tlist, diags)=dec.clements(U)
        qrec=np.identity(n)
        for i in tilist:
            qrec=dec.T(*i)@qrec
        qrec=np.diag(diags) @ qrec
        for i in reversed(tlist):
            qrec=dec.Ti(*i) @qrec

        self.assertAllAlmostEqual(U, qrec, delta=self.tol)

    def test_clements_random_unitary(self):
        self.logTestName()
        error=np.empty(nsamples)
        for k in range(nsamples):
            n=20
            V=haar_measure(n)
            (tilist,tlist, diags)=dec.clements(V)
            qrec=np.identity(n)
            for i in tilist:
                qrec=dec.T(*i)@qrec
            qrec=np.diag(diags) @ qrec
            for i in reversed(tlist):
                qrec=dec.Ti(*i) @qrec

            error[k]=np.linalg.norm(V-qrec)
        self.assertAlmostEqual(error.mean() , 0)

    def test_williamson_BM_random_circuit(self):
        self.logTestName()
        for k in range(nsamples):
            n=3
            U1=haar_measure(n)
            U2=haar_measure(n)
            state=gaussiancircuit.GaussianModes(n,hbar=2)
            ns=[0.1*i+0.1 for i in range(n)]

            for i in range(n):
                state.init_thermal(ns[i],i)
            state.apply_u(U1)
            for i in range(n):
                state.squeeze(np.log(0.2*i+2),0,i)
            state.apply_u(U2)

            V=state.scovmatxp()

            D, S = dec.williamson(V)
            omega = dec.sympmat(n)

            self.assertAlmostEqual(np.linalg.norm(S @ omega @ S.T -omega),0)
            self.assertAlmostEqual(np.linalg.norm(S @ D @ S.T -V),0)

            O, s, Oo = dec.bloch_messiah(S)
            self.assertAlmostEqual(np.linalg.norm(np.transpose(O) @ omega @ O -omega),0)
            self.assertAlmostEqual(np.linalg.norm(np.transpose(O) @ O -np.identity(2*n)),0)

            self.assertAlmostEqual(np.linalg.norm(np.transpose(Oo) @ omega @ Oo -omega),0)
            self.assertAlmostEqual(np.linalg.norm(np.transpose(Oo) @ Oo -np.identity(2*n)),0)
            self.assertAlmostEqual(np.linalg.norm(np.transpose(s) @ omega @ s -omega),0)
            self.assertAlmostEqual(np.linalg.norm(O @ s @Oo -S),0)

    def test_williamson_BM_random_circuit_pure(self):
        self.logTestName()
        for k in range(nsamples):
            n=3
            U2=haar_measure(n)
            state=gaussiancircuit.GaussianModes(n,hbar=2)
            for i in range(n):
                state.squeeze(np.log(0.2*i+2),0,i)
            state.apply_u(U2)

            V=state.scovmatxp()

            D, S = dec.williamson(V)
            omega = dec.sympmat(n)

            self.assertAlmostEqual(np.linalg.norm(S @ omega @ S.T -omega),0)
            self.assertAlmostEqual(np.linalg.norm(S @ D @ S.T -V),0)



            O, s, Oo = dec.bloch_messiah(S)
            self.assertAlmostEqual(np.linalg.norm(np.transpose(O) @ omega @ O -omega),0)
            self.assertAlmostEqual(np.linalg.norm(np.transpose(O) @ O -np.identity(2*n)),0)

            self.assertAlmostEqual(np.linalg.norm(np.transpose(Oo) @ omega @ Oo -omega),0)
            self.assertAlmostEqual(np.linalg.norm(np.transpose(Oo) @ Oo -np.identity(2*n)),0)
            self.assertAlmostEqual(np.linalg.norm(np.transpose(s) @ omega @ s -omega),0)
            self.assertAlmostEqual(np.linalg.norm(O @ s @Oo -S),0)


class FrontendGaussianDecompositions(GaussianBaseTest):
    num_subsystems = 3

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend
        self.u1 = random_interferometer(self.num_subsystems)
        self.u2 = random_interferometer(self.num_subsystems)
        self.S = random_symplectic(self.num_subsystems)
        self.V_mixed = random_covariance(self.num_subsystems, hbar=self.hbar, pure=False)
        self.V_pure = random_covariance(self.num_subsystems, hbar=self.hbar, pure=True)

    def test_merge_interferometer(self):
        self.logTestName()
        I1 = Interferometer(self.u1)
        I1inv = Interferometer(self.u1.conj().T)
        I2 = Interferometer(self.u2)

        # unitary merged with its conjugate transpose is identity
        self.assertTrue(I1.merge(I1inv) is None)
        # two merged unitaries are the same as their product
        self.assertAllAlmostEqual(I1.merge(I2).p[0].x, self.u2@self.u1, delta=self.tol)

    def test_merge_covariance(self):
        self.logTestName()
        V1 = Gaussian(self.V_mixed, hbar=self.hbar)
        V2 = Gaussian(self.V_pure, hbar=self.hbar)

        # only the second applied covariance state is kept
        self.assertEqual(V1.merge(V2), V2)
        # the same is true of state operations
        self.assertEqual(Squeezed(2).merge(V2), V2)

    def test_covariance_random_state_mixed(self):
        self.logTestName()
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Gaussian(self.V_mixed) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), self.V_mixed, delta=self.tol)

    def test_covariance_random_state_pure(self):
        self.logTestName()
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Gaussian(self.V_pure) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), self.V_pure, delta=self.tol)

    def test_gaussian_transform(self):
        self.logTestName()
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            GaussianTransform(self.S) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), self.S@self.S.T*self.hbar/2, delta=self.tol)

    def test_passive_gaussian_transform(self):
        self.logTestName()
        self.eng.reset()
        q = self.eng.register
        O = np.vstack([np.hstack([self.u1.real, -self.u1.imag]),
                       np.hstack([self.u1.imag, self.u1.real])])

        with self.eng:
            All(Squeezed(0.5)) | q
            init = self.eng.run()
            GaussianTransform(O) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), O @ init.cov() @ O.T, delta=self.tol)

    def test_active_gaussian_transform_on_vacuum(self):
        self.logTestName()
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            GaussianTransform(self.S, vacuum=True) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), self.S@self.S.T*self.hbar/2, delta=self.tol)

    def test_interferometer(self):
        self.logTestName()
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            All(Squeezed(0.5)) | q
            init = self.eng.run()
            Interferometer(self.u1) | q

        state = self.eng.run()
        O = np.vstack([np.hstack([self.u1.real, -self.u1.imag]),
                       np.hstack([self.u1.imag, self.u1.real])])
        self.assertAllAlmostEqual(state.cov(), O @ init.cov() @ O.T, delta=self.tol)

    def test_identity_interferometer(self):
        self.logTestName()
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Interferometer(np.identity(6)) | q

        state = self.eng.run()
        self.assertEqual(len(self.eng.cmd_applied[0]), 0)


class FrontendGaussians(GaussianBaseTest):
    num_subsystems = 3

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend
        self.eng.reset()

    def test_covariance_vacuum(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Gaussian(np.identity(6)*self.hbar/2, decomp=False) | q

        state = self.eng.run()
        cov = state.cov()
        means = state.means()
        self.assertAllEqual(cov, np.identity(6))
        self.assertAllEqual(means, np.zeros([6]))

    def test_covariance_squeezed(self):
        self.logTestName()
        q = self.eng.register
        cov = (self.hbar/2)*np.diag([np.exp(-0.1)]*3 + [np.exp(0.1)]*3)

        with self.eng:
            Gaussian(cov, decomp=False) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)

    def test_covariance_displaced_squeezed(self):
        self.logTestName()
        q = self.eng.register
        cov = (self.hbar/2)*np.diag([np.exp(-0.1)]*3 + [np.exp(0.1)]*3)

        with self.eng:
            Gaussian(cov, r=[0, 0.1, 0.2, -0.1, 0.3, 0], decomp=False) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)

    def test_covariance_thermal(self):
        self.logTestName()
        q = self.eng.register
        cov = np.diag(self.hbar*(np.array([0.3,0.4,0.2]*2)+0.5))

        with self.eng:
            Gaussian(cov, decomp=False) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)

    def test_covariance_rotated_squeezed(self):
        self.logTestName()
        q = self.eng.register

        r = 0.1
        phi = 0.2312
        v1 = (self.hbar/2)*np.diag([np.exp(-r),np.exp(r)])
        A = changebasis(3)
        cov = A.T @ block_diag(*[rot(phi) @ v1 @ rot(phi).T]*3) @ A

        with self.eng:
            Gaussian(cov, decomp=False) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)

    def test_decomp_covariance_vacuum(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Gaussian(np.identity(6)*self.hbar/2) | q

        state = self.eng.run()
        cov = state.cov()
        means = state.means()
        self.assertAllEqual(cov, np.identity(6))
        self.assertAllEqual(means, np.zeros([6]))
        self.assertEqual(len(self.eng.cmd_applied[0]), 0)

    def test_decomp_covariance_squeezed(self):
        self.logTestName()
        q = self.eng.register
        cov = (self.hbar/2)*np.diag([np.exp(-0.1)]*3 + [np.exp(0.1)]*3)

        with self.eng:
            Gaussian(cov) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)
        self.assertAllEqual(len(self.eng.cmd_applied[0]), 3)

    def test_decomp_covariance_displaced_squeezed(self):
        self.logTestName()
        q = self.eng.register
        cov = (self.hbar/2)*np.diag([np.exp(-0.1)]*3 + [np.exp(0.1)]*3)

        with self.eng:
            Gaussian(cov, r=[0, 0.1, 0.2, -0.1, 0.3, 0]) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)
        self.assertAllEqual(len(self.eng.cmd_applied[0]), 7)

    def test_decomp_covariance_thermal(self):
        self.logTestName()
        q = self.eng.register
        cov = np.diag(self.hbar*(np.array([0.3,0.4,0.2]*2)+0.5))

        with self.eng:
            Gaussian(cov) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)
        self.assertAllEqual(len(self.eng.cmd_applied[0]), 3)

    def test_decomp_covariance_rotated_squeezed(self):
        self.logTestName()
        q = self.eng.register

        r = 0.1
        phi = 0.2312
        v1 = (self.hbar/2)*np.diag([np.exp(-r),np.exp(r)])
        A = changebasis(3)
        cov = A.T @ block_diag(*[rot(phi) @ v1 @ rot(phi).T]*3) @ A

        with self.eng:
            Gaussian(cov) | q

        state = self.eng.run()
        self.assertAllAlmostEqual(state.cov(), cov, delta=self.tol)
        self.assertAllEqual(len(self.eng.cmd_applied[0]), 3)


class FrontendFockGaussians(FockBaseTest):
    """Fidelity tests."""
    num_subsystems = 3

    def setUp(self):
        super().setUp()
        self.eng, q = sf.Engine(self.num_subsystems, hbar=self.hbar)
        self.eng.backend = self.backend
        self.eng.reset()

    def test_covariance_vacuum(self):
        self.logTestName()
        q = self.eng.register

        with self.eng:
            Gaussian(np.identity(6)*self.hbar/2) | q

        state = self.eng.run(**self.kwargs)
        self.assertEqual(len(self.eng.cmd_applied[0]), 0)
        self.assertAllAlmostEqual(state.fidelity_vacuum(), 1, delta=self.tol)

    def test_covariance_squeezed(self):
        self.logTestName()
        q = self.eng.register
        r = 0.05
        phi = 0
        cov = (self.hbar/2)*np.diag([np.exp(-2*r)]*3 + [np.exp(2*r)]*3)
        in_state = squeezed_state(r, phi, basis='fock', fock_dim=self.D)

        with self.eng:
            Gaussian(cov) | q

        state = self.eng.run(**self.kwargs)
        self.assertAllEqual(len(self.eng.cmd_applied[0]), 3)

        for n in range(3):
            self.assertAllAlmostEqual(state.fidelity(in_state, n), 1, delta=self.tol)

    def test_covariance_rotated_squeezed(self):
        self.logTestName()
        q = self.eng.register

        r = 0.1
        phi = 0.2312
        in_state = squeezed_state(r, phi, basis='fock', fock_dim=self.D)

        v1 = (self.hbar/2)*np.diag([np.exp(-2*r),np.exp(2*r)])
        A = changebasis(3)
        cov = A.T @ block_diag(*[rot(phi) @ v1 @ rot(phi).T]*3) @ A

        with self.eng:
            Gaussian(cov) | q

        state = self.eng.run(**self.kwargs)
        self.assertAllEqual(len(self.eng.cmd_applied[0]), 3)
        for n in range(3):
            self.assertAllAlmostEqual(state.fidelity(in_state, n), 1, delta=self.tol)


if __name__ == '__main__':
    print('Testing Strawberry Fields decompositions.py')

    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [
        DecompositionsModule,
        FrontendGaussianDecompositions,
        FrontendGaussianGaussians,
        FrontendFockGaussians
    ]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

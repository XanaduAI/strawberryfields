"""
Unit tests for decompositions in :class:`strawberryfields.backends.decompositions`.
"""

import os
import sys
import signal

import unittest

import numpy as np
from scipy.linalg import qr

import strawberryfields.decompositions as dec
from defaults import BaseTest

from strawberryfields.backends.shared_ops import haar_measure
from strawberryfields.backends.gaussianbackend import gaussiancircuit

nsamples=10


def random_degenerate_symmetric():
    iis=[1+np.random.randint(2),1+np.random.randint(3),1+np.random.randint(3),1]
    vv=[[i]*iis[i] for i in range(len(iis))]
    dd=np.array(sum(vv, []))
    n=len(dd)
    U=haar_measure(n)
    symmat=U @ np.diag(dd) @ np.transpose(U)
    return symmat


class TestTakagi(BaseTest):
    num_subsystems = 1
    def test_random_symm(self):
        error=np.empty(nsamples)
        for i in range(nsamples):
            X=random_degenerate_symmetric()
            rl, U = dec.takagi(X)
            Xr= U @ np.diag(rl) @ np.transpose(U)
            diff= np.linalg.norm(Xr-X)

            error[i]=diff

        self.assertAlmostEqual(error.mean() , 0)


class TestClements(BaseTest):
    num_subsystems = 1
    def test_identity(self):
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

    def test_random_unitary(self):
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


class TestWilliamsonAndBlochMessiah(BaseTest):
    num_subsystems = 1
    def test_random_circuit(self):
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


class TestWilliamsonAndBlochMessiahPure(BaseTest):
    num_subsystems = 1
    def test_random_circuit(self):
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


if __name__ == '__main__':
    print('Testing Strawberry Fields decompositions.py')

    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [
        TestTakagi,
        TestClements,
        TestWilliamsonAndBlochMessiah,
        TestWilliamsonAndBlochMessiahPure
    ]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

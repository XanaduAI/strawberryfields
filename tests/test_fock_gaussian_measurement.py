##############################################################################
#
# Unit tests for probabilities of Fock autocomes in the Gaussian backend
# This DOES NOT test for sampling of the said probabilities
#
##############################################################################

import unittest
from itertools import combinations
from math import factorial

import numpy as np

from defaults import BaseTest, FockBaseTest, strawberryfields as sf
from strawberryfields import backends


num_repeats = 50

###################################################################

def squeezed_matelem(r,n):
  if n%2==0:
    return (1/np.cosh(r))*(np.tanh(r)**(n))*factorial(n)/(2**(n/2)*factorial(n/2))**2
  else:
      return 0.0
  


class FockGaussianMeasurementTests(BaseTest):
  num_subsystems = 1

  def test_coherent_state(self):
    """Tests Fock probabilities on a coherent state."""
    self.logTestName()
    alpha=1
    nmax=self.D - 1
    self.circuit.prepare_coherent_state(alpha,0)
    state = self.circuit.state()
    if isinstance(self.backend, backends.BaseFock):
      if state.is_pure:
        bsd = state.ket()
      else:
        bsd = state.dm()
      if self.kwargs['pure']:
        if self.args.batched:
          bsd = np.array([np.outer(b, np.conj(b)) for b in bsd])
        else:
          bsd = np.outer(bsd, np.conj(bsd))
    else:
      bsd = state.reduced_dm(0,cutoff=nmax+1)
    bsd_diag = np.array([np.diag(b) for b in bsd]) if self.args.batched else np.diag(bsd)

    gbs = np.empty((self.bsize,self.D))
    for i in range(self.D):
      gbs[:,i] = state.fock_prob([i])

    alpha2=np.abs(alpha)**2
    exact=np.array([ np.exp(-alpha2)*(alpha2**i)/factorial(i) for i in range(self.D)])
    exact = np.tile(exact, self.bsize)

    self.assertAllAlmostEqual(gbs.flatten(),bsd_diag.real.flatten(),delta=self.tol)
    self.assertAllAlmostEqual(exact.flatten(),bsd_diag.real.flatten(),delta=self.tol)


  def test_squeezed_state(self):
    """Tests Fock probabilities on a squeezed state."""
    self.logTestName()
    r=1.0
    nmax = self.D - 1
    self.circuit.prepare_squeezed_state(1.0,0,0)
    state = self.circuit.state()
    if isinstance(self.backend, backends.BaseFock):
      if state.is_pure:
        bsd = state.ket()
      else:
        bsd = state.dm()
      if self.kwargs['pure']:
        if self.args.batched:
          bsd = np.array([np.outer(b, np.conj(b)) for b in bsd])
        else:
          bsd = np.outer(bsd, np.conj(bsd))
    else:
      bsd = state.reduced_dm(0,cutoff=nmax+1)
    bsd_diag = np.array([np.diag(b) for b in bsd]) if self.args.batched else np.diag(bsd)

    gbs = np.empty((self.bsize,self.D))
    for i in range(self.D):
      gbs[:,i] = state.fock_prob([i])

    exact = np.array([squeezed_matelem(r,n) for n in range(self.D)] )
    exact = np.tile(exact, self.bsize)

    self.assertAllAlmostEqual(gbs.flatten(),bsd_diag.real.flatten(),delta=self.tol)
    self.assertAllAlmostEqual(exact.flatten(),bsd_diag.real.flatten(),delta=self.tol)


  def test_displaced_squeezed_state(self):
    """Tests Fock probabilities on a state that is squeezed then displaced."""
    self.logTestName()
    r=1.0
    alpha=3+4*1j
    nmax = self.D - 1
    self.circuit.prepare_squeezed_state(1.0,0,0)
    self.circuit.displacement(alpha,0)
    state = self.circuit.state()
    if isinstance(self.backend, backends.BaseFock):
      if state.is_pure:
        bsd = state.ket()
      else:
        bsd = state.dm()
      if self.kwargs['pure']:
        if self.args.batched:
          bsd = np.array([np.outer(b, np.conj(b)) for b in bsd])
        else:
          bsd = np.outer(bsd, np.conj(bsd))
    else:
      bsd = state.reduced_dm(0,cutoff=nmax+1)
    bsd_diag = np.array([np.diag(b) for b in bsd]) if self.args.batched else np.diag(bsd)

    gbs = np.empty((self.bsize,self.D))
    for i in range(self.D):
      gbs[:,i] = state.fock_prob([i])

    self.assertAllAlmostEqual(gbs.flatten(),bsd_diag.real.flatten(),delta=self.tol)


class FockGaussianMeasurementTestsMulti(BaseTest):
  num_subsystems = 2
  def test_two_mode_squeezed(self):
    self.logTestName()
    r = np.arcsinh(np.sqrt(1))
    nmax = 3
    self.circuit.prepare_squeezed_state(r,0,0)
    self.circuit.prepare_squeezed_state(-r,0,1)
    self.circuit.beamsplitter(np.sqrt(0.5),np.sqrt(0.5),0,1)
    
    gbs = np.empty((self.bsize,nmax,nmax))
    state = self.circuit.state()
    for i in range(nmax):
      for j in range(nmax):
        gbs[:,i,j] = state.fock_prob(np.array([i,j]))

    n = np.arange(nmax)
    exact = np.diag((1/np.cosh(r)**2) * np.tanh(r)**(2*n))
    exact = np.tile(exact.flatten(), self.bsize)
    self.assertAllAlmostEqual(gbs.flatten(),exact.flatten(),delta=self.tol)



class FockGaussianMeasurementTestsMultiComplex(BaseTest):
  num_subsystems = 2
  def test_two_mode_squeezed(self):
    self.logTestName()
    r=np.arcsinh(np.sqrt(1))
    nmax=3
    self.circuit.prepare_squeezed_state(r,0,0)
    self.circuit.prepare_squeezed_state(r,0,1)
    self.circuit.beamsplitter(np.sqrt(0.5),1j*np.sqrt(0.5),0,1)
    
    gbs=np.empty((self.bsize,nmax,nmax))
    state = self.circuit.state()
    for i in range(nmax):
      for j in range(nmax):
        gbs[:,i,j] = state.fock_prob(np.array([i,j]))

    n = np.arange(nmax)
    exact = np.diag((1/np.cosh(r)**2) * np.tanh(r)**(2*n))
    exact = np.tile(exact.flatten(), self.bsize)
    self.assertAllAlmostEqual(gbs.flatten(),exact.flatten(),delta=self.tol)



if __name__=="__main__":
  # run the tests in this file
  suite = unittest.TestSuite()
  for t in (FockGaussianMeasurementTests, FockGaussianMeasurementTestsMulti, FockGaussianMeasurementTestsMultiComplex):
    ttt = unittest.TestLoader().loadTestsFromTestCase(t)
    suite.addTests(ttt)

  unittest.TextTestRunner().run(suite)



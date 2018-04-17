# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gaussian operations"""
from types import GeneratorType
from collections import OrderedDict
from itertools import tee

import numpy as np
from scipy.linalg import sqrtm
from scipy.special import binom, factorial


def fock_amplitudes_one_mode(alpha, mat, cutoff, tol=1e-8):
    """ Returns the Fock space density matrix of gaussian state characterized
    by a complex displacement alpha and a (symmetric) covariance matrix
    cutoff determines what is the maximum Fock state  and tol is a value used
    to calculate how many terms should be included in the calculation of matrix elements
    if the state is mixed"""
    if mat.shape != (2, 2):
        raise ValueError("Covariance matrix mat must be 2x2")

    (nth, theta, r) = bm_reduction(mat)

    if abs(nth) < tol:
        psi = np.conjugate(np.array([one_mode_matelem(-alpha, -r, -2*theta, m, 0) for m in range(cutoff+1)]))
        return np.outer(psi, np.conj(psi))

    rat = nth/(1+nth)
    ### The following heuristic determines how many terms to take based
    ### on the temperature of the thermal state and the tolerance prescribed by the user in tol
    mmax = int(-1+np.log(tol)/np.log(rat))+1
    rho = np.zeros((cutoff+1, cutoff+1), dtype=complex)
    ss = 1.0
    for n in range(mmax):
        psi = np.sqrt(ss)*np.conjugate(np.array([one_mode_matelem(-alpha, -r, -2*theta, m, n) for m in range(cutoff+1)]))
        rho += np.outer(psi, np.conj(psi))
        ss = ss*rat
    return rho/(1+nth)


def one_mode_matelem(beta, r, theta, m, n, tol=1.0e-8):
    """ Calculates the function f_{m, n}(r, theta, beta) as defined in the conventions file
    If abs(r)<tol then r is taken to be exactly zero and an optimized routine is used
    """
    # pylint: disable=too-many-arguments
    if np.abs(r) > tol:
        nu = np.exp(-1j*theta)*np.sinh(r)
        mu = np.cosh(r)
        alpha = beta*mu-np.conjugate(beta)*nu
        mini = min(n, m)
        hermiteni = hermite(beta/np.sqrt(2*mu*nu), n+1)
        hermitemi = hermite(-np.conjugate(alpha)/np.sqrt(-2*mu*np.conjugate(nu)), m+1)


        ssum = 0.0+0.0*1j
        for i in range(mini+1):
            prod = (binom(m, i)/factorial(n-i))*((2/(mu*nu))**(i/2))*((-np.conjugate(nu)/(2*mu))**((m-i)/2))*hermiteni[n-i]*hermitemi[m-i]
            ssum += prod

        matel = ssum*np.sqrt(factorial(n)/(factorial(m)*mu))*(nu/(2*mu))**(n/2)*np.exp(-0.5*(np.abs(beta)**2-np.conjugate(nu)*beta**2/mu))
        return matel

    alpha = beta
    mini = min(n, m)
    ssum = 0.0+0.0*1j
    for i in range(mini+1):
        ssum += ((-1)**(m-i))*(beta**(n-i))*(np.conjugate(beta)**(m-i))*binom(m, i)/factorial(n-i)

    matel = ssum*np.sqrt(factorial(n)/(factorial(m)))*np.exp(-0.5*np.abs(beta)**2)
    return matel


def hermite(x, n):
    """ Returns the hermite polynomials up to degree n-1 evaluated at x"""
    if n == 0:
        return np.array([1.0])

    if n == 1:
        return np.array([1.0, 2*x])

    p = np.empty(n, dtype=complex)
    p[0] = 1
    p[1] = 2*x

    for i in range(2, n):
        p[i] = 2*x*p[i-1]-2*(i-1)*p[i-2]

    return p


def bm_reduction(mat):
    """ Performs the Bloch-Messiah decomposition of single mode thermal state.
    Said decomposition writes a gaussian state as a  a thermal squeezed-rotated-displaced state
    The function returns the thermal population, rotation angle and squeezing parameters
    """

    if mat.shape != (2, 2):
        raise ValueError("Covariance matrix mat must be 2x2")

    detm = np.linalg.det(mat)
    nth = 0.5*(np.sqrt(detm)-1)
    mm = mat/np.sqrt(detm)
    a = mm[0, 0]
    b = mm[0, 1]
    r = -0.5*np.arccosh((1+a*a+b*b)/(2*a))
    theta = 0.5*np.arctan2((2*a*b), (-1+a*a-b*b))
    return nth, theta, r


def sm_fidelity(mu1, mu2, cov1, cov2, tol=1e-8):
    """ Calculates the squared fidelity between the gaussian states s1 and s2. It uses the formulas from
    Quantum Fidelity for Arbitrary Gaussian States
    Leonardo Banchi, Samuel L. Braunstein, and Stefano Pirandola
    Phys. Rev. Lett. 115, 260501 – Published 22 December 2015
    The function returns the square of the quantity defined in the reference cited above.
    Note however that our matrices need to be multiplied by 1/2 to get theirs and our vectors
    need to be divided by sqrt(1/2) equivalently the factor in the exponential is not multiplied
    by 2*1/4 but instead by 2*1/8=0.25
    """
    # pylint: disable=duplicate-code
    v1 = 0.5*cov1
    v2 = 0.5*cov2
    deltar = mu1-mu2
    n = 1
    W = omega(2*n)

    si12 = np.linalg.inv(v1+v2)
    vaux = np.dot(np.dot(np.transpose(W), si12), 0.25*W+np.dot(v2, np.dot(W, v1)))

    p1 = np.dot(vaux, W)
    p1 = np.dot(p1, p1)
    p1 = np.identity(2*n)+0.25*np.linalg.inv(p1)
    if np.linalg.norm(p1) < tol:
        p1 = np.zeros_like(p1)
    else:
        p1 = sqrtm(p1)
    p1 = 2*(p1+np.identity(2*n))
    p1 = np.dot(p1, vaux).real
    f = np.sqrt(np.linalg.det(si12)*np.linalg.det(p1))*np.exp(-0.25*np.dot(np.dot(deltar, si12), deltar).real)
    return f


def chop_in_blocks(m, idtodelete):
    """
    Splits a (symmetric) matrix into 3 blocks, A, B, C
    Blocks A and B are diagonal blocks and C is the offdiagonal block
    idtodelete specifies which indices go into B.
    """
    A = np.copy(m)
    A = np.delete(A, idtodelete, axis=0)
    A = np.delete(A, idtodelete, axis=1)
    B = np.delete(m[:, idtodelete], idtodelete, axis=0)
    C = np.empty((len(idtodelete), (len(idtodelete))))
    for localindex, globalindex in enumerate(idtodelete):
        for localindex1, globalindex1 in enumerate(idtodelete):
            C[localindex, localindex1] = m[globalindex, globalindex1]
    return (A, B, C)


def chop_in_blocks_vector(v, idtodelete):
    """
    Splits a vector into two vectors, where idtodelete specifies
    which elements go into vb
    """
    idtokeep = list(set(np.arange(len(v)))-set(idtodelete))
    va = v[idtokeep]
    vb = v[idtodelete]
    return (va, vb)


def reassemble(A, idtodelete):
    """
    Puts the matrix A inside a larger matrix of dimensions
    dim(A)+len(idtodelete)
    The empty space are filled with zeros (offdiagonal) and ones (diagonals)
    """
    ntot = len(A)+len(idtodelete)
    ind = set(np.arange(ntot))-set(idtodelete)
    newmat = np.zeros((ntot, ntot))
    for i, i1 in enumerate(ind):
        for j, j1 in enumerate(ind):
            newmat[i1, j1] = A[i, j]

    for i in idtodelete:
        newmat[i, i] = 1.0
    return newmat


def reassemble_vector(va, idtodelete):
    ntot = len(va)+len(idtodelete)
    ind = set(np.arange(ntot))-set(idtodelete)
    newv = np.zeros(ntot)
    for j, j1 in enumerate(ind):
        newv[j1] = va[j]
    return newv


def omega(n):
    """ Utility function to calculate fidelities"""
    x = np.zeros(n)
    x[0::2] = 1
    A = np.diag(x[0:-1], 1)
    W = A-np.transpose(A)
    return W


def xmat(n):
    """ Returns the matrix ((0, I_n), (I, 0_n))"""
    idm = np.identity(n)
    return np.concatenate((np.concatenate((0*idm, idm), axis=1),
                           np.concatenate((idm, 0*idm), axis=1)), axis=0).real


class LimitedSizeDict(OrderedDict):
    """Defines a limited sizes dictionary.
    Used to limit the cache size.
    """
    def __init__(self, *args, **kwargs):
        self.size_limit = kwargs.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwargs)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


MAXSIZE = 1000
Tee = tee([], 1)[0].__class__

def memoized(f, maxsize=MAXSIZE):
    """Used to memoize a generator"""
    cache = LimitedSizeDict(maxsize=maxsize)
    def ret(*args):
        if args not in cache:
            cache[args] = f(*args)
        if isinstance(cache[args], (GeneratorType, Tee)):
            # the original can't be used any more,
            # so we need to change the cache as well
            cache[args], r = tee(cache[args])
            return r
        return cache[args]
    return ret


@memoized
def partitions(s, singles=True, pairs=True):
    """Returns the partitions necessary to calculate click probabilities."""
    # pylint: disable=too-many-branches
    #print(s)
    if len(s) == 2:
        if singles:
            yield (s[0],), (s[1],)
        if pairs:
            yield s
    else:
        # pull off a single item and partition the rest
        if singles:
            if len(s) > 1:
                item_partition = (s[0],)
                rest = s[1:]
                rest_partitions = partitions(rest, singles, pairs)
                for p in rest_partitions:
                    if isinstance(p[0], tuple):
                        yield ((item_partition),) + p
                    else:
                        yield (item_partition, p)
            else:
                yield s
        # pull off a pair of items and partition the rest
        if pairs:
            # idx0 = 0
            for idx1 in range(1, len(s)):
                item_partition = (s[0], s[idx1])
                rest = s[1:idx1] + s[idx1+1:]
                rest_partitions = partitions(rest, singles, pairs)
                for p in rest_partitions:
                    if isinstance(p[0], tuple):
                        yield ((item_partition),) + p
                    else:
                        yield (item_partition, p)


def gen_indices(l):
    """
    Given a list of integers [i_0,i_1,...,i_m] it returns a list of int8 integers as follows
    [0,0,...,1,1,...,2,2....,....,m,m,m...]
    where 0 is repated i_0 times, 1 is repated i_1 times and so on and so forth
    """
    x = np.zeros(sum(l), dtype=np.uint8)
    ss = 0
    for i, ii in enumerate(l):
        for j in range(ii):
            x[j+ss] = i
        ss += l[i]
    return x


def fock_prob(s2, ocp, tol=1.0e-13):
    """
    Calculates the probability of measuring the gaussian state s2 in the photon number
    occupation pattern ocp"""
    beta = np.concatenate((s2.mean, np.conjugate(s2.mean)))
    nmodes = s2.nlen
    sqinv = np.linalg.inv(s2.qmat())
    pref = np.exp(-0.5*np.dot(np.dot(beta, sqinv), np.conjugate(beta)))
    sqd = np.sqrt(1/np.linalg.det(s2.qmat()).real)
    if not all(p == 0 for p in ocp):
        gamma = np.dot(np.dot(xmat(nmodes), np.conjugate(sqinv)), beta)
        ind = gen_indices(ocp)
        ina = tuple(np.concatenate((ind, ind+nmodes)))
        A = s2.Amat()
        doubles = True
        if np.linalg.norm(s2.mean)*np.sqrt(2) < tol:
            # This is equivalent to np.linalg.norm(beta) < tol but twice as fast.
            # Is the sqrt(2) really needed?
            singles = False
        else:
            singles = True
        part1 = partitions(ina, singles, doubles)

        ssum = 0.0
        for i in part1:
            pp = 1.0
            if isinstance(i[0], np.uint8):
                i = (i,)

            for j in i:
                if len(j) == 1:
                    pp *= gamma[j]
                if len(j) == 2:
                    pp *= A[j]

            ssum += pp

        return (pref*sqd*ssum).real/np.prod(factorial(ocp))
    else:
        return (pref*sqd).real/np.prod(factorial(ocp))

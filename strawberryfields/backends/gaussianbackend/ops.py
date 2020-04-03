# Copyright 2019 Xanadu Quantum Technologies Inc.

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

import numpy as np

from scipy.linalg import sqrtm

from thewalrus.quantum import (
    density_matrix_element,
    is_pure_cov,
    pure_state_amplitude,
    state_vector,
    density_matrix,
)


def fock_amplitudes_one_mode(alpha, cov, cutoff):
    """ Returns the Fock space density matrix of gaussian state characterized
    by a complex displacement alpha and a (symmetric) covariance matrix
    The Fock ladder ladder goes from 0 to cutoff-1"""
    r = 2 * np.array([alpha.real, alpha.imag])
    if is_pure_cov(cov):
        psi = state_vector(r, cov, normalize=True, cutoff=cutoff)
        rho = np.outer(psi, psi.conj())
        return rho
    return density_matrix(r, cov, normalize=True, cutoff=cutoff)


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
    v1 = 0.5 * cov1
    v2 = 0.5 * cov2
    deltar = mu1 - mu2
    n = 1
    W = omega(2 * n)

    si12 = np.linalg.inv(v1 + v2)
    vaux = np.dot(np.dot(np.transpose(W), si12), 0.25 * W + np.dot(v2, np.dot(W, v1)))

    p1 = np.dot(vaux, W)
    p1 = np.dot(p1, p1)
    p1 = np.identity(2 * n) + 0.25 * np.linalg.inv(p1)
    if np.linalg.norm(p1) < tol:
        p1 = np.zeros_like(p1)
    else:
        p1 = sqrtm(p1)
    p1 = 2 * (p1 + np.identity(2 * n))
    p1 = np.dot(p1, vaux).real
    f = np.sqrt(np.linalg.det(si12) * np.linalg.det(p1)) * np.exp(
        -0.25 * np.dot(np.dot(deltar, si12), deltar).real
    )
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
    idtokeep = list(set(np.arange(len(v))) - set(idtodelete))
    va = v[idtokeep]
    vb = v[idtodelete]
    return (va, vb)


def reassemble(A, idtodelete):
    """
    Puts the matrix A inside a larger matrix of dimensions
    dim(A)+len(idtodelete)
    The empty space are filled with zeros (offdiagonal) and ones (diagonals)
    """
    ntot = len(A) + len(idtodelete)
    ind = set(np.arange(ntot)) - set(idtodelete)
    newmat = np.zeros((ntot, ntot))
    for i, i1 in enumerate(ind):
        for j, j1 in enumerate(ind):
            newmat[i1, j1] = A[i, j]

    for i in idtodelete:
        newmat[i, i] = 1.0
    return newmat


def reassemble_vector(va, idtodelete):
    r"""Creates a vector with zeros indices idtodelete
    and everywhere else it puts the entries of va
    """
    ntot = len(va) + len(idtodelete)
    ind = set(np.arange(ntot)) - set(idtodelete)
    newv = np.zeros(ntot)
    for j, j1 in enumerate(ind):
        newv[j1] = va[j]
    return newv


def omega(n):
    """ Utility function to calculate fidelities"""
    x = np.zeros(n)
    x[0::2] = 1
    A = np.diag(x[0:-1], 1)
    W = A - np.transpose(A)
    return W


def xmat(n):
    """ Returns the matrix ((0, I_n), (I, 0_n))"""
    idm = np.identity(n)
    return np.concatenate(
        (np.concatenate((0 * idm, idm), axis=1), np.concatenate((idm, 0 * idm), axis=1)), axis=0
    ).real


def fock_prob(mu, cov, ocp):
    """
    Calculates the probability of measuring the gaussian state s2 in the photon number
    occupation pattern ocp"""
    if is_pure_cov(cov):
        return np.abs(pure_state_amplitude(mu, cov, ocp, check_purity=False)) ** 2

    return density_matrix_element(mu, cov, list(ocp), list(ocp)).real

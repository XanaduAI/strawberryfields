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
"""Common shared decompositions that can be used by backends"""

from itertools import groupby

import numpy as np
from scipy.linalg import block_diag, sqrtm, polar, schur

from .backends.shared_ops import sympmat, changebasis


def takagi(N, tol=13):
    r"""Computes the Autonne-Takagi decomposition of a complex symmetric (not Hermitian!) matrix.

    Note that singular values of N are considered equal if they are equal after np.round(values, tol).

    See Cariolaro et al. Phys. Rev. A 94, 062109 (2016) [10.1103/PhysRevA.94.062109]
    and references therein for a derivation

    Args:
        N (array): square, symmetric complex numpy array N
        tol (int): the number of decimal places to use when rounding the singular values of N
    Returns:
        tuple(array,array): Returns the tuple (rl, U), where rl are the
            (rounded) singular values, and U is the Takagi unitary ``N = U @ np.diag(l) @ np.transpose(U)``
    """
    (n, m) = N.shape
    if n != m:
        raise ValueError("The input matrix must be square")
    if np.round(np.linalg.norm(N-np.transpose(N)), tol) != 0:
        raise ValueError("The input matrix is not symmetric")

    v, l, ws = np.linalg.svd(N)
    w = np.transpose(np.conjugate(ws))
    rl = np.round(l, tol)

    ## Generate list with degenerancies
    result = []
    for k, g in groupby(rl):
        result.append(list(g))

    ## Generate lists containing the columns that correspond to degenerancies
    kk = 0
    for k in result:
        for ind, j in enumerate(k): # pylint: disable=unused-variable
            k[ind] = kk
            kk = kk+1

    ## Generate the lists with the degenerate column subspaces
    vas = []
    was = []
    for i in result:
        vas.append(v[:, i])
        was.append(w[:, i])

    ## Generate the matrices qs of the degenerate subspaces
    qs = []
    for i in range(len(result)):
        qs.append(sqrtm(np.transpose(vas[i]) @ was[i]))

    ## Construct the Takagi unitary
    qb = block_diag(*qs)

    U = v @ np.conj(qb)
    return rl, U


def T(m, n, theta, phi, nmax):
    r"""The Clements T matrix"""
    mat = np.identity(nmax, dtype=np.complex128)
    mat[m, m] = np.exp(1j*phi)*np.cos(theta)
    mat[m, n] = -np.sin(theta)
    mat[n, m] = np.exp(1j*phi)*np.sin(theta)
    mat[n, n] = np.cos(theta)
    return mat

def Ti(m, n, theta, phi, nmax):
    r"""The inverse Clements T matrix"""
    return np.transpose(T(m, n, theta, -phi, nmax))

def nullTi(m, n, U):
    r"""Nullifies element m,n of U using Ti"""
    (nmax, mmax) = U.shape

    if nmax != mmax:
        raise ValueError("nmax and mmax must be equal")

    if U[m, n+1] == 0:
        thetar = np.pi/2
        phir = 0
    else:
        r = U[m, n] / U[m, n+1]
        thetar = np.arctan(np.abs(r))
        phir = np.angle(r)

    return [n, n+1, thetar, phir, nmax]

def nullT(n, m, U):
    r"""Nullifies element n,m of U using T"""
    (nmax, mmax) = U.shape

    if nmax != mmax:
        raise ValueError("nmax and mmax must be equal")

    if U[n-1, m] == 0:
        thetar = np.pi/2
        phir = 0
    else:
        r = -U[n, m] / U[n-1, m]
        thetar = np.arctan(np.abs(r))
        phir = np.angle(r)

    return [n-1, n, thetar, phir, nmax]

def clements(V, tol=11):
    r"""Performs the Clements decomposition of a Unitary complex matrix.

    See Clements et al. Optica 3, 1460 (2016) [10.1364/OPTICA.3.001460] for more details.

    Args:
        V (array): Unitary matrix of size n_size
        tol (int): the number of decimal places to use when determining if the matrix is symmetric

    Returns:
        tuple[array]: returns a tuple of the form ``(tilist,tlist,np.diag(localV))``
            where:

            * ``tilist``: list containing ``[n,m,theta,phi,n_size]`` of the Ti unitaries needed
            * ``tlist``: list containing ``[n,m,theta,phi,n_size]`` of the T unitaries needed
            * ``localV``: Diagonal unitary sitting sandwhiched by Ti's and the T's
    """
    localV = V
    (nsize, _) = localV.shape

    diffn = np.linalg.norm(V @ V.conj().T - np.identity(nsize))
    if np.round(diffn, tol) != 0.0:
        raise ValueError("The input matrix is not unitary")

    tilist = []
    tlist = []
    for k, i in enumerate(range(nsize-2, -1, -1)):
        if k%2 == 0:
            for j in reversed(range(nsize-1-i)):
                tilist.append(nullTi(i+j+1, j, localV))
                localV = localV @ Ti(*tilist[-1])
        else:
            for j in range(nsize-1-i):
                tlist.append(nullT(i+j+1, j, localV))
                localV = T(*tlist[-1]) @ localV

    return tilist, tlist, np.diag(localV)


def williamson(V, tol=11):
    r"""Performs the Williamson decomposition of positive definite (real) symmetric matrix.

    Note that it is assumed that the symplectic form is

    ..math:: \Omega = \begin{bmatrix}0&I\\-I&0\end{bmatrix}

    where :math:`I` is the identity matrix and :math:`0` is the zero matrix.

    See https://math.stackexchange.com/questions/1171842/finding-the-symplectic-matrix-in-williamsons-theorem/2682630#2682630

    Args:
        V (array): A positive definite symmetric (real) matrix V
        tol (int): the number of decimal places to use when determining if the matrix is symmetric

    Returns:
        tuple(array,array): Returns a tuple ``(Db, S)`` where ``Db`` is a diagonal matrix
            and ``S`` is a symplectic matrix such that :math:`V = S^T Db S`
    """
    (n, m) = V.shape
    diffn = np.linalg.norm(V-np.transpose(V))

    if np.round(diffn, tol) != 0.0:
        raise ValueError("The input matrix is not symmetric")
    if n != m:
        raise ValueError("The input matrix is not square")
    if n%2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n//2
    omega = sympmat(n)
    rotmat = changebasis(n)
    vals = np.linalg.eigvalsh(V)

    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    Mm12 = sqrtm(np.linalg.inv(V)).real
    r1 = Mm12 @ omega @ Mm12
    s1, K = schur(r1)
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    seq = []

    ## In what follows I construct a permutation matrix p  so that the Schur matrix has
    ## only positive elements above the diagonal
    ## Also the Schur matrix uses the x_1,p_1, ..., x_n,p_n  ordering thus I use rotmat to
    ## go to the ordering x_1, ..., x_n, p_1, ... , p_n

    for i in range(n):
        if s1[2*i, 2*i+1] > 0:
            seq.append(I)
        else:
            seq.append(X)

    p = block_diag(*seq)
    Kt = K @ p
    s1t = p @ s1 @ p
    dd = np.transpose(rotmat) @ s1t @rotmat
    Ktt = Kt @ rotmat
    Db = np.diag([1/dd[i, i+n] for i in range(n)] + [1/dd[i, i+n] for i in range(n)])
    S = Mm12 @ Ktt @ sqrtm(Db)
    return Db, np.linalg.inv(S).T


def bloch_messiah(S, tol=10):
    r""" Performs the Bloch-Messiah decomposition of a symplectic matrix in terms of
    two symplectic unitaries and squeezing transformation.

    It automatically sorts the squeezers so that they respect the canonical symplectic form.

    Note that it is assumed that the symplectic form is

    ..math:: \Omega = \begin{bmatrix}0&I\\-I&0\end{bmatrix}

    where :math:`I` is the identity matrix and :math:`0` is the zero matrix.

    For more info see:
    https://math.stackexchange.com/questions/1886038/finding-euler-decomposition-of-a-symplectic-matrix

    Args:
        S (array): A symplectic matrix S
        tol (int): the number of decimal places to use when determining if the matrix is symplectic

    Returns:
        tuple[array]: Returns the tuple ``(ut1, st1, vt1)``. ``ut1`` and ``vt1`` are symplectic unitaries,
            and ``st1`` is diagonal and of the form :math:`= \text{diag}(s1,\dots,s_n, 1/s_1,\dots,1/s_n)`
            such that :math:`S = ut1  st1  v1`
    """
    (n, m) = S.shape

    if n != m:
        raise ValueError("The input matrix is not square")
    if n%2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n//2
    omega = sympmat(n)
    if np.round(np.linalg.norm(np.transpose(S) @ omega @ S - omega), tol) != 0.0:
        raise ValueError("The input matrix is not symplectic")

    u, sigma = polar(S, side='left')
    ss, uss = takagi(sigma)

    ## Apply a permutation matrix so that the squeezers appear in the order
    ## s_1,...,s_n, 1/s_1,...1/s_n
    perm = np.array(list(range(0, n)) + list(reversed(range(n, 2*n))))

    pmat = np.identity(2*n)[perm, :]
    ut = uss @ pmat

    ## Apply a second permutation matrix to permute s
    ## (and their corresonding inverses) to get the canonical symplectic form
    qomega = np.transpose(ut) @ (omega) @ ut
    perm1 = list(range(2*n))

    for i in range(n):
        if qomega[i, i+n] < 0:
            perm1[i] = i+n
            perm1[i+n] = i

    perm1 = np.array(perm1)
    pmat1 = np.identity(2*n)[perm1, :]
    st1 = pmat1 @ pmat @ np.diag(ss) @ pmat @ pmat1
    ut1 = uss @ pmat @ pmat1
    v1 = np.transpose(ut1) @ u
    return ut1, st1, v1


def covmat_to_hamil(V, tol=10): # pragma: no cover
    r"""Converts a covariance matrix to a Hamiltonian.

    Given a covariance matrix V of a Gaussian state :math:`\rho` in the xp ordering,
    finds a positive matrix :math:`H` such that

    .. math:: \rho = \exp(-Q^T H Q/2)/Z

    where :math:`Q = (x_1,\dots,x_n,p_1,\dots,p_n)` are the canonical
    operators, and Z is the partition function.

    For more details, see https://arxiv.org/pdf/1507.01941.pdf

    Args:
        V (array): Gaussian covariance matrix
        tol (int): the number of decimal places to use when determining if the matrix is symmetric

    Returns:
        array: positive definite Hamiltonian matrix
    """
    (n, m) = V.shape
    if n != m:
        raise ValueError("Input matrix must be square")
    if np.round(np.linalg.norm(V-V.T), tol) != 0:
        raise ValueError("Input matrix must be symmetric")

    n = n//2
    omega = sympmat(n)

    vals = np.linalg.eigvalsh(V)
    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    W = 1j*V @ omega
    l, v = np.linalg.eig(W)
    H = (1j * omega @ (v @ np.diag(np.arctanh(1.0/l.real)) @ np.linalg.inv(v))).real

    return H


def hamil_to_covmat(H, tol=10): # pragma: no cover
    r"""Converts a Hamiltonian matrix to a covariance matrix.

    Given a Hamiltonian matrix of a Gaussian state H, finds the equivalent covariance matrix
    V in the xp ordering.

    For more details, see https://arxiv.org/pdf/1507.01941.pdf

    Args:
        H (array): positive definite Hamiltonian matrix
        tol (int): the number of decimal places to use when determining if the Hamiltonian is symmetric

    Returns:
        array: Gaussian covariance matrix
    """
    (n, m) = H.shape
    if n != m:
        raise ValueError("Input matrix must be square")
    if np.round(np.linalg.norm(H-H.T), tol) != 0:
        raise ValueError("Input matrix must be symmetric")

    vals = np.linalg.eigvalsh(H)
    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    n = n//2
    omega = sympmat(n)

    Wi = 1j*omega @ H
    l, v = np.linalg.eig(Wi)
    V = (1j * (v @ np.diag(1.0/np.tanh(l.real)) @ np.linalg.inv(v)) @ omega).real
    return V

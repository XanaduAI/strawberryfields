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
"""Gaussian circuit operations"""
# pylint: disable=duplicate-code,attribute-defined-outside-init
import numpy as np

from . import ops
from ..shared_ops import changebasis


class GaussianModes:
    """ Base class for representing and operating on a collection of
    continuous variable modes in the symplectic basis as encoded in a
    covariance matrix and a mean vector.
    The modes are initialized in the (multimode) vacuum state,
    The state of the modes is manipulated by calling the various methods."""
    # pylint: disable=too-many-public-methods

    def __init__(self, num_subsystems, hbar):
        r"""The class is initialized by providing an integer indicating the number of modes
        Unlike the "standard" covariance matrix for the Wigner function that uses symmetric ordering
        as defined in e.g.
        [1] Gaussian quantum information
        Christian Weedbrook, Stefano Pirandola, Raúl García-Patrón, Nicolas J. Cerf, Timothy C. Ralph, Jeffrey H. Shapiro, and Seth Lloyd
        Rev. Mod. Phys. 84, 621 – Published 1 May 2012
        we define covariance matrices in terms of the following two quantities:
        $$
        N_{i,j} =\langle a_i^\dagger a_j \rangle
        M_{i,j} = \langle a_i a_j \rangle
        $$
        Note that the matrix $N$ is hermitian and the matrix M is symmetric.
        The mean displacements are stored as expectation values of the destruction operator $\alpha_i  = \langle a_i \rangle$
        We also provide functions that give the symmetric ordered covariance matrices and the mean displacement for the quadrature
        operators $q = a+a^\dagger$ and $p = i(a^\dagger -a)$. Note that with these conventions $[q,p]=2 i$.
        For vacuum one has $N_{i,j}=M_{i,j}=alpha_i =0$,
        The quantities $N,M,\alpha$ are stored in the variable nmat, mmat, mean respectively
        """
        # Check validity
        if not isinstance(num_subsystems, int):
            raise ValueError("Number of modes must be an integer")

        self.hbar = hbar
        self.reset(num_subsystems)

    def add_mode(self, n=1):
        """add mode to the circuit"""
        newnlen = self.nlen+n
        newnmat = np.zeros((newnlen, newnlen), dtype=complex)
        newmmat = np.zeros((newnlen, newnlen), dtype=complex)
        newmean = np.zeros(newnlen, dtype=complex)
        newactive = list(np.arange(newnlen, dtype=int))

        for i in range(self.nlen):
            newmean[i] = self.mean[i]
            newactive[i] = self.active[i]
            for j in range(self.nlen):
                newnmat[i, j] = self.nmat[i, j]
                newmmat[i, j] = self.mmat[i, j]

        self.mean = newmean
        self.nmat = newnmat
        self.mmat = newmmat
        self.active = newactive
        self.nlen = newnlen

    def del_mode(self, modes):
        """ delete mode from the circuit"""
        if isinstance(modes, int):
            modes = [modes]

        for mode in modes:
            if self.active[mode] is None:
                raise ValueError("Cannot delete mode, mode does not exist")

            self.loss(0.0, mode)
            self.active[mode] = None

    def reset(self, num_subsystems=None):
        """Resets the simulation state.

        Args:
            num_subsystems (int, optional): Sets the number of modes in the reset
                circuit. None means unchanged.
        """
        if num_subsystems is not None:
            if not isinstance(num_subsystems, int):
                raise ValueError("Number of modes must be an integer")
            self.nlen = num_subsystems

        self.nmat = np.zeros((self.nlen, self.nlen), dtype=complex)
        self.mmat = np.zeros((self.nlen, self.nlen), dtype=complex)
        self.mean = np.zeros(self.nlen, dtype=complex)
        self.active = list(np.arange(self.nlen, dtype=int))

    def get_modes(self):
        """return the modes currently active"""
        return [x for x in self.active if x is not None]

    def displace(self, beta, i):
        """ Implements a displacement operation by the complex number beta in mode i"""
        #Update displacement of mode i by the complex amount bet
        if self.active[i] is None:
            raise ValueError("Cannot displace mode, mode does not exist")

        self.mean[i] += beta

    def squeeze(self, r, phi, k):
        """ Implements a squeezing operation in mode k by the amount z = r*exp(1j*phi)."""
        if self.active[k] is None:
            raise ValueError("Cannot squeeze mode, mode does not exist")

        phase = np.exp(1j*phi)
        phase2 = phase*phase
        sh = np.sinh(r)
        ch = np.cosh(r)
        sh2 = sh*sh
        ch2 = ch*ch
        shch = sh*ch
        nk = np.copy(self.nmat[k])
        mk = np.copy(self.mmat[k])

        alphak = np.copy(self.mean[k])
        # Update displacement of mode k
        self.mean[k] = alphak*ch-phase*np.conj(alphak)*sh
        # Update covariance matrix elements. Only the k column and row of nmat and mmat need to be updated.
        # First update the diagonal elements
        self.nmat[k, k] = sh2 - phase*shch*np.conj(mk[k]) - shch*np.conj(phase)*mk[k] + ch2*nk[k] + sh2*nk[k]
        self.mmat[k, k] = -(phase*shch) + phase2*sh2*np.conj(mk[k]) + ch2*mk[k] - 2*phase*shch*nk[k]

        # Update the column k
        for l in np.delete(np.arange(self.nlen), k):
            self.nmat[k, l] = -(sh*np.conj(phase)*mk[l]) + ch*nk[l]
            self.mmat[k, l] = ch*mk[l] - phase*sh*nk[l]

        # Update row k
        self.nmat[:, k] = np.conj(self.nmat[k])
        self.mmat[:, k] = self.mmat[k]

    def phase_shift(self, phi, k):
        """ Implements a phase shift in mode k by the amount phi."""
        if self.active[k] is None:
            raise ValueError("Cannot phase shift mode, mode does not exist")

        phase = np.exp(1j*phi)
        phase2 = phase*phase
        # Update displacement of mode k
        self.mean[k] = self.mean[k]*phase

        # Update covariance matrix elements. Only the k column and row of nmat and mmat need to be updated.
        # First update the diagonal elements
        self.mmat[k][k] = phase2*self.mmat[k][k]

        # Update the column k
        for l in np.delete(np.arange(self.nlen), k):
            self.nmat[k][l] = np.conj(phase)*self.nmat[k][l]
            self.mmat[k][l] = phase*self.mmat[k][l]

        # Update row k
        self.nmat[:, k] = np.conj(self.nmat[k])
        self.mmat[:, k] = self.mmat[k]

    def beamsplitter(self, theta, phi, k, l):
        """ Implements a beam splitter operation between modes k and l by the amount theta, phi"""
        if self.active[k] is None or self.active[l] is None:
            raise ValueError("Cannot perform beamsplitter, mode(s) do not exist")

        if k == l:
            raise ValueError("Cannot use the same mode for beamsplitter inputs")

        phase = np.exp(1j*phi)
        phase2 = phase*phase
        sh = np.sin(theta)
        ch = np.cos(theta)
        sh2 = sh*sh
        ch2 = ch*ch
        shch = sh*ch
        # alpha1 = self.mean[0]

        nk = np.copy(self.nmat[k])
        mk = np.copy(self.mmat[k])
        nl = np.copy(self.nmat[l])
        ml = np.copy(self.mmat[l])
        # Update displacement of mode k and l
        alphak = np.copy(self.mean[k])
        alphal = np.copy(self.mean[l])
        self.mean[k] = ch*alphak+phase*sh*alphal
        self.mean[l] = ch*alphal-np.conj(phase)*sh*alphak
        # Update covariance matrix elements. Only the k and l columns and rows of nmat and mmat need to be updated.
        # First update the (k,k), (k,l), (l,l), and (l,l) elements
        self.nmat[k][k] = ch2*nk[k] + phase*shch*nk[l] + shch*np.conj(phase)*nl[k] + sh2*nl[l]
        self.nmat[k][l] = -(shch*np.conj(phase)*nk[k]) + ch2*nk[l] - sh2*np.conj(phase2)*nl[k] + shch*np.conj(phase)*nl[l]
        self.nmat[l][k] = np.conj(self.nmat[k][l])
        self.nmat[l][l] = sh2*nk[k] - phase*shch*nk[l] - shch*np.conj(phase)*nl[k] + ch2*nl[l]

        self.mmat[k][k] = ch2*mk[k] + 2*phase*shch*ml[k] + phase2*sh2*ml[l]
        self.mmat[k][l] = -(shch*np.conj(phase)*mk[k]) + ch2*ml[k] - sh2*ml[k] + phase*shch*ml[l]
        self.mmat[l][k] = self.mmat[k][l]
        self.mmat[l][l] = sh2*np.conj(phase2)*mk[k] - 2*shch*np.conj(phase)*ml[k] + ch2*ml[l]

        # Update columns k and l
        for i in np.delete(np.arange(self.nlen), (k, l)):
            self.nmat[k][i] = ch*nk[i] + sh*np.conj(phase)*nl[i]
            self.mmat[k][i] = ch*mk[i] + phase*sh*ml[i]
            self.nmat[l][i] = -(phase*sh*nk[i]) + ch*nl[i]
            self.mmat[l][i] = -(sh*np.conj(phase)*mk[i]) + ch*ml[i]

        # Update rows k and l
        self.nmat[:, k] = np.conj(self.nmat[k])
        self.mmat[:, k] = self.mmat[k]
        self.nmat[:, l] = np.conj(self.nmat[l])
        self.mmat[:, l] = self.mmat[l]

    def scovmatxp(self):
        r"""Constructs and returns the symmetric ordered covariance matrix in the xp ordering.
        The ordered for the canonical operators is $ q_1,..,q_n, p_1,...,p_n$.
        This differes from the ordering used in [1] which is $q_1,p_1,q_2,p_2,...,q_n,p_n$
        Note that one ordering can be obtained from the other by using a permutation matrix.
        Said permutation matrix is implemented in the function changebasis(n) where n is
        the number of modes.
        """
        mm11 = self.nmat+np.transpose(self.nmat)+self.mmat+np.conj(self.mmat)+np.identity(self.nlen)
        mm12 = 1j*(-np.transpose(self.mmat)+np.transpose(np.conj(self.mmat))+np.transpose(self.nmat)-self.nmat)
        mm22 = self.nmat+np.transpose(self.nmat)-self.mmat-np.conj(self.mmat)+np.identity(self.nlen)
        return np.concatenate((np.concatenate((mm11, mm12), axis=1),
                               np.concatenate((np.transpose(mm12), mm22), axis=1)), axis=0).real

    def scovmat(self):
        """Constructs and returns the symmetric ordered covariance matrix as defined in [1]
        """
        rotmat = changebasis(self.nlen)
        return np.dot(np.dot(rotmat, self.scovmatxp()), np.transpose(rotmat))

    def smean(self):
        r"""the symmetric mean $[q_1,p_1,q_2,p_2,...,q_n,p_n]$"""
        r = np.empty(2*self.nlen)
        for i in range(self.nlen):
            r[2*i] = 2*self.mean[i].real
            r[2*i+1] = 2*self.mean[i].imag
        return r

    def fromsmean(self, r, modes=None):
        r"""Populates the means from a provided vector of means with hbar=2 assumed.

        Args:
            r (array): vector of means in :math:`(x_1,p_1,x_2,p_2,\dots)` ordering
            modes (Sequence): sequence of modes corresponding to the vector of means
        """
        mode_list = modes
        if modes is None:
            mode_list = range(self.nlen)

        for idx, mode in enumerate(mode_list):
            self.mean[mode] = 0.5*(r[2*idx]+1j*r[2*idx+1])

    def fromscovmat(self, V, modes=None):
        r"""Updates the circuit's state when a standard covariance matrix is provided.

        Args:
            V (array): covariance matrix in symmetric ordering
            modes (Sequence): sequence of modes corresponding to the covariance matrix
        """
        if modes is None:
            n = len(V)//2
            modes = np.arange(self.nlen)

            if n != self.nlen:
                raise ValueError("Covariance matrix is the incorrect size, does not match means vector.")
        else:
            n = len(modes)
            modes = np.array(modes)
            if n > self.nlen:
                raise ValueError("Covariance matrix is larger than the number of subsystems.")

        # convert to xp ordering
        rotmat = changebasis(n)
        VV = np.dot(np.dot(np.transpose(rotmat), V), rotmat)

        A = VV[0:n, 0:n]
        B = VV[0:n, n:2*n]
        C = VV[n:2*n, n:2*n]
        Bt = np.transpose(B)

        if n < self.nlen:
            # reset modes to be prepared back to the vacuum state
            for mode in modes:
                self.loss(0.0, mode)

        rows = modes.reshape(-1, 1)
        cols = modes.reshape(1, -1)
        self.nmat[rows, cols] = 0.25*(A+C+1j*(B-Bt)-2*np.identity(n))
        self.mmat[rows, cols] = 0.25*(A-C+1j*(B+Bt))

    def qmat(self, modes=None):
        """ Construct the covariance matrix for the Q function"""
        if modes is None:
            modes = list(range(self.nlen))

        rows = np.reshape(modes, [-1, 1])
        cols = np.reshape(modes, [1, -1])

        sigmaq = np.concatenate((np.concatenate((self.nmat[rows, cols], np.conjugate(self.mmat[rows, cols])), axis=1),
                                 np.concatenate((self.mmat[rows, cols], np.conjugate(self.nmat[rows, cols])), axis=1)),
                                axis=0)+np.identity(2*len(modes))
        return sigmaq

    def fidelity_coherent(self, alpha, modes=None):
        """ Returns a function that evaluates the Q function of the given state """
        if modes is None:
            modes = list(range(self.nlen))

        Q = self.qmat(modes)
        Qi = np.linalg.inv(Q)
        delta = self.mean[modes]-alpha

        delta = np.concatenate((delta, np.conjugate(delta)))
        return np.sqrt(np.linalg.det(Qi).real)*np.exp(-0.5*np.dot(delta, np.dot(Qi, np.conjugate(delta))).real)

    def fidelity_vacuum(self, modes=None):
        """fidelity of the current state with the vacuum state"""
        if modes is None:
            modes = list(range(self.nlen))

        alpha = np.zeros(len(modes))
        return self.fidelity_coherent(alpha)

    def Amat(self):
        """ Constructs the A matrix from Hamilton's paper"""
        ######### this needs to be conjugated
        sigmaq = np.concatenate((np.concatenate((np.transpose(self.nmat), self.mmat), axis=1), np.concatenate((np.transpose(np.conjugate(self.mmat)), self.nmat), axis=1)), axis=0)+np.identity(2*self.nlen)
        return np.dot(ops.xmat(self.nlen), np.identity(2*self.nlen)-np.linalg.inv(sigmaq))

    def loss(self, T, k):
        r"""Implements a loss channel in mode k by amplitude loss amount \sqrt{T}
        (energy loss amount T)"""
        if self.active[k] is None:
            raise ValueError("Cannot apply loss channel, mode does not exist")

        sqrtT = np.sqrt(T)
        self.nmat[k] = sqrtT*self.nmat[k]
        self.mmat[k] = sqrtT*self.mmat[k]

        self.nmat[k][k] = sqrtT*self.nmat[k][k]
        self.mmat[k][k] = sqrtT*self.mmat[k][k]

        self.nmat[:, k] = np.conj(self.nmat[k])
        self.mmat[:, k] = self.mmat[k]
        self.mean[k] = sqrtT*self.mean[k]

    def init_thermal(self, population, mode):
        """ Initializes a state of mode in a thermal state with the given population"""
        self.loss(0.0, mode)
        self.nmat[mode][mode] = population

    def is_vacuum(self, tol=0.0):
        """ Checks if the state is vacuum by calculating its fidelity with vacuum """
        fid = self.fidelity_vacuum()
        return np.abs(fid-1) <= tol

    def measure_dyne(self, covmat, indices):
        """ Performs the general-dyne measurement specified in covmat, the indices should correspond
        with the ordering of the covmat of the measurement
        covmat specifies a gaussian effect via its covariance matrix. For more information see
        Quantum Continuous Variables: A Primer of Theoretical Methods
        by Alessio Serafini page 129
        """
        if covmat.shape != (2*len(indices), 2*len(indices)):
            raise ValueError("Covariance matrix size does not match indices provided")

        for i in indices:
            if self.active[i] is None:
                raise ValueError("Cannot apply homodyne measurement, mode does not exist")

        expind = np.concatenate((2*np.array(indices), 2*np.array(indices)+1))
        mp = self.scovmat()
        (A, B, C) = ops.chop_in_blocks(mp, expind)
        V = A-np.dot(np.dot(B, np.linalg.inv(C+covmat)), np.transpose(B))
        V1 = ops.reassemble(V, expind)
        self.fromscovmat(V1)

        r = self.smean()
        (va, vc) = ops.chop_in_blocks_vector(r, expind)
        vm = np.random.multivariate_normal(vc, C)

        va = va+np.dot(np.dot(B, np.linalg.inv(C+covmat)), vm-vc)
        va = ops.reassemble_vector(va, expind)
        self.fromsmean(va)
        return vm

    def homodyne(self, n, eps=0.0002):
        """ Performs a homodyne measurement by calling measure dyne an giving it the
        covariance matrix of a squeezed state whose x quadrature has variance eps**2"""
        covmat = np.diag(np.array([eps**2, 1./eps**2]))
        res = self.measure_dyne(covmat, [n])

        return res

    def post_select_homodyne(self, n, val, eps=0.0002):
        """ Performs a homodyne measurement but postelecting on the value vals for mode n """
        if self.active[n] is None:
            raise ValueError("Cannot apply homodyne measurement, mode does not exist")
        covmat = np.diag(np.array([eps**2, 1./eps**2]))
        indices = [n]
        expind = np.concatenate((2*np.array(indices), 2*np.array(indices)+1))
        mp = self.scovmat()
        (A, B, C) = ops.chop_in_blocks(mp, expind)
        V = A-np.dot(np.dot(B, np.linalg.inv(C+covmat)), np.transpose(B))
        V1 = ops.reassemble(V, expind)
        self.fromscovmat(V1)

        r = self.smean()
        (va, vc) = ops.chop_in_blocks_vector(r, expind)
        vm1 = np.random.normal(vc[1], np.sqrt(C[1][1]))
        vm = np.array([val, vm1])
        va = va+np.dot(np.dot(B, np.linalg.inv(C+covmat)), vm-vc)
        va = ops.reassemble_vector(va, expind)
        self.fromsmean(va)
        return val

    def post_select_heterodyne(self, n, alpha_val):
        """ Performs a homodyne measurement but postelecting on the value vals for mode n """
        if self.active[n] is None:
            raise ValueError("Cannot apply heterodyne measurement, mode does not exist")

        covmat = np.identity(2)
        indices = [n]
        expind = np.concatenate((2*np.array(indices), 2*np.array(indices)+1))
        mp = self.scovmat()
        (A, B, C) = ops.chop_in_blocks(mp, expind)
        V = A-np.dot(np.dot(B, np.linalg.inv(C+covmat)), np.transpose(B))
        V1 = ops.reassemble(V, expind)
        self.fromscovmat(V1)

        r = self.smean()
        (va, vc) = ops.chop_in_blocks_vector(r, expind)
        vm = 2.0*np.array([np.real(alpha_val), np.imag(alpha_val)])
        va = va+np.dot(np.dot(B, np.linalg.inv(C+covmat)), vm-vc)
        va = ops.reassemble_vector(va, expind)
        self.fromsmean(va)
        return alpha_val

    def apply_u(self, U):
        """ Transforms the state according to the linear optical unitary that maps a[i] \to U[i, j]^*a[j]"""
        self.mean = np.dot(np.conj(U), self.mean)
        self.nmat = np.dot(np.dot(U, self.nmat), np.conj(np.transpose(U)))
        self.mmat = np.dot(np.dot(np.conj(U), self.mmat), np.conj(np.transpose(U)))

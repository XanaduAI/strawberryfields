# Copyright 2020 Xanadu Quantum Technologies Inc.

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
from thewalrus.quantum import Xmat
import thewalrus.symplectic as symp


from . import ops
from ..shared_ops import changebasis


# Shape of the weights, means, and covs arrays.
def w_shape(nmodes, ngauss):
    return (ngauss ** nmodes)


def m_shape(nmodes, ngauss):
    return (ngauss ** nmodes, 2 * nmodes)


def c_shape(nmodes, ngauss):
    return (ngauss ** nmodes, 2 * nmodes, 2 * nmodes)


def to_xp(n):
    return np.concatenate((np.arange(0, 2*n, 2), np.arange(0, 2*n, 2) + 1))


def from_xp(n):
    perm_inds_list = [(i, i + n) for i in range(n)]
    perm_inds = [a for tup in perm_inds_list for a in tup]
    return perm_inds


def update_means(means, X, perm_out):
    X_perm = X[:, perm_out][perm_out,:]
    return (X_perm @ means.T).T


def update_covs(covs, X, perm_out, Y=0):
    X_perm = X[:, perm_out][perm_out, :]
    if not isinstance(Y, int):
        Y = Y[:, perm_out][perm_out, :]
    return (X_perm @ covs @ X_perm.T) + Y


class BosonicModes:
    """Base class for representing and operating on a collection of
    continuous variable modes in the symplectic basis as encoded in a
    covariance matrix and a mean vector.
    The modes are initialized in the (multimode) vacuum state,
    The state of the modes is manipulated by calling the various methods."""

    # pylint: disable=too-many-public-methods

    def __init__(self, num_subsystems, num_weights=1):
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

        self.hbar = 2
        self.reset(num_subsystems, num_weights)

    def add_mode(self, n=1):
        """Add n modes to the circuit."""
        self.nlen += n
        self.active.append(self.nlen)

        # Updated mode index permutation list
        self.to_xp = to_xp(self.nlen)
        self.from_xp = from_xp(self.nlen)

        new_weights = np.ones(w_shape(self.nlen, self._trunc)) / (self._trunc ** self.nlen)
        new_weights[:self._trunc ** self.nlen] = self.weights
        self.weights = new_weights

        rows = np.arange(self._trunc ** self.nlen)
        cols = np.arange(2 * self.nlen)

        new_means = np.zeros(m_shape(self.nlen, self._trunc))
        new_means[np.ix_(rows, cols)] = self.means
        self.means = new_means

        id_covs = [np.identity(2 * self.nlen, dtype=complex)
                    for i in range(self._trunc ** self.nlen)]
        new_covs = np.array(id_covs)
        new_covs[np.ix_(rows, cols, cols)] = self.covs
        self.covs = new_covs

    def del_mode(self, modes):
        """Delete modes modes from the circuit."""
        if isinstance(modes, int):
            modes = [modes]

        for mode in modes:
            if self.active[mode] is None:
                raise ValueError("Cannot delete mode, mode does not exist")

            self.loss(0.0, mode)
            self.active[mode] = None

    def reset(self, num_subsystems=None, num_weights=None):
        """Reset the simulation state.

        Args:
            num_subsystems (int, optional): Sets the number of modes in the reset
                circuit. None means unchanged.
            num_weights (int): Sets the number of gaussians per mode in the
                superposition. None means unchanged.
        """
        if num_subsystems is not None:
            if not isinstance(num_subsystems, int):
                raise ValueError("Number of modes must be an integer")
            self.nlen = num_subsystems

        if num_weights is not None:
            if not isinstance(num_weights, int):
                raise ValueError("Number of weights must be an integer")
            self._trunc = num_weights

        self.active = list(np.arange(self.nlen, dtype=int))
        # Mode index permutation list to and from XP ordering.
        self.to_xp = to_xp(self.nlen)
        self.from_xp = from_xp(self.nlen)

        self.weights = np.ones(w_shape(self.nlen, self._trunc), dtype=complex)
        self.weights = self.weights / (self._trunc ** self.nlen)

        self.means = np.zeros(m_shape(self.nlen, self._trunc), dtype=complex)
        id_covs = [np.identity(2*self.nlen, dtype=complex)
                   for i in range(self._trunc ** self.nlen)]
        self.covs = np.array(id_covs)

    def get_modes(self):
        """Return the modes currently active."""
        return [x for x in self.active if x is not None]

    def displace(self, r, phi, i):
        """Displace mode i by the amount r*np.exp(1j*phi)."""
        if self.active[i] is None:
            raise ValueError("Cannot displace mode, mode does not exist")

        self.means += symp.expand_vector(r * np.exp(1j * phi), i, self.nlen)[self.from_xp]

    def squeeze(self, r, phi, k):
        """Squeeze mode k by the amount r*exp(1j*phi)."""
        if self.active[k] is None:
            raise ValueError("Cannot squeeze mode, mode does not exist")

        sq = symp.expand(symp.squeezing(r, phi), k, self.nlen)
        self.means = update_means(self.means, sq, self.from_xp)
        self.covs = update_covs(self.covs, sq, self.from_xp)
        
    def mbsqueeze(self, k, r, phi, r_anc, eta_anc, avg):
        """Squeeze mode k by the amount r*exp(1j*phi) using measurement-based squeezing.
        The squeezing of the ancilla resource is r_anc, and the detection efficiency of
        the homodyne on the ancilla mode is eta_anc. Average map or single shot map 
        can be applied."""
        
        if self.active[k] is None:
            raise ValueError("Cannot squeeze mode, mode does not exist")
        
        phi = phi + (1 - np.sign(r)) * np.pi / 2
        r = np.abs(r)
        theta = np.arccos(np.exp(- r))
        self.phase_shift(phi / 2,k)
        
        if avg:
            X = np.diag([np.cos(theta), 1 / np.cos(theta)])
            Y = np.diag([(np.sin(theta) ** 2) * (np.exp(- 2 * r_anc)), 
                          (np.tan(theta) ** 2) * (1 - eta_anc) / eta_anc])
            X2, Y2 = self.expandXY([k], X, Y)
            self.apply_channel(X2, Y2)   
            
        if not avg:
            self.add_mode()
            new_mode = self.nlen
            self.squeeze(r_anc, 0, new_mode)
            self.beamsplitter(theta, 0, k, new_mode)
            self.loss(1 - eta_anc)
            val = self.measure_homodyne(np.pi / 2, new_mode)
            self.delete_mode(new_mode)
            prefac = - np.tan(theta) / np.sqrt(2 * self.circuit.hbar * eta_anc)
            self.displacement(prefac * val, np.pi / 2, k)
            
        self.phase_shift(-phi / 2,k)
        
        if not avg:
            return val

    def phase_shift(self, phi, k):
        """Implement a phase shift in mode k by the amount phi."""
        if self.active[k] is None:
            raise ValueError("Cannot phase shift mode, mode does not exist")

        rot = symp.expand(symp.rotation(phi), k, self.nlen)
        self.means = update_means(self.means, rot, self.from_xp)
        self.covs = update_covs(self.covs, rot, self.from_xp)

    def beamsplitter(self, theta, phi, k, l):
        """Implement a beam splitter operation between modes k and l by the amount theta, phi."""
        if self.active[k] is None or self.active[l] is None:
            raise ValueError("Cannot perform beamsplitter, mode(s) do not exist")

        if k == l:
            raise ValueError("Cannot use the same mode for beamsplitter inputs")

        # Cross-check with Gaussian backend.
        bs = symp.expand(symp.beam_splitter(theta, phi), [k, l], self.nlen)
        self.means = update_means(self.means, bs, self.from_xp)
        self.covs = update_covs(self.covs, bs, self.from_xp)

    def scovmatxp(self):
        r"""Constructs and returns the symmetric ordered covariance matrix in the xp ordering.

        The order for the canonical operators is :math:`q_1,..,q_n, p_1,...,p_n`.
        This differs from the ordering used in [1] which is :math:`q_1,p_1,q_2,p_2,...,q_n,p_n`
        Note that one ordering can be obtained from the other by using a permutation matrix.

        Said permutation matrix is implemented in the function changebasis(n) where n is
        the number of modes.
        """
        return self.covs[:, self.perm_inds][..., self.prem_inds]

    def smeanxp(self):
        r"""Return the symmetric-ordered vector of mean in the xp ordering.

        The order for the canonical operators is :math:`q_1, \ldots, q_n, p_1, \ldots, p_n`.
        This differs from the ordering used in [1] which is :math:`q_1, p_1, q_2, p_2, \ldots, q_n, p_n`.
        Note that one ordering can be obtained from the other by using a permutation matrix.
        """
        return self.means.T[self.perm_inds].T

    def scovmat(self):
        """Return the symmetric-ordered covariance matrix as defined in [1]"""
        # rotmat = changebasis(self.nlen)
        # return np.dot(np.dot(rotmat, self.scovmatxp()), np.transpose(rotmat))
        return self.covs

    def smean(self):
        r"""The symmetric mean $[q_1,p_1,q_2,p_2,...,q_n,p_n]$"""
        return self.means

    def sweights(self):
        '''Returns the matrix of weights.'''
        return self.weights

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
            self.mean[mode] = 0.5 * (r[2 * idx] + 1j * r[2 * idx + 1])

    def fromscovmat(self, V, modes=None):
        r"""Updates the circuit's state when a standard covariance matrix is provided.

        Args:
            V (array): covariance matrix in symmetric ordering
            modes (sequence): sequence of modes corresponding to the covariance matrix
        """
        if modes is None:
            n = len(V) // 2
            modes = np.arange(self.nlen)

            if n != self.nlen:
                raise ValueError(
                    "Covariance matrix is the incorrect size, does not match means vector."
                )
        else:
            n = len(modes)
            modes = np.array(modes)
            if n > self.nlen:
                raise ValueError("Covariance matrix is larger than the number of subsystems.")

        # convert to xp ordering
        rotmat = changebasis(n)
        VV = np.dot(np.dot(np.transpose(rotmat), V), rotmat)

        A = VV[0:n, 0:n]
        B = VV[0:n, n : 2 * n]
        C = VV[n : 2 * n, n : 2 * n]
        Bt = np.transpose(B)

        if n < self.nlen:
            # reset modes to be prepared back to the vacuum state
            for mode in modes:
                self.loss(0.0, mode)

        rows = modes.reshape(-1, 1)
        cols = modes.reshape(1, -1)
        self.nmat[rows, cols] = 0.25 * (A + C + 1j * (B - Bt) - 2 * np.identity(n))
        self.mmat[rows, cols] = 0.25 * (A - C + 1j * (B + Bt))

    def qmat(self, modes=None):
        """ Construct the covariance matrix for the Q function"""
        if modes is None:
            modes = list(range(self.nlen))

        rows = np.reshape(modes, [-1, 1])
        cols = np.reshape(modes, [1, -1])

        sigmaq = (
            np.concatenate(
                (
                    np.concatenate(
                        (self.nmat[rows, cols], np.conjugate(self.mmat[rows, cols])), axis=1
                    ),
                    np.concatenate(
                        (self.mmat[rows, cols], np.conjugate(self.nmat[rows, cols])), axis=1
                    ),
                ),
                axis=0,
            )
            + np.identity(2 * len(modes))
        )
        return sigmaq

    def fidelity_coherent(self, alpha, modes=None):
        """ Returns a function that evaluates the Q function of the given state """
        if modes is None:
            modes = list(range(self.nlen))

        Q = self.qmat(modes)
        Qi = np.linalg.inv(Q)
        delta = self.mean[modes] - alpha

        delta = np.concatenate((delta, np.conjugate(delta)))
        return np.sqrt(np.linalg.det(Qi).real) * np.exp(
            -0.5 * np.dot(delta, np.dot(Qi, np.conjugate(delta))).real
        )

    def fidelity_vacuum(self, modes=None):
        """fidelity of the current state with the vacuum state"""
        if modes is None:
            modes = list(range(self.nlen))

        alpha = np.zeros(len(modes))
        return self.fidelity_coherent(alpha)

    def Amat(self):
        """ Constructs the A matrix from Hamilton's paper"""
        ######### this needs to be conjugated
        sigmaq = (
            np.concatenate(
                (
                    np.concatenate((np.transpose(self.nmat), self.mmat), axis=1),
                    np.concatenate((np.transpose(np.conjugate(self.mmat)), self.nmat), axis=1),
                ),
                axis=0,
            )
            + np.identity(2 * self.nlen)
        )
        return np.dot(Xmat(self.nlen), np.identity(2 * self.nlen) - np.linalg.inv(sigmaq))

    def loss(self, T, k):
        r"""Implements a loss channel in mode k by amplitude loss amount \sqrt{T}
        (energy loss amount T)"""

        if self.active[k] is None:
            raise ValueError("Cannot apply loss channel, mode does not exist")

        X = np.sqrt(T) * np.identity(2)
        Y = (1 - T) * np.identity(2)
        X2, Y2 = self.expandXY(self, [k], X, Y)     
        self.apply_channel(X2, Y2)

    def thermal_loss(self, T, nbar, k):
        r"""Implements the thermal loss channel in mode k by amplitude loss amount \sqrt{T}
        unlike the loss channel, here the ancilliary mode that goes into the second arm of the
        beam splitter is prepared in a thermal state with mean photon number nth"""
        if self.active[k] is None:
            raise ValueError("Cannot apply loss channel, mode does not exist")
        
        X = np.sqrt(T) * np.identity(2)
        Y = (1 - T) * nbar * np.identity(2)
        X2, Y2 = self.expandXY(self, [k], X, Y)     
        self.apply_channel(X2, Y2)

    def init_thermal(self, population, mode):
        """ Initializes a state of mode in a thermal state with the given population"""
        # self.loss(0.0, mode)
        # self.nmat[mode][mode] = population

    def is_vacuum(self, tol=0.0):
        """ Checks if the state is vacuum by calculating its fidelity with vacuum """
        fid = self.fidelity_vacuum()
        return np.abs(fid - 1) <= tol

    def measure_dyne(self, covmat, indices, shots=1):
        """Performs the general-dyne measurement specified in covmat, the indices should correspond
        with the ordering of the covmat of the measurement
        covmat specifies a gaussian effect via its covariance matrix. For more information see
        Quantum Continuous Variables: A Primer of Theoretical Methods
        by Alessio Serafini page 129
        """
        # if covmat.shape != (2 * len(indices), 2 * len(indices)):
        #     raise ValueError("Covariance matrix size does not match indices provided")

        # for i in indices:
        #     if self.active[i] is None:
        #         raise ValueError("Cannot apply homodyne measurement, mode does not exist")

        # expind = np.concatenate((2 * np.array(indices), 2 * np.array(indices) + 1))
        # mp = self.scovmat()
        # (A, B, C) = ops.chop_in_blocks(mp, expind)
        # V = A - np.dot(np.dot(B, np.linalg.inv(C + covmat)), np.transpose(B))
        # V1 = ops.reassemble(V, expind)
        # self.fromscovmat(V1)

        # r = self.smean()
        # (va, vc) = ops.chop_in_blocks_vector(r, expind)
        # vm = np.random.multivariate_normal(vc, C, size=shots)
        # # The next line is a hack in that it only updates conditioned on the first samples value
        # # should still work if shots = 1
        # va = va + np.dot(np.dot(B, np.linalg.inv(C + covmat)), vm[0] - vc)
        # va = ops.reassemble_vector(va, expind)
        # self.fromsmean(va)
        return

    def homodyne(self, n, shots=1, eps=0.0002):
        """Performs a homodyne measurement by calling measure dyne an giving it the
        covariance matrix of a squeezed state whose x quadrature has variance eps**2"""
        # covmat = np.diag(np.array([eps ** 2, 1.0 / eps ** 2]))
        # res = self.measure_dyne(covmat, [n], shots=shots)
        return

    def post_select_homodyne(self, n, val, eps=0.0002):
        """ Performs a homodyne measurement but postelecting on the value vals for mode n """
        if self.active[n] is None:
            raise ValueError("Cannot apply homodyne measurement, mode does not exist")
        # covmat = np.diag(np.array([eps ** 2, 1.0 / eps ** 2]))
        # indices = [n]
        # expind = np.concatenate((2 * np.array(indices), 2 * np.array(indices) + 1))
        # mp = self.scovmat()
        # (A, B, C) = ops.chop_in_blocks(mp, expind)
        # V = A - np.dot(np.dot(B, np.linalg.inv(C + covmat)), np.transpose(B))
        # V1 = ops.reassemble(V, expind)
        # self.fromscovmat(V1)

        # r = self.smean()
        # (va, vc) = ops.chop_in_blocks_vector(r, expind)
        # vm1 = np.random.normal(vc[1], np.sqrt(C[1][1]))
        # vm = np.array([val, vm1])
        # va = va + np.dot(np.dot(B, np.linalg.inv(C + covmat)), vm - vc)
        # va = ops.reassemble_vector(va, expind)
        # self.fromsmean(va)
        return

    def post_select_heterodyne(self, n, alpha_val):
        """ Performs a homodyne measurement but postelecting on the value vals for mode n """
        if self.active[n] is None:
            raise ValueError("Cannot apply heterodyne measurement, mode does not exist")

        # covmat = np.identity(2)
        # indices = [n]
        # expind = np.concatenate((2 * np.array(indices), 2 * np.array(indices) + 1))
        # mp = self.scovmat()
        # (A, B, C) = ops.chop_in_blocks(mp, expind)
        # V = A - np.dot(np.dot(B, np.linalg.inv(C + covmat)), np.transpose(B))
        # V1 = ops.reassemble(V, expind)
        # self.fromscovmat(V1)

        # r = self.smean()
        # (va, vc) = ops.chop_in_blocks_vector(r, expind)
        # vm = 2.0 * np.array([np.real(alpha_val), np.imag(alpha_val)])
        # va = va + np.dot(np.dot(B, np.linalg.inv(C + covmat)), vm - vc)
        # va = ops.reassemble_vector(va, expind)
        # self.fromsmean(va)
        return

    def apply_u(self, U):
        """ Transforms the state according to the linear optical unitary that maps a[i] \to U[i, j]^*a[j]"""
        Us = symp.interferometer(U)
        self.means = update_means(self.means, Us, self.from_xp)
        self.covs = update_covs(self.covs, Us, self.from_xp)

    def apply_channel(self, X, Y):
        self.means = update_means(self.means, X, self.from_xp, Y)
        self.covs = update_covs(self.covs, X, self.from_xp, Y)
    
    def expandS(self, modes, S):
        """ Expands symplectic matrix for modes to symplectic matrix for the whole system. """
        S2 = symp.expand(S, modes, self.nlen)
        return S2
    
    def expandXY(self, modes, X, Y):
        """ Expands X and Y matrices for modes to X and Y matrices for the whole system. """
        X2 = symp.expand(X, modes, self.nlen)
        M = len(Y) // 2
        Y2 = np.zeros((2 * self.nlen, 2 * self.nlen), dtype=Y.dtype)
        w = np.array(modes)
    
        Y2[w.reshape(-1, 1), w.reshape(1, -1)] = Y[:M, :M].copy()
        Y2[(w + self.nlen).reshape(-1, 1), (w + self.nlen).reshape(1, -1)] = Y[M:, M:].copy()
        Y2[w.reshape(-1, 1), (w + self.nlen).reshape(1, -1)] = Y[:M, M:].copy()
        Y2[(w + self.nlen).reshape(-1, 1), w.reshape(1, -1)] = Y[M:, :M].copy()
        
        return X2, Y2

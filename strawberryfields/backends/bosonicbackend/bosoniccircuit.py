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

"""Bosonic circuit operations"""
# pylint: disable=duplicate-code,attribute-defined-outside-init
import itertools as it
import numpy as np
from scipy.linalg import block_diag
import thewalrus.symplectic as symp

from strawberryfields.backends.bosonicbackend import ops


# Shape of the weights, means, and covs arrays.
def w_shape(num_modes, num_gauss):
    r"""Calculate total number of weights. Assumes same number of weights per mode.

    Args:
        num_modes (int): number of modes
        num_gauss (int): number of gaussian peaks per mode

    Returns:
        int: number of weights for all Gaussian peaks in phase space
    """
    return num_gauss ** num_modes


def m_shape(num_modes, num_gauss):
    r"""Shape of array of mean vectors. Assumes same number of weights per mode.

    Args:
        num_modes (int): number of modes
        num_gauss (int): number of gaussian peaks per mode

    Returns:
        tuple: (number of weights, number of quadratures)
    """
    return (num_gauss ** num_modes, 2 * num_modes)


def c_shape(num_modes, num_gauss):
    r"""Shape of array of covariance matricess. Assumes same number of weights per mode.

    Args:
        num_modes (int): number of modes
        num_gauss (int): number of gaussian peaks per mode

    Returns:
        tuple: (number of weights, number of quadratures, number of quadratures)
    """
    return (num_gauss ** num_modes, 2 * num_modes, 2 * num_modes)


def to_xp(num_modes):
    r"""Provides array of indices to order quadratures as all x
    followed by all p, starting from (x1,p1,...,xn,pn) ordering.

    Args:
        num_modes (int): number of modes

    Returns:
        array: quadrature ordering for all x followed by all p
    """
    return np.concatenate((np.arange(0, 2 * num_modes, 2), np.arange(0, 2 * num_modes, 2) + 1))


def from_xp(num_modes):
    r"""Provides array of indices to order quadratures as (x1,p1,...,xn,pn)
    starting from all x followed by all p.

    Args:
        num_modes (int): number of modes

    Returns:
        list: quadrature ordering for (x1,p1,...,xn,pn)
    """
    perm_inds_list = [(i, i + num_modes) for i in range(num_modes)]
    perm_inds = [a for tup in perm_inds_list for a in tup]
    return perm_inds


def update_means(means, X, perm_out):
    r"""Apply a linear transformation ``X`` to the array of means. The
    quadrature ordering can be specified by ``perm_out`` to match ordering
    of ``X`` to means.

    Args:
        means (array): array of mean vectors
        X (array): matrix for linear transformation
        perm_out (array): indices for quadrature ordering

    Returns:
        array: transformed array of mean vectors
    """
    X_perm = X[:, perm_out][perm_out, :]
    return (X_perm @ means.T).T


def update_covs(covs, X, perm_out, Y=0):
    r"""Apply a linear transformation parametrized by ``(X,Y)`` to the
    array of covariance matrices. The  quadrature ordering can be specified
    by ``perm_out`` to match ordering of ``(X,Y)`` to the covariance matrices.

    If Y is not specified, it defaults to 0.

    Args:
        covs (array): array of covariance matrices
        X (array): matrix for mutltiplicative part of transformation
        perm_out (array): indices for quadrature ordering
        Y (array or 0): matrix for additive part of transformation.

    Returns:
        array: transformed array of covariance matrices
    """
    X_perm = X[:, perm_out][perm_out, :]
    if not isinstance(Y, int):
        Y = Y[:, perm_out][perm_out, :]
    return (X_perm @ covs @ X_perm.T) + Y

# pylint: too-many-instance-attributes
class BosonicModes:
    """A Bosonic circuit class."""

    # pylint: disable=too-many-public-methods

    def __init__(self):
        self.hbar = 2

    def add_mode(self, peak_list=None):
        r"""Add len(peak_list) modes to the circuit. Each mode has a number of
        weights specified by peak_list, and the means and covariances are set
        to vacuum.

        Args:
            peak_list (list): list of weights per mode.
        """
        if peak_list is None:
            peak_list = [1]

        num_modes = len(peak_list)
        num_gauss = np.prod(peak_list)
        self.nlen += num_modes

        # Updated mode index permutation list
        self.to_xp = to_xp(self.nlen)
        self.from_xp = from_xp(self.nlen)
        self.active.append(self.nlen - 1)

        # Weights are set equal to each other and normalized
        vac_weights = np.array([1 / num_gauss for i in range(num_gauss)], dtype=complex)
        # New mode means and covs set to vacuum
        vac_means = np.zeros((num_gauss, 2 * num_modes)).tolist()
        vac_covs = [
            ((self.hbar / 2) * np.identity(2 * num_modes)).tolist() for i in range(num_gauss)
        ]

        # Find all possible combinations of means and combs of the
        # Gaussians between the modes.
        mean_combs = it.product(self.means.tolist(), vac_means)
        cov_combs = it.product(self.covs.tolist(), vac_covs)

        # Tensor product of the weights.
        weights = np.kron(self.weights, vac_weights)
        # De-nest the means iterator.
        means = np.array([[a for b in tup for a in b] for tup in mean_combs])
        # Stack covs appropriately.
        covs = np.array([block_diag(*tup) for tup in cov_combs])

        self.weights = weights
        self.means = means
        self.covs = covs

    def del_mode(self, modes):
        r"""Delete modes from the circuit.

        Args:
            modes (int or list): modes to be deleted.
        """
        if isinstance(modes, int):
            modes = [modes]

        for mode in modes:
            if self.active[mode] is None:
                raise ValueError("Cannot delete mode, mode does not exist")

            self.loss(0.0, mode)
            self.active[mode] = None

    def reset(self, num_subsystems=None, num_weights=1):
        """Reset the simulation state.

        Args:
            num_subsystems (int, optional): Sets the number of modes in the reset
                circuit. ``None`` means unchanged.
            num_weights (int): Sets the number of gaussians per mode in the
                superposition.
        """
        if num_subsystems is not None:
            if not isinstance(num_subsystems, int):
                raise ValueError("Number of modes must be an integer")
            self.nlen = num_subsystems

        if not isinstance(num_weights, int):
            raise ValueError("Number of weights must be an integer")

        self.active = list(np.arange(self.nlen, dtype=int))
        # Mode index permutation list to and from XP ordering.
        self.to_xp = to_xp(self.nlen)
        self.from_xp = from_xp(self.nlen)

        self.weights = np.ones(w_shape(self.nlen, num_weights), dtype=complex)
        self.weights = self.weights / (num_weights ** self.nlen)

        self.means = np.zeros(m_shape(self.nlen, num_weights), dtype=complex)
        id_covs = [np.identity(2 * self.nlen, dtype=complex) for i in range(len(self.weights))]
        self.covs = np.array(id_covs)

    def get_modes(self):
        r"""Return the modes currently active."""
        return [x for x in self.active if x is not None]

    def displace(self, r, phi, i):
        r"""Displace mode ``i`` by the amount ``r*np.exp(1j*phi)``.

        Args:
            r (float): displacement magnitude
            phi (float): displacement phase
            i (int): mode to be displaced
        """
        if self.active[i] is None:
            raise ValueError("Cannot displace mode, mode does not exist")
        self.means += symp.expand_vector(r * np.exp(1j * phi), i, self.nlen)[self.from_xp]

    def squeeze(self, r, phi, k):
        r"""Squeeze mode ``k`` by the amount ``r*exp(1j*phi)``.

        Args:
            r (float): squeezing magnitude
            phi (float): squeezing phase
            k (int): mode to be squeezed
        """
        if self.active[k] is None:
            raise ValueError("Cannot squeeze mode, mode does not exist")

        sq = symp.expand(symp.squeezing(r, phi), k, self.nlen)
        self.means = update_means(self.means, sq, self.from_xp)
        self.covs = update_covs(self.covs, sq, self.from_xp)

    def mbsqueeze(self, k, r, phi, r_anc, eta_anc, avg):
        r"""Squeeze mode ``k`` by the amount ``r*exp(1j*phi)`` using measurement-based squeezing.

        Either the average map, described by a Gaussian CPTP transformation, or a single-shot map
        with ancillary measurement outcomes can be simulated.

        Args:
            k (int): mode to be squeezed
            r (float): target squeezing magnitude
            phi (float): target squeezing phase
            r_anc (float): squeezing magnitude of the ancillary mode
            eta_anc(float): detection efficiency of the ancillary mode
            avg (bool): whether to apply the average map or single-shot
        Returns:
            float: if single-shot map selected, returns the measurement outcome of the ancilla
        """

        if self.active[k] is None:
            raise ValueError("Cannot squeeze mode, mode does not exist")

        # antisqueezing corresponds to an extra phase shift
        if r < 0:
            phi += np.pi
        r = np.abs(r)
        # beamsplitter angle
        theta = np.arccos(np.exp(-r))
        self.phase_shift(-phi / 2, k)

        # Construct (X,Y) for Gaussian CPTP of average map
        if avg:
            X = np.diag([np.cos(theta), 1 / np.cos(theta)])
            Y = np.diag(
                [
                    (np.sin(theta) ** 2) * (np.exp(-2 * r_anc)),
                    (np.tan(theta) ** 2) * (1 - eta_anc) / eta_anc,
                ]
            )
            Y *= self.hbar / 2
            X2, Y2 = self.expandXY([k], X, Y)
            self.apply_channel(X2, Y2)

        # Add new ancilla mode, interfere it and measure it
        # Delete ancilla mode from active list
        elif not avg:
            self.add_mode()
            new_mode = self.nlen - 1
            self.squeeze(r_anc, 0, new_mode)
            self.beamsplitter(theta, 0, k, new_mode)
            self.loss(eta_anc, new_mode)
            self.phase_shift(np.pi / 2, new_mode)
            val = self.homodyne(new_mode)
            self.del_mode(new_mode)
            self.covs = np.delete(self.covs, [2 * new_mode, 2 * new_mode + 1], axis=1)
            self.covs = np.delete(self.covs, [2 * new_mode, 2 * new_mode + 1], axis=2)
            self.means = np.delete(self.means, [2 * new_mode, 2 * new_mode + 1], axis=1)
            self.nlen = self.nlen - 1
            self.from_xp = from_xp(self.nlen)
            self.to_xp = to_xp(self.nlen)
            self.active = self.active[:new_mode]
            prefac = -np.tan(theta) / np.sqrt(2 * self.hbar * eta_anc)
            self.displace(prefac * val[0][0], np.pi / 2, k)

        self.phase_shift(phi / 2, k)

        if not avg:
            return val

        return None

    def phase_shift(self, phi, k):
        r"""Implement a phase shift in mode k.

        Args:
           phi (float): phase
           k (int): mode to be phase shifted
        """
        if self.active[k] is None:
            raise ValueError("Cannot phase shift mode, mode does not exist")

        rot = symp.expand(symp.rotation(phi), k, self.nlen)
        self.means = update_means(self.means, rot, self.from_xp)
        self.covs = update_covs(self.covs, rot, self.from_xp)

    def beamsplitter(self, theta, phi, k, l):
        r"""Implement a beam splitter operation between modes k and l.

        Args:
            theta (float): real beamsplitter angle
            phi (float): complex beamsplitter angle
            k (int): first mode
            l (int): second mode
        """
        if self.active[k] is None or self.active[l] is None:
            raise ValueError("Cannot perform beamsplitter, mode(s) do not exist")

        if k == l:
            raise ValueError("Cannot use the same mode for beamsplitter inputs")

        # Cross-check with Gaussian backend.
        bs = symp.expand(symp.beam_splitter(theta, phi), [k, l], self.nlen)
        self.means = update_means(self.means, bs, self.from_xp)
        self.covs = update_covs(self.covs, bs, self.from_xp)

    def scovmatxp(self):
        r"""Returns the symmetric ordered array of covariance matrices
        in the :math:`q_1,...,q_n,p_1,...,p_n` ordering.
        """
        return self.covs[:, self.to_xp][..., self.to_xp]

    def smeanxp(self):
        r"""Returns the symmetric ordered array of means in the
        :math:`q_1,...,q_n,p_1,...,p_n` ordering.
        """
        return self.means.T[self.to_xp].T

    def scovmat(self):
        r"""Returns the symmetric ordered array of covariance matrices
        in the :math:`q_1,p_1,...,q_n,p_n` ordering.
        """
        return self.covs

    def smean(self):
        r"""Returns the symmetric ordered array of means
        in the :math:`q_1,p_1,...,q_n,p_n` ordering.
        """
        return self.means

    def sweights(self):
        """Returns the array of weights."""
        return self.weights

    def fromsmean(self, r, modes=None):
        r"""Populates the array of means from a provided array of means.

        The input must already have performed the scaling of the means by self.hbar,
        and must be sorted in ascending order.

        Args:
            r (array): vector of means in :math:`(x_1,p_1,x_2,p_2,\dots)` ordering
            modes (Sequence): sequence of modes corresponding to the vector of means
        """
        if modes is None:
            modes = range(self.nlen)

        mode_ind = np.sort(np.append(2 * np.array(modes), 2 * np.array(modes) + 1))
        self.means[:, mode_ind] = r

    def fromscovmat(self, V, modes=None):
        r"""Populates the array of covariance matrices from a provided array of covariance matrices.

        The input must already have performed the scaling of the means by self.hbar,
        and must be sorted in ascending order.

        Args:
            V (array): covariance matrix in :math:`(x_1,p_1,x_2,p_2,\dots)` ordering
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

        mode_ind = np.sort(np.append(2 * np.array(modes), 2 * np.array(modes) + 1))
        # Delete current entries of covariance and any correlations to other modes
        self.covs[:, mode_ind, :] = 0
        self.covs[:, :, mode_ind] = 0
        # Set new covariance elements
        self.covs[np.ix_(np.arange(self.covs.shape[0], dtype=int), mode_ind, mode_ind)] = V

    def fidelity_coherent(self, alpha, modes=None):
        r"""Returns the fidelity to a coherent state.

        Args:
            alpha (array): amplitudes for coherent states
            modes (list or None): modes to use for fidelity calculation
        """
        if modes is None:
            modes = self.get_modes()
        # Sort by (q1,p1,q2,p2,...)
        mode_ind = np.sort(np.append(2 * np.array(modes), 2 * np.array(modes) + 1))
        # Construct mean vector for coherent state
        alpha_mean = np.array([])
        for i in range(len(modes)):
            alpha_mean = np.append(alpha_mean, alpha.real[i] * np.sqrt(2 * self.hbar))
            alpha_mean = np.append(alpha_mean, alpha.imag[i] * np.sqrt(2 * self.hbar))
        # Construct difference of coherent state mean vector with means of all peaks in the state
        deltas = self.means[:, mode_ind] - alpha_mean
        # Construct sum of coherent state covariance matrix and all covariances in the state
        cov_sum = (
            self.covs[:, mode_ind, :][:, :, mode_ind] + self.hbar * np.eye((len(mode_ind))) / 2
        )
        # Sum all Gaussian peaks
        exp_arg = np.einsum("...j,...jk,...k", deltas, np.linalg.inv(cov_sum), deltas)
        weighted_exp = (
            np.array(self.weights)
            * self.hbar ** len(modes)
            * np.exp(-0.5 * exp_arg)
            / np.sqrt(np.linalg.det(cov_sum))
        )
        fidelity = np.sum(weighted_exp)
        return fidelity

    def fidelity_vacuum(self, modes=None):
        r"""Returns the fidelity to the vacuum.

        Args:
            modes (list or None): modes to use for fidelity calculation
        """
        if modes is None:
            modes = self.get_modes()
        alpha = np.zeros(len(modes))
        fidelity = self.fidelity_coherent(alpha, modes=modes)
        return fidelity

    def parity_val(self, modes=None):
        r"""Returns the expectation value of the parity operator, which is the
        value of the Wigner function at the origin.

        Args:
            modes (list or None): modes to use for parity calculation
        """
        if modes is None:
            modes = list(range(self.nlen))
        # Sort by (q1,p1,q2,p2,...)
        mode_ind = np.sort(np.append(2 * np.array(modes), 2 * np.array(modes) + 1))
        exp_arg = np.einsum(
            "...j,...jk,...k",
            self.means[:, mode_ind],
            np.linalg.inv(self.covs[:, mode_ind, :][:, :, mode_ind]),
            self.means[:, mode_ind],
        )
        weighted_exp = (
            np.array(self.weights)
            * np.exp(-0.5 * exp_arg)
            / np.sqrt(np.linalg.det(self.covs[:, mode_ind, :][:, :, mode_ind]))
        )
        parity = np.sum(weighted_exp) * (self.hbar / 2) ** len(modes)
        return parity

    def loss(self, T, k):
        r"""Implements a loss channel in mode k.

        Args:
            T (float between 0 and 1): loss amount is \sqrt{T}
            k (int): mode that loses energy
        """

        if self.active[k] is None:
            raise ValueError("Cannot apply loss channel, mode does not exist")

        X = np.sqrt(T) * np.identity(2)
        Y = self.hbar * (1 - T) * np.identity(2) / 2
        X2, Y2 = self.expandXY([k], X, Y)
        self.apply_channel(X2, Y2)

    def thermal_loss(self, T, nbar, k):
        r"""Implements the thermal loss channel in mode k.

        Args:
            T (float between 0 and 1): loss amount is \sqrt{T}
            nbar (float): mean photon number of the thermal bath
            k (int): mode that undegoes thermal loss
        """
        if self.active[k] is None:
            raise ValueError("Cannot apply loss channel, mode does not exist")
        X = np.sqrt(T) * np.identity(2)
        Y = self.hbar * (1 - T) * (2 * nbar + 1) * np.identity(2) / 2
        X2, Y2 = self.expandXY([k], X, Y)
        self.apply_channel(X2, Y2)

    def init_thermal(self, nbar, mode):
        r"""Initializes a state of mode in a thermal state with the given population.

        Args:
            nbar (float): mean photon number of the thermal state
            mode (int): mode that get initialized
        """
        self.thermal_loss(0.0, nbar, mode)

    def is_vacuum(self, tol=0.0):
        r"""Checks if the state is vacuum by calculating its fidelity with vacuum.

        Args:
            tol (float): the closeness tolerance to fidelity of 1
        """
        fid = self.fidelity_vacuum()
        return np.abs(fid - 1) <= tol

    def measure_dyne(self, covmat, indices, shots=1):
        r"""Performs general-dyne measurements on a set of modes.

        For more information see Quantum Continuous Variables: A Primer of Theoretical Methods
        by Alessio Serafini page 129.

        Args:
            covmat (array): covariance matrix of the generaldyne measurement
            indices (list): modes to be measured
            shots (int): how many measurements are performed

        Returns:
            array: measurement outcome corresponding to a point in phase space
        """
        if covmat.shape != (2 * len(indices), 2 * len(indices)):
            raise ValueError("Covariance matrix size does not match indices provided")

        if np.linalg.det(covmat) < (self.hbar / 2) ** (2 * len(indices)):
            raise ValueError("Measurement covariance matrix is unphysical.")

        if self.covs.imag.any():
            raise NotImplementedError("Covariance matrices must be real")

        for i in indices:
            if self.active[i] is None:
                raise ValueError("Cannot apply measurement, mode does not exist")

        expind = np.concatenate((2 * np.array(indices), 2 * np.array(indices) + 1))
        vals = np.zeros((shots, 2 * len(indices)))
        imag_means_ind = np.where(self.means[:, expind].imag.any(axis=1))[0]
        nonneg_weights_ind = np.where(np.angle(self.weights) != np.pi)[0]
        ub_ind = np.union1d(imag_means_ind, nonneg_weights_ind)
        ub_weights = np.abs(np.array(self.weights))
        if imag_means_ind:
            ub_weights[imag_means_ind] *= np.exp(
                0.5
                * np.einsum(
                    "...j,...jk,...k",
                    (self.means[imag_means_ind, :][:, expind].imag),
                    np.linalg.inv(
                        self.covs[imag_means_ind, :, :][:, expind, :][:, :, expind].real + covmat
                    ),
                    (self.means[imag_means_ind, :][:, expind].imag),
                )
            )
        ub_weights = ub_weights[ub_ind]
        ub_weights_prob = ub_weights / np.sum(ub_weights)

        for i in range(shots):
            drawn = False
            while not drawn:
                peak_ind_sample = np.random.choice(ub_ind, size=1, p=ub_weights_prob)[0]

                cov_meas = self.covs[peak_ind_sample, expind, :][:, expind].real + covmat
                peak_sample = np.random.multivariate_normal(
                    self.means[peak_ind_sample, expind].real, cov_meas
                )

                exp_arg = np.einsum(
                    "...j,...jk,...k",
                    (peak_sample - self.means[:, expind]),
                    np.linalg.inv(self.covs[:, expind, :][:, :, expind].real + covmat),
                    (peak_sample - self.means[:, expind]),
                )
                ub_exp_arg = np.copy(exp_arg)
                if imag_means_ind:
                    ub_exp_arg[imag_means_ind] = np.einsum(
                        "...j,...jk,...k",
                        (peak_sample - self.means[imag_means_ind, :][:, expind].real),
                        np.linalg.inv(
                            self.covs[imag_means_ind, :, :][:, expind, :][:, :, expind].real
                            + covmat
                        ),
                        (peak_sample - self.means[imag_means_ind, :][:, expind].real),
                    )
                prob_dist_val = np.real_if_close(
                    np.sum(
                        (
                            np.array(self.weights)
                            / np.sqrt(
                                np.linalg.det(
                                    2
                                    * np.pi
                                    * (self.covs[:, expind, :][:, :, expind].real + covmat)
                                )
                            )
                        )
                        * np.exp(-0.5 * exp_arg)
                    )
                )
                prob_upbnd = np.real_if_close(
                    np.sum(
                        (
                            ub_weights
                            / np.sqrt(
                                np.linalg.det(
                                    2
                                    * np.pi
                                    * (
                                        self.covs[ub_ind, :, :][:, expind, :][:, :, expind].real
                                        + covmat
                                    )
                                )
                            )
                        )
                        * np.exp(-0.5 * ub_exp_arg[ub_ind])
                    )
                )
                vertical_sample = np.random.random(size=1) * prob_upbnd
                if vertical_sample < prob_dist_val:
                    drawn = True
                    vals[i] = peak_sample
        # The next line is a hack in that it only updates conditioned on the first samples value
        # should still work if shots = 1
        if len(indices) < len(self.active):
            self.post_select_generaldyne(covmat, indices, vals[0])

        # If all modes are measured, set them to vacuum
        if len(indices) == len(self.active):
            for i in indices:
                self.loss(0, i)

        return vals

    def homodyne(self, n, shots=1, eps=0.0002):
        r"""Performs an x-homodyne measurement on a mode, simulated by a generaldyne
        onto a highly squeezed state.

        Args:
            n (int): mode to be measured
            shots (int): how many measurements are performed
            eps (int): squeezing of the measurement state

        Returns:
            array: homodyne outcome
        """
        covmat = self.hbar * np.diag(np.array([eps ** 2, 1.0 / eps ** 2])) / 2
        return self.measure_dyne(covmat, [n], shots=shots)

    def heterodyne(self, n, shots=1):
        r"""Performs a heterodyne measurement on a mode, simulated by a generaldyne
        onto a coherent state.

        Args:
            n (int): mode to be measured
            shots (int): how many measurements are performed

        Returns:
            array: heterodyne outcome
        """
        covmat = self.hbar * np.eye(2) / 2
        return self.measure_dyne(covmat, [n], shots=shots)

    def post_select_generaldyne(self, covmat, indices, vals):
        r"""Simulates general-dyne measurement on a set of modes with a specified measurement
        outcome.

        Args:
            covmat (array): covariance matrix of the generaldyne measurement
            indices (list): modes to be measured
            vals (array): measurement outcome to postselect
        """
        if covmat.shape != (2 * len(indices), 2 * len(indices)):
            raise ValueError("Covariance matrix size does not match indices provided")

        for i in indices:
            if self.active[i] is None:
                raise ValueError("Cannot apply measurement, mode does not exist")

        expind = np.concatenate((2 * np.array(indices), 2 * np.array(indices) + 1))
        mp = self.scovmat()
        (A, B, C) = ops.chop_in_blocks_multi(mp, expind)
        V = A - B @ np.linalg.inv(C + covmat) @ B.transpose(0, 2, 1)
        self.covs = ops.reassemble_multi(V, expind)

        r = self.smean()
        (va, vc) = ops.chop_in_blocks_vector_multi(r, expind)
        va = va + np.einsum("...ij,...j", B @ np.linalg.inv(C + covmat), (vals - vc))
        self.means = ops.reassemble_vector_multi(va, expind)

        reweights_exp_arg = np.einsum(
            "...j,...jk,...k", (vals - vc), np.linalg.inv(C + covmat), (vals - vc)
        )
        reweights = np.exp(-0.5 * reweights_exp_arg) / (
            np.sqrt(np.linalg.det(2 * np.pi * (C + covmat)))
        )
        self.weights *= reweights
        self.weights /= np.sum(self.weights)

        self.means = self.means[abs(self.weights) > 0]
        self.covs = self.covs[abs(self.weights) > 0]
        self.weights = self.weights[abs(self.weights) > 0]

    def post_select_homodyne(self, n, val, eps=0.0002, phi=0):
        r"""Simulates a homodyne measurement on a mode, postselecting on an outcome.

        Args:
            n (int): mode to be measured
            val (array): measurement value to post-select
            eps (int): squeezing of the measurement state
            phi (float): homodyne angle
        """
        if self.active[n] is None:
            raise ValueError("Cannot apply homodyne measurement, mode does not exist")
        self.phase_shift(phi, n)
        covmat = self.hbar * np.diag(np.array([eps ** 2, 1.0 / eps ** 2])) / 2
        indices = [n]
        vals = np.array([val, 0])
        self.post_select_generaldyne(covmat, indices, vals)

    def post_select_heterodyne(self, n, alpha_val):
        r"""Simulates a heterodyne measurement on a mode, postselecting on an outcome.

        Args:
            n (int): mode to be measured
            alpha_val (array): measurement value to post-select
        """
        if self.active[n] is None:
            raise ValueError("Cannot apply heterodyne measurement, mode does not exist")

        covmat = self.hbar * np.identity(2) / 2
        indices = [n]
        vals = np.array([alpha_val.real, alpha_val.imag])
        self.post_select_generaldyne(covmat, indices, vals)

    def apply_u(self, U):
        r"""Transforms the state according to the linear optical unitary that
        maps a[i] \to U[i, j]^*a[j].

        Args:
            U (array): linear opical unitary matrix
        """
        Us = symp.interferometer(U)
        self.means = update_means(self.means, Us, self.from_xp)
        self.covs = update_covs(self.covs, Us, self.from_xp)

    def apply_channel(self, X, Y):
        r"""Transforms the state according to a deterministic Gaussian CPTP map.

        Args:
            X (array): matrix for mutltiplicative part of transformation
            Y (array): matrix for additive part of transformation.
        """
        self.means = update_means(self.means, X, self.from_xp)
        self.covs = update_covs(self.covs, X, self.from_xp, Y)

    def expandS(self, modes, S):
        """Expands symplectic matrix on subset of modes to symplectic matrix for the whole system.

        Args:
            modes (list): list of modes on which S acts
            S (array): symplectic matrix
        """
        return symp.expand(S, modes, self.nlen)

    def expandXY(self, modes, X, Y):
        """Expands deterministic Gaussian CPTP matrices ``(X,Y)`` on subset of modes to
        transformations for the whole system.

        Args:
            modes (list): list of modes on which ``(X,Y)``
            X (array): matrix for mutltiplicative part of transformation
            Y (array): matrix for additive part of transformation.
        """
        X2 = symp.expand(X, modes, self.nlen)
        M = len(Y) // 2
        Y2 = np.zeros((2 * self.nlen, 2 * self.nlen), dtype=Y.dtype)
        w = np.array(modes)

        Y2[w.reshape(-1, 1), w.reshape(1, -1)] = Y[:M, :M].copy()
        Y2[(w + self.nlen).reshape(-1, 1), (w + self.nlen).reshape(1, -1)] = Y[M:, M:].copy()
        Y2[w.reshape(-1, 1), (w + self.nlen).reshape(1, -1)] = Y[:M, M:].copy()
        Y2[(w + self.nlen).reshape(-1, 1), w.reshape(1, -1)] = Y[M:, :M].copy()

        return X2, Y2

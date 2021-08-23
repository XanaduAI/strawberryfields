# Copyright 2021 Xanadu Quantum Technologies Inc.

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
    """Calculate total number of weights. Assumes same number of weights per mode.

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
    num_quad = 2 * num_modes
    return (num_gauss ** num_modes, num_quad, num_quad)


def to_xp(num_modes):
    """Provides an array of indices to order quadratures as all x
    followed by all p i.e., (x1,...,xn,p1,..., pn), starting from the
    (x1,p1,...,xn,pn) ordering.

    Args:
        num_modes (int): number of modes

    Returns:
        array: quadrature ordering for all x followed by all p
    """
    quad_order = np.arange(0, 2 * num_modes, 2)
    return np.concatenate((quad_order, quad_order + 1))


def from_xp(num_modes):
    r"""Provides array of indices to order quadratures as (x1,p1,...,xn,pn)
    starting from all x followed by all p i.e., (x1,...,xn,p1,..., pn).

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


def update_covs(covs, X, perm_out, Y=None):
    r"""Apply a linear transformation parametrized by ``(X,Y)`` to the
    array of covariance matrices. The  quadrature ordering can be specified
    by ``perm_out`` to match ordering of ``(X,Y)`` to the covariance matrices.

    If Y is not specified, it defaults to 0.

    Args:
        covs (array): array of covariance matrices
        X (array): matrix for multiplicative part of transformation
        perm_out (array): indices for quadrature ordering
        Y (array): matrix for additive part of transformation.

    Returns:
        array: transformed array of covariance matrices
    """
    X_perm = X[:, perm_out][perm_out, :]
    if Y is not None:
        Y = Y[:, perm_out][perm_out, :]
    else:
        Y = 0.0
    return (X_perm @ covs @ X_perm.T) + Y


# pylint: disable=too-many-instance-attributes
class BosonicModes:
    """A Bosonic circuit class."""

    # pylint: disable=too-many-public-methods

    def __init__(self, num_subsystems=1, num_weights=1):
        self.hbar = 2
        self.reset(num_subsystems, num_weights)

    def add_mode(self, peak_list=None):
        r"""Add len(peak_list) modes to the circuit. Each mode has a number of
        weights specified by peak_list, and the means and covariances are set
        to vacuum.

        Args:
            peak_list (list): list of weights per mode
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
            modes (int or list): modes to be deleted

        Raises:
            ValueError: if any of the modes are not in the list of active modes
        """
        if isinstance(modes, int):
            modes = [modes]

        for mode in modes:
            if self.active[mode] is None:
                raise ValueError("Cannot delete mode, as the mode does not exist.")

            self.loss(0.0, mode)
            self.active[mode] = None

    def reset(self, num_subsystems=None, num_weights=1):
        """Reset the simulation state.

        Args:
            num_subsystems (int): Sets the number of modes in the reset
                circuit. ``None`` means unchanged.
            num_weights (int): Sets the number of gaussians per mode in the
                superposition.

        Raises:
            ValueError: if num_subsystems or num_weights is not an integer
        """
        if num_subsystems is not None:
            if not isinstance(num_subsystems, int):
                raise ValueError("Number of modes must be an integer.")
            self.nlen = num_subsystems

        if not isinstance(num_weights, int):
            raise ValueError("Number of weights must be an integer.")

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
        r"""Return the modes that are currently active. Active modes
        are those created by the user which have not been deleted.
        If a mode is deleted, its entry in the list is ``None``."""
        return [x for x in self.active if x is not None]

    def displace(self, r, phi, i):
        r"""Displace mode ``i`` by the amount ``r*np.exp(1j*phi)``.

        Args:
            r (float): displacement magnitude
            phi (float): displacement phase
            i (int): mode to be displaced

        Raises:
            ValueError: if the mode is not in the list of active modes
        """
        if self.active[i] is None:
            raise ValueError("Cannot displace mode, as the mode does not exist.")
        self.means += symp.expand_vector(r * np.exp(1j * phi), i, self.nlen)[self.from_xp]

    def squeeze(self, r, phi, k):
        r"""Squeeze mode ``k`` by the amount ``r*exp(1j*phi)``.

        Args:
            r (float): squeezing magnitude
            phi (float): squeezing phase
            k (int): mode to be squeezed

        Raises:
            ValueError: if the mode is not in the list of active modes
        """
        if self.active[k] is None:
            raise ValueError("Cannot squeeze mode, mode does not exist")

        sq = symp.expand(symp.squeezing(r, phi), k, self.nlen)
        self.means = update_means(self.means, sq, self.from_xp)
        self.covs = update_covs(self.covs, sq, self.from_xp)

    def mb_squeeze_avg(self, k, r, phi, r_anc, eta_anc):
        r"""Squeeze mode ``k`` by the amount ``r*exp(1j*phi)`` using measurement-based squeezing.
        This applies the average map, described by a Gaussian CPTP transformation.

        Args:
            k (int): mode to be squeezed
            r (float): target squeezing magnitude
            phi (float): target squeezing phase
            r_anc (float): squeezing magnitude of the ancillary mode
            eta_anc(float): detection efficiency of the ancillary mode

        Raises:
            ValueError: if the mode is not in the list of active modes
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
        self.phase_shift(phi / 2, k)

    def mb_squeeze_single_shot(self, k, r, phi, r_anc, eta_anc):
        r"""Squeeze mode ``k`` by the amount ``r*exp(1j*phi)`` using measurement-based squeezing.
        This applies a single-shot map, returning the ancillary measurement outcome.

        Args:
            k (int): mode to be squeezed
            r (float): target squeezing magnitude
            phi (float): target squeezing phase
            r_anc (float): squeezing magnitude of the ancillary mode
            eta_anc(float): detection efficiency of the ancillary mode

        Returns:
            float: the measurement outcome of the ancilla

        Raises:
            ValueError: if the mode is not in the list of active modes
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

        # Add new ancilla mode, interfere it and measure it
        self.add_mode()
        new_mode = self.nlen - 1
        self.squeeze(r_anc, 0, new_mode)
        self.beamsplitter(theta, 0, k, new_mode)
        self.loss(eta_anc, new_mode)
        self.phase_shift(np.pi / 2, new_mode)
        val = self.homodyne(new_mode)[0][0]

        # Delete all record of ancilla mode
        self.del_mode(new_mode)
        self.covs = np.delete(self.covs, [2 * new_mode, 2 * new_mode + 1], axis=1)
        self.covs = np.delete(self.covs, [2 * new_mode, 2 * new_mode + 1], axis=2)
        self.means = np.delete(self.means, [2 * new_mode, 2 * new_mode + 1], axis=1)
        self.nlen -= 1
        self.from_xp = from_xp(self.nlen)
        self.to_xp = to_xp(self.nlen)
        self.active = self.active[:new_mode]

        # Feedforward displacement
        prefac = -np.tan(theta) / np.sqrt(2 * self.hbar * eta_anc)
        self.displace(prefac * val, np.pi / 2, k)

        self.phase_shift(phi / 2, k)

        return val

    def phase_shift(self, phi, k):
        r"""Implement a phase shift in mode k.

        Args:
           phi (float): phase
           k (int): mode to be phase shifted

        Raises:
            ValueError: if the mode is not in the list of active modes
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

        Raises:
            ValueError: if any of the two modes is not in the list of active modes
            ValueError: if the first mode equals the second mode
        """
        if self.active[k] is None or self.active[l] is None:
            raise ValueError("Cannot perform beamsplitter, mode(s) do not exist")

        if k == l:
            raise ValueError("Cannot use the same mode for beamsplitter inputs.")

        bs = symp.expand(symp.beam_splitter(theta, phi), [k, l], self.nlen)
        self.means = update_means(self.means, bs, self.from_xp)
        self.covs = update_covs(self.covs, bs, self.from_xp)

    def get_covmat_xp(self):
        r"""Returns the symmetric ordered array of covariance matrices
        in the :math:`q_1,...,q_n,p_1,...,p_n` ordering, given that
        they were stored in the :math:`q_1,p_1,...,q_n,p_n` ordering.
        """
        return self.covs[:, self.to_xp][..., self.to_xp]

    def get_mean_xp(self):
        r"""Returns the symmetric ordered array of means in the
        :math:`q_1,...,q_n,p_1,...,p_n` ordering, given that they
        were stored in the :math:`q_1,p_1,...,q_n,p_n` ordering.
        """
        return self.means.T[self.to_xp].T

    def get_covmat(self):
        r"""Returns the symmetric ordered array of covariance matrices
        in the :math:`q_1,p_1,...,q_n,p_n` ordering, just as
        they were stored.
        """
        return self.covs

    def get_mean(self):
        r"""Returns the symmetric ordered array of means
        in the :math:`q_1,p_1,...,q_n,p_n` ordering, just as
        they were stored.
        """
        return self.means

    def get_weights(self):
        """Returns the array of weights."""
        return self.weights

    def from_mean(self, r, modes=None):
        r"""Populates the array of means from a provided array of means.

        The input must already have performed the scaling of the means by self.hbar,
        and must be sorted :math:`(x_1,p_1,x_2,p_2,\dots)` order.

        Args:
            r (array): vector of means in :math:`(x_1,p_1,x_2,p_2,\dots)` ordering
            modes (Sequence): sequence of modes corresponding to the vector of means
        """
        if modes is None:
            modes = range(self.nlen)

        mode_ind = np.sort(np.append(2 * np.array(modes), 2 * np.array(modes) + 1))
        self.means[:, mode_ind] = r

    def from_covmat(self, V, modes=None):
        r"""Populates the array of covariance matrices from a provided array of covariance matrices.

        The input must already have performed the scaling of the covariances by self.hbar,
        and must be sorted in ascending order.

        Args:
            V (array): covariance matrix in :math:`(x_1,p_1,x_2,p_2,\dots)` ordering
            modes (sequence): sequence of modes corresponding to the covariance matrix

        Raises:
            ValueError: if the covariance matrix dimension does not match the
                number of quadratures targeted
        """
        if modes is None:
            n = len(V) // 2
            modes = np.arange(self.nlen)

            if n != self.nlen:
                raise ValueError(
                    "The covariance matrix has an incorrect size, does not match means vector."
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

    def fidelity_coherent(self, alpha, modes=None, tol=1e-15):
        r"""Returns the fidelity to a coherent state.

        Args:
            alpha (array): amplitudes for coherent states
            modes (list): modes to use for fidelity calculation

        Returns:
            float: fidelity of the state in modes to the coherent state alpha
        """
        if modes is None:
            modes = self.get_modes()

        # Shortcut if there are no active modes. Only allowable alpha is of length zero,
        # which is the vacuum, so its fidelity to the state will be 1.
        if len(modes) == 0:
            return 1.0

        # Sort by (q1,p1,q2,p2,...)
        mode_ind = np.sort(np.append(2 * np.array(modes), 2 * np.array(modes) + 1))
        # Construct mean vector for coherent state
        alpha_mean = []
        for i in range(len(modes)):
            alpha_mean.append(alpha.real[i] * np.sqrt(2 * self.hbar))
            alpha_mean.append(alpha.imag[i] * np.sqrt(2 * self.hbar))
        alpha_mean = np.array(alpha_mean)
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

        # Numerical error can yield fidelity marginally greater than 1
        if 1 - fidelity < 0 and fidelity - 1 < tol:
            fidelity = 1
        return fidelity

    def fidelity_vacuum(self, modes=None):
        r"""Returns the fidelity to the vacuum.

        Args:
            modes (list): modes to use for fidelity calculation

        Returns:
            float: fidelity of the state in modes to the vacuum
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
            modes (list): modes to use for parity calculation

        Returns:
            float: parity of the state for the subsystem defined by modes
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
        r"""Implements a loss channel in mode k. T is the loss parameter that must be
        between 0 and 1.

        Args:
            T (float): loss amount is \sqrt{T}
            k (int): mode that loses energy

        Raises:
            ValueError: if the mode is not in the list of active modes
        """

        if self.active[k] is None:
            raise ValueError("Cannot apply loss channel, mode does not exist")
        if (T < 0) or (T > 1):
            raise ValueError("Loss parameter must be between 0 and 1.")

        X = np.sqrt(T) * np.identity(2)
        Y = self.hbar * (1 - T) * np.identity(2) / 2
        X2, Y2 = self.expandXY([k], X, Y)
        self.apply_channel(X2, Y2)

    def thermal_loss(self, T, nbar, k):
        r"""Implements the thermal loss channel in mode k. T is the loss parameter that must
        be between 0 and 1.

        Args:
            T (float): loss amount is \sqrt{T}
            nbar (float): mean photon number of the thermal bath
            k (int): mode that undegoes thermal loss

        Raises:
            ValueError: if the mode is not in the list of active modes
        """
        if self.active[k] is None:
            raise ValueError("Cannot apply loss channel, mode does not exist")
        X = np.sqrt(T) * np.identity(2)
        Y = self.hbar * (1 - T) * (2 * nbar + 1) * np.identity(2) / 2
        X2, Y2 = self.expandXY([k], X, Y)
        self.apply_channel(X2, Y2)

    def init_thermal(self, nbar, mode):
        r"""Initializes a state in a thermal state with the given population.

        Args:
            nbar (float): mean photon number of the thermal state
            mode (int): mode that gets initialized
        """
        self.thermal_loss(0.0, nbar, mode)

    def is_vacuum(self, tol=1e-10):
        r"""Checks if the state is vacuum by calculating its fidelity with vacuum.

        Args:
            tol (float): the closeness tolerance to fidelity of 1

        Return:
            bool: whether the state is vacuum or not
        """
        fid = self.fidelity_vacuum()
        return np.abs(fid - 1) <= tol

    def measure_dyne(self, covmat, modes, shots=1):
        r"""Performs general-dyne measurements on a set of modes.

        For more information on definition of general-dyne see
        Quantum Continuous Variables: A Primer of Theoretical Methods
        by Alessio Serafini page 129.

        TODO: add reference for sampling algorithm once available

        Args:
            covmat (array): covariance matrix of the generaldyne measurement
            modes (list): modes to be measured
            shots (int): how many measurements are performed

        Returns:
            array: measurement outcome corresponding to a point in phase space

        Raises:
            ValueError: if the dimension of covmat does not match number of quadratures
                        associated with modes
            ValueError: if covmat does not respect the uncertainty relation
            NotImplementedError: if any of the covariances of the state are imaginary
            ValueError: if any of the modes are not in the list of active modes
        """
        # pylint: disable=too-many-branches
        if covmat.shape != (2 * len(modes), 2 * len(modes)):
            raise ValueError("Covariance matrix size does not match indices provided.")

        if np.linalg.det(covmat) < (self.hbar / 2) ** (2 * len(modes)):
            raise ValueError("Measurement covariance matrix is unphysical.")

        if self.covs.imag.any():
            raise NotImplementedError("Covariance matrices must be real.")

        for i in modes:
            if self.active[i] is None:
                raise ValueError("Cannot apply measurement, mode does not exist")

        # Indices for the relevant quadratures
        quad_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))
        # Associated means and covs, already adding the covmat to the state covariances
        means_quad = self.means[:, quad_ind]
        covs_quad = self.covs[:, quad_ind, :][:, :, quad_ind].real + covmat
        # Array to be filled with measurement samples
        vals = np.zeros((shots, 2 * len(modes)))

        # Indices of the Gaussians in the linear combination with imaginary means
        imag_means_ind = np.where(means_quad.imag.any(axis=1))[0]
        # Indices of the Gaussians in the linear combination with weights
        # that are not purely negative (i.e. positive or imaginary)
        nonneg_weights_ind = np.where(np.angle(self.weights) != np.pi)[0]
        # Union of the two sets forms the set of indices used to construct the
        # upper bounding function
        ub_ind = np.union1d(imag_means_ind, nonneg_weights_ind)

        # Build weights for the upper bounding function
        # Take absolute value of all weights
        ub_weights = np.abs(np.array(self.weights))
        # If there are terms with complex means, multiply the associated weights
        # by an extra prefactor, which comes from the cross term between the
        # imaginary parts of the means
        if imag_means_ind.size > 0:
            # Get imaginary parts of means
            imag_means = means_quad[imag_means_ind].imag
            # Get associated covariances
            imag_covs = covs_quad[imag_means_ind]
            # Construct prefactor
            imag_exp_arg = np.einsum(
                "...j,...jk,...k",
                imag_means,
                np.linalg.inv(imag_covs),
                imag_means,
            )
            imag_prefactor = np.exp(0.5 * imag_exp_arg)
            # Multiply weights by prefactor
            ub_weights[imag_means_ind] *= imag_prefactor
        # Keep only the weights that are indexed by ub_ind
        ub_weights = ub_weights[ub_ind]
        # To define a probability dsitribution, normalize the set of weights
        ub_weights_prob = ub_weights / np.sum(ub_weights)

        # Perform the rejection sampling technique until the desired number of shots
        # are acquired
        for i in range(shots):
            drawn = False
            while not drawn:
                # Sample an index for a peak from the upperbounding function
                # according to ub_weights_prob
                peak_ind_sample = np.random.choice(ub_ind, size=1, p=ub_weights_prob)[0]
                # Get the associated mean covariance for that peak
                mean_sample = means_quad[peak_ind_sample].real
                cov_sample = covs_quad[peak_ind_sample]
                # Sample a phase space value from the peak
                peak_sample = np.random.multivariate_normal(mean_sample, cov_sample)

                # Differences between the sample and the means
                diff_sample = peak_sample - means_quad
                # Calculate arguments for the Gaussian functions used to calculate
                # the exact probability distribution at the sampled point
                exp_arg = np.einsum(
                    "...j,...jk,...k",
                    (diff_sample),
                    np.linalg.inv(covs_quad),
                    (diff_sample),
                )

                # Make a copy to calculate the exponential arguments of the
                # upper bounding function at the point
                ub_exp_arg = np.copy(exp_arg)
                # If there are complex means, make sure to only use real part of
                # the mean in the upper bounding function
                if imag_means_ind.size > 0:
                    # Difference between the sample and the means of the upper bound
                    diff_sample_ub = peak_sample - means_quad[imag_means_ind, :].real
                    # Replace arguments associated with complex means with real-valued expression
                    ub_exp_arg[imag_means_ind] = np.einsum(
                        "...j,...jk,...k",
                        (diff_sample_ub),
                        np.linalg.inv(imag_covs),
                        (diff_sample_ub),
                    )
                # Keep only terms associated with upper bound indices ub_ind
                ub_exp_arg = ub_exp_arg[ub_ind]

                # Calculate the value of the probability distribution at the sampled point
                # Prefactors for each exponential in the sum
                prefactors = 1 / np.sqrt(np.linalg.det(2 * np.pi * covs_quad))
                # Sum Gaussians
                prob_dist_val = np.sum(self.weights * prefactors * np.exp(-0.5 * exp_arg))
                # Should be real-valued
                prob_dist_val = np.real_if_close(prob_dist_val)

                # Calculate the upper bounding function at the sampled point
                # Sum Gaussians, keeping only prefactors associated with ub_ind
                prob_upbnd = np.sum(ub_weights * prefactors[ub_ind] * np.exp(-0.5 * ub_exp_arg))
                # Should be real-valued
                np.real_if_close(prob_upbnd)

                # Sample point between 0 and upperbound function at the phase space sample
                vertical_sample = np.random.random(size=1) * prob_upbnd
                # Keep or reject phase space sample based on whether vertical_sample falls
                # above or below the value of the probability distribution
                if vertical_sample < prob_dist_val:
                    drawn = True
                    vals[i] = peak_sample

        # The next line is a hack in that it only updates conditioned on the first samples value
        # should still work if shots = 1
        if len(modes) < len(self.active):
            # Update other modes based on phase space sample
            self.post_select_generaldyne(covmat, modes, vals[0])

        # If all modes are measured, set them to vacuum
        if len(modes) == len(self.active):
            for i in modes:
                self.loss(0, i)

        return vals

    def measure_threshold(self, modes):
        r"""Performs photon number measurement on the given modes"""
        if len(modes) == 1:
            if self.active[modes[0]] is None:
                raise ValueError("Cannot apply measurement, mode does not exist")

            Idmat = self.hbar * np.eye(2) / 2
            vacuum_fidelity = np.abs(self.fidelity_vacuum(modes))
            measurement = np.random.choice((0, 1), p=[vacuum_fidelity, 1 - vacuum_fidelity])
            samples = measurement

            # If there are no more modes to measure simply set everything to vacuum
            if len(modes) == len(self.active):
                for mode in modes:
                    self.loss(0, mode)
            # If there are other active modes simply update based on measurement
            else:
                mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))
                sigma_A, sigma_AB, sigma_B = ops.chop_in_blocks_multi(self.covs, mode_ind)
                sigma_A_prime = sigma_A - sigma_AB @ np.linalg.inv(
                    sigma_B + Idmat
                ) @ sigma_AB.transpose(0, 2, 1)
                r_A, r_B = ops.chop_in_blocks_vector_multi(self.means, mode_ind)
                r_A_prime = r_A - np.einsum(
                    "...ij,...j", sigma_AB @ np.linalg.inv(sigma_B + Idmat), r_B
                )

                reweights_exp_arg = np.einsum(
                    "...j,...jk,...k", -r_B, np.linalg.inv(sigma_B + Idmat), -r_B
                )
                reweights = np.exp(-0.5 * reweights_exp_arg) / (
                    np.sqrt(np.linalg.det(2 * np.pi * (sigma_B + Idmat)))
                )

                if measurement == 1:
                    self.means = np.append(
                        ops.reassemble_vector_multi(r_A, mode_ind),
                        ops.reassemble_vector_multi(r_A_prime, mode_ind),
                        axis=0,
                    )
                    self.covs = np.append(
                        ops.reassemble_multi(sigma_A, mode_ind),
                        ops.reassemble_multi(sigma_A_prime, mode_ind),
                        axis=0,
                    )
                    self.weights = np.append(
                        self.weights / (1 - vacuum_fidelity),
                        self.weights * (reweights * 2 * np.pi * self.hbar / (vacuum_fidelity - 1)),
                        axis=0,
                    )
                else:
                    self.post_select_heterodyne(modes[0], 0)
            self.loss(0, modes[0])
            return samples

        raise ValueError("Measure Threshold can only be applied to one mode at a time")

    def homodyne(self, mode, shots=1, eps=0.0002):
        r"""Performs an x-homodyne measurement on a mode, simulated by a generaldyne
        onto a highly squeezed state.

        Args:
            mode (int): mode to be measured
            shots (int): how many measurements are performed
            eps (int): squeezing of the measurement state

        Returns:
            array: homodyne outcome
        """
        covmat = self.hbar * np.diag(np.array([eps ** 2, 1.0 / eps ** 2])) / 2
        return self.measure_dyne(covmat, [mode], shots=shots)

    def heterodyne(self, mode, shots=1):
        r"""Performs a heterodyne measurement on a mode, simulated by a generaldyne
        onto a coherent state.

        Args:
            mode (int): mode to be measured
            shots (int): how many measurements are performed

        Returns:
            array: heterodyne outcome
        """
        covmat = self.hbar * np.eye(2) / 2
        return self.measure_dyne(covmat, [mode], shots=shots)

    def post_select_generaldyne(self, covmat, modes, vals):
        r"""Simulates a general-dyne measurement on a set of modes with a specified measurement
        outcome.

        Args:
            covmat (array): covariance matrix of the generaldyne measurement
            modes (list): modes to be measured
            vals (array): measurement outcome to postselect

        Raises:
            ValueError: if the dimension of covmat does not match the number of quadratures
                        associated with modes
            ValueError: if any of the modes are not in the list of active modes
        """
        if covmat.shape != (2 * len(modes), 2 * len(modes)):
            raise ValueError(
                "The size of the covariance matrix does not match the indices provided."
            )

        for i in modes:
            if self.active[i] is None:
                raise ValueError("Cannot apply measurement, mode does not exist")

        expind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))
        mp = self.get_covmat()
        A, B, C = ops.chop_in_blocks_multi(mp, expind)
        V = A - B @ np.linalg.inv(C + covmat) @ B.transpose(0, 2, 1)
        self.covs = ops.reassemble_multi(V, expind)

        r = self.get_mean()
        (va, vc) = ops.chop_in_blocks_vector_multi(r, expind)
        va = va + np.einsum("...ij,...j", B @ np.linalg.inv(C + covmat), (vals - vc))
        self.means = ops.reassemble_vector_multi(va, expind)

        # Reweight each peak based on how likely a given peak was to have
        # contributed to the observed outcome
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

    def post_select_homodyne(self, mode, val, eps=0.0002, phi=0):
        r"""Simulates a homodyne measurement on a mode, postselecting on an outcome.

        Args:
            mode (int): mode to be measured
            val (array): measurement value to postselect
            eps (int): squeezing of the measurement state
            phi (float): homodyne angle

        Raises:
            ValueError: if the mode are not in the list of active modes
        """
        if self.active[mode] is None:
            raise ValueError("Cannot apply homodyne measurement, mode does not exist.")
        self.phase_shift(phi, mode)
        covmat = self.hbar * np.diag(np.array([eps ** 2, 1.0 / eps ** 2])) / 2
        indices = [mode]
        vals = np.array([val, 0])
        self.post_select_generaldyne(covmat, indices, vals)

    def post_select_heterodyne(self, mode, alpha_val):
        r"""Simulates a heterodyne measurement on a mode, postselecting on an outcome.

        Args:
            mode (int): mode to be measured
            alpha_val (array): measurement value to postselect

        Raises:
            ValueError: if the mode are not in the list of active modes
        """
        if self.active[mode] is None:
            raise ValueError("Cannot apply heterodyne measurement, mode does not exist.")

        covmat = self.hbar * np.identity(2) / 2
        vals = np.array([alpha_val.real, alpha_val.imag])
        self.post_select_generaldyne(covmat, [mode], vals)

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
            X (array): matrix for multiplicative part of transformation
            Y (array): matrix for additive part of transformation
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
            modes (list): list of modes on which ``(X,Y)`` act
            X (array): matrix for mutltiplicative part of transformation
            Y (array): matrix for additive part of transformation
        """
        X2 = symp.expand(X, modes, self.nlen)
        Y2 = symp.expand(Y, modes, self.nlen)
        for i in range(self.nlen):
            if i not in modes:
                Y2[i, i] = 0
                Y2[i + self.nlen, i + self.nlen] = 0

        return X2, Y2

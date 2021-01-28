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
# pylint: disable=too-many-public-methods
"""Bosonic backend"""
import warnings
import numpy as np

from scipy.special import comb
from scipy.linalg import block_diag
from thewalrus.samples import hafnian_sample_state, torontonian_sample_state
import itertools as it

from strawberryfields.backends import BaseBosonic
from strawberryfields.backends.shared_ops import changebasis
from strawberryfields.backends.states import BaseBosonicState

from .bosoniccircuit import BosonicModes
from ..base import NotApplicableError


def to_xp(n):
    """Permutation to quadrature-like (x_1,...x_n, p_1...p_n) ordering.

    Args:
        n (int): number of modes

    Returns:
        list[int]: the permutation of of mode indices.
    """
    return np.concatenate((np.arange(0, 2 * n, 2), np.arange(0, 2 * n, 2) + 1))


def from_xp(n):
    """Permutation to mode-like (x_1,p_1...x_n,p_n) ordering.

    Args:
        n (int): number of modes

    Returns:
        list[int]: the permutation of of mode indices.
    """
    perm_inds_list = [(i, i + n) for i in range(n)]
    perm_inds = [a for tup in perm_inds_list for a in tup]
    return perm_inds


def kron_list(l):
    """Take Kronecker products of a list of lists."""
    if len(l) == 1:
        return l[0]
    return np.kron(l[0], kron_list(l[1:]))


class BosonicBackend(BaseBosonic):
    """Bosonic backend class."""

    short_name = "bosonic"
    circuit_spec = "bosonic"

    def __init__(self):
        """Initialize the backend."""
        super().__init__()
        self._supported["mixed_states"] = True
        self._init_modes = None
        self.circuit = None

    def run_prog(self, prog, batches, **kwargs):

        from strawberryfields.ops import (
            Bosonic,
            Catstate,
            Comb,
            DensityMatrix,
            Fock,
            GKP,
            Ket,
            MbSgate,
        )

        # Initialize the circuit.
        self.init_circuit(prog)

        # Apply operations to circuit. For now, copied from LocalEngine;
        # only change is to ignore preparation classes
        # TODO: Deal with Preparation classes in the middle of a circuit.
        applied = []
        samples_dict = {}
        all_samples = {}
        for cmd in prog.circuit:
            nongausspreps = (Bosonic, Catstate, Comb, DensityMatrix, Fock, GKP, Ket)
            ancilla_gates = (MbSgate,)
            if type(cmd.op) in ancilla_gates:
                try:
                    # try to apply it to the backend and, if op is a measurement, store it in values
                    val = cmd.op.apply(cmd.reg, self, **kwargs)
                    if val is not None:
                        for i, r in enumerate(cmd.reg):
                            if r.ind not in self.ancillae_samples_dict.keys():
                                self.ancillae_samples_dict[r.ind] = []
                            if batches:
                                self.ancillae_samples_dict[r.ind].append(val[:, :, i])
                            else:
                                self.ancillae_samples_dict[r.ind].append(val[:, i])

                    applied.append(cmd)

                except NotApplicableError:
                    # command is not applicable to the current backend type
                    raise NotApplicableError(
                        "The operation {} cannot be used with {}.".format(cmd.op, self.backend)
                    ) from None

                except NotImplementedError:
                    # command not directly supported by backend API
                    raise NotImplementedError(
                        "The operation {} has not been implemented in {} for the arguments {}.".format(
                            cmd.op, self.backend, kwargs
                        )
                    ) from None
            if type(cmd.op) not in (nongausspreps + ancilla_gates):
                try:
                    # try to apply it to the backend and, if op is a measurement, store it in values
                    val = cmd.op.apply(cmd.reg, self, **kwargs)
                    if val is not None:
                        for i, r in enumerate(cmd.reg):
                            if batches:
                                samples_dict[r.ind] = val[:, :, i]

                                # Internally also store all the measurement outcomes
                                if r.ind not in all_samples:
                                    all_samples[r.ind] = list()
                                all_samples[r.ind].append(val[:, :, i])
                            else:
                                samples_dict[r.ind] = val[:, i]

                                # Internally also store all the measurement outcomes
                                if r.ind not in all_samples:
                                    all_samples[r.ind] = list()
                                all_samples[r.ind].append(val[:, i])

                    applied.append(cmd)

                except NotApplicableError:
                    # command is not applicable to the current backend type
                    raise NotApplicableError(
                        "The operation {} cannot be used with {}.".format(cmd.op, self.backend)
                    ) from None

                except NotImplementedError:
                    # command not directly supported by backend API
                    raise NotImplementedError(
                        "The operation {} has not been implemented in {} for the arguments {}.".format(
                            cmd.op, self.backend, kwargs
                        )
                    ) from None

        return applied, samples_dict, all_samples

    def init_circuit(self, prog, **kwargs):
        """Instantiate the circuit and initialize weights, means, and covs
        depending on the Preparation classes."""

        from strawberryfields.ops import (
            Bosonic,
            Catstate,
            Comb,
            DensityMatrix,
            Fock,
            GKP,
            Ket,
        )

        nmodes = prog.num_subsystems
        self.ancillae_samples_dict = {}
        self.circuit = BosonicModes()
        init_weights, init_means, init_covs = [[0] * nmodes for i in range(3)]

        vac_means = np.zeros((1, 2), dtype=complex)  # .tolist()
        vac_covs = np.array([0.5 * self.circuit.hbar * np.identity(2)])

        # List of modes that have been traversed through
        reg_list = []

        # Go through the operations in the circuit
        for cmd in prog.circuit:
            # Check if an operation has already acted on these modes.
            labels = [label.ind for label in cmd.reg]
            isitnew = 1 - np.isin(labels, reg_list)
            if np.any(isitnew):
                # Operation parameters
                pars = cmd.op.p
                for reg in labels:
                    # All the possible preparations should go in this loop
                    if type(cmd.op) == Bosonic:
                        w, m, c = [pars[i].tolist() for i in range(3)]

                    elif type(cmd.op) == Catstate:
                        w, m, c = self.prepare_cat(*pars)

                    elif type(cmd.op) == GKP:
                        w, m, c = self.prepare_gkp(*pars)

                    elif type(cmd.op) == Comb:
                        w, m, c = self.prepare_comb(*pars)

                    elif type(cmd.op) == Fock:
                        w, m, c = self.prepare_fock(*pars)

                    elif type(cmd.op) in (Ket, DensityMatrix):
                        raise Exception("Not yet implemented!")

                    # The rest of the preparations are gaussian.
                    # TODO: initialize with Gaussian |vacuum> state
                    # directly by asking preparation methods below for
                    # the right weights, means, covs.
                    else:
                        w, m, c = np.array([1], dtype=complex), vac_means, vac_covs

                    init_weights[reg] = w
                    init_means[reg] = m
                    init_covs[reg] = c

                reg_list += labels

        # Assume unused modes in the circuit are vacua.
        for i in set(range(nmodes)).difference(reg_list):
            init_weights[i], init_means[i], init_covs[i] = np.array([1]), vac_means, vac_covs

        # Find all possible combinations of means and combs of the
        # Gaussians between the modes.
        mean_combs = it.product(*init_means)
        cov_combs = it.product(*init_covs)

        # Tensor product of the weights.
        weights = kron_list(init_weights)
        # De-nest the means iterator.
        means = np.array([[a for b in tup for a in b] for tup in mean_combs], dtype=complex)
        # Stack covs appropriately.
        covs = np.array([block_diag(*tup) for tup in cov_combs])

        # Declare circuit attributes.
        self.circuit.nlen = nmodes
        self.circuit.to_xp = to_xp(nmodes)
        self.circuit.from_xp = from_xp(nmodes)
        self.circuit.active = list(np.arange(nmodes, dtype=int))

        self.circuit.weights = weights
        self.circuit.means = means
        self.circuit.covs = covs

    def begin_circuit(self, num_subsystems, **kwargs):
        self._init_modes = num_subsystems
        self.circuit = BosonicModes()
        self.circuit.reset(num_subsystems, 1)

    def add_mode(self, n=1):
        self.circuit.add_mode([n])

    def del_mode(self, modes):
        self.circuit.del_mode(modes)

    def get_modes(self):
        return self.circuit.get_modes()

    def reset(self, **kwargs):
        self.circuit.reset(self._init_modes, 1)

    def prepare_thermal_state(self, nbar, mode):
        self.circuit.init_thermal(nbar, mode)

    def prepare_vacuum_state(self, mode):
        self.circuit.loss(0.0, mode)

    def prepare_coherent_state(self, r, phi, mode):
        self.circuit.loss(0.0, mode)
        self.circuit.displace(r, phi, mode)

    def prepare_squeezed_state(self, r, phi, mode):
        self.circuit.loss(0.0, mode)
        self.circuit.squeeze(r, phi, mode)

    def prepare_displaced_squeezed_state(self, r_d, phi_d, r_s, phi_s, mode):
        self.circuit.loss(0.0, mode)
        self.circuit.squeeze(r_s, phi_s, mode)
        self.circuit.displace(r_d, phi_d, mode)

    def prepare_cat(self, alpha, phi, cutoff, desc, D):
        """ Prepares the arrays of weights, means and covs for a cat state"""

        # case alpha = 0 -> prepare vacuum
        if np.isclose(np.absolute(alpha), 0):
            return (
                np.array([1], dtype=complex),
                np.array([[0, 0]], dtype=complex),
                np.array([0.5 * self.circuit.hbar * np.identity(2)]),
            )

        norm = 1 / (2 * (1 + np.exp(-2 * np.absolute(alpha) ** 2) * np.cos(phi)))
        phi = np.pi * phi

        if desc == "complex":
            rplus = np.sqrt(2 * self.circuit.hbar) * np.array([alpha.real, alpha.imag])
            cplx_coef = np.exp(-2 * np.absolute(alpha) ** 2 - 1j * phi)
            rcomplex = np.sqrt(2 * self.circuit.hbar) * np.array(
                [1j * alpha.imag, -1j * alpha.real]
            )
            weights = norm * np.array([1, 1, cplx_coef, np.conjugate(cplx_coef)])
            weights /= np.sum(weights)
            means = np.array([rplus, -rplus, rcomplex, np.conjugate(rcomplex)])
            cov = 0.5 * self.circuit.hbar * np.identity(2, dtype=float)
            cov = np.repeat(cov[None, :], weights.size, axis=0)
            return weights, means, cov

        elif desc == "real":
            # Defining useful constants
            a = np.absolute(alpha)
            phase = np.angle(alpha)
            E = np.pi ** 2 * D * self.circuit.hbar / (16 * a ** 2)
            v = self.circuit.hbar / 2
            num_mean = 8 * a * np.sqrt(self.circuit.hbar) / (np.pi * D * np.sqrt(2))
            denom_mean = 16 * a ** 2 / (np.pi ** 2 * D) + 2
            coef_sigma = np.pi ** 2 * self.circuit.hbar / (8 * a ** 2 * (E + v))
            prefac = (
                np.sqrt(np.pi * self.circuit.hbar)
                * np.exp(0.25 * np.pi ** 2 * D)
                / (4 * a)
                / (np.sqrt(E + v))
            )
            z_max = int(
                np.ceil(
                    2
                    * np.sqrt(2)
                    * a
                    / (np.pi * np.sqrt(self.circuit.hbar))
                    * np.sqrt((-2 * (E + v) * np.log(cutoff / prefac)))
                )
            )

            x_means = np.zeros(4 * z_max + 1, dtype=float)
            p_means = 0.5 * np.array(range(-2 * z_max, 2 * z_max + 1), dtype=float)

            # Creating and calculating the weigths array for oscillating terms
            odd_terms = np.array(range(-2 * z_max, 2 * z_max + 1), dtype=int) % 2
            even_terms = (odd_terms + 1) % 2
            even_phases = (-1) ** ((np.array(range(-2 * z_max, 2 * z_max + 1), dtype=int) % 4) // 2)
            odd_phases = (-1) ** (
                1 + ((np.array(range(-2 * z_max, 2 * z_max + 1), dtype=int) + 2) % 4) // 2
            )
            weights = np.cos(phi) * even_terms * even_phases * np.exp(
                -0.5 * coef_sigma * p_means ** 2
            ) - np.sin(phi) * odd_terms * odd_phases * np.exp(-0.5 * coef_sigma * p_means ** 2)
            weights *= prefac
            weights_real = np.ones(2, dtype=float)
            weights = norm * np.concatenate((weights_real, weights))

            # making sure the state is properly normalized
            weights /= np.sum(weights)

            # computing the means array
            means = np.concatenate(
                (
                    np.reshape(x_means, (-1, 1)),
                    np.reshape(p_means, (-1, 1)),
                ),
                axis=1,
            )
            means *= num_mean / denom_mean
            means_real = np.sqrt(2 * self.circuit.hbar) * np.array([[a, 0], [-a, 0]], dtype=float)
            means = np.concatenate((means_real, means))

            # computing the covariance array
            cov = np.array([[0.5 * self.circuit.hbar, 0], [0, (E * v) / (E + v)]])
            cov = np.repeat(cov[None, :], 4 * z_max + 1, axis=0)
            cov_real = (
                0.5
                * self.circuit.hbar
                * np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=float)
            )
            cov = np.concatenate((cov_real, cov))

            # filter out 0 components
            filt = ~np.isclose(weights, 0, atol=cutoff)
            weights = weights[filt]
            means = means[filt]
            cov = cov[filt]

            # applying a rotation if necessary
            if not np.isclose(phase, 0):
                S = np.array([[np.cos(phase), -np.sin(phase)], [np.sin(phase), np.cos(phase)]])
                means = np.dot(S, means.T).T
                cov = S @ cov @ S.T

            return weights, means, cov

        else:
            raise ValueError('desc accept only "real" or "complex" arguments')

    def prepare_gkp(self, state, epsilon, cutoff, desc="real", shape="square"):
        """ Prepares the arrays of weights, means and covs for a gkp state """

        theta, phi = state[0], state[1]

        if shape == "square":
            if desc == "real":

                def coef(arr):
                    l, m = arr[:, 0], arr[:, 1]
                    t = np.zeros(arr.shape[0], dtype=complex)
                    t += np.logical_and(l % 2 == 0, m % 2 == 0)
                    t += np.logical_and(l % 4 == 0, m % 2 == 1) * (
                        np.cos(0.5 * theta) ** 2 - np.sin(0.5 * theta) ** 2
                    )
                    t += np.logical_and(l % 4 == 2, m % 2 == 1) * (
                        np.sin(0.5 * theta) ** 2 - np.cos(0.5 * theta) ** 2
                    )
                    t += np.logical_and(l % 4 % 2 == 1, m % 4 == 0) * np.sin(theta) * np.cos(phi)
                    t -= np.logical_and(l % 4 % 2 == 1, m % 4 == 2) * np.sin(theta) * np.cos(phi)
                    t -= (
                        np.logical_or(
                            np.logical_and(l % 4 == 3, m % 4 == 3),
                            np.logical_and(l % 4 == 1, m % 4 == 1),
                        )
                        * np.sin(theta)
                        * np.sin(phi)
                    )
                    t += (
                        np.logical_or(
                            np.logical_and(l % 4 == 3, m % 4 == 1),
                            np.logical_and(l % 4 == 1, m % 4 == 3),
                        )
                        * np.sin(theta)
                        * np.sin(phi)
                    )

                    return t * np.exp(
                        -np.pi
                        * 0.25
                        * (l ** 2 + m ** 2)
                        * (1 - np.exp(-2 * epsilon))
                        / (1 + np.exp(-2 * epsilon))
                    )

                z_max = int(
                    np.ceil(
                        np.sqrt(
                            -4
                            / np.pi
                            * np.log(cutoff)
                            * (1 + np.exp(-2 * epsilon))
                            / (1 - np.exp(-2 * epsilon))
                        )
                    )
                )
                damping = 2 * np.exp(-epsilon) / (1 + np.exp(-2 * epsilon))

                means_gen = it.tee(
                    it.starmap(
                        lambda l, m: l + 1j * m, it.product(range(-z_max, z_max + 1), repeat=2)
                    ),
                    2,
                )
                means = np.concatenate(
                    (
                        np.reshape(
                            np.fromiter(means_gen[0], complex, count=(2 * z_max + 1) ** 2), (-1, 1)
                        ).real,
                        np.reshape(
                            np.fromiter(means_gen[1], complex, count=(2 * z_max + 1) ** 2), (-1, 1)
                        ).imag,
                    ),
                    axis=1,
                )

                weights = coef(means)
                filt = abs(weights) > cutoff
                weights = weights[filt]
                means = means[filt]
                weights /= np.sum(weights)
                means *= 0.5 * damping * np.sqrt(np.pi * self.circuit.hbar)
                cov = (
                    0.5
                    * self.circuit.hbar
                    * (1 - np.exp(-2 * epsilon))
                    / (1 + np.exp(-2 * epsilon))
                    * np.identity(2)
                )
                cov = np.repeat(cov[None, :], weights.size, axis=0)

                return weights, means, cov

            elif desc == "complex":
                raise ValueError("The complex description of GKP is not implemented")
        else:
            raise ValueError("Only square GKP are implemented for now")

    def prepare_fock(self, n, r=0.0001):
        """ Prepares the arrays of weights, means and covs of a Fock state"""
        if 1 / r ** 2 < n:
            raise ValueError(
                "The parameter 1 / r ** 2={} is smaller than n={}".format(1 / r ** 2, n)
            )
        # A simple function to calculate the parity
        parity = lambda n: 1 if n % 2 == 0 else -1
        # All the means are zero
        means = np.zeros([n + 1, 2])
        covs = np.array(
            [
                0.5
                * self.circuit.hbar
                * np.identity(2)
                * (1 + (n - j) * r ** 2)
                / (1 - (n - j) * r ** 2)
                for j in range(n + 1)
            ]
        )
        weights = np.array(
            [
                (1 - n * (r ** 2)) / (1 - (n - j) * (r ** 2)) * comb(n, j) * parity(j)
                for j in range(n + 1)
            ]
        )
        weights = weights / np.sum(weights)
        return weights, means, covs

    def prepare_comb(self, n, d, r, cutoff):
        """ Prepares the arrays of weights, means and covs of a squeezed comb state"""
        raise ValueError("Squeezed comb states not implemented")

    def rotation(self, phi, mode):
        self.circuit.phase_shift(phi, mode)

    def displacement(self, r, phi, mode):
        self.circuit.displace(r, phi, mode)

    def squeeze(self, r, phi, mode):
        self.circuit.squeeze(r, phi, mode)

    def mbsqueeze(self, mode, r, phi, r_anc, eta_anc, avg):
        if avg:
            self.circuit.mbsqueeze(mode, r, phi, r_anc, eta_anc, avg)
        if not avg:
            ancilla_val = self.circuit.mbsqueeze(mode, r, phi, r_anc, eta_anc, avg)
            return ancilla_val

    def beamsplitter(self, theta, phi, mode1, mode2):
        self.circuit.beamsplitter(theta, phi, mode1, mode2)

    def gaussian_cptp(self, modes, X, Y):
        if not isinstance(Y, int):
            X2, Y2 = self.circuit.expandXY(modes, X, Y)
            self.circuit.apply_channel(X2, Y2)
        else:
            X2 = self.circuit.expandS(modes, X)
            self.circuit.apply_channel(X, 0)

    def measure_homodyne(self, phi, mode, shots=1, select=None, **kwargs):
        r"""Measure a :ref:`phase space quadrature <homodyne>` of the given mode.

        See :meth:`.BaseBackend.measure_homodyne`.

        Keyword Args:
            eps (float): Homodyne amounts to projection onto a quadrature eigenstate.
                This eigenstate is approximated by a squeezed state whose variance has been
                squeezed to the amount ``eps``, :math:`V_\text{meas} = \texttt{eps}^2`.
                Perfect homodyning is obtained when ``eps`` :math:`\to 0`.

        Returns:
            float: measured value
        """
        if shots != 1:
            if select is not None:
                raise NotImplementedError(
                    "Gaussian backend currently does not support "
                    "postselection if shots != 1 for homodyne measurement"
                )

            raise NotImplementedError(
                "Gaussian backend currently does not support " "shots != 1 for homodyne measurement"
            )

        # phi is the rotation of the measurement operator, hence the minus
        self.circuit.phase_shift(-phi, mode)

        if select is None:
            val = self.circuit.homodyne(mode, **kwargs)[0, 0]
        else:
            val = select * 2 / np.sqrt(2 * self.circuit.hbar)
            self.circuit.post_select_homodyne(mode, val, **kwargs)

        # `qs` will always be a single value since multiple shots is not supported
        return np.array([[val * np.sqrt(2 * self.circuit.hbar) / 2]])

    def measure_heterodyne(self, mode, shots=1, select=None):

        if shots != 1:
            if select is not None:
                raise NotImplementedError(
                    "Gaussian backend currently does not support "
                    "postselection if shots != 1 for heterodyne measurement"
                )

            raise NotImplementedError(
                "Gaussian backend currently does not support "
                "shots != 1 for heterodyne measurement"
            )

        if select is None:
            m = np.identity(2)
            res = 0.5 * self.circuit.measure_dyne(m, [mode], shots=shots)
            return np.array([[res[0, 0] + 1j * res[0, 1]]])

        res = select
        self.circuit.post_select_heterodyne(mode, select)

        # `res` will always be a single value since multiple shots is not supported
        return np.array([[res]])

    def prepare_gaussian_state(self, r, V, modes):
        if isinstance(modes, int):
            modes = [modes]

        # make sure number of modes matches np.shape of r and V
        N = len(modes)
        if len(r) != 2 * N:
            raise ValueError("Length of means vector must be twice the number of modes.")
        if V.shape != (2 * N, 2 * N):
            raise ValueError(
                "Shape of covariance matrix must be [2N, 2N], where N is the number of modes."
            )

        # Include these lines to accomodate out of order modes, e.g.[1,0]
        ordering = np.append(np.argsort(modes), np.argsort(modes) + len(modes))
        V = V[ordering, :][:, ordering]
        r = r[ordering]

        # convert xp-ordering to symmetric ordering
        means = np.vstack([r[:N], r[N:]]).reshape(-1, order="F")
        C = changebasis(N)
        cov = C @ V @ C.T

        self.circuit.fromscovmat(cov, modes)
        self.circuit.fromsmean(means, modes)

    def is_vacuum(self, tol=1e-12, **kwargs):
        fid = self.state().fidelity_vacuum()
        return np.abs(fid - 1) <= tol

    def loss(self, T, mode):
        self.circuit.loss(T, mode)

    def thermal_loss(self, T, nbar, mode):
        self.circuit.thermal_loss(T, nbar, mode)

    def measure_fock(self, modes, shots=1, select=None, **kwargs):
        # if select is not None:
        #     raise NotImplementedError(
        #         "Gaussian backend currently does not support " "postselection"
        #     )
        # if shots != 1:
        #     warnings.warn(
        #         "Cannot simulate non-Gaussian states. "
        #         "Conditional state after Fock measurement has not been updated."
        #     )

        # mu = self.circuit.mean
        # mean = self.circuit.smeanxp()
        # cov = self.circuit.scovmatxp()

        # x_idxs = np.array(modes)
        # p_idxs = x_idxs + len(mu)
        # modes_idxs = np.concatenate([x_idxs, p_idxs])
        # reduced_cov = cov[np.ix_(modes_idxs, modes_idxs)]
        # reduced_mean = mean[modes_idxs]

        # # check we are sampling from a gaussian state with zero mean
        # if np.allclose(mu, np.zeros_like(mu)):
        #     samples = hafnian_sample_state(reduced_cov, shots)
        # else:
        #     samples = hafnian_sample_state(reduced_cov, shots, mean=reduced_mean)

        # return samples

        # TODO
        pass

    def measure_threshold(self, modes, shots=1, select=None, **kwargs):
        # if shots != 1:
        #     if select is not None:
        #         raise NotImplementedError(
        #             "Gaussian backend currently does not support " "postselection"
        #         )
        #     warnings.warn(
        #         "Cannot simulate non-Gaussian states. "
        #         "Conditional state after Threshold measurement has not been updated."
        #     )

        # mu = self.circuit.mean
        # cov = self.circuit.scovmatxp()
        # # check we are sampling from a gaussian state with zero mean
        # if not np.allclose(mu, np.zeros_like(mu)):
        #     raise NotImplementedError(
        #         "Threshold measurement is only supported for " "Gaussian states with zero mean"
        #     )
        # x_idxs = np.array(modes)
        # p_idxs = x_idxs + len(mu)
        # modes_idxs = np.concatenate([x_idxs, p_idxs])
        # reduced_cov = cov[np.ix_(modes_idxs, modes_idxs)]
        # samples = torontonian_sample_state(reduced_cov, shots)

        # return samples

        # TODO
        pass

    def state(self, modes=None, peaks=None, **kwargs):
        """Returns the state of the quantum simulation.

        See :meth:`.BaseBackend.state`.

        Returns:
            BosonicState: state description
        """
        if isinstance(modes, int):
            modes = [modes]

        if modes is None:
            modes = self.get_modes()

        mode_names = ["q[{}]".format(i) for i in modes]

        if len(modes) == 0:
            return BaseBosonicState(
                (np.array([[]]), np.array([[]]), np.array([])), len(modes), 0, mode_names=mode_names
            )

        mode_ind = np.sort(np.append(2 * np.array(modes), 2 * np.array(modes) + 1))

        weights = self.circuit.weights

        # Generate dictionary between tuples of the form (peek_0, ... peek_i)
        # where the subscript denotes the mode, and the corresponding index
        # in the cov object.
        # if peaks is None:
        #     peaks = tuple(np.zeros(len(modes)))
        # g_list = [np.arange(len(w)) for i in range(len(modes))]
        # combs = it.product(*g_list)
        # covs_dict = {tuple: index for (index, tuple) in enumerate(combs)}

        covmats = self.circuit.covs[:, mode_ind, :][:, :, mode_ind]
        means = self.circuit.means[:, mode_ind]

        return BaseBosonicState(
            (means, covmats, weights), len(modes), len(weights), mode_names=mode_names
        )

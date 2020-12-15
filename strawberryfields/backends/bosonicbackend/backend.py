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

        vac_means = np.zeros(2, dtype=complex).tolist()
        vac_covs = np.identity(2, dtype=complex).tolist()

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
                        w, m, c = [1], [vac_means[:]], [vac_covs[:]]

                    init_weights[reg] = w
                    init_means[reg] = m
                    init_covs[reg] = c

                reg_list += labels

        # Assume unused modes in the circuit are vacua.
        for i in set(range(nmodes)).difference(reg_list):
            init_weights[i], init_means[i], init_covs[i] = [1], [vac_means[:]], [vac_covs[:]]

        # Find all possible combinations of means and combs of the
        # Gaussians between the modes.
        mean_combs = it.product(*init_means)
        cov_combs = it.product(*init_covs)

        # Tensor product of the weights.
        weights = kron_list(init_weights)
        # De-nest the means iterator.
        means = np.array([[a for b in tup for a in b] for tup in mean_combs])
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
        self.circuit = BosonicModes(num_subsystems)

    def add_mode(self, n=1):
        self.circuit.add_mode(n)

    def del_mode(self, modes):
        self.circuit.del_mode(modes)

    def get_modes(self):
        return self.circuit.get_modes()

    def reset(self, pure=True, **kwargs):
        self.circuit.reset(self._init_modes)

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

    def prepare_cat(self, alpha, phi, desc):
        """ Prepares the arrays of weights, means and covs for a cat state"""
        return

    def prepare_gkp(self, state, epsilon, cutoff, desc="real", shape="square"):
        """ Prepares the arrays of weights, means and covs for a gkp state """
        return

    def prepare_fock(self, n, r=0.0001):
        """ Prepares the arrays of weights, means and covs of a Fock state"""
        return

    def prepare_comb(self, n, d, r, cutoff):
        """ Prepares the arrays of weights, means and covs of a squeezed comb state"""
        return

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
        self.circuit.beamsplitter(-theta, -phi, mode1, mode2)

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
            qs = self.circuit.homodyne(mode, **kwargs)[0, 0]
        else:
            val = select * 2 / np.sqrt(2 * self.circuit.hbar)
            qs = self.circuit.post_select_homodyne(mode, val, **kwargs)

        # `qs` will always be a single value since multiple shots is not supported
        return np.array([[qs * np.sqrt(2 * self.circuit.hbar) / 2]])

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

        # make sure number of modes matches shape of r and V
        N = len(modes)
        if len(r) != 2 * N:
            raise ValueError("Length of means vector must be twice the number of modes.")
        if V.shape != (2 * N, 2 * N):
            raise ValueError(
                "Shape of covariance matrix must be [2N, 2N], where N is the number of modes."
            )

        # convert xp-ordering to symmetric ordering
        means = np.vstack([r[:N], r[N:]]).reshape(-1, order="F")
        C = changebasis(N)
        cov = C @ V @ C.T

        self.circuit.fromscovmat(cov, modes)
        self.circuit.fromsmean(means, modes)

    def is_vacuum(self, tol=0.0, **kwargs):
        return self.circuit.is_vacuum(tol)

    def loss(self, T, mode):
        self.circuit.loss(T, mode)

    def thermal_loss(self, T, nbar, mode):
        self.circuit.thermal_loss(T, nbar, mode)

    def measure_fock(self, modes, shots=1, select=None, **kwargs):
        if select is not None:
            raise NotImplementedError(
                "Gaussian backend currently does not support " "postselection"
            )
        if shots != 1:
            warnings.warn(
                "Cannot simulate non-Gaussian states. "
                "Conditional state after Fock measurement has not been updated."
            )

        mu = self.circuit.mean
        mean = self.circuit.smeanxp()
        cov = self.circuit.scovmatxp()

        x_idxs = np.array(modes)
        p_idxs = x_idxs + len(mu)
        modes_idxs = np.concatenate([x_idxs, p_idxs])
        reduced_cov = cov[np.ix_(modes_idxs, modes_idxs)]
        reduced_mean = mean[modes_idxs]

        # check we are sampling from a gaussian state with zero mean
        if np.allclose(mu, np.zeros_like(mu)):
            samples = hafnian_sample_state(reduced_cov, shots)
        else:
            samples = hafnian_sample_state(reduced_cov, shots, mean=reduced_mean)

        return samples

    def measure_threshold(self, modes, shots=1, select=None, **kwargs):
        if shots != 1:
            if select is not None:
                raise NotImplementedError(
                    "Gaussian backend currently does not support " "postselection"
                )
            warnings.warn(
                "Cannot simulate non-Gaussian states. "
                "Conditional state after Threshold measurement has not been updated."
            )

        mu = self.circuit.mean
        cov = self.circuit.scovmatxp()
        # check we are sampling from a gaussian state with zero mean
        if not np.allclose(mu, np.zeros_like(mu)):
            raise NotImplementedError(
                "Threshold measurement is only supported for " "Gaussian states with zero mean"
            )
        x_idxs = np.array(modes)
        p_idxs = x_idxs + len(mu)
        modes_idxs = np.concatenate([x_idxs, p_idxs])
        reduced_cov = cov[np.ix_(modes_idxs, modes_idxs)]
        samples = torontonian_sample_state(reduced_cov, shots)

        return samples

    def state(self, modes=None, peaks=None, **kwargs):
        """Returns the state of the quantum simulation.

        See :meth:`.BaseBackend.state`.

        Returns:
            BosonicState: state description
        """
        if modes is None:
            modes = list(range(len(self.get_modes())))

        w = self.circuit.weights

        # Generate dictionary between tuples of the form (peek_0, ... peek_i)
        # where the subscript denotes the mode, and the corresponding index
        # in the cov object.
        # if peaks is None:
        #     peaks = tuple(zeros(len(modes)))
        # g_list = [arange(len(w)) for i in range(len(modes))]
        # combs = it.product(*g_list)
        # covs_dict = {tuple: index for (index, tuple) in enumerate(combs)}

        listmodes = list(np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1)))

        covmat = self.circuit.covs
        means = self.circuit.means
        if len(w) == 1:
            m = covmat[0]
            r = means[0]

            covmat = np.empty((2 * len(modes), 2 * len(modes)))
            means = r[listmodes]

            for i, ii in enumerate(listmodes):
                for j, jj in enumerate(listmodes):
                    covmat[i, j] = m[ii, jj]

            means *= np.sqrt(2 * self.circuit.hbar) / 2
            covmat *= self.circuit.hbar / 2

        mode_names = ["q[{}]".format(i) for i in np.array(self.get_modes())[modes]]
        num_w = int(len(w) ** (1 / len(modes)))
        return BaseBosonicState((means, covmat, w), len(modes), num_w, mode_names=mode_names)

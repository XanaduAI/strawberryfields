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
# pylint: disable=too-many-public-methods
"""Bosonic backend"""
import itertools as it

import numpy as np

from scipy.special import comb
from scipy.linalg import block_diag

from strawberryfields.backends import BaseBosonic
from strawberryfields.backends.shared_ops import changebasis
from strawberryfields.backends.states import BaseBosonicState

from .bosoniccircuit import BosonicModes
from ..base import NotApplicableError


def kron_list(l):
    """Take Kronecker products of a list of lists."""
    if len(l) == 1:
        return l[0]
    return np.kron(l[0], kron_list(l[1:]))


class BosonicBackend(BaseBosonic):
    r"""The BosonicBackend implements a simulation of quantum optical circuits
    in NumPy by representing states as linear combinations of Gaussian functions
    in phase space., returning a :class:`~.BosonicState` state object.

    The primary component of the BosonicBackend is a
    :attr:`~.BosonicModes` object which is used to simulate a multi-mode quantum optical system.
    :class:`~.BosonicBackend` provides the basic API-compatible interface to the simulator, while the
    :attr:`~.BosonicModes` object actually carries out the mathematical simulation.

    The :attr:`BosonicModes` simulators maintain internal sets of weights, means and
    covariance matrices for all the Gaussian functions in the linear combination. Note
    these can be complex-valued quantities.

    ..
        .. currentmodule:: strawberryfields.backends.bosonicbackend
        .. autosummary::
            :toctree: api

            ~bosoniccircuit.BosonicModes
            ~ops
    """

    short_name = "bosonic"
    circuit_spec = None

    def __init__(self):
        """Initialize the backend."""
        super().__init__()
        self._supported["mixed_states"] = True
        self._init_modes = None
        self.circuit = None

    def run_prog(self, prog, **kwargs):
        """Runs a strawberryfields program using the bosonic backend.

        Args:
            prog (object): sf.Program instance

        Returns:
            tuple: list of applied commands,
                    dictionary of measurement samples,
                    dictionary of ancilla measurement samples

        Raises:
            NotApplicableError: if an op in the program does not apply to
                                the bosonic backend
            NotImplementedError: if an op in the program is not implemented
                                 in the bosonic backend
        """

        from strawberryfields.ops import (
            Bosonic,
            Catstate,
            DensityMatrix,
            Fock,
            GKP,
            Ket,
            MbSgate,
        )

        # Initialize the circuit. This applies all non-Gaussian state-prep
        self.init_circuit(prog)

        # Apply operations to circuit. For now, copied from LocalEngine;
        # only change is to ignore preparation classes and ancilla-assisted gates
        # TODO: Deal with Preparation classes in the middle of a circuit.
        applied = []
        samples_dict = {}
        all_samples = {}
        for cmd in prog.circuit:
            nongausspreps = (Bosonic, Catstate, DensityMatrix, Fock, GKP, Ket)
            ancilla_gates = (MbSgate,)
            # For ancilla-assisted gates, if they return measurement values, store
            # them in ancillae_samples_dict
            if type(cmd.op) in ancilla_gates:
                # if the op returns a measurement outcome store it in a dictionary
                val = cmd.op.apply(cmd.reg, self, **kwargs)
                if val is not None:
                    for i, r in enumerate(cmd.reg):
                        if r.ind not in self.ancillae_samples_dict.keys():
                            self.ancillae_samples_dict[r.ind] = [val[:, i]]
                        else:
                            self.ancillae_samples_dict[r.ind].append(val[:, i])

                applied.append(cmd)

            # Rest of operations applied as normal
            if type(cmd.op) not in (nongausspreps + ancilla_gates):
                try:
                    # try to apply it to the backend and, if op is a measurement, store outcome in values
                    val = cmd.op.apply(cmd.reg, self, **kwargs)
                    if val is not None:
                        for i, r in enumerate(cmd.reg):
                            samples_dict[r.ind] = val[:, i]

                            # Internally also store all the measurement outcomes
                            if r.ind not in all_samples:
                                all_samples[r.ind] = list()
                            all_samples[r.ind].append(val[:, i])

                    applied.append(cmd)

                except NotApplicableError:
                    # command is not applicable to the current backend type
                    raise NotApplicableError(
                        "The operation {} cannot be used with the Bosonic Backend.".format(cmd.op)
                    ) from None

                except NotImplementedError:
                    # command not directly supported by backend API
                    raise NotImplementedError(
                        "The operation {} has not been implemented in the Bosonic Backend for the arguments {}.".format(
                            cmd.op, kwargs
                        )
                    ) from None

        return applied, samples_dict, all_samples

    def init_circuit(self, prog, **kwargs):
        """Instantiate the circuit and initialize weights, means, and covs
        depending on the Preparation classes.

        Args:
            prog (object): sf.Program instance

        Raises:
            NotImplementedError: if Ket or DensityMatrix preparation used
        """

        from strawberryfields.ops import (
            Bosonic,
            Catstate,
            DensityMatrix,
            Fock,
            GKP,
            Ket,
        )

        nmodes = prog.num_subsystems
        self.begin_circuit(nmodes)
        self.ancillae_samples_dict = {}
        # Dummy initial weights, means and covs
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
                        weights, means, covs = [pars[i].tolist() for i in range(3)]

                    elif type(cmd.op) == Catstate:
                        weights, means, covs = self.prepare_cat(*pars)

                    elif type(cmd.op) == GKP:
                        weights, means, covs = self.prepare_gkp(*pars)

                    elif type(cmd.op) == Fock:
                        weights, means, covs = self.prepare_fock(*pars)

                    elif type(cmd.op) in (Ket, DensityMatrix):
                        raise NotImplementedError(
                            "Ket and DensityMatrix preparation not implemented in bosonic backend."
                        )

                    # The rest of the preparations are gaussian.
                    # TODO: initialize with Gaussian |vacuum> state
                    # directly by asking preparation methods below for
                    # the right weights, means, covs.
                    else:
                        weights, means, covs = np.array([1], dtype=complex), vac_means, vac_covs

                    init_weights[reg] = weights
                    init_means[reg] = means
                    init_covs[reg] = covs

                reg_list += labels

        # Assume unused modes in the circuit are vacua.
        # If there are any Gaussian state preparations, these will be handled
        # by run_prog
        for i in set(range(nmodes)).difference(reg_list):
            init_weights[i], init_means[i], init_covs[i] = np.array([1]), vac_means, vac_covs

        # Find all possible combinations of means and combs of the
        # Gaussians between the modes.
        mean_combos = it.product(*init_means)
        cov_combos = it.product(*init_covs)

        # Tensor product of the weights.
        weights = kron_list(init_weights)
        # De-nest the means iterator.
        means = np.array([[a for b in tup for a in b] for tup in mean_combos], dtype=complex)
        # Stack covs appropriately.
        covs = np.array([block_diag(*tup) for tup in cov_combos])

        # Declare circuit attributes.
        self.circuit.weights = weights
        self.circuit.means = means
        self.circuit.covs = covs

    def begin_circuit(self, num_subsystems, **kwargs):
        self._init_modes = num_subsystems
        self.circuit = BosonicModes(num_subsystems)

    def add_mode(self, n=1, **kwargs):
        r"""Adds new modes to the circuit each with a number of Gaussian peaks
        specified by peaks.

        Args:
            n (int): number of new modes to add

        Keyword Args:
            peaks (list): number of Gaussian peaks for each new mode

        Raises:
            ValueError: if the length of the list of peaks is different than
                the number of modes
        """
        peaks = kwargs.get("peaks", None)
        if peaks is None:
            peaks = list(np.ones(n, dtype=int))
        if n != len(peaks):
            raise ValueError("Please specify the number of peaks per new mode.")
        self.circuit.add_mode(peaks)

    def del_mode(self, modes):
        self.circuit.del_mode(modes)

    def get_modes(self):
        return self.circuit.get_modes()

    def reset(self, pure=True, **kwargs):
        self.circuit.reset(num_subsystems=self._init_modes, num_weights=1)

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

        # Include these lines to accommodate out of order modes, e.g.[1,0]
        ordering = np.append(np.argsort(modes), np.argsort(modes) + len(modes))
        V = V[ordering, :][:, ordering]
        r = r[ordering]

        # convert xp-ordering to symmetric ordering
        means = np.vstack([r[:N], r[N:]]).reshape(-1, order="F")
        C = changebasis(N)
        cov = C @ V @ C.T

        self.circuit.from_covmat(cov, modes)
        self.circuit.from_mean(means, modes)

    def prepare_cat(self, alpha, phi, cutoff, desc, D):
        r"""Prepares the arrays of weights, means and covs for a cat state:
            ``(|alpha> + exp(i*phi*pi)|-alpha>)/N``.

        Args:
            alpha (float): alpha value of cat state
            phi (float): phi value of cat state
            cutoff (float): if using the 'real' representation, this determines 
                how many terms to keep
            desc (string): whether to use the 'real' or 'complex' representation
            D (float): for 'real rep., quality parameter of approximation

        Returns:
            tuple: arrays of the weights, means and covariances for the state
        """

        # Case alpha = 0, prepare vacuum
        if np.isclose(np.absolute(alpha), 0):
            weights = np.array([1], dtype=complex)
            means = np.array([[0, 0]], dtype=complex)
            covs = np.array([0.5 * self.circuit.hbar * np.identity(2)])
            return (weights, means, covs)

        # Normalization factor
        norm = 1 / (2 * (1 + np.exp(-2 * np.absolute(alpha) ** 2) * np.cos(phi)))
        phi = np.pi * phi
        hbar = self.circuit.hbar

        if desc == "complex":
            # Mean of |alpha><alpha| term
            rplus = np.sqrt(2 * hbar) * np.array([alpha.real, alpha.imag])
            # Mean of |alpha><-alpha| term
            rcomplex = np.sqrt(2 * hbar) * np.array([1j * alpha.imag, -1j * alpha.real])
            # Coefficient for complex Gaussians
            cplx_coef = np.exp(-2 * np.absolute(alpha) ** 2 - 1j * phi)
            # Arrays of weights, means and covs
            weights = norm * np.array([1, 1, cplx_coef, np.conjugate(cplx_coef)])
            weights /= np.sum(weights)
            means = np.array([rplus, -rplus, rcomplex, np.conjugate(rcomplex)])
            covs = 0.5 * hbar * np.identity(2, dtype=float)
            covs = np.repeat(covs[None, :], weights.size, axis=0)
            return weights, means, covs

        elif desc == "real":
            # Defining useful constants
            a = np.absolute(alpha)
            phase = np.angle(alpha)
            E = np.pi ** 2 * D * hbar / (16 * a ** 2)
            v = hbar / 2
            num_mean = 8 * a * np.sqrt(hbar) / (np.pi * D * np.sqrt(2))
            denom_mean = 16 * a ** 2 / (np.pi ** 2 * D) + 2
            coef_sigma = np.pi ** 2 * hbar / (8 * a ** 2 * (E + v))
            prefac = (
                np.sqrt(np.pi * hbar) * np.exp(0.25 * np.pi ** 2 * D) / (4 * a) / (np.sqrt(E + v))
            )
            z_max = int(
                np.ceil(
                    2
                    * np.sqrt(2)
                    * a
                    / (np.pi * np.sqrt(hbar))
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
            means_real = np.sqrt(2 * hbar) * np.array([[a, 0], [-a, 0]], dtype=float)
            means = np.concatenate((means_real, means))

            # computing the covariance array
            cov = np.array([[0.5 * hbar, 0], [0, (E * v) / (E + v)]])
            cov = np.repeat(cov[None, :], 4 * z_max + 1, axis=0)
            cov_real = 0.5 * hbar * np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=float)
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
        """Prepares the arrays of weights, means and covs for a gkp state:
            ``cos(theta/2)|0>_{gkp} + exp(-i*phi)sin(theta/2)|1>_{gkp}``

        Args:
            state (list): [theta,phi] for qubit definition above
            epsilon (float): finite energy parameter of the state
            cutoff (float): if using the 'real' representation, this determines 
                how many terms to keep
            desc (string): 'real' or 'complex' reprsentation
            shape (string): 'square' lattice or otherwise

        Returns:
            tuple: arrays of the weights, means and covariances for the state

        Raises:
            NotImplementedError: if the complex representation or a non-square lattice
                                is attempted
        """

        theta, phi = state[0], state[1]

        if shape == "square":
            if desc == "real":

                def coef(peak_loc):
                    """Returns the value of the weight for a given peak.

                    Args:
                        peak_loc (array): location of the ideal peak in phase space

                    Returns:
                        float: weight of the peak
                    """
                    l, m = peak_loc[:, 0], peak_loc[:, 1]
                    t = np.zeros(peak_loc.shape[0], dtype=complex)
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
                    prefactor = np.exp(
                        -np.pi
                        * 0.25
                        * (l ** 2 + m ** 2)
                        * (1 - np.exp(-2 * epsilon))
                        / (1 + np.exp(-2 * epsilon))
                    )
                    weight = t * prefactor
                    return weight

                # Set the max peak value
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

                # Create set of means before finite energy effects
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

                # Calculate the weights for each peak
                weights = coef(means)
                filt = abs(weights) > cutoff
                weights = weights[filt]
                weights /= np.sum(weights)
                # Apply finite energy effect to means
                means = means[filt]
                means *= 0.5 * damping * np.sqrt(np.pi * self.circuit.hbar)
                # Covariances all the same
                covs = (
                    0.5
                    * self.circuit.hbar
                    * (1 - np.exp(-2 * epsilon))
                    / (1 + np.exp(-2 * epsilon))
                    * np.identity(2)
                )
                covs = np.repeat(covs[None, :], weights.size, axis=0)

                return weights, means, covs

            elif desc == "complex":
                raise NotImplementedError("The complex description of GKP is not implemented")
        else:
            raise ValueError("Only square GKP are implemented for now")

    def prepare_fock(self, n, r=0.05):
        """Prepares the arrays of weights, means and covs of a Fock state.

        Args:
            n (int): photon number
            r (float): quality parameter for the approximation

        Returns:
            tuple: arrays of the weights, means and covariances for the state

        Raises:
            ValueError: if 1/r**2 is less than n
        """
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

    def rotation(self, phi, mode):
        self.circuit.phase_shift(phi, mode)

    def displacement(self, r, phi, mode):
        self.circuit.displace(r, phi, mode)

    def squeeze(self, r, phi, mode):
        self.circuit.squeeze(r, phi, mode)

    def mb_squeeze_avg(self, mode, r, phi, r_anc, eta_anc):
        r"""Squeeze mode by the amount ``r*exp(1j*phi)`` using measurement-based squeezing.
        Here, the average, deterministic Gaussian CPTP map is applied.

        Args:
            k (int): mode to be squeezed
            r (float): target squeezing magnitude
            phi (float): target squeezing phase
            r_anc (float): squeezing magnitude of the ancillary mode
            eta_anc(float): detection efficiency of the ancillary mode
        """
        self.circuit.mb_squeeze_avg(mode, r, phi, r_anc, eta_anc)
    
    def mb_squeeze_single_shot(self, mode, r, phi, r_anc, eta_anc):
        r"""Squeeze mode by the amount ``r*exp(1j*phi)`` using measurement-based squeezing.
        Here, the single-shot map is applied, returning the ancillary measurement outcome.

        Args:
            k (int): mode to be squeezed
            r (float): target squeezing magnitude
            phi (float): target squeezing phase
            r_anc (float): squeezing magnitude of the ancillary mode
            eta_anc(float): detection efficiency of the ancillary mode

        Returns:
            float: the measurement outcome of the ancilla
        """
        ancilla_val = self.circuit.mb_squeeze_single_shot(mode, r, phi, r_anc, eta_anc)
        return ancilla_val

    def beamsplitter(self, theta, phi, mode1, mode2):
        self.circuit.beamsplitter(theta, phi, mode1, mode2)

    def gaussian_cptp(self, modes, X, Y=None):
        r"""Transforms the state according to a deterministic Gaussian CPTP map.

        Args:
            modes (list): list of modes on which ``(X,Y)`` act
            X (array): matrix for multiplicative part of transformation
            Y (array): matrix for additive part of transformation
        """
        if isinstance(modes, int):
            modes = [modes]
        if Y is not None:
            X2, Y2 = self.circuit.expandXY(modes, X, Y)
            self.circuit.apply_channel(X2, Y2)
        else:
            X2 = self.circuit.expandS(modes, X)
            self.circuit.apply_channel(X, Y)

    def measure_homodyne(self, phi, mode, shots=1, select=None, **kwargs):
        # Phi is the rotation of the measurement operator, hence the minus
        self.circuit.phase_shift(-phi, mode)

        if select is None:
            val = self.circuit.homodyne(mode, **kwargs)[0, 0]
        else:
            val = select * 2 / np.sqrt(2 * self.circuit.hbar)
            self.circuit.post_select_homodyne(mode, val, **kwargs)

        return np.array([[val * np.sqrt(2 * self.circuit.hbar) / 2]])

    def measure_heterodyne(self, mode, shots=1, select=None):
        if select is None:
            res = 0.5 * self.circuit.heterodyne(mode, shots=shots)
            return np.array([[res[:, 0] + 1j * res[:, 1]]])

        res = select
        self.circuit.post_select_heterodyne(mode, select)
        return res

    def is_vacuum(self, tol=1e-10, **kwargs):
        return self.circuit.is_vacuum(tol=tol)

    def loss(self, T, mode):
        self.circuit.loss(T, mode)

    def thermal_loss(self, T, nbar, mode):
        self.circuit.thermal_loss(T, nbar, mode)

    def measure_fock(self, modes, shots=1, select=None, **kwargs):
        raise NotImplementedError("Bosonic backend does not yet support Fock measurements.")

    def measure_threshold(self, modes, shots=1, select=None, **kwargs):
        raise NotImplementedError("Bosonic backend does not yet support threshold measurements.")

    def state(self, modes=None, **kwargs):
        """Returns the state of the quantum simulation.

        See :meth:`.BaseBackend.state`.

        For the bosonic backend, mode indices are sorted in ascending order.

        Returns:
            :class:`~.BosonicState`: object containing all state information
        """
        if isinstance(modes, int):
            modes = [modes]

        if modes is None:
            modes = self.get_modes()

        mode_names = ["q[{}]".format(i) for i in modes]

        # This next check is a hack if the user has deleted all the modes
        if len(modes) == 0:
            return BaseBosonicState(
                (np.array([[]]), np.array([[]]), np.array([])), len(modes), 0, mode_names=mode_names
            )

        mode_ind = np.sort(np.append(2 * np.array(modes), 2 * np.array(modes) + 1))

        weights = self.circuit.weights
        covmats = self.circuit.covs[:, mode_ind, :][:, :, mode_ind]
        means = self.circuit.means[:, mode_ind]

        return BaseBosonicState(
            (means, covmats, weights), len(modes), len(weights), mode_names=mode_names
        )

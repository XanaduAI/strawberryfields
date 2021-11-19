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
from functools import reduce
from collections.abc import Iterable

import numpy as np

from scipy.special import comb
from scipy.linalg import block_diag

from thewalrus.symplectic import xxpp_to_xpxp

from strawberryfields.backends import BaseBosonic
from strawberryfields.backends.states import BaseBosonicState

from strawberryfields.backends.bosonicbackend.bosoniccircuit import BosonicModes
from strawberryfields.backends.base import NotApplicableError
from strawberryfields.program_utils import CircuitError
import sympy


def kron_list(l):
    """Take Kronecker products of a list of lists."""
    return reduce(np.kron, l)


def parameter_checker(parameters):
    """Checks if any items in an iterable are sympy objects."""
    for item in parameters:
        if isinstance(item, sympy.Expr):
            return True

        # This checks all the nested items if item is an iterable
        if isinstance(item, Iterable) and not isinstance(item, str):
            if parameter_checker(item):
                return True
    return False


# pylint: disable=abstract-method
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
        self.ancillae_samples_dict = {}

    # pylint: disable=too-many-branches
    # pylint: disable=import-outside-toplevel
    def run_prog(self, prog, **kwargs):
        """Runs a strawberryfields program using the bosonic backend.

        Args:
            prog (object): sf.Program instance

        Returns:
            tuple: a tuple of the list of applied commands and the dictionary of measurement samples

        Raises:
            NotApplicableError: if an op in the program does not apply to the bosonic backend
            NotImplementedError: if an op in the program is not implemented in the bosonic backend
        """
        from strawberryfields.ops import (
            Bosonic,
            Catstate,
            DensityMatrix,
            Fock,
            GKP,
            Ket,
            MSgate,
            _New_modes,
        )

        # If a circuit exists, initialize the circuit. This applies all non-Gaussian state-prep
        if prog.circuit:
            self.init_circuit(prog)

        # Apply operations to circuit. For now, copied from LocalEngine;
        # only change is to ignore preparation classes and ancilla-assisted gates
        # TODO: Deal with Preparation classes in the middle of a circuit.
        applied = []
        samples_dict = {}

        non_gauss_preps = (
            Bosonic,
            Catstate,
            DensityMatrix,
            Fock,
            GKP,
            Ket,
            _New_modes,
        )
        ancilla_gates = (MSgate,)
        for cmd in prog.circuit:
            # For ancilla-assisted gates, if they return measurement values, store
            # them in ancillae_samples_dict
            if isinstance(cmd.op, ancilla_gates):
                # if the op returns a measurement outcome store it in a dictionary
                val = cmd.op.apply(cmd.reg, self, **kwargs)
                if val is not None:
                    for i, r in enumerate(cmd.reg):
                        if r.ind not in self.ancillae_samples_dict.keys():
                            self.ancillae_samples_dict[r.ind] = [val]
                        else:
                            self.ancillae_samples_dict[r.ind].append(val)

                applied.append(cmd)

            # Rest of operations applied as normal
            elif not isinstance(cmd.op, non_gauss_preps):
                try:
                    # try to apply it to the backend and if op is a measurement, store outcome in values
                    val = cmd.op.apply(cmd.reg, self, **kwargs)
                    if val is not None:
                        for i, r in enumerate(cmd.reg):
                            # Internally also store all the measurement outcomes
                            if r.ind not in samples_dict:
                                samples_dict[r.ind] = []
                            samples_dict[r.ind].append(val[:, i])

                    applied.append(cmd)

                except NotApplicableError as e:
                    # command is not applicable to the current backend type
                    raise NotApplicableError(
                        "The operation {} cannot be used with the Bosonic Backend.".format(cmd.op)
                    ) from e

                except NotImplementedError as e:
                    # command not directly supported by backend API
                    raise NotImplementedError(
                        "The operation {} has not been implemented in the Bosonic Backend for the arguments {}.".format(
                            cmd.op, kwargs
                        )
                    ) from e
        return applied, samples_dict

    # pylint: disable=import-outside-toplevel
    def init_circuit(self, prog):
        """Instantiate the circuit and initialize weights, means, and covs
        depending on the ``Preparation`` classes.

        Args:
            prog (object): :class:`~.Program` instance

        Raises:
            NotImplementedError: if ``Ket`` or ``DensityMatrix`` preparation used
            CircuitError: if any of the parameters for non-Gaussian state preparation
                are symbolic
        """
        from strawberryfields.ops import (
            Bosonic,
            Catstate,
            DensityMatrix,
            Fock,
            GKP,
            Ket,
            _New_modes,
        )

        # _New_modes is what gets checked when New() is called in a program circuit.
        # It is included here since it could be used to instantiate a mode for non-Gaussian
        # state preparation, and it's best to initialize any new modes from the outset.
        non_gauss_preps = (Bosonic, Catstate, DensityMatrix, Fock, GKP, Ket, _New_modes)
        nmodes = prog.init_num_subsystems
        self.begin_circuit(nmodes)
        # Dummy initial weights, means and covs
        init_weights, init_means, init_covs = [[0] * nmodes for _ in range(3)]

        vac_means = np.zeros((1, 2), dtype=complex)
        vac_covs = np.expand_dims(0.5 * self.circuit.hbar * np.identity(2), axis=0)

        # List of modes that have been traversed through
        reg_list = []

        # Go through the operations in the circuit
        for cmd in prog.circuit:
            # Check if an operation other than New() has already acted on these modes.
            labels = [label.ind for label in cmd.reg]
            isitnew = 1 - np.isin(labels, reg_list)
            new_labels = np.asarray(labels)[np.logical_not(np.isin(labels, reg_list))]
            if np.any(isitnew):
                # Operation parameters
                pars = cmd.op.p
                # Check if any of the preparations rely on symbolic quantities
                if isinstance(cmd.op, non_gauss_preps) and parameter_checker(pars):
                    raise CircuitError(
                        "Symbolic non-Gaussian preparations have not been implemented "
                        "in the bosonic backend."
                    )
                for reg in new_labels:
                    # All the possible preparations should go in this loop
                    if isinstance(cmd.op, Bosonic):
                        weights, means, covs = [pars[i] for i in range(3)]

                    elif isinstance(cmd.op, Catstate):
                        weights, means, covs = self.prepare_cat(*pars)

                    elif isinstance(cmd.op, GKP):
                        weights, means, covs = self.prepare_gkp(*pars)

                    elif isinstance(cmd.op, Fock):
                        weights, means, covs = self.prepare_fock(*pars)

                    elif isinstance(cmd.op, (Ket, DensityMatrix)):
                        raise NotImplementedError(
                            "Ket and DensityMatrix preparation not implemented in the bosonic backend."
                        )

                    # If a new mode is added in the program context, then add it here
                    elif isinstance(cmd.op, _New_modes):
                        cmd.op.apply(cmd.reg, self)
                        init_weights.append([0])
                        init_means.append([0])
                        init_covs.append([0])

                    # The rest of the preparations are gaussian.
                    # TODO: initialize with Gaussian |vacuum> state
                    # directly by asking preparation methods below for
                    # the right weights, means, covs.
                    else:
                        weights, means, covs = np.array([1], dtype=complex), vac_means, vac_covs

                    init_weights[reg] = weights
                    init_means[reg] = means
                    init_covs[reg] = covs

                # Add the mode to the list of already prepared modes, unless the command was
                # just to create the new mode, in which case it checks again to see if there is
                # a subsequent non-Gaussian state creation
                if not isinstance(cmd.op, _New_modes):
                    reg_list += labels

            else:
                if type(cmd.op) in non_gauss_preps:
                    raise NotImplementedError(
                        "Non-gaussian state preparations must be the first operation for each register."
                    )

        # Assume unused modes in the circuit are vacuum states.
        # If there are any Gaussian state preparations, these will be handled
        # by the run_prog method
        for i in set(range(nmodes)).difference(reg_list):
            init_weights[i], init_means[i], init_covs[i] = np.array([1]), vac_means, vac_covs

        # Find all possible combinations of means and combs of the
        # Gaussians between the modes.
        mean_combos = it.product(*init_means)
        cov_combos = it.product(*init_covs)

        # Tensor product of the weights.
        tensored_weights = kron_list(init_weights)
        # De-nest the means iterator.
        tensored_means = np.array([np.concatenate(tup) for tup in mean_combos], dtype=complex)
        # Stack covs appropriately.
        tensored_covs = np.array([block_diag(*tup) for tup in cov_combos])

        # Declare circuit attributes.
        self.circuit.weights = tensored_weights
        self.circuit.means = tensored_means
        self.circuit.covs = tensored_covs

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
        cov = xxpp_to_xpxp(V)

        self.circuit.from_covmat(cov, modes)
        self.circuit.from_mean(means, modes)

    def prepare_cat(self, a, theta, p, representation, ampl_cutoff, D):
        r"""Prepares the arrays of weights, means and covs for a cat state:

        :math:`\ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{i\phi} \ket{-\alpha})`,

        where :math:`\alpha = ae^{i\theta}`.

        Args:
            a (float): displacement magnitude :math:`|\alpha|`
            theta (float): displacement angle :math:`\theta`
            p (float): Parity, where :math:`\phi=p\pi`. ``p=0`` corresponds to an even
                cat state, and ``p=1`` an odd cat state.
            representation (str): whether to use the ``'real'`` or ``'complex'`` representation
            ampl_cutoff (float): if using the ``'real'`` representation, this determines
                 how many terms to keep
            D (float): for ``'real'`` representation, quality parameter of approximation

        Returns:
            tuple: arrays of the weights, means and covariances for the state
        """

        phi = np.pi * p

        if representation not in ("complex", "real"):
            raise ValueError(
                'The representation argument accepts only "real" or "complex" as arguments.'
            )

        # Case alpha = 0, prepare vacuum
        if np.isclose(a, 0):
            weights = np.array([1], dtype=complex)
            means = np.array([[0, 0]], dtype=complex)
            covs = np.array([0.5 * self.circuit.hbar * np.identity(2)])
            return weights, means, covs

        # Normalization factor
        norm = 1 / (2 * (1 + np.exp(-2 * a ** 2) * np.cos(phi)))
        hbar = self.circuit.hbar

        if representation == "complex":

            alpha = a * np.exp(1j * theta)

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

        # The only remaining option is a real-valued cat state
        return self.prepare_cat_real_rep(a, theta, p, ampl_cutoff, D)

    def prepare_cat_real_rep(self, a, theta, p, ampl_cutoff, D):
        r"""Prepares the arrays of weights, means and covs for a cat state:

        :math:`\ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{i\phi\pi} \ket{-\alpha})`,

        where :math:`\alpha = ae^{i\theta}`.

        For this representation, weights, means and covariances are real-valued.

        Args:
            a (float): displacement magnitude :math:`|\alpha|`
            theta (float): displacement angle :math:`\theta`
            p (float): Parity, where :math:`\phi=p\pi`. ``p=0`` corresponds to an even
                cat state, and ``p=1`` an odd cat state.
            ampl_cutoff (float): this determines how many terms to keep
            D (float): quality parameter of approximation

        Returns:
            tuple: arrays of the weights, means and covariances for the state
        """

        # Normalization factor
        phi = np.pi * p
        norm = 1 / (2 * (1 + np.exp(-2 * a ** 2) * np.cos(phi)))
        hbar = self.circuit.hbar

        # Defining useful constants
        E = np.pi ** 2 * D * hbar / (16 * a ** 2)
        v = hbar / 2
        num_mean = 8 * a * np.sqrt(hbar) / (np.pi * D * np.sqrt(2))
        denom_mean = 16 * a ** 2 / (np.pi ** 2 * D) + 2
        coef_sigma = np.pi ** 2 * hbar / (8 * a ** 2 * (E + v))
        prefac = np.sqrt(np.pi * hbar) * np.exp(0.25 * np.pi ** 2 * D) / (4 * a) / (np.sqrt(E + v))
        z_max = int(
            np.ceil(
                2
                * np.sqrt(2)
                * a
                / (np.pi * np.sqrt(hbar))
                * np.sqrt((-2 * (E + v) * np.log(ampl_cutoff / prefac)))
            )
        )

        x_means = np.zeros(4 * z_max + 1, dtype=float)
        p_means = 0.5 * np.array(range(-2 * z_max, 2 * z_max + 1), dtype=float)

        # Creating and calculating the weights array for oscillating terms
        term_inds = np.array(range(-2 * z_max, 2 * z_max + 1), dtype=int)
        odd_terms = term_inds % 2
        even_terms = (odd_terms + 1) % 2
        even_phases = (-1) ** ((term_inds % 4) // 2)
        odd_phases = (-1) ** (1 + ((term_inds + 2) % 4) // 2)
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
        filt = ~np.isclose(weights, 0, atol=ampl_cutoff)
        weights = weights[filt]
        means = means[filt]
        cov = cov[filt]

        # applying a rotation if necessary
        if not np.isclose(theta, 0):
            S = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            means = np.dot(S, means.T).T
            cov = S @ cov @ S.T

        return weights, means, cov

    def prepare_gkp(self, state, epsilon, ampl_cutoff, representation="real", shape="square"):
        r"""Prepares the arrays of weights, means and covs for a finite energy GKP state.

        GKP states are qubits, with the qubit state defined by:

        :math:`\ket{\psi}_{gkp} = \cos\frac{\theta}{2}\ket{0}_{gkp} + e^{-i\phi}\sin\frac{\theta}{2}\ket{1}_{gkp}`

        where the computational basis states are :math:`\ket{\mu}_{gkp} = \sum_{n} \ket{(2n+\mu)\sqrt{\pi\hbar}}_{q}`.

        Args:
            state (list): ``[theta,phi]`` for qubit definition above
            epsilon (float): finite energy parameter of the state
            ampl_cutoff (float): this determines how many terms to keep
            representation (str): ``'real'`` or ``'complex'`` reprsentation
            shape (str): shape of the lattice; default 'square'

        Returns:
            tuple: arrays of the weights, means and covariances for the state

        Raises:
            NotImplementedError: if the complex representation or a non-square lattice is attempted
        """

        if representation == "complex":
            raise NotImplementedError("The complex description of GKP is not implemented")

        if shape != "square":
            raise NotImplementedError("Only square GKP are implemented for now")

        theta, phi = state[0], state[1]

        def coeff(peak_loc):
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
                    * np.log(ampl_cutoff)
                    * (1 + np.exp(-2 * epsilon))
                    / (1 - np.exp(-2 * epsilon))
                )
            )
        )
        damping = 2 * np.exp(-epsilon) / (1 + np.exp(-2 * epsilon))

        # Create set of means before finite energy effects
        means_gen = it.tee(
            it.starmap(lambda l, m: l + 1j * m, it.product(range(-z_max, z_max + 1), repeat=2)),
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
        weights = coeff(means)
        filt = abs(weights) > ampl_cutoff
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

    def prepare_fock(self, n, r=0.05):
        """Prepares the arrays of weights, means and covs of a Fock state.

        Args:
            n (int): photon number
            r (float): quality parameter for the approximation

        Returns:
            tuple: arrays of the weights, means and covariances for the state

        Raises:
            ValueError: if :math:`1/r^2` is less than :math:`n`
        """
        if 1 / r ** 2 < n:
            raise ValueError(f"The parameter 1 / r ** 2={1 / r ** 2} is smaller than n={n}")
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
            ],
            dtype=complex,
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
        r"""Squeeze mode by the amount :math:`re^{i\phi}` using measurement-based squeezing.

        Here, the average, deterministic Gaussian CPTP map is applied.

        Args:
            mode (int): mode to be squeezed
            r (float): target squeezing magnitude
            phi (float): target squeezing phase
            r_anc (float): squeezing magnitude of the ancillary mode
            eta_anc (float): detection efficiency of the ancillary mode
        """
        self.circuit.mb_squeeze_avg(mode, r, phi, r_anc, eta_anc)

    def mb_squeeze_single_shot(self, mode, r, phi, r_anc, eta_anc):
        r"""Squeeze mode by the amount :math:`re^{i\phi}` using measurement-based squeezing.

        Here, the single-shot map is applied, returning the ancillary measurement outcome.

        Args:
            mode (int): mode to be squeezed
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
            val = self.circuit.homodyne(mode, shots=shots, **kwargs)[:, 0]
        else:
            val = select * 2 / np.sqrt(2 * self.circuit.hbar)
            self.circuit.post_select_homodyne(mode, val)
            val = np.array([val])

        return np.array([val]).T * np.sqrt(2 * self.circuit.hbar) / 2

    def measure_heterodyne(self, mode, shots=1, select=None):
        if select is None:
            res = 0.5 * self.circuit.heterodyne(mode, shots=shots)
            return np.array([res[:, 0] + 1j * res[:, 1]]).T

        res = select
        self.circuit.post_select_heterodyne(mode, select)
        return np.array([[res]])

    def is_vacuum(self, tol=1e-10, **kwargs):
        return self.circuit.is_vacuum(tol=tol)

    def loss(self, T, mode):
        self.circuit.loss(T, mode)

    def thermal_loss(self, T, nbar, mode):
        self.circuit.thermal_loss(T, nbar, mode)

    def measure_fock(self, modes, shots=1, select=None, **kwargs):
        raise NotImplementedError("Bosonic backend does not yet support Fock measurements.")

    def measure_threshold(self, modes, shots=1, select=None, **kwargs):
        if select is not None:
            raise NotImplementedError("Bosonic backend currently does not support " "postselection")
        if shots != 1:
            raise NotImplementedError(
                "Bosonic backend currently does not support " "multiple shots"
            )

        samples = []
        for mode in modes:
            samples.append(self.circuit.measure_threshold([mode]))
        return np.array([samples])

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

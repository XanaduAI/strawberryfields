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

"""
Fock backend proper
======================

Contains most of the code for managing the simulator state and offloading
operations to the utilities in ops.

Hyperlinks: :class:`QReg`

.. currentmodule:: strawberryfields.backends.fockbackend.circuit

Contents
----------------------
.. autosummary::
     QReg

"""
# pylint: disable=too-many-arguments,len-as-condition,attribute-defined-outside-init
# pylint: disable=too-many-branches,too-many-locals,too-many-public-methods

import copy
import string
import numbers
from itertools import product

import numpy as np
from numpy import sqrt, pi
from scipy.special import factorial as bang

from . import ops

indices = string.ascii_lowercase
MAX_MODES = len(indices) - 3
def_mode = 'blas'

class QReg():
    """
    Class implementing a basic simulator for a collection of modes
    in the fock basis.
    """

    def __init__(self, num, trunc, hbar=2, pure=True, do_checks=False, mode=def_mode):
        r"""Class initializer.

        Args:
            num (non-negative int): Number of modes in the register.
            trunc (positive int): Truncation parameter. Modes with up to trunc-1 modes
                                                        are representable
            hbar (int): The value of :math:`\hbar` to initialise the circuit with, depending on the conventions followed.
                By default, :math:`\hbar=2`. See :ref:`conventions` for more details.
            pure (bool, optional): Whether states are pure (True) or mixed (False)
            do_checks (bool, optional): Whether arguments are to be checked first
        """

        # Check validity
        if num < 0:
            raise ValueError("Number of modes must be non-negative -- got {}".format(num))
        elif num > MAX_MODES:
            raise ValueError("Fock simulator has a maximum of {} modes".format(MAX_MODES))
        elif trunc <= 0:
            raise ValueError("Truncation must be positive -- got {}".format(trunc))

        self._num_modes = num
        self._trunc = trunc
        self._hbar = hbar
        self._checks = do_checks
        self._pure = pure
        self._mode = mode
        self.reset(None)

    def _apply_gate(self, mat, modes):
        """Master gate application function. Selects between implementations based
        on the `_mode` class parameter.

        Args:
            mat (array): The matrix to apply
            modes (list<non-negative int>): The modes to apply `mat` to
        """

        args = [mat, self._state, self._pure, modes, self._num_modes, self._trunc]
        if self._mode == 'blas':
            self._state = ops.apply_gate_BLAS(*args)
        elif self._mode == 'einsum':
            self._state = ops.apply_gate_einsum(*args)
        else:
            raise NotImplementedError

    def _apply_channel(self, kraus_ops, modes):
        """Master channel application function. Applies a channel represented by
        Kraus operators.

        .. note::
                Always results in a mixed state.

        Args:
            kraus_ops (list<array>): A list of Kraus operators
            modes (list<non-negative int>): The modes to apply the channel to
        """

        if self._pure:
            self._state = ops.mix(self._state, self._num_modes)
            self._pure = False

        if len(kraus_ops) == 0:
            self._state = np.zeros([self._trunc for i in range(self._num_modes*2)], dtype=ops.def_type)
        elif self._mode == 'blas':
            states = [ops.apply_gate_einsum(k, np.copy(self._state), False, modes, self._num_modes, self._trunc)\
                                    for k in kraus_ops]
            self._state = sum(states)
        elif self._mode == 'einsum':
            states = [ops.apply_gate_einsum(k, self._state, False, modes, self._num_modes, self._trunc)\
                                    for k in kraus_ops]
            self._state = sum(states)


    def reset(self, pure=None, num_subsystems=None):
        """Resets the simulation state.

        Args:
            pure (bool, optional): Sets the purity setting. Default is unchanged.
            num_subsystems (int, optional): Sets the number of modes in the reset
                circuit. Default is unchanged.
        """
        if pure is not None:
            self._pure = pure

        if num_subsystems is not None:
            self._num_modes = num_subsystems

        if self._pure:
            self._state = ops.vacuumState(self._num_modes, self._trunc)
        else:
            self._state = ops.vacuumStateMixed(self._num_modes, self._trunc)

    def norm(self):
        """returns the norm of the state"""
        if self._pure:
            return sqrt(np.vdot(self._state, self._state).real)
        return ops.trace(self._state, self._num_modes)

    def alloc(self, n=1):
        """allocate a number of modes at the end of the state."""
        # base_shape = [self._trunc for i in range(n)]
        if self._pure:
            vac = ops.vacuumState(n, self._trunc)
        else:
            vac = ops.vacuumStateMixed(n, self._trunc)

        self._state = ops.tensor(self._state, vac, self._num_modes, self._pure)
        self._num_modes = self._num_modes + n

    def dealloc(self, modes):
        """Traces out and deallocates the modes in `modes`"""
        if self._pure:
            self._state = ops.mix(self._state, self._num_modes)
            self._pure = False

        self._state = ops.partial_trace(self._state, self._num_modes, modes)
        self._num_modes = self._num_modes - len(modes)

    def prepare(self, state, mode):
        r"""
        Prepares a given mode in a given state.

        Assumes the state of the entire system is of the form
        :math:`\ket{0}\otimes\ket{\psi}`, up to mode permutations.
        In particular, the mode may retain any previous phase shift.

        Args:
            state (array or matrix): The new state in the fock basis
            mode (non-negative int): The overwritten mode
        """

        # Do consistency checks
        pure_shape = (self._trunc,)
        # mixed_shape = (self._trunc, self._trunc)

        if self._checks:
            if (self._pure and state.shape != (self._trunc,)) or \
                 (not (self._pure) and state.shape != (self._trunc, self._trunc)):
                raise ValueError("Incorrect shape for state preparation")

        if self._num_modes == 1:
            # Hack for marginally faster state preparation
            self._state = state.astype(ops.def_type)
            self._pure = bool(state.shape == pure_shape)
        else:
            if self._pure:
                self._state = ops.mix(self._state, self._num_modes)
                self._pure = False

            if state.shape != (self._trunc, self._trunc):
                state = np.outer(state, state.conj())

            # Take the partial trace
            reduced_state = ops.partial_trace(self._state, self._num_modes, [mode])

            # Insert state
            self._state = ops.tensor(reduced_state, state, self._num_modes-1, self._pure, pos=mode)

    def prepare_mode_fock(self, n, mode):
        """
        Prepares a mode in a fock state.
        """

        if self._pure:
            self.prepare(ops.fockState(n, self._trunc), mode)
        else:
            st = ops.fockState(n, self._trunc)
            self.prepare(np.outer(st, st.conjugate()), mode)

    def prepare_mode_coherent(self, alpha, mode):
        """
        Prepares a mode in a coherent state.
        """
        if self._pure:
            self.prepare(ops.coherentState(alpha, self._trunc), mode)
        else:
            st = ops.coherentState(alpha, self._trunc)
            self.prepare(np.outer(st, st.conjugate()), mode)

    def prepare_mode_squeezed(self, r, theta, mode):
        """
        Prepares a mode in a squeezed state.
        """
        if self._pure:
            self.prepare(ops.squeezedState(r, theta, self._trunc), mode)
        else:
            st = ops.squeezedState(r, theta, self._trunc)
            self.prepare(np.outer(st, st.conjugate()), mode)

    def prepare_mode_displaced_squeezed(self, alpha, r, phi, mode):
        """
        Prepares a mode in a displaced squeezed state.
        """
        if self._pure:
            self.prepare(ops.displacedSqueezed(alpha, r, phi, self._trunc), mode)
        else:
            st = ops.displacedSqueezed(alpha, r, phi, self._trunc)
            self.prepare(np.outer(st, st.conjugate()), mode)

    def prepare_mode_thermal(self, nbar, mode):
        """
        Prepares a mode in a thermal state.
        """
        st = ops.thermalState(nbar, self._trunc)
        self.prepare(st, mode)

    def phase_shift(self, theta, mode):
        """
        Applies a phase shifter.
        """
        self._apply_gate(ops.phase(theta, self._trunc), [mode])

    def displacement(self, alpha, mode):
        """
        Applies a displacement gate.
        """
        self._apply_gate(ops.displacement(alpha, self._trunc), [mode])

    def beamsplitter(self, t, r, phi, mode1, mode2):
        """
        Applies a beamsplitter.
        """
        self._apply_gate(ops.beamsplitter(t, r, phi, self._trunc), [mode1, mode2])

    def squeeze(self, r, theta, mode):
        """
        Applies a squeezing gate.
        """
        self._apply_gate(ops.squeezing(r, theta, self._trunc), [mode])

    def kerr_interaction(self, kappa, mode):
        """
        Applies a Kerr interaction gate.
        """
        self._apply_gate(ops.kerr(kappa, self._trunc), [mode])

    def cubic_phase_shift(self, gamma, mode):
        """
        Applies a cubic phase shift gate.
        """
        self._apply_gate(ops.cubicPhase(gamma, self._hbar, self._trunc), [mode])

    def is_vacuum(self, tol):
        """
        Tests whether the system is in the vacuum state.
        """
        # base_shape = [self._trunc for i in range(self._num_modes)]
        if self._pure:
            vac = ops.vacuumState(self._num_modes, self._trunc)
        else:
            vac = ops.vacuumStateMixed(self._num_modes, self._trunc)
        return np.linalg.norm((self._state - vac).ravel()) < tol

    def get_state(self):
        """
        Returns the state of the system in the fock basis along with its purity.
        """
        return self._state, self._pure

    def loss(self, T, mode):
        """
        Applies a loss channel to the state.
        """
        self._apply_channel(ops.lossChannel(T, self._trunc), [mode])

    def measure_fock(self, modes, select=None):
        """
        Measures a list of modes.
        """
        # pylint: disable=singleton-comparison
        if select is not None and np.any(np.array(select) == None):
            raise NotImplementedError("Post-selection lists must only contain numerical values.")

        # Make sure the state is mixed
        if self._pure:
            state = ops.mix(self._state, self._num_modes)
        else:
            state = self._state

        if select is not None:
            # perform post-selection

            # make sure modes and select are the same length
            if len(select) != len(modes):
                raise ValueError("When performing post-selection, the number of "
                                 "selected values (including None) must match the number of measured modes")

            # make sure the select values are all integers or nones
            if not all(isinstance(s, int) or s is None for s in select):
                raise TypeError("The post-select list elements either be integers or None")

            # modes to measure
            measure = [i for i, s in zip(modes, select) if s is None]

            # modes already post-selected:
            selected = [i for i, s in zip(modes, select) if s is not None]
            select_values = [s for s in select if s is not None]

            # project out postselected modes
            self._state = ops.project_reset(selected, select_values, self._state, self._pure, self._num_modes, self._trunc)

            if self.norm() == 0:
                raise ZeroDivisionError("Measurement has zero probability.")

            self._state = self._state / self.norm()

        else:
            # no post-selection; modes to measure are the modes provided
            measure = modes

        if len(measure) > 0:
            # sampling needs to be performed
            # Compute distribution by tracing out modes not measured, then computing the diagonal
            unmeasured = [i for i in range(self._num_modes) if i not in measure]
            reduced = ops.partial_trace(state, self._num_modes, unmeasured)
            dist = np.ravel(ops.diagonal(reduced, len(measure)).real)

            # Make a random choice
            if sum(dist) != 1:
                # WARNING: distribution is not normalized, could hide errors
                i = np.random.choice(list(range(len(dist))), p=dist / sum(dist))
            else:
                i = np.random.choice(list(range(len(dist))), p=dist)

            permuted_outcome = ops.unIndex(i, len(measure), self._trunc)

            # Permute the outcome to match the order of the modes in 'measure'
            permutation = np.argsort(measure)
            outcome = [0] * len(measure)
            for i in range(len(measure)):
                outcome[permutation[i]] = permuted_outcome[i]

            # Project the state onto the measurement outcome & reset in vacuum
            self._state = ops.project_reset(measure, outcome, self._state, self._pure, self._num_modes, self._trunc)

            if self.norm() == 0:
                raise ZeroDivisionError("Measurement has zero probability.")

            self._state = self._state / self.norm()

        # include post-selected values in measurement outcomes
        if select is not None:
            outcome = copy.copy(select)

        return outcome

    def measure_homodyne(self, phi, mode, select=None, **kwargs):
        """
        Performs a homodyne measurement on a mode.
        """
        m_omega_over_hbar = 1/self._hbar

        # Make sure the state is mixed for reduced density matrix
        if self._pure:
            state = ops.mix(self._state, self._num_modes)
        else:
            state = self._state

        if select is not None:
            meas_result = select
            if isinstance(meas_result, numbers.Number):
                homodyne_sample = float(meas_result)
            else:
                raise TypeError("Selected measurement result must be of numeric type.")
        else:
             # Compute reduced density matrix
            unmeasured = [i for i in range(self._num_modes) if not i == mode]
            reduced = ops.partial_trace(state, self._num_modes, unmeasured)

            # Rotate to measurement basis
            args = [ops.phase(-phi, self._trunc), reduced, False, [0], 1, self._trunc]
            if self._mode == 'blas':
                reduced = ops.apply_gate_BLAS(*args)
            elif self._mode == 'einsum':
                reduced = ops.apply_gate_einsum(*args)

            # Create pdf. Same as tf implementation, but using
            # the recursive relation H_0(x) = 1, H_1(x) = 2x, H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)
            if "max" in kwargs:
                q_mag = kwargs["max"]
            else:
                q_mag = 10
            if "num_bins" in kwargs:
                num_bins = kwargs["num_bins"]
            else:
                num_bins = 100000

            q_tensor, Hvals = ops.hermiteVals(q_mag, num_bins, m_omega_over_hbar, self._trunc)
            H_matrix = np.zeros((self._trunc, self._trunc, num_bins))
            for n, m in product(range(self._trunc), repeat=2):
                H_matrix[n][m] = 1 / sqrt(2**n * bang(n) * 2**m * bang(m)) * Hvals[n] * Hvals[m]
            H_terms = np.expand_dims(reduced, -1) * np.expand_dims(H_matrix, 0)
            rho_dist = np.sum(H_terms, axis=(1, 2)) \
                                 * (m_omega_over_hbar/pi)**0.5 \
                                 * np.exp(-m_omega_over_hbar * q_tensor**2) \
                                 * (q_tensor[1] - q_tensor[0]) # Delta_q for normalization (only works if the bins are equally spaced)

            # Sample from rho_dist. This is a bit different from tensorflow due to how
            # numpy treats multinomial sampling. In particular, numpy returns a
            # histogram of the samples whereas tensorflow gives the list of samples.
            # Numpy also does not use the log probabilities
            probs = rho_dist.flatten().real
            probs /= np.sum(probs)
            sample_hist = np.random.multinomial(1, probs)
            sample_idx = list(sample_hist).index(1)
            homodyne_sample = q_tensor[sample_idx]

        # Project remaining modes into the conditional state
        inf_squeezed_vac = \
            np.array([(-0.5)**(n//2) * sqrt(bang(n)) / bang(n//2) if n%2 == 0 else 0.0 + 0.0j \
                for n in range(self._trunc)], dtype=ops.def_type)
        alpha = homodyne_sample * sqrt(m_omega_over_hbar / 2)

        composed = np.dot(ops.phase(phi, self._trunc), ops.displacement(alpha, self._trunc))
        args = [composed, inf_squeezed_vac, True, [0], 1, self._trunc]
        if self._mode == 'blas':
            eigenstate = ops.apply_gate_BLAS(*args)
        elif self._mode == 'einsum':
            eigenstate = ops.apply_gate_einsum(*args)

        vac_state = np.array([1.0 + 0.0j if i == 0 else 0.0 + 0.0j for i in range(self._trunc)], dtype=ops.def_type)
        projector = np.outer(vac_state, eigenstate.conj())
        self._apply_gate(projector, [mode])

        # Normalize
        self._state = self._state / self.norm()

        return homodyne_sample

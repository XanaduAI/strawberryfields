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
Fock backend interface
=======================

"""
# pylint: disable=protected-access,too-many-public-methods

import string
from cmath import phase
import numpy as np

from strawberryfields.backends import BaseFock, ModeMap
from .circuit import Circuit
from ..states import BaseFockState

indices = string.ascii_lowercase


class FockBackend(BaseFock):
    """ Backend in the Fock basis """

    def __init__(self):
        """Instantiate a FockBackend object."""
        super().__init__()
        self._supported["mixed_states"] = True
        self._short_name = "fock"

    def _remap_modes(self, modes):
        if isinstance(modes, int):
            modes = [modes]
            was_int = True
        else:
            was_int = False
        map_ = self._modemap.show()
        submap = [map_[m] for m in modes]
        if not self._modemap.valid(modes) or None in submap:
            raise ValueError('The specified modes are not valid.')
        else:
            remapped_modes = self._modemap.remap(modes)
        if was_int:
            remapped_modes = remapped_modes[0]
        return remapped_modes

    def begin_circuit(self, num_subsystems, cutoff_dim=None, hbar=2, pure=True, **kwargs):
        r"""
        Create a quantum circuit (initialized in vacuum state) with the number of modes
        equal to num_subsystems and a Fock-space cutoff dimension of cutoff_dim.

        Args:
            num_subsystems (int): number of modes the circuit should begin with
            cutoff_dim (int): numerical cutoff dimension in Fock space for each mode.
                ``cutoff_dim=D`` represents the Fock states :math:`|0\rangle,\dots,|D-1\rangle`.
                This argument is **required** for the Fock backend.
            hbar (float): The value of :math:`\hbar` to initialise the circuit with, depending on the conventions followed.
                By default, :math:`\hbar=2`. See :ref:`conventions` for more details.
            pure (bool): whether to begin the circuit in a pure state representation
        """
        # pylint: disable=attribute-defined-outside-init
        if cutoff_dim is None:
            raise ValueError("Argument 'cutoff_dim' must be passed to the Fock backend")
        if not isinstance(cutoff_dim, int):
            raise ValueError("Argument 'cutoff_dim' must be a positive integer")
        if not isinstance(num_subsystems, int):
            raise ValueError("Argument 'num_subsystems' must be a positive integer")
        if not isinstance(pure, bool):
            raise ValueError("Argument 'pure' must be either True or False")

        self._init_modes = num_subsystems
        self.circuit = Circuit(num_subsystems, cutoff_dim, hbar, pure)
        self._modemap = ModeMap(num_subsystems)

    def add_mode(self, n=1):
        """Add num_modes new modes to the underlying circuit state. Indices for new modes
        always occur at the end of the state tensor.
        Note: this will increase the number of indices used for the state representation.

        Args:
            n (int): the number of modes to be added to the circuit
        """
        self.circuit.alloc(n)
        self._modemap.add(n)

    def del_mode(self, modes):
        """Trace out the specified modes from the underlying circuit state.
        Note: this will reduce the number of indices used for the state representation,
        and also convert the state representation to mixed.

        Args:
            modes (list[int]): the modes to be removed from the circuit

        """
        remapped_modes = self._remap_modes(modes)
        if isinstance(remapped_modes, int):
            remapped_modes = [remapped_modes]
        self.circuit.dealloc(remapped_modes)
        self._modemap.delete(modes)

    def get_modes(self):
        """Return a list of the active mode indices for the circuit.

        Returns:
            list[int]: sorted list of active (assigned, not invalid) mode indices
        """
        return [i for i, j in enumerate(self._modemap._map) if j is not None]

    def reset(self, pure=True, **kwargs):
        """Resets the circuit state back to an all-vacuum state.

        Args:
            pure (bool): whether to use a pure state representation upon reset
        """
        cutoff = kwargs.get('cutoff_dim', self.circuit._trunc)
        self._modemap.reset()
        self.circuit.reset(pure, num_subsystems=self._init_modes, cutoff_dim=cutoff)

    def prepare_vacuum_state(self, mode):
        """Prepare the vacuum state on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            mode (int): index of mode where state is prepared
        """
        self.circuit.prepare_mode_fock(0, self._remap_modes(mode))

    def prepare_coherent_state(self, alpha, mode):
        """Prepare a coherent state with parameter alpha on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            alpha (complex): coherent state displacement parameter
            mode (int): index of mode where state is prepared
        """
        self.circuit.prepare_mode_coherent(alpha, self._remap_modes(mode))

    def prepare_squeezed_state(self, r, phi, mode):
        r"""Prepare a squeezed vacuum state in the specified mode.
        Note: this may convert the state representation to mixed.

        The requested mode is traced out and replaced with the squeezed state :math:`\ket{z}`,
        where :math:`z=re^{i\phi}`.
        As a result the state may have to be described using a density matrix.

        Args:
            r (float): squeezing amplitude
            phi (float): squeezing angle
            mode (int): which mode to prepare the squeezed state in
        """
        self.circuit.prepare_mode_squeezed(r, phi, self._remap_modes(mode))

    def prepare_displaced_squeezed_state(self, alpha, r, phi, mode):
        """Prepare a displaced squezed state with parameters (alpha, r, phi) on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            alpha (complex): displacement parameter
            r (float): squeezing amplitude
            phi (float): squeezing phase
            mode (int): index of mode where state is prepared

        """
        self.circuit.prepare_mode_displaced_squeezed(alpha, r, phi, self._remap_modes(mode))

    def prepare_thermal_state(self, nbar, mode):
        """Prepare the thermal state with mean photon number nbar on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            nbar (float): mean thermal population of the mode
            mode (int): which mode to prepare the thermal state in
        """
        self.circuit.prepare_mode_thermal(nbar, self._remap_modes(mode))

    def rotation(self, phi, mode):
        """Apply the phase-space rotation operation to the specified mode.

        Args:
            phi (float): rotation angle
            mode (int): which mode to apply the rotation to
        """
        self.circuit.phase_shift(phi, self._remap_modes(mode))

    def displacement(self, alpha, mode):
        """Perform a displacement operation on the specified mode.

        Args:
            alpha (float): displacement parameter
            mode (int): index of mode where operation is carried out

        """
        self.circuit.displacement(alpha, self._remap_modes(mode))

    def squeeze(self, z, mode):
        """Perform a squeezing operation on the specified mode.

        Args:
            z (complex): squeezing parameter
            mode (int): index of mode where operation is carried out

        """
        self.circuit.squeeze(abs(z), phase(z), self._remap_modes(mode))

    def beamsplitter(self, t, r, mode1, mode2):
        """Perform a beamsplitter operation on the specified modes.

        Args:
            t (float): transmittivity parameter
            r (complex): reflectivity parameter
            mode1 (int): index of first mode where operation is carried out
            mode2 (int): index of second mode where operation is carried out

        """
        if isinstance(t, complex):
            raise ValueError("Beamsplitter transmittivity t must be a float.")
        self.circuit.beamsplitter(t, abs(r), phase(r), self._remap_modes(mode1), self._remap_modes(mode2))

    def kerr_interaction(self, kappa, mode):
        r"""Apply the Kerr interaction :math:`\exp{(i\kappa \hat{n}^2)}` to the specified mode.

        Args:
            kappa (float): strength of the interaction
            mode (int): which mode to apply it to
        """
        self.circuit.kerr_interaction(kappa, self._remap_modes(mode))

    def cross_kerr_interaction(self, kappa, mode1, mode2):
        r"""Apply the two mode cross-Kerr interaction :math:`\exp{(i\kappa \hat{n}_1\hat{n}_2)}` to the specified modes.

        Args:
            kappa (float): strength of the interaction
            mode1 (int): first mode that cross-Kerr interaction acts on
            mode2 (int): second mode that cross-Kerr interaction acts on
        """
        self.circuit.cross_kerr_interaction(kappa, self._remap_modes(mode1), self._remap_modes(mode2))

    def cubic_phase(self, gamma, mode):
        r"""Apply the cubic phase operation to the specified mode.

        .. warning:: The cubic phase gate can suffer heavily from numerical inaccuracies due to finite-dimensional cutoffs in the Fock basis.
                     The gate implementation in Strawberry Fields is unitary, but it does not implement an exact cubic phase gate.
                     The Kerr gate provides an alternative non-Gaussian gate.

        Args:
            gamma (float): cubic phase shift
            mode (int): which mode to apply it to
        """
        self.circuit.cubic_phase_shift(gamma, self._remap_modes(mode))

    def measure_homodyne(self, phi, mode, select=None, **kwargs):
        """Perform a homodyne measurement on the specified mode.

        Args:
            phi (float): angle (relative to x-axis) for the measurement
            mode (int): index of mode where operation is carried out
            select (float): (Optional) desired values of measurement results
            **kwargs: Can be used to (optionally) pass user-specified numerical parameters `max` and `num_bins`.
                                These are used numerically to build the probability distribution function (pdf) for the homdyne measurement
                                Specifically, the pdf is discretized onto the 1D grid [-max,max], with num_bins equally spaced bins

        Returns:
            float: measurement outcome
        """
        return self.circuit.measure_homodyne(phi, self._remap_modes(mode), select=select, **kwargs)

    def loss(self, T, mode):
        """Perform a loss channel operation on the specified mode.

        Args:
            T: loss parameter
            mode (int): index of mode where operation is carried out

        """
        self.circuit.loss(T, self._remap_modes(mode))

    def is_vacuum(self, tol=0.0, **kwargs):
        r"""Test whether the current circuit state is in vacuum (up to tolerance tol).

        Args:
            tol (float): numerical tolerance for how close state must be to true vacuum state

        Returns:
            bool: True if vacuum state up to tolerance tol
        """
        return self.circuit.is_vacuum(tol)

    def get_cutoff_dim(self):
        """Returns the Hilbert space cutoff dimension used.

        Returns:
            int: cutoff dimension
        """
        return self.circuit._trunc


    def state(self, modes=None, **kwargs):
        r"""Returns the state of the quantum simulation, restricted to the subsystems defined by `modes`.

        Args:
            modes (int, Sequence[int], None): specifies the mode or modes to restrict the return state to.
                If none returns the state containing all modes.
        Returns:
            BaseFockState: an instance of the Strawberry Fields FockState class.
        """
        s, pure = self.circuit.get_state()

        if modes is None:
            # reduced state is full state
            red_state = s
            num_modes = len(s.shape) if pure else len(s.shape) // 2
            modes = [m for m in range(num_modes)]
        else:
            # convert to mixed state representation
            if pure:
                num_modes = len(s.shape)
                left_str = [indices[i] for i in range(0, 2 * num_modes, 2)]
                right_str = [indices[i] for i in range(1, 2 * num_modes, 2)]
                out_str = [indices[: 2 * num_modes]]
                einstr = ''.join(left_str + [','] + right_str + ['->'] + out_str)
                rho = np.einsum(einstr, s, s.conj())
            else:
                rho = s

            # reduce rho down to specified subsystems
            if isinstance(modes, int):
                modes = [modes]

            if len(modes) != len(set(modes)):
                raise ValueError("The specified modes cannot be duplicated.")

            num_modes = len(rho.shape) // 2
            if len(modes) > num_modes:
                raise ValueError("The number of specified modes cannot be larger than the number of subsystems.")

            keep_indices = indices[: 2 * len(modes)]
            trace_indices = indices[2 * len(modes) : len(modes) + num_modes]
            ind = [i * 2 for i in trace_indices]
            ctr = 0

            for m in range(num_modes):
                if m in modes:
                    ind.insert(m, keep_indices[2 * ctr : 2 * (ctr + 1)])
                    ctr += 1

            indStr = ''.join(ind) + '->' + keep_indices
            red_state = np.einsum(indStr, rho)

            # permute indices of returned state to reflect the ordering of modes (we know and hence can assume that red_state is a mixed state)
        if modes != sorted(modes):
            mode_permutation = np.argsort(modes)
            index_permutation = [2*x+i for x in mode_permutation for i in (0, 1)]
            red_state = np.transpose(red_state, np.argsort(index_permutation))

        hbar = self.circuit._hbar
        cutoff = self.circuit._trunc # pylint: disable=protected-access
        mode_names = ["q[{}]".format(i) for i in np.array(self.get_modes())[modes]]
        state = BaseFockState(red_state, len(modes), pure, cutoff, hbar, mode_names)
        return state

    # ==============================================
    # Fock state specific
    # ==============================================

    def prepare_fock_state(self, n, mode):
        """Prepare a Fock state on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            n (int): number state to prepare
            mode (int): index of mode where state is prepared

        """
        self.circuit.prepare_mode_fock(n, self._remap_modes(mode))

    def prepare_ket_state(self, state, modes):
        """Prepare an arbitrary pure state on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            state (array): vector representation of ket state to prepare
            mode (int): index of mode where state is prepared
        """
        self.circuit.prepare_multimode(state, self._remap_modes(modes))

    def prepare_dm_state(self, state, modes):
        """Prepare an arbitrary mixed state on the specified mode.
        Note: this will convert the state representation to mixed.

        Args:
            state (array): density matrix representation of state to prepare
            mode (int): index of mode where state is prepared
        """
        self.circuit.prepare_multimode(state, self._remap_modes(modes))

    def measure_fock(self, modes, select=None, **kwargs):
        """Perform a Fock measurement on the specified modes.

        Args:
            modes (list[int]): indices of mode where operation is carried out
            select (list[int]): (Optional) desired values of measurement results.
                                                     The length of this list must match the length of the modes list.

        Returns:
            list[int]: measurement outcomes
        """
        return self.circuit.measure_fock(self._remap_modes(modes), select=select)

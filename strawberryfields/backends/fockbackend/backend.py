# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fock backend simulator"""
# pylint: disable=protected-access,too-many-public-methods

import string
from cmath import phase
import numpy as np

from strawberryfields.backends import BaseFock, ModeMap
from strawberryfields.backends.states import BaseFockState

from .circuit import Circuit

indices = string.ascii_lowercase


class FockBackend(BaseFock):
    r"""Implements a simulation of quantum optical circuits in a truncated
    Fock basis using NumPy, returning a :class:`~.BaseFock`
    state object.

    The primary component of the FockBackend is a
    :attr:`~.FockBackend.circuit` object which is used to simulate a multi-mode quantum optical system. The
    :class:`FockBackend` provides the basic API-compatible interface to the simulator, while the
    :attr:`~.FockBackend.circuit` object actually carries out the mathematical simulation.

    The :attr:`~.FockBackend.circuit` simulator maintains an internal tensor representation of the quantum state of a multi-mode quantum optical system
    using a (truncated) Fock basis representation. As its various state manipulation methods are called, the quantum state is updated
    to reflect these changes. The simulator will try to keep the internal state in a pure (vector) representation
    for as long as possible. Unitary gates will not change the type of representation, while state preparations and measurements will.

    A number of factors determine the shape and dimensionality of the state tensor:

    * the underlying state representation being used (either a ket vector or a density matrix)
    * the number of modes :math:`n` actively being simulated
    * the cutoff dimension :math:`D` for the Fock basis

    The state tensor corresponds to a multimode quantum system. If the
    representation is a pure state, the state tensor has shape
    :math:`(\underbrace{D,...,D}_{n~\text{times}})`.
    In a mixed state representation, the state tensor has shape
    :math:`(\underbrace{D,D,...,D,D}_{2n~\text{times}})`.
    Indices for the same mode appear consecutively. Hence, for a mixed state, the first two indices
    are for the first mode, the second are for the second mode, etc.

    ..
        .. currentmodule:: strawberryfields.backends.fockbackend
        .. autosummary::
            :toctree:

            ~circuit.Circuit
            ~ops
    """

    short_name = "fock"
    circuit_spec = "fock"

    def __init__(self):
        """Instantiate a FockBackend object."""
        super().__init__()
        self._supported["mixed_states"] = True
        self._init_modes = None  #: int: initial number of modes in the circuit
        self._modemap = None  #: Modemap: maps external mode indices to internal ones
        self.circuit = (
            None  #: ~.fockbackend.circuit.Circuit: representation of the simulated quantum state
        )

    def _remap_modes(self, modes):
        if isinstance(modes, int):
            modes = [modes]
            was_int = True
        else:
            was_int = False
        map_ = self._modemap.show()
        submap = [map_[m] for m in modes]
        if not self._modemap.valid(modes) or None in submap:
            raise ValueError("The specified modes are not valid.")

        remapped_modes = self._modemap.remap(modes)
        if was_int:
            remapped_modes = remapped_modes[0]
        return remapped_modes

    def begin_circuit(self, num_subsystems, **kwargs):
        r"""Instantiate a quantum circuit.

        Instantiates a representation of a quantum optical state with ``num_subsystems`` modes.
        The state is initialized to vacuum.

        The modes in the circuit are indexed sequentially using integers, starting from zero.
        Once an index is assigned to a mode, it can never be re-assigned to another mode.
        If the mode is deleted its index becomes invalid.
        An operation acting on an invalid or unassigned mode index raises an ``IndexError`` exception.

        Args:
            num_subsystems (int): number of modes in the circuit

        Keyword Args:
            cutoff_dim (int): Numerical Hilbert space cutoff dimension for the modes.
                For each mode, the simulator can represent the Fock states :math:`\ket{0}, \ket{1}, \ldots, \ket{\text{cutoff_dim}-1}`.
            pure (bool): If True (default), use a pure state representation (otherwise will use a mixed state representation).
        """
        cutoff_dim = kwargs.get("cutoff_dim", None)
        pure = kwargs.get("pure", True)
        if cutoff_dim is None:
            raise ValueError("Argument 'cutoff_dim' must be passed to the Fock backend")
        if not isinstance(cutoff_dim, int):
            raise ValueError("Argument 'cutoff_dim' must be a positive integer")
        if not isinstance(num_subsystems, int):
            raise ValueError("Argument 'num_subsystems' must be a positive integer")
        if not isinstance(pure, bool):
            raise ValueError("Argument 'pure' must be either True or False")

        self._init_modes = num_subsystems
        self.circuit = Circuit(num_subsystems, cutoff_dim, pure)
        self._modemap = ModeMap(num_subsystems)

    def add_mode(self, n=1):
        self.circuit.alloc(n)
        self._modemap.add(n)

    def del_mode(self, modes):
        remapped_modes = self._remap_modes(modes)
        if isinstance(remapped_modes, int):
            remapped_modes = [remapped_modes]
        self.circuit.dealloc(remapped_modes)
        self._modemap.delete(modes)

    def get_modes(self):
        return [i for i, j in enumerate(self._modemap._map) if j is not None]

    def reset(self, pure=True, **kwargs):
        cutoff = kwargs.get("cutoff_dim", self.circuit._trunc)
        self._modemap.reset()
        self.circuit.reset(pure, num_subsystems=self._init_modes, cutoff_dim=cutoff)

    def prepare_vacuum_state(self, mode):
        self.circuit.prepare_mode_fock(0, self._remap_modes(mode))

    def prepare_coherent_state(self, r, phi, mode):
        self.circuit.prepare_mode_coherent(r, phi, self._remap_modes(mode))

    def prepare_squeezed_state(self, r, phi, mode):
        self.circuit.prepare_mode_squeezed(r, phi, self._remap_modes(mode))

    def prepare_displaced_squeezed_state(self, r_d, phi_d, r_s, phi_s, mode):
        self.circuit.prepare_mode_displaced_squeezed(
            r_d, phi_d, r_s, phi_s, self._remap_modes(mode)
        )

    def prepare_thermal_state(self, nbar, mode):
        self.circuit.prepare_mode_thermal(nbar, self._remap_modes(mode))

    def rotation(self, phi, mode):
        self.circuit.phase_shift(phi, self._remap_modes(mode))

    def displacement(self, r, phi, mode):
        self.circuit.displacement(r, phi, self._remap_modes(mode))

    def squeeze(self, r, phi, mode):
        self.circuit.squeeze(r, phi, self._remap_modes(mode))

    def two_mode_squeeze(self, r, phi, mode1, mode2):
        self.circuit.two_mode_squeeze(r, phi, self._remap_modes(mode1), self._remap_modes(mode2))

    def beamsplitter(self, theta, phi, mode1, mode2):
        self.circuit.beamsplitter(theta, phi, self._remap_modes(mode1), self._remap_modes(mode2))

    def measure_homodyne(self, phi, mode, shots=1, select=None, **kwargs):
        """Perform a homodyne measurement on the specified mode.

        See :meth:`.BaseBackend.measure_homodyne`.

        Keyword Args:
            num_bins (int): Number of equally spaced bins for the probability distribution function
                (pdf) simulating the homodyne measurement (default: 100000).
            max (float): The pdf is discretized onto the 1D grid [-max,max] (default: 10).
        """
        if shots != 1:
            raise NotImplementedError(
                "fock backend currently does not support " "shots != 1 for homodyne measurement"
            )
        return self.circuit.measure_homodyne(phi, self._remap_modes(mode), select=select, **kwargs)

    def loss(self, T, mode):
        self.circuit.loss(T, self._remap_modes(mode))

    def is_vacuum(self, tol=0.0, **kwargs):
        return self.circuit.is_vacuum(tol)

    def get_cutoff_dim(self):
        return self.circuit._trunc

    def state(self, modes=None, **kwargs):
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
                einstr = "".join(left_str + [","] + right_str + ["->"] + out_str)
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
                raise ValueError(
                    "The number of specified modes cannot be larger than the number of subsystems."
                )

            keep_indices = indices[: 2 * len(modes)]
            trace_indices = indices[2 * len(modes) : len(modes) + num_modes]
            ind = [i * 2 for i in trace_indices]
            ctr = 0

            for m in range(num_modes):
                if m in modes:
                    ind.insert(m, keep_indices[2 * ctr : 2 * (ctr + 1)])
                    ctr += 1

            indStr = "".join(ind) + "->" + keep_indices
            red_state = np.einsum(indStr, rho)

            # permute indices of returned state to reflect the ordering of modes (we know and hence can assume that red_state is a mixed state)
        if modes != sorted(modes):
            mode_permutation = np.argsort(modes)
            index_permutation = [2 * x + i for x in mode_permutation for i in (0, 1)]
            red_state = np.transpose(red_state, np.argsort(index_permutation))

        cutoff = self.circuit._trunc
        mode_names = ["q[{}]".format(i) for i in np.array(self.get_modes())[modes]]
        state = BaseFockState(red_state, len(modes), pure, cutoff, mode_names)
        return state

    # ==============================================
    # Fock state specific
    # ==============================================

    def prepare_fock_state(self, n, mode):
        self.circuit.prepare_mode_fock(n, self._remap_modes(mode))

    def prepare_ket_state(self, state, modes):
        self.circuit.prepare_multimode(state, self._remap_modes(modes))

    def prepare_dm_state(self, state, modes):
        self.circuit.prepare_multimode(state, self._remap_modes(modes))

    def cubic_phase(self, gamma, mode):
        self.circuit.cubic_phase_shift(gamma, self._remap_modes(mode))

    def kerr_interaction(self, kappa, mode):
        self.circuit.kerr_interaction(kappa, self._remap_modes(mode))

    def cross_kerr_interaction(self, kappa, mode1, mode2):
        self.circuit.cross_kerr_interaction(
            kappa, self._remap_modes(mode1), self._remap_modes(mode2)
        )

    def measure_fock(self, modes, shots=1, select=None, **kwargs):
        if shots != 1:
            raise NotImplementedError(
                "fock backend currently does not support " "shots != 1 for Fock measurement"
            )
        return self.circuit.measure_fock(self._remap_modes(modes), select=select)

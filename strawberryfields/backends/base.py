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

r"""
.. _backends:

Backend API
===================================================

**Module name:** :mod:`strawberryfields.backends.base`

.. currentmodule:: strawberryfields.backends.base

This module implements the backend API. It contains the classes

* :class:`BaseBackend`
* :class:`BaseFock`
* :class:`BaseGaussian`

as well as a few methods which apply only to the Gaussian backend.

.. note:: The Strawberry Fields backends by default assume :math:`\hbar=2`, however
    different conventions may be chosen when calling :meth:`~.BaseBackend.begin_circuit`

.. note::
    Keyword arguments are denoted ``**kwargs``, and allow additional
    options to be passed to the backends - these are documented where
    available. For more details on available keyword arguments, please
    consult the backends directly.

Hierarchy for backends
------------------------

.. currentmodule:: strawberryfields.backends

.. inheritance-diagram:: base.BaseBackend
    fockbackend.backend.FockBackend
    gaussianbackend.backend.GaussianBackend
    tfbackend.backend.TFBackend
    :parts: 1


Base backend
-----------------------------------

.. currentmodule:: strawberryfields.backends.base.BaseBackend

.. autosummary::
    supports
    begin_circuit
    add_mode
    del_mode
    get_modes
    reset
    prepare_vacuum_state
    prepare_coherent_state
    prepare_squeezed_state
    prepare_displaced_squeezed_state
    prepare_thermal_state
    rotation
    displacement
    squeeze
    beamsplitter
    loss
    measure_homodyne
    state
    is_vacuum

Fock backends
------------------

.. currentmodule:: strawberryfields.backends.base


Some methods are only implemented in the subclass :class:`FockBackend`,
which is the base class for simulators using a Fock-state representation
for quantum optical circuits.

.. currentmodule:: strawberryfields.backends.base.BaseFock

.. autosummary::
    get_cutoff_dim
    prepare_fock_state
    prepare_ket_state
    prepare_dm_state
    cubic_phase
    kerr_interaction
    cross_kerr_interaction
    measure_fock

Gaussian backends
---------------------

Likewise, some methods are only implemented in subclass :class:`BaseGaussian`,
which is the base class for simulators using a Gaussian symplectic representation
for quantum optical circuits.

.. currentmodule:: strawberryfields.backends.base.BaseGaussian

.. autosummary::
    measure_heterodyne

Code details
~~~~~~~~~~~~

"""

# todo If we move to Sphinx 1.7, the docstrings of the methods in the derived classes FockBackend,
# TFBackend and GaussianBackend that are declared in BaseBackend should be removed entirely.
# This way they are inherited directly from the parent class BaseBackend and thus kept automatically up-to-date.
# The derived classes should provide a docstring for these methods only if they change their behavior for some reason.

# pylint: disable=no-self-use


class NotApplicableError(TypeError):
    """Exception raised by the backend when the user attempts an unsupported operation.
    E.g. :meth:`measure_fock` on a Gaussian backend.
    Conceptually different from NotImplementedError (which means "not implemented, but at some point may be").
    """
    pass


class ModeMap:
    """
    Simple internal class for maintaining a map of existing modes.
    """
    def __init__(self, num_subsystems):
        self._init = num_subsystems
        #: list[int]: _map[k] is the internal index used by the backend for
        # computational mode k, or None if the mode has been deleted
        self._map = [k for k in range(num_subsystems)]

    def reset(self):
        """reset the modemap to the initial state"""
        self._map = [k for k in range(self._init)]

    def _single_mode_valid(self, mode):
        if mode is None:
            return False

        if mode >= 0 and mode < len(self._map):
            return True

        return False

    def _reduce_to_existing_modes(self, modes):
        # Reduces modes to only those which are not None in the map
        if isinstance(modes, int):
            modes = [modes]
        return [m for m in modes if m in self._map]

    def remap(self, modes):
        """Remaps the mode list"""
        if isinstance(modes, int):
            modes = [modes]
            was_int = True
        else:
            was_int = False

        modes_list = [self._map[m] for m in modes]

        if was_int:
            return modes_list[0]

        return modes_list

    def valid(self, modes):
        """checks if the mode list is valid"""
        if modes is None:
            return False

        if isinstance(modes, int):
            modes = [modes]

        # pylint: disable=len-as-condition
        if len(modes) == 0 or len(modes) > len(self._map):
            return False

        for m in modes:
            if not self._single_mode_valid(m):
                return False

        return True

    def show(self):
        """Returns the mapping"""
        return self._map

    def delete(self, modes):
        """Deletes a mode"""
        if isinstance(modes, int):
            modes = [modes]
        if self.valid(modes):
            new_map = []
            ctr = 0
            for m in range(len(self._map)):
                if m in modes or self._map[m] is None:
                    new_map.append(None)
                else:
                    new_map.append(ctr)
                    ctr += 1
            self._map = new_map
        else:
            raise ValueError("Specified modes for deleting are invalid.")

    def add(self, num_modes):
        """Adds a mode"""
        num_active_modes = len([m for m in self._map if m is not None])
        self._map += [k for k in range(num_active_modes, num_active_modes + num_modes)]


class BaseBackend:
    """Abstract base class for backends."""
    # pylint: disable=too-many-public-methods

    def __init__(self):
        self._supported = {}

    def __str__(self):
        """String representation."""
        # defaults to the class name
        return self.__class__.__name__

    def supports(self, name):
        """Check whether the backend supports the given operating mode.

        Currently supported operating modes are:

        * "gaussian": for manipulations in the Gaussian representation using the
          displacements and covariance matrices
        * "fock_basis": for manipulations in the Fock representation
        * "mixed_states": for representations where the quantum state is mixed
        * "batched": allows for a multiple circuits to be simulated in parallel

        Args:
            name (str): name of the operating mode which we are checking support for

        Returns:
            bool: True if this backend supports that operating mode.
        """
        return self._supported.get(name, False)

    def begin_circuit(self, num_subsystems, cutoff_dim=None, hbar=2, pure=True, **kwargs):
        r"""Instantiate a quantum circuit.

        Instantiates a circuit with num_subsystems modes to track and update a quantum optical state.
        The state of the circuit is initialized to vacuum.

        The modes in the circuit are indexed sequentially using integers, starting from zero.
        Once an index is assigned to a mode, it can never be re-assigned to another mode.
        If the mode is deleted its index becomes invalid.
        An operation acting on an invalid or unassigned mode index raises an IndexError exception.

        Args:
            num_subsystems (int): number of modes in the circuit
            cutoff_dim (int): numerical Hilbert space cutoff dimension (used for circuits operating in Fock basis)
            hbar (float): The value of :math:`\hbar` to initialise the circuit with, depending on the conventions followed.
                By default, :math:`\hbar=2`. See :ref:`conventions` for more details.
            pure (bool): whether to initialize the circuit in a pure state (will use a mixed state if pure is False)
        """
        pass  # BaseBackend can be instantiated for testing purposes, even though it does not do anything.

    def add_mode(self, n=1):
        """Add one or more modes to the circuit.

        The new modes are initialized to the vacuum state.
        They are assigned mode indices sequentially, starting from the first unassigned index.

        Args:
            n (int): number of modes to add

        Returns:
            list[int]: indices of the newly added modes
        """
        raise NotImplementedError

    def del_mode(self, modes):
        """Delete one or more modes from the circuit.

        The deleted modes are traced out.
        As a result the state may have to be described using a density matrix.

        The indices of the deleted modes become invalid for the lifetime of the circuit object.
        They will never be reassigned to other modes.
        Deleting a mode that has already been deleted raises an IndexError exception.

        Args:
            modes (Sequence[int]): list of mode numbers to delete
        """
        raise NotImplementedError

    def get_modes(self):
        """Return a list of the active mode indices for the circuit.

        Returns:
            list[int]: sorted list of active (assigned, not invalid) mode indices
        """
        raise NotImplementedError

    def reset(self, pure=True, **kwargs):
        """Reset the circuit so that all the modes are in the vacuum state.

        After the reset the circuit is in the same state as it was after
        the last :meth:`begin_circuit` call. It will have the original number
        of modes, all initialized in the vacuum state. Some circuit parameters
        may be changed during the reset, see the keyword args below.

        Args:
            pure (bool): if True, initialize the circuit in a pure state (will use a mixed state if pure is False)

        Keyword Args:
            cutoff_dim (int): new Hilbert space truncation dimension (for Fock basis backends only)
        """
        raise NotImplementedError

    def prepare_vacuum_state(self, mode):
        """Prepare the vacuum state in the specified mode.

        The requested mode is traced out and replaced with the vacuum state.
        As a result the state may have to be described using a density matrix.

        Args:
            mode (int): which mode to prepare the vacuum state in
        """
        raise NotImplementedError

    def prepare_coherent_state(self, alpha, mode):
        r"""Prepare a coherent state in the specified mode.

        The requested mode is traced out and replaced with the coherent state :math:`\ket{\alpha}`.
        As a result the state may have to be described using a density matrix.

        Args:
            alpha (complex): coherent state displacement parameter
            mode (int): which mode to prepare the coherent state in
        """
        raise NotImplementedError

    def prepare_squeezed_state(self, r, phi, mode):
        r"""Prepare a squeezed vacuum state in the specified mode.

        The requested mode is traced out and replaced with the squeezed state :math:`\ket{z}`,
        where :math:`z=re^{i\phi}`.
        As a result the state may have to be described using a density matrix.

        Args:
            r (float): squeezing amplitude
            phi (float): squeezing angle
            mode (int): which mode to prepare the squeezed state in
        """
        raise NotImplementedError

    def prepare_displaced_squeezed_state(self, alpha, r, phi, mode):
        r"""Prepare a displaced squeezed state in the specified mode.

        The requested mode is traced out and replaced with the displaced squeezed
        state state :math:`\ket{\alpha, z}`, where :math:`z=re^{i\phi}`.
        As a result the state may have to be described using a density matrix.

        Args:
            alpha (complex): displacement parameter
            r (float): squeezing amplitude
            phi (float): squeezing angle
            mode (int): which mode to prepare the squeezed state in
        """
        raise NotImplementedError

    def prepare_thermal_state(self, nbar, mode):
        r"""Prepare a thermal state in the specified mode.

        The requested mode is traced out and replaced with the thermal state :math:`\rho(nbar)`.
        As a result the state will be described using a density matrix.

        Args:
            nbar (float): thermal population of the mode
            mode (int): which mode to prepare the thermal state in
        """
        raise NotImplementedError

    def rotation(self, phi, mode):
        """Apply the phase-space rotation operation to the specified mode.

        Args:
            phi (float): rotation angle
            mode (int): which mode to apply the rotation to
        """
        raise NotImplementedError

    def displacement(self, alpha, mode):
        """Apply the displacement operation to the specified mode.

        Args:
            alpha (complex): displacement parameter
            mode (int): which mode to apply the displacement to
        """
        raise NotImplementedError

    def squeeze(self, z, mode):
        """Apply the squeezing operation to the specified mode.

        Args:
            z (complex): squeezing parameter
            mode (int): which mode to apply the squeeze to
        """
        raise NotImplementedError

    def beamsplitter(self, t, r, mode1, mode2):
        """Apply the beamsplitter operation to the specified modes.

        Args:
            t (float): transmitted amplitude
            r (complex): reflected amplitude (with phase)
            mode1 (int): first mode that beamsplitter acts on
            mode2 (int): second mode that beamsplitter acts on
        """
        raise NotImplementedError

    def loss(self, T, mode):
        r"""Perform a loss channel operation on the specified mode.

        Args:
            T (float): loss parameter, :math:`0\leq T\leq 1`.
            mode (int): index of mode where operation is carried out
        """
        raise NotImplementedError

    def measure_homodyne(self, phi, mode, select=None, **kwargs):
        r"""Measure a :ref:`phase space quadrature <homodyne>` of the given mode.

        For the measured mode, samples the probability distribution
        :math:`f(q) = \bra{q}_x R^\dagger(\phi) \rho R(\phi) \ket{q}_x`
        and returns the sampled value.

        Updates the current state of the circuit such that the measured mode is reset
        to the vacuum state. This is because we cannot represent exact position or
        momentum eigenstates in any of the backends, and experimentally the photons
        are destroyed in a homodyne measurement.

        Args:
            phi (float): phase angle of the quadrature to measure (x: :math:`\phi=0`, p: :math:`\phi=\pi/2`)
            mode (int): which mode to measure
            select (float): (Optional) desired values of measurement results.
                Allows user to post-select on specific measurement results instead of randomly sampling.
            **kwargs: can be used to pass user-specified numerical parameters to the backend.
                Options for such arguments will be documented in the respective subclasses.

        Returns:
            float: measured value
        """
        raise NotImplementedError

    def is_vacuum(self, tol=0.0, **kwargs):
        r"""Test whether the current circuit state is in vacuum (up to tolerance tol).

        Args:
            tol (float): numerical tolerance for how close state must be to true vacuum state

        Returns:
            bool: True if vacuum state up to tolerance tol
        """
        raise NotImplementedError

    def state(self, modes=None, **kwargs):
        r"""Returns the state of the quantum simulation, restricted to the subsystems defined by `modes`.

        Args:
            modes (int or Sequence[int]): specifies the mode(s) to restrict the return state to
                This argument is optional; the default value ``modes=None`` returns the state containing all modes.
                If modes is not ordered, the returned state contains the requested modes in the given order, i.e.,
                requesting the modes=[3,1] results in a two mode state being returned with the first mode being
                subsystem 3 and the second mode being subsystem 1 of simulator.
        Returns:
            An instance of the Strawberry Fields State class, suited to the particular backend.
        """
        raise NotImplementedError


#=============================
# Fock-basis backends
#=============================

class BaseFock(BaseBackend):
    """Abstract base class for backends capable of Fock state manipulation."""

    def __init__(self):
        super().__init__()
        self._supported["fock_basis"] = True

    def get_cutoff_dim(self):
        """Returns the Hilbert space cutoff dimension used.

        Returns:
            int: cutoff dimension
        """
        raise NotImplementedError

    def prepare_fock_state(self, n, mode):
        r"""Prepare a Fock state in the specified mode.

        The requested mode is traced out and replaced with the Fock state :math:`\ket{n}`.
        As a result the state may have to be described using a density matrix.

        Args:
            n (int): Fock state to prepare
            mode (int): which mode to prepare the fock state in
        """
        raise NotImplementedError

    def prepare_ket_state(self, state, modes):
        r"""Prepare the given ket state (in the Fock basis) in the specified modes.

        The requested mode(s) is/are traced out and replaced with the given ket state
        (in the Fock basis). As a result the state may have to be described using a
        density matrix.

        Args:
            state (array): state in the Fock basis
                The state can be given in either vector form, with one index,
                or tensor form, with one index per mode. For backends supporting batched
                mode, state can be a batch of such vectors or tensors.
            modes (int or Sequence[int]): which mode to prepare the state in
                If modes is not ordered this is taken into account when preparing the state,
                i.e., when a two mode state is prepared in modes=[3,1], then the first
                mode of state goes into mode 3 and the second mode goes into mode 1 of the simulator.
        """
        raise NotImplementedError

    def prepare_dm_state(self, state, modes):
        r"""Prepare the given dm state (in the Fock basis) in the specified modes.

        The requested mode(s) is/are traced out and replaced with the given dm state (in the Fock basis).
        As a result the state will be described using a density matrix.

        Args:
            state (array): state in the Fock basis
                The state can be given in either matrix form, with two indices, or tensor
                form, with two indices per mode. For backends supporting batched mode,
                state can be a batch of such matrices or tensors.
            modes (int or Sequence[int]): which mode to prepare the state in
                If modes is not ordered this is take into account when preparing the
                state, i.e., when a two mode state is prepared in modes=[3,1], then
                the first mode of state goes into mode 3 and the second mode goes
                into mode 1 of the simulator.
        """
        raise NotImplementedError


    def cubic_phase(self, gamma, mode):
        r"""Apply the cubic phase operation to the specified mode.

        .. warning::
            The cubic phase gate can suffer heavily from numerical inaccuracies
            due to finite-dimensional cutoffs in the Fock basis. The gate
            implementation in Strawberry Fields is unitary, but it
            does not implement an exact cubic phase gate. The Kerr gate
            provides an alternative non-Gaussian gate.

        Args:
            gamma (float): cubic phase shift
            mode (int): which mode to apply it to
        """
        raise NotImplementedError

    def kerr_interaction(self, kappa, mode):
        r"""Apply the Kerr interaction :math:`\exp{(i\kappa \hat{n}^2)}` to the specified mode.

        Args:
            kappa (float): strength of the interaction
            mode (int): which mode to apply it to
        """
        raise NotImplementedError

    def cross_kerr_interaction(self, kappa, mode1, mode2):
        r"""Apply the two mode cross-Kerr interaction :math:`\exp{(i\kappa \hat{n}_1\hat{n}_2)}` to the specified modes.

        Args:
            kappa (float): strength of the interaction
            mode1 (int): first mode that cross-Kerr interaction acts on
            mode2 (int): second mode that cross-Kerr interaction acts on
        """
        raise NotImplementedError

    def measure_fock(self, modes, select=None, **kwargs):
        """Measure the given modes in the Fock basis.

        Updates the current state of the circuit to the conditional state of this measurement result.

        Args:
            modes (Sequence[int]): which modes to measure
            select (Sequence[int]): (Optional) desired values of measurement results.
                Allows user to post-select on specific measurement results instead of randomly sampling.

        Returns:
            tuple[int]: corresponding measurement results
        """
        raise NotImplementedError

    def state(self, modes=None, **kwargs):
        r"""Returns the state of the quantum simulation, restricted to the subsystems defined by `modes`.

        Args:
            modes (int or Sequence[int]): specifies the mode or modes to restrict the return state to.
                This argument is optional; the default value ``modes=None`` returns the state containing all modes.
        Returns:
            An instance of the Strawberry Fields FockState class.
        """
        raise NotImplementedError

#==============================
# Gaussian-formulation backends
#==============================

class BaseGaussian(BaseBackend):
    """Abstract base class for backends that are only capable of Gaussian state manipulation."""

    def __init__(self):
        super().__init__()
        self._supported["gaussian"] = True

    def measure_heterodyne(self, mode, select=None):
        r"""Perform a heterodyne measurement on the given mode.

        Updates the current state of the circuit such that the measured mode is reset to the vacuum state.

        Args:
            modes (Sequence[int]): which modes to measure
            select (complex): (Optional) desired values of measurement result.
                Allows user to post-select on specific measurement results instead of randomly sampling.

        Returns:
            complex: measured values
        """
        raise NotImplementedError

    def prepare_gaussian_state(self, r, V, modes):
        r"""Prepare the given Gaussian state (via the provided vector of
        means and the covariance matrix) in the specified modes.

        The requested mode(s) is/are traced out and replaced with the given Gaussian state.

        Args:
            r (array): the vector of means in xp ordering.
            V (array): the covariance matrix in xp ordering.
            modes (int or Sequence[int]): which mode to prepare the state in
                If the modes are not sorted, this is take into account when preparing the state.
                i.e., when a two mode state is prepared in modes=[3,1], then the first
                mode of state goes into mode 3 and the second mode goes into mode 1 of the simulator.
        """
        raise NotImplementedError

    def get_cutoff_dim(self):
        # pylint: disable=unused-argument,missing-docstring
        raise NotApplicableError

    def prepare_fock_state(self, n, mode):
        # pylint: disable=unused-argument,missing-docstring
        raise NotApplicableError

    def prepare_ket_state(self, state, mode):
        # pylint: disable=unused-argument,missing-docstring
        raise NotApplicableError

    def prepare_dm_state(self, state, mode):
        # pylint: disable=unused-argument,missing-docstring
        raise NotApplicableError

    def cubic_phase(self, gamma, mode):
        # pylint: disable=unused-argument,missing-docstring
        raise NotApplicableError

    def kerr_interaction(self, kappa, mode):
        # pylint: disable=unused-argument,missing-docstring
        raise NotApplicableError

    def cross_kerr_interaction(self, kappa, mode1, mode2):
        # pylint: disable=unused-argument,missing-docstring
        raise NotApplicableError

    def measure_fock(self, modes, select=None):
        # pylint: disable=unused-argument,missing-docstring
        raise NotApplicableError

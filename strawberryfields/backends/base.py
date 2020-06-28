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
r"""This module contains the abstract base classes that define Strawberry Fields
compatible statevector simulator backends."""

# pylint: disable=no-self-use,missing-docstring


class NotApplicableError(TypeError):
    """Exception raised by the backend when the user attempts an unsupported operation.
    E.g. :meth:`measure_fock` on a Gaussian backend.
    Conceptually different from NotImplementedError (which means "not implemented, but at some point may be").
    """


class ModeMap:
    """
    Simple internal class for maintaining a map of existing modes.
    """

    def __init__(self, num_subsystems):
        self._init = num_subsystems
        #: list[int]: _map[k] is the internal index used by the backend for
        # computational mode k, or None if the mode has been deleted
        self._map = list(range(num_subsystems))

    def reset(self):
        """reset the modemap to the initial state"""
        self._map = list(range(self._init))

    def _single_mode_valid(self, mode):
        if mode is None:
            return False

        if 0 <= mode < len(self._map):
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
        self._map += list(range(num_active_modes, num_active_modes + num_modes))


class BaseBackend:
    """Abstract base class for backends."""

    # pylint: disable=too-many-public-methods

    #: str: short name of the backend
    short_name = "base"
    #: str, None: Short name of the Compiler class used to validate Programs for this backend. None if no validation is required.
    compiler = None

    def __init__(self):
        self._supported = {}

    def __str__(self):
        """String representation."""
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
            cutoff_dim (int): Hilbert space truncation dimension (for Fock basis backends only)
            batch_size (int): (optional) batch-axis dimension, enables batched operation if > 1 (for the TF backend only)
        """
        # BaseBackend can be instantiated for testing purposes, even though it does not do anything.

    def add_mode(self, n=1):
        """Add modes to the circuit.

        The new modes are initialized to the vacuum state.
        They are assigned mode indices sequentially, starting from the first unassigned index.

        Args:
            n (int): number of modes to add

        Returns:
            list[int]: indices of the newly added modes
        """
        raise NotImplementedError

    def del_mode(self, modes):
        """Delete modes from the circuit.

        The deleted modes are traced out.
        As a result the state may have to be described using a density matrix.

        The indices of the deleted modes become invalid for the lifetime of the circuit object.
        They will never be reassigned to other modes.
        Deleting a mode that has already been deleted raises an ``IndexError`` exception.

        Args:
            modes (Sequence[int]): mode numbers to delete
        """
        raise NotImplementedError

    def get_modes(self):
        """Return a list of the active modes for the circuit.

        A mode is active if it has been created and has not been deleted.

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
            pure (bool): if True, initialize the circuit in a pure state representation
                (will use a mixed state representation if pure is False)

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

    def prepare_coherent_state(self, r, phi, mode):
        r"""Prepare a coherent state in the specified mode.

        The requested mode is traced out and replaced with the coherent state :math:`\ket{r e^{i\phi}}`.
        As a result the state may have to be described using a density matrix.

        Args:
            r (float): coherent state displacement amplitude
            phi (float): coherent state displacement phase
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

    def prepare_displaced_squeezed_state(self, r_d, phi_d, r_s, phi_s, mode):
        r"""Prepare a displaced squeezed state in the specified mode.

        The requested mode is traced out and replaced with the displaced
        squeezed state :math:`\ket{\alpha, z}`, where :math:`\alpha=r_d
        e^{i\phi_d}` and :math:`z=r_s e^{i\phi_s}`.
        As a result the state may have to be described using a density matrix.

        Args:
            r_d (float): displacement amplitude
            phi_d (float): displacement angle
            r_s (float): squeezing amplitude
            phi_s (float): squeezing angle
            mode (int): which mode to prepare the squeezed state in
        """
        raise NotImplementedError

    def prepare_thermal_state(self, nbar, mode):
        r"""Prepare a thermal state in the specified mode.

        The requested mode is traced out and replaced with the thermal state :math:`\rho(nbar)`.
        As a result the state may have to be described using a density matrix.

        Args:
            nbar (float): thermal population (mean photon number) of the mode
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

    def displacement(self, r, phi, mode):
        """Apply the displacement operation to the specified mode.

        Args:
            r (float): displacement amplitude
            phi(float): displacement angle
            mode (int): which mode to apply the displacement to
        """
        raise NotImplementedError

    def squeeze(self, r, phi, mode):
        """Apply the squeezing operation to the specified mode.

        Args:
            r (float): squeezing amplitude
            phi(float): squeezing angle
            mode (int): which mode to apply the squeeze to
        """
        raise NotImplementedError

    def beamsplitter(self, theta, phi, mode1, mode2):
        """Apply the beamsplitter operation to the specified modes.

        Args:
            theta (float): transmissivity is cos(theta)
            phi (float): phase angle
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

    def thermal_loss(self, T, nbar, mode):
        r"""Perform a thermal loss channel operation on the specified mode.

        Args:
            T (float): loss parameter, :math:`0\leq T\leq 1`.
            nbar (float): mean photon number of the environment thermal state
            mode (int): index of mode where operation is carried out
        """
        raise NotImplementedError

    def measure_homodyne(self, phi, mode, shots=1, select=None, **kwargs):
        r"""Measure a :ref:`phase space quadrature <homodyne>` of the given mode.

        For the measured mode, samples the probability distribution
        :math:`f(q) = \bra{q_\phi} \rho \ket{q_\phi}`
        and returns the sampled value.
        Here :math:`\ket{q_\phi}` is the eigenstate of the operator

        .. math::
           \hat{q}_\phi = \sqrt{2/\hbar}(\cos(\phi)\hat{x} +\sin(\phi)\hat{p}) = e^{-i\phi} \hat{a} +e^{i\phi} \hat{a}^\dagger.

        .. note::
           This method is :math:`\hbar` independent.
           The returned values can be converted to conventional position/momentum
           eigenvalues by multiplying them with :math:`\sqrt{\hbar/2}`.

        Updates the current state such that the measured mode is reset
        to the vacuum state. This is because we cannot represent exact position or
        momentum eigenstates in any of the backends, and experimentally the photons
        are destroyed in a homodyne measurement.

        Args:
            phi (float): phase angle of the quadrature to measure (x: :math:`\phi=0`, p: :math:`\phi=\pi/2`)
            mode (int): which mode to measure
            shots (int): number of measurement samples to obtain
            select (None or float): If not None: desired value of the measurement result.
                Enables post-selection on specific measurement results instead of random sampling.

        Keyword arguments can be used to pass additional parameters to the backend.
        Options for such arguments will be documented in the respective subclasses.

        Returns:
            float: measured value
        """
        raise NotImplementedError

    def measure_fock(self, modes, shots=1, select=None, **kwargs):
        """Measure the given modes in the Fock basis.

        .. note::
          When ``shots == 1``, updates the current system state to the
          conditional state of that measurement result. When ``shots > 1``, the
          system state is not updated.

        Args:
            modes (Sequence[int]): which modes to measure
            shots (int): number of measurement samples to obtain
            select (None or Sequence[int]): If not None: desired values of the measurement results.
                Enables post-selection on specific measurement results instead of random sampling.
                ``len(select) == len(modes)`` is required.
        Returns:
            tuple[int]: measurement results
        """
        raise NotImplementedError

    def measure_threshold(self, modes, shots=1, select=None, **kwargs):
        """Measure the given modes in the thresholded Fock basis, i.e., zero or nonzero photons).

        .. note::

            When :code:``shots == 1``, updates the current system state to the conditional state of that
            measurement result. When :code:``shots > 1``, the system state is not updated.

        Args:
            modes (Sequence[int]): which modes to measure
            shots (int): number of measurement samples to obtain
            select (None or Sequence[int]): If not None: desired values of the measurement results.
                Enables post-selection on specific measurement results instead of random sampling.
                ``len(select) == len(modes)`` is required.
        Returns:
            tuple[int]: measurement results
        """
        raise NotImplementedError

    def is_vacuum(self, tol=0.0, **kwargs):
        r"""Test whether the current circuit state is vacuum (up to given tolerance).

        Returns True iff :math:`|\bra{0} \rho \ket{0} -1| \le` ``tol``, i.e.,
        the fidelity of the current circuit state with the vacuum state is within
        the given tolerance from 1.

        Args:
            tol (float): numerical tolerance

        Returns:
            bool: True iff current state is vacuum up to tolerance tol
        """
        raise NotImplementedError

    def state(self, modes=None, **kwargs):
        r"""Returns the state of the quantum simulation.

        Args:
            modes (int or Sequence[int] or None): Specifies the modes to restrict the return state to.
                None returns the state containing all the modes.
                The returned state contains the requested modes in the given order, i.e.,
                ``modes=[3,0]`` results in a two mode state being returned with the first mode being
                subsystem 3 and the second mode being subsystem 0.
        Returns:
            BaseState: state description, specific child class depends on the backend
        """
        raise NotImplementedError


# =============================
# Fock-basis backends
# =============================


class BaseFock(BaseBackend):
    """Abstract base class for backends capable of Fock state manipulation."""

    compiler = "fock"

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
            mode (int): which mode to prepare the Fock state in
        """
        raise NotImplementedError

    def prepare_ket_state(self, state, modes):
        r"""Prepare the given ket state in the specified modes.

        The requested modes are traced out and replaced with the given ket state
        (in the Fock basis). As a result the state may have to be described using a
        density matrix.

        Args:
            state (array): Ket state in the Fock basis.
                The state can be given in either vector form, with one index,
                or tensor form, with one index per mode. For backends supporting batched
                mode, state can be a batch of such vectors or tensors.
            modes (int or Sequence[int]): Modes to prepare the state in.
                If modes is not ordered this is taken into account when preparing the state,
                i.e., when a two mode state is prepared in modes=[3,1], then the first
                mode of state goes into mode 3 and the second mode goes into mode 1 of the simulator.
        """
        raise NotImplementedError

    def prepare_dm_state(self, state, modes):
        r"""Prepare the given mixed state in the specified modes.

        The requested modes are traced out and replaced with the given density matrix
        state (in the Fock basis).
        As a result the state will be described using a density matrix.

        Args:
            state (array): Density matrix in the Fock basis.
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

        Applies the operation

        .. math::
           \exp\left(i \frac{\gamma}{6} (\hat{a} +\hat{a}^\dagger)^3\right)

        to the specified mode.

        .. note::
           This method is :math:`\hbar` independent.
           The usual definition of the cubic phase gate is :math:`\hbar` dependent:

           .. math::
              V(\gamma') = \exp\left(i \frac{\gamma'}{3\hbar} \hat{x}^3\right) = \exp\left(i \frac{\gamma' \sqrt{\hbar/2}}{6} (\hat{a} +\hat{a}^\dagger)^3\right).

           Hence the cubic phase gate :math:`V(\gamma')` is executed on a backend by scaling the
           :math:`\gamma'` parameter by :math:`\sqrt{\hbar/2}` and then passing it to this method,
           much in the way the :math:`\hbar` dependent `X` and `Z` gates are implemented through the
           :math:`\hbar` independent :meth:`~BaseBackend.displacement` method.

        .. warning::
            The cubic phase gate can suffer heavily from numerical inaccuracies
            due to finite-dimensional cutoffs in the Fock basis. The gate
            implementation in Strawberry Fields is unitary, but it
            does not implement an exact cubic phase gate. The Kerr gate
            provides an alternative non-Gaussian gate.

        Args:
            gamma (float): scaled cubic phase shift, :math:`\gamma = \gamma' \sqrt{\hbar/2}`
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

    def state(self, modes=None, **kwargs):
        r"""Returns the state of the quantum simulation.

        See :meth:`.BaseBackend.state`.

        Returns:
            BaseFockState: state description
        """
        raise NotImplementedError


# ==============================
# Gaussian-formulation backends
# ==============================


class BaseGaussian(BaseBackend):
    """Abstract base class for backends that are only capable of Gaussian state manipulation."""

    compiler = "gaussian"

    def __init__(self):
        super().__init__()
        self._supported["gaussian"] = True

    def measure_heterodyne(self, mode, shots=1, select=None):
        r"""Perform a heterodyne measurement on the given mode.

        Updates the current state of the circuit such that the measured mode is reset to the vacuum state.

        Args:
            mode (int): which mode to measure
            shots (int): number of measurement samples to obtain
            select (None or complex): If not None: desired value of the measurement result.
                Enables post-selection on specific measurement results instead of random sampling.

        Returns:
            complex: measured value
        """
        raise NotImplementedError

    def prepare_gaussian_state(self, r, V, modes):
        r"""Prepare a Gaussian state.

        The specified modes are traced out and replaced with a Gaussian state
        provided via a vector of means and a covariance matrix.

        .. note::
           This method is :math:`\hbar` independent.
           The input arrays are the means and covariance of the
           :math:`a+a^\dagger` and :math:`-i(a-a^\dagger)` operators.
           They are obtained by dividing the xp means by :math:`\sqrt{\hbar/2}`
           and the xp covariance by :math:`\hbar/2`.

        Args:
            r (array): vector of means in xp ordering
            V (array): covariance matrix in xp ordering
            modes (int or Sequence[int]): Which modes to prepare the state in.
                If the modes are not sorted, this is taken into account when preparing the state.
                I.e., when a two mode state is prepared with ``modes=[3,1]``, the first
                mode of the given state goes into mode 3 and the second mode goes into mode 1.
        """
        raise NotImplementedError

    def get_cutoff_dim(self):
        raise NotApplicableError

    def prepare_fock_state(self, n, mode):
        raise NotApplicableError

    def prepare_ket_state(self, state, mode):
        raise NotApplicableError

    def prepare_dm_state(self, state, mode):
        raise NotApplicableError

    def cubic_phase(self, gamma, mode):
        raise NotApplicableError

    def kerr_interaction(self, kappa, mode):
        raise NotApplicableError

    def cross_kerr_interaction(self, kappa, mode1, mode2):
        raise NotApplicableError

    def state(self, modes=None, **kwargs):
        """Returns the state of the quantum simulation.

        See :meth:`.BaseBackend.state`.

        Returns:
            BaseGaussianState: state description
        """
        raise NotImplementedError

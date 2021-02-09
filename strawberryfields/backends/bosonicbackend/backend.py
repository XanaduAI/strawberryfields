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
import numpy as np

from strawberryfields.backends import BaseBosonic
from strawberryfields.backends.shared_ops import changebasis
from strawberryfields.backends.states import BaseBosonicState

from .bosoniccircuit import BosonicModes


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

    def begin_circuit(self, num_subsystems, **kwargs):
        """Populates the circuit attribute with a BosonicModes object.

        Args:
            num_subsystems (int): Sets the number of modes in the circuit.
        """
        self._init_modes = num_subsystems
        self.circuit = BosonicModes(num_subsystems)

    def add_mode(self, modes=1, **kwargs):
        r"""Adds new modes to the circuit each a with number of Gaussian peaks
        specified by peaks.

        Args:
             peaks (list): number of Gaussian peaks for each new mode

        Raises:
            ValueError: if the length of the list of peaks is different than
            the number of modes.
        """
        peaks = kwargs.get("peaks", None)
        if peaks is None:
            peaks = list(np.ones(modes))
        if modes != len(peaks):
            raise ValueError("Please specify the number of peaks per new mode.")
        self.circuit.add_mode(peaks)

    def del_mode(self, modes):
        r"""Delete modes from the circuit.

        Args:
            modes (int or list): modes to be deleted.
        """
        self.circuit.del_mode(modes)

    def get_modes(self):
        r"""Return the modes that are currently active. Active modes
        are those created by the user which have not been deleted.
        If a mode is deleted, its entry in the list is ``None``."""
        return self.circuit.get_modes()

    def reset(self, pure=True, **kwargs):
        """Reset all modes in the circuit to vacuum."""
        self.circuit.reset(self._init_modes, 1)

    def prepare_thermal_state(self, nbar, mode):
        r"""Initializes a state of mode in a thermal state with the given population.

        Args:
            nbar (float): mean photon number of the thermal state
            mode (int): mode that get initialized
        """
        self.circuit.init_thermal(nbar, mode)

    def prepare_vacuum_state(self, mode):
        """Prepares a vacuum state in mode.

        Args:
            mode (int): mode to be converted to vacuum.
        """
        self.circuit.loss(0.0, mode)

    def prepare_coherent_state(self, r, phi, mode):
        r"""Create a coherent state in mode with alpha=``r*np.exp(1j*phi)``.

        Args:
            r (float): coherent state magnitude
            phi (float): coherent state phase
            mode (int): mode to be made into coherent state
        """
        self.circuit.loss(0.0, mode)
        self.circuit.displace(r, phi, mode)

    def prepare_squeezed_state(self, r, phi, mode):
        r"""Create a squeezed state in mode with squeezing ``r*exp(1j*phi)``.

        Args:
            r (float): squeezing magnitude
            phi (float): squeezing phase
            mode (int): mode to be made into a squeezed state

        Raises:
            ValueError: if the mode is not in the list of active modes
        """
        self.circuit.loss(0.0, mode)
        self.circuit.squeeze(r, phi, mode)

    def prepare_displaced_squeezed_state(self, r_d, phi_d, r_s, phi_s, mode):
        r"""Create a displaced, squeezed state in mode with squeezing
        ``r_s*exp(1j*phi_s)`` and displacement ``r_d*exp(1j*phi_d)``.

        Args:
            r_d (float): displacement magnitude
            phi_d (float): displacement phase
            r_s (float): squeezing magnitude
            phi_s (float): squeezing phase
            mode (int): mode to be made into a displaced, squeezed state
        """
        self.circuit.loss(0.0, mode)
        self.circuit.squeeze(r_s, phi_s, mode)
        self.circuit.displace(r_d, phi_d, mode)

    def rotation(self, phi, mode):
        r"""Implement a phase shift in mode by phi.

        Args:
           phi (float): phase
           mode (int): mode to be phase shifted
        """
        self.circuit.phase_shift(phi, mode)

    def displacement(self, r, phi, mode):
        r"""Displace mode by the amount ``r*np.exp(1j*phi)``.

        Args:
            r (float): displacement magnitude
            phi (float): displacement phase
            mode (int): mode to be displaced
        """
        self.circuit.displace(r, phi, mode)

    def squeeze(self, r, phi, mode):
        r"""Squeeze mode by the amount ``r*exp(1j*phi)``.

        Args:
            r (float): squeezing magnitude
            phi (float): squeezing phase
            mode (int): mode to be squeezed
        """
        self.circuit.squeeze(r, phi, mode)

    def mb_squeeze(self, mode, r, phi, r_anc, eta_anc, avg):
        r"""Squeeze mode by the amount ``r*exp(1j*phi)`` using measurement-based squeezing.
        Depending on avg, this applies the average or single-shot map, returning the ancillary measurement outcome.

        Args:
            k (int): mode to be squeezed
            r (float): target squeezing magnitude
            phi (float): target squeezing phase
            r_anc (float): squeezing magnitude of the ancillary mode
            eta_anc(float): detection efficiency of the ancillary mode
            avg (bool): whether to apply the average or single shot map

        Returns:
            float: if not avg, the measurement outcome of the ancilla
        """
        if avg:
            self.circuit.mb_squeeze_avg(mode, r, phi, r_anc, eta_anc)
            return None
        ancilla_val = self.circuit.mb_squeeze_single_shot(mode, r, phi, r_anc, eta_anc)
        return ancilla_val

    def beamsplitter(self, theta, phi, mode1, mode2):
        r"""Implement a beam splitter operation between mode1 and mode2.

        Args:
            theta (float): real beamsplitter angle
            phi (float): complex beamsplitter angle
            mode1 (int): first mode
            mode2 (int): second mode
        """
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
        r"""Measure a :ref:`phase space quadrature <homodyne>` of the given mode.

        See :meth:`.BaseBackend.measure_homodyne`.
        Args:
            phi (float): angle in phase space for the homodyne
            mode (int): mode to be measured
            shots (int): how many samples to collect
            select (float): if supplied, what value to postselect
        Keyword Args:
            eps (float): Homodyne amounts to projection onto a quadrature eigenstate.
                This eigenstate is approximated by a squeezed state whose variance has been
                squeezed to the amount ``eps``, :math:`V_\text{meas} = \texttt{eps}^2`.
                Perfect homodyning is obtained when ``eps`` :math:`\to 0`.

        Returns:
            array: measured values
        """
        # Phi is the rotation of the measurement operator, hence the minus
        self.circuit.phase_shift(-phi, mode)

        if select is None:
            val = self.circuit.homodyne(mode, **kwargs)[0, 0]
        else:
            val = select * 2 / np.sqrt(2 * self.circuit.hbar)
            self.circuit.post_select_homodyne(mode, val, **kwargs)

        return np.array([val * np.sqrt(2 * self.circuit.hbar) / 2])

    def measure_heterodyne(self, mode, shots=1, select=None):
        r"""Measure heterodyne of the given mode.

        Args:
            mode (int): mode to be measured
            shots (int): how many samples to collect
            select (complex): if supplied, what value to postselect

        Returns:
            array: measured values
        """
        if select is None:
            res = 0.5 * self.circuit.heterodyne(mode, shots=shots)
            return np.array([res[:, 0] + 1j * res[:, 1]])
        res = select
        self.circuit.post_select_heterodyne(mode, select)
        return res

    def prepare_gaussian_state(self, r, V, modes):
        r"""Prepares a Gaussian state on modes from the mean vector and covariance
        matrix.

        Args:
            r (array): vector of means in :math:`(x_1,p_1,x_2,p_2,\dots)` ordering
            V (array): covariance matrix in :math:`(x_1,p_1,x_2,p_2,\dots)` ordering
            modes (list): modes corresponding to the covariance matrix entries

        Raises:
            ValueError: if the shapes of r or V do not match the number of modes.
        """
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

        self.circuit.from_covmat(cov, modes)
        self.circuit.from_mean(means, modes)

    def is_vacuum(self, tol=1e-12, **kwargs):
        """Determines whether or not the state is vacuum.

        Args:
            tol (float): how close to 1 the fidelity must be

        Returns:
            bool: whether the state is vacuum
        """
        fid = self.state().fidelity_vacuum()
        return np.abs(fid - 1) <= tol

    def loss(self, T, mode):
        r"""Implements a loss channel in mode. T is the loss parameter that must be
        between 0 and 1.

        Args:
            T (float): loss amount is \sqrt{T}
            mode (int): mode that loses energy
        """
        self.circuit.loss(T, mode)

    def thermal_loss(self, T, nbar, mode):
        r"""Implements the thermal loss channel in mode. T is the loss parameter that must
        be between 0 and 1.

        Args:
            T (float): loss amount is \sqrt{T}
            nbar (float): mean photon number of the thermal bath
            mode (int): mode that undegoes thermal loss
        """
        self.circuit.thermal_loss(T, nbar, mode)

    def measure_fock(self, modes, shots=1, select=None, **kwargs):
        raise NotImplementedError("Bosonic backend does not yet support Fock" "measurements")

    def measure_threshold(self, modes, shots=1, select=None, **kwargs):
        raise NotImplementedError("Bosonic backend does not yet support threshold" "measurements")

    def state(self, modes=None, **kwargs):
        """Returns the state of the quantum simulation.

        See :meth:`.BaseBackend.state`.

        Returns:
            BosonicState: object containing all state information
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

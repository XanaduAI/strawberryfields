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
            the number of modes.
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

    def rotation(self, phi, mode):
        self.circuit.phase_shift(phi, mode)

    def displacement(self, r, phi, mode):
        self.circuit.displace(r, phi, mode)

    def squeeze(self, r, phi, mode):
        self.circuit.squeeze(r, phi, mode)

    def mb_squeeze(self, mode, r, phi, r_anc, eta_anc, avg):
        r"""Squeeze mode by the amount ``r*exp(1j*phi)`` using measurement-based squeezing.

        Depending on avg, this applies the average or single-shot map, returning the ancillary
        measurement outcome.

        Args:
            k (int): mode to be squeezed
            r (float): target squeezing magnitude
            phi (float): target squeezing phase
            r_anc (float): squeezing magnitude of the ancillary mode
            eta_anc(float): detection efficiency of the ancillary mode
            avg (bool): whether to apply the average or single-shot map

        Returns:
            float or None: if not avg, the measurement outcome of the ancilla
        """
        if avg:
            self.circuit.mb_squeeze_avg(mode, r, phi, r_anc, eta_anc)
            return None
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

        return np.array([val * np.sqrt(2 * self.circuit.hbar) / 2])

    def measure_heterodyne(self, mode, shots=1, select=None):
        if select is None:
            res = 0.5 * self.circuit.heterodyne(mode, shots=shots)
            return np.array([res[:, 0] + 1j * res[:, 1]])
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
        raise NotImplementedError("Bosonic backend does not yet support Fock measurements")

    def measure_threshold(self, modes, shots=1, select=None, **kwargs):
        raise NotImplementedError("Bosonic backend does not yet support threshold measurements")

    def state(self, modes=None, **kwargs):
        """Returns the state of the quantum simulation.

        See :meth:`.BaseBackend.state`.

        For the bosonic backend, mode indices are sorted in ascending order.

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

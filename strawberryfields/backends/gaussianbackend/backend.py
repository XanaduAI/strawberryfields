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
"""Gaussian backend"""
from numpy import empty, concatenate, array, identity, arctan2, angle, sqrt, dot, vstack
from numpy.linalg import inv

from strawberryfields.backends import BaseGaussian
from strawberryfields.backends.shared_ops import changebasis

from .ops import xmat
from .gaussiancircuit import GaussianModes
from .states import GaussianState


class GaussianBackend(BaseGaussian):
    """Gaussian backend implementation"""
    def __init__(self):
        """
        Instantiate a GaussianBackend object.
        """
        super().__init__()
        self._supported["mixed_states"] = True
        self._short_name = "gaussian"

    def begin_circuit(self, num_subsystems, cutoff_dim=None, hbar=2, pure=None, **kwargs):
        r"""
        Create a quantum circuit (initialized in vacuum state) with number of modes
        equal to num_subsystems.

        Args:
            num_subsystems (int): number of modes the circuit should begin with
            hbar (float): The value of :math:`\hbar` to initialise the circuit with, depending on the conventions followed.
                By default, :math:`\hbar=2`. See :ref:`conventions` for more details.
        """
        # pylint: disable=attribute-defined-outside-init
        self._init_modes = num_subsystems
        self.circuit = GaussianModes(num_subsystems, hbar)

    def add_mode(self, n=1):
        """
        Add n new modes to the underlying circuit state. Indices for new modes
        always occur at the end of the covariance matrix.

        Note: this will increase the number of indices used for the state representation.

        Args:
            n (int): the number of modes to be added to the circuit.
        """
        self.circuit.add_mode(n)

    def del_mode(self, modes):
        """
        Trace out the specified modes from the underlying circuit state.

        Note: this will reduce the number of indices used for the state representation.

        Args:
            modes (list[int]): the modes to be removed from the circuit.
        """
        self.circuit.del_mode(modes)

    def get_modes(self):
        """
        Return the indices of the modes that are active (the ones that have not been deleted).

        Returns:
            active modes (list[int]): the active modes in the circuit
        """
        return self.circuit.get_modes()

    def reset(self, pure=True, **kwargs):
        """
        Resets the circuit state back to an all-vacuum state.
        """
        self.circuit.reset(self._init_modes)

    def prepare_thermal_state(self, nbar, mode):
        """
        Prepare the vacuum state on the specified mode.

        Note: this may convert the state representation to mixed.

        Args:
            nbar (float): mean number of photons in the mode
            mode (int): index of mode where state is prepared
        """
        self.circuit.init_thermal(nbar, mode)

    def prepare_vacuum_state(self, mode):
        """
        Prepare the vacuum state on the specified mode.

        Note: this may convert the state representation to mixed.

        Args:
            mode (int): index of mode where state is prepared
        """
        self.circuit.loss(0.0, mode)

    def prepare_coherent_state(self, alpha, mode):
        """
        Prepare a coherent state with parameter alpha on the specified mode.

        Args:
            alpha (complex): coherent state displacement parameter
            mode (int): index of mode where state is prepared
        """

        self.circuit.loss(0.0, mode)
        self.circuit.displace(alpha, mode)

    def prepare_squeezed_state(self, r, phi, mode):
        """
        Prepare a coherent state with parameters (r, phi) on the specified mode.

        Args:
            r (float): squeezing amplitude
            phi (float): squeezing phase
            mode (int): index of mode where state is prepared
        """
        self.circuit.loss(0.0, mode)
        self.circuit.squeeze(r, phi, mode)

    def rotation(self, phi, mode):
        """
        Perform a phase-space rotation by angle phi on the specified mode.

        Args:
            phi (float):
            mode (int): index of mode where operation is carried out
        """
        self.circuit.phase_shift(phi, mode)

    def displacement(self, alpha, mode):
        """
        Perform a displacement operation on the specified mode.

        Args:
            alpha (float): displacement parameter
            mode (int): index of mode where operation is carried out
        """
        self.circuit.displace(alpha, mode)

    def squeeze(self, z, mode):
        """
        Perform a squeezing operation on the specified mode.

        Args:
            z (complex): squeezing parameter
            mode (int): index of mode where operation is carried out

        """
        phi = angle(z)
        r = abs(z)
        self.circuit.squeeze(r, phi, mode)

    def prepare_displaced_squeezed_state(self, alpha, r, phi, mode):
        """
        Prepare a displaced squezed state with parameters (alpha, r, phi) on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            alpha (complex): displacement parameter
            r (float): squeezing amplitude
            phi (float): squeezing phase
            mode (int): index of mode where state is prepared
        """

        self.circuit.squeeze(r, phi, mode)
        self.circuit.displace(alpha, mode)

    def beamsplitter(self, t, r, mode1, mode2):
        """
        Perform a beamsplitter operation on the specified modes.

        It is assumed that :math:`|r|^2+|t|^2 = t^2+|r|^2=1`, i.e that t is real.

        Args:
            t (float): transmittivity parameter
            r (complex): reflectivity parameter
            mode1 (int): index of first mode where operation is carried out
            mode2 (int): index of second mode where operation is carried out
        """
        if isinstance(t, complex):
            raise ValueError("Beamsplitter transmittivity t must be a float.")
        theta = arctan2(abs(r), t)
        phi = angle(r)
        self.circuit.beamsplitter(-theta, -phi, mode1, mode2)

    def measure_homodyne(self, phi, mode, select=None, **kwargs):
        """
        Perform a homodyne measurement on the specified modes.

        .. note:: The rotation has its sign flipped since the rotation in this case
            is defined to act on the measurement effect and not in the state

        Args:
            phi (float): angle (relative to x-axis) for the measurement
            mode (int): index of mode where operation is carried out
            **kwargs: Can be used to (optionally) pass user-specified numerical parameter `eps`.
                                Homodyne amounts to projection onto a quadrature eigenstate. This eigenstate is approximated
                                by a squeezed state whose variance has been squeezed to the amount eps, V_(meas) = eps**2.
                                Perfect homodyning is obtained when eps \to 0.
        Returns:
            float or Tensor: measurement outcomes
        """
        if "eps" in kwargs:
            eps = kwargs["eps"]
        else:
            eps = 0.0002

        self.circuit.phase_shift(-phi, mode)

        if select is None:
            qs = self.circuit.homodyne(mode, eps)[0]
        else:
            val = select * 2/sqrt(2*self.circuit.hbar)
            qs = self.circuit.post_select_homodyne(mode, val, eps)

        return qs * sqrt(2*self.circuit.hbar)/2

    def measure_heterodyne(self, mode, select=None):
        """
        Perform a heterodyne measurement on the specified modes.

        Args:
            mode (int): index of mode where operation is carried out
        Returns:
            complex : measurement outcome
        """
        if select is None:
            m = identity(2)
            res = 0.5*self.circuit.measure_dyne(m, [mode])
            return res[0]+1j*res[1]

        res = select
        self.circuit.post_select_heterodyne(mode, select)

        return res

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
        if isinstance(modes, int):
            modes = [modes]

        # make sure number of modes matches shape of r and V
        N = len(modes)
        if len(r) != 2*N:
            raise ValueError("Length of means vector must be twice the number of modes.")
        if V.shape != (2*N, 2*N):
            raise ValueError("Shape of covariance matrix must be [2N, 2N], where N is the number of modes.")

        # convert xp-ordering to symmetric ordering
        means = vstack([r[:N], r[N:]]).reshape(-1, order='F')
        C = changebasis(N)
        cov = C @ V @ C.T

        self.circuit.fromscovmat(cov, modes)
        self.circuit.fromsmean(means, modes)

    def is_vacuum(self, tol=0.0, **kwargs):
        """
        Numerically check that the fidelity of the circuit state with vacuum is within
        tol of 1.

        Args:
            tol(float): value of the tolerance.
        """
        return self.circuit.is_vacuum(tol)

    def loss(self, T, mode):
        """
        Perform a loss channel operation on the specified mode.

        Args:
            T: loss parameter
            mode (int): index of mode where operation is carried out

        """
        self.circuit.loss(T, mode)

    def state(self, modes=None, **kwargs):
        """ Returns the vector of means and the covariance matrix of the specified modes.

        Args:
            modes (list[int]) : indices of the requested modes. If none, all modes returned.
        Returns:
            tuple (means, covmat) where means is a numpy array containing the mean values
            of the qaudratures and covmats is a numpy square array containing the covariance
            matrix of said modes
        """

        m = self.circuit.scovmat()
        r = self.circuit.smean()

        if modes is None:
            modes = list(range(len(self.get_modes())))

        listmodes = list(concatenate((2*array(modes), 2*array(modes)+1)))
        covmat = empty((2*len(modes), 2*len(modes)))
        means = r[listmodes]

        for i, ii in enumerate(listmodes):
            for j, jj in enumerate(listmodes):
                covmat[i, j] = m[ii, jj]

        means *= sqrt(2*self.circuit.hbar)/2
        covmat *= self.circuit.hbar/2

        mode_names = ["q[{}]".format(i) for i in array(self.get_modes())[modes]]

        # qmat and amat
        qmat = self.circuit.qmat()
        N = qmat.shape[0]//2

        # work out if qmat and Amat need to be reduced
        if 1 <= len(modes) < N:
            # reduce qmat
            ind = concatenate([array(modes), N+array(modes)])
            rows = ind.reshape((-1, 1))
            cols = ind.reshape((1, -1))
            qmat = qmat[rows, cols]

            # calculate reduced Amat
            N = qmat.shape[0]//2
            Amat = dot(xmat(N), identity(2*N)-inv(qmat))
        else:
            Amat = self.circuit.Amat()

        return GaussianState((means, covmat), len(modes), qmat, Amat,
                             hbar=self.circuit.hbar, mode_names=mode_names)

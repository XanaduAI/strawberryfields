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
# pylint: disable=too-many-public-methods
"""Gaussian backend"""
import warnings

from numpy import (
    empty,
    concatenate,
    array,
    identity,
    arctan2,
    angle,
    sqrt,
    dot,
    vstack,
    zeros_like,
    allclose,
    ix_,
)
from numpy.linalg import inv
from thewalrus.samples import hafnian_sample_state, torontonian_sample_state

from strawberryfields.backends import BaseGaussian
from strawberryfields.backends.shared_ops import changebasis

from .ops import xmat
from .gaussiancircuit import GaussianModes
from .states import GaussianState


class GaussianBackend(BaseGaussian):
    r"""The GaussianBackend implements a simulation of quantum optical circuits
    in NumPy using the Gaussian formalism, returning a :class:`~.GaussianState`
    state object.

    The primary component of the GaussianBackend is a
    :attr:`~.GaussianModes` object which is used to simulate a multi-mode quantum optical system.
    :class:`~.GaussianBackend` provides the basic API-compatible interface to the simulator, while the
    :attr:`~.GaussianModes` object actually carries out the mathematical simulation.

    The :attr:`GaussianModes` simulators maintain an internal covariance matrix & vector of means
    representation of a multi-mode quantum optical system.

    Note that unlike commonly used covariance matrix representations we encode our state in two complex
    matrices :math:`N` and :math:`M` that are defined as follows
    :math:`N_{i,j} = \langle a^\dagger _i a_j \rangle`
    :math:`M_{i,j} = \langle a _i a_j \rangle`
    and a vector of means :math:`\alpha_i =\langle a_i \rangle`.

    .. 
        .. currentmodule:: strawberryfields.backends.gaussianbackend
        .. autosummary::
            :toctree: api

            ~gaussiancircuit.GaussianModes
            ~ops
    """

    short_name = "gaussian"
    circuit_spec = "gaussian"

    def __init__(self):
        """Initialize the backend.
        """
        super().__init__()
        self._supported["mixed_states"] = True
        self._init_modes = None
        self.circuit = None

    def begin_circuit(self, num_subsystems, **kwargs):
        self._init_modes = num_subsystems
        self.circuit = GaussianModes(num_subsystems)

    def add_mode(self, n=1):
        self.circuit.add_mode(n)

    def del_mode(self, modes):
        self.circuit.del_mode(modes)

    def get_modes(self):
        return self.circuit.get_modes()

    def reset(self, pure=True, **kwargs):
        self.circuit.reset(self._init_modes)

    def prepare_thermal_state(self, nbar, mode):
        self.circuit.init_thermal(nbar, mode)

    def prepare_vacuum_state(self, mode):
        self.circuit.loss(0.0, mode)

    def prepare_coherent_state(self, alpha, mode):
        self.circuit.loss(0.0, mode)
        self.circuit.displace(alpha, mode)

    def prepare_squeezed_state(self, r, phi, mode):
        self.circuit.loss(0.0, mode)
        self.circuit.squeeze(r, phi, mode)

    def prepare_displaced_squeezed_state(self, alpha, r, phi, mode):
        self.circuit.loss(0.0, mode)
        self.circuit.squeeze(r, phi, mode)
        self.circuit.displace(alpha, mode)

    def rotation(self, phi, mode):
        self.circuit.phase_shift(phi, mode)

    def displacement(self, alpha, mode):
        self.circuit.displace(alpha, mode)

    def squeeze(self, z, mode):
        phi = angle(z)
        r = abs(z)
        self.circuit.squeeze(r, phi, mode)

    def beamsplitter(self, t, r, mode1, mode2):
        if isinstance(t, complex):
            raise ValueError("Beamsplitter transmittivity t must be a float.")
        theta = arctan2(abs(r), t)
        phi = angle(r)
        self.circuit.beamsplitter(-theta, -phi, mode1, mode2)

    def measure_homodyne(self, phi, mode, shots=1, select=None, **kwargs):
        r"""Measure a :ref:`phase space quadrature <homodyne>` of the given mode.

        See :meth:`.BaseBackend.measure_homodyne`.

        Keyword Args:
            eps (float): Homodyne amounts to projection onto a quadrature eigenstate.
                This eigenstate is approximated by a squeezed state whose variance has been
                squeezed to the amount ``eps``, :math:`V_\text{meas} = \texttt{eps}^2`.
                Perfect homodyning is obtained when ``eps`` :math:`\to 0`.

        Returns:
            float: measured value
        """
        if shots != 1:
            if select is not None:
                raise NotImplementedError(
                    "Gaussian backend currently does not support "
                    "postselection if shots != 1 for homodyne measurement"
                )

            raise NotImplementedError(
                "Gaussian backend currently does not support " "shots != 1 for homodyne measurement"
            )

        # phi is the rotation of the measurement operator, hence the minus
        self.circuit.phase_shift(-phi, mode)

        if select is None:
            qs = self.circuit.homodyne(mode, **kwargs)[0, 0]
        else:
            val = select * 2 / sqrt(2 * self.circuit.hbar)
            qs = self.circuit.post_select_homodyne(mode, val, **kwargs)

        return qs * sqrt(2 * self.circuit.hbar) / 2

    def measure_heterodyne(self, mode, shots=1, select=None):

        if shots != 1:
            if select is not None:
                raise NotImplementedError(
                    "Gaussian backend currently does not support "
                    "postselection if shots != 1 for heterodyne measurement"
                )

            raise NotImplementedError(
                "Gaussian backend currently does not support "
                "shots != 1 for heterodyne measurement"
            )

        if select is None:
            m = identity(2)
            res = 0.5 * self.circuit.measure_dyne(m, [mode], shots=shots)
            return res[0, 0] + 1j * res[0, 1]

        res = select
        self.circuit.post_select_heterodyne(mode, select)

        return res

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

        # convert xp-ordering to symmetric ordering
        means = vstack([r[:N], r[N:]]).reshape(-1, order="F")
        C = changebasis(N)
        cov = C @ V @ C.T

        self.circuit.fromscovmat(cov, modes)
        self.circuit.fromsmean(means, modes)

    def is_vacuum(self, tol=0.0, **kwargs):
        return self.circuit.is_vacuum(tol)

    def loss(self, T, mode):
        self.circuit.loss(T, mode)

    def thermal_loss(self, T, nbar, mode):
        self.circuit.thermal_loss(T, nbar, mode)

    def measure_fock(self, modes, shots=1, select=None):
        if shots != 1:
            if select is not None:
                raise NotImplementedError(
                    "Gaussian backend currently does not support " "postselection"
                )
            warnings.warn(
                "Cannot simulate non-Gaussian states. "
                "Conditional state after Fock measurement has not been updated."
            )

        mu = self.circuit.mean
        mean = self.circuit.smeanxp()
        cov = self.circuit.scovmatxp()

        x_idxs = array(modes)
        p_idxs = x_idxs + len(mu)
        modes_idxs = concatenate([x_idxs, p_idxs])
        reduced_cov = cov[ix_(modes_idxs, modes_idxs)]
        reduced_mean = mean[modes_idxs]
        # check we are sampling from a gaussian state with zero mean
        if allclose(mu, zeros_like(mu)):
            samples = hafnian_sample_state(reduced_cov, shots)
        else:
            samples = hafnian_sample_state(reduced_cov, shots, mean=reduced_mean)
        # for backward compatibility with previous measurement behaviour,
        # if only one shot, then we drop the shots axis
        if shots == 1:
            samples = samples.reshape((len(modes),))
        return samples

    def measure_threshold(self, modes, shots=1, select=None):
        if shots != 1:
            if select is not None:
                raise NotImplementedError(
                    "Gaussian backend currently does not support " "postselection"
                )
            warnings.warn(
                "Cannot simulate non-Gaussian states. "
                "Conditional state after Threshold measurement has not been updated."
            )

        mu = self.circuit.mean
        cov = self.circuit.scovmatxp()
        # check we are sampling from a gaussian state with zero mean
        if not allclose(mu, zeros_like(mu)):
            raise NotImplementedError(
                "Threshold measurement is only supported for " "Gaussian states with zero mean"
            )
        x_idxs = array(modes)
        p_idxs = x_idxs + len(mu)
        modes_idxs = concatenate([x_idxs, p_idxs])
        reduced_cov = cov[ix_(modes_idxs, modes_idxs)]
        samples = torontonian_sample_state(reduced_cov, shots)
        # for backward compatibility with previous measurement behaviour,
        # if only one shot, then we drop the shots axis
        if shots == 1:
            samples = samples.reshape((len(modes),))
        return samples

    def state(self, modes=None, **kwargs):
        """Returns the state of the quantum simulation.

        See :meth:`.BaseBackend.state`.

        Returns:
            GaussianState: state description
        """
        m = self.circuit.scovmat()
        r = self.circuit.smean()

        if modes is None:
            modes = list(range(len(self.get_modes())))

        listmodes = list(concatenate((2 * array(modes), 2 * array(modes) + 1)))
        covmat = empty((2 * len(modes), 2 * len(modes)))
        means = r[listmodes]

        for i, ii in enumerate(listmodes):
            for j, jj in enumerate(listmodes):
                covmat[i, j] = m[ii, jj]

        means *= sqrt(2 * self.circuit.hbar) / 2
        covmat *= self.circuit.hbar / 2

        mode_names = ["q[{}]".format(i) for i in array(self.get_modes())[modes]]

        # qmat and amat
        qmat = self.circuit.qmat()
        N = qmat.shape[0] // 2

        # work out if qmat and Amat need to be reduced
        if 1 <= len(modes) < N:
            # reduce qmat
            ind = concatenate([array(modes), N + array(modes)])
            rows = ind.reshape((-1, 1))
            cols = ind.reshape((1, -1))
            qmat = qmat[rows, cols]

            # calculate reduced Amat
            N = qmat.shape[0] // 2
            Amat = dot(xmat(N), identity(2 * N) - inv(qmat))
        else:
            Amat = self.circuit.Amat()

        return GaussianState((means, covmat), len(modes), qmat, Amat, mode_names=mode_names)

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
    sqrt,
    vstack,
    zeros_like,
    allclose,
    ix_,
)
from thewalrus.samples import hafnian_sample_state, torontonian_sample_state
from thewalrus.symplectic import xxpp_to_xpxp
from thewalrus.quantum import Qmat, Xmat, Amat
import numpy as np
import numba
from strawberryfields.backends import BaseGaussian
from strawberryfields.backends.states import BaseGaussianState

from .gaussiancircuit import GaussianModes


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
        """Initialize the backend."""
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

    def rotation(self, phi, mode):
        self.circuit.phase_shift(phi, mode)

    def displacement(self, r, phi, mode):
        self.circuit.displace(r, phi, mode)

    def squeeze(self, r, phi, mode):
        self.circuit.squeeze(r, phi, mode)

    def beamsplitter(self, theta, phi, mode1, mode2):
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

        # `qs` will always be a single value since multiple shots is not supported
        return array([[qs * sqrt(2 * self.circuit.hbar) / 2]])

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
            return array([[res[0, 0] + 1j * res[0, 1]]])

        res = select
        self.circuit.post_select_heterodyne(mode, select)

        # `res` will always be a single value since multiple shots is not supported
        return array([[res]])

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
        cov = xxpp_to_xpxp(V)

        self.circuit.fromscovmat(cov, modes)
        self.circuit.fromsmean(means, modes)

    def is_vacuum(self, tol=0.0, **kwargs):
        return self.circuit.is_vacuum(tol)

    def loss(self, T, mode):
        self.circuit.loss(T, mode)

    def thermal_loss(self, T, nbar, mode):
        self.circuit.thermal_loss(T, nbar, mode)

    def measure_fock(self, modes, shots=1, select=None, **kwargs):
        if select is not None:
            raise NotImplementedError(
                "Gaussian backend currently does not support " "postselection"
            )
        if shots != 1:
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

        return samples

    @numba.jit(nopython=True)
    def numba_ix(arr, rows, cols):

        return arr[rows][:, cols]

    @numba.jit(nopython=True)
    def nb_block(X):  # pragma: no cover
        """Numba implementation of `np.block`.
        Only suitable for 2x2 blocks.
        Taken from: https://stackoverflow.com/a/57562911
        Args:
        X (array) : arrays for blocks of matrix
        Return:
        array : the block matrix from X
        """
        xtmp1 = np.concatenate(X[0], axis=1)
        xtmp2 = np.concatenate(X[1], axis=1)
        return np.concatenate((xtmp1, xtmp2), axis=0)

    @numba.jit(nopython=True)
    def Qmat_numba(cov, hbar=2):  # pragma: no cover
        r"""Numba compatible version of `thewalrus.quantum.Qmat`
        Returns the :math:`Q` Husimi matrix of the Gaussian state.
        Args:
        cov (array): :math:`2N\times 2N xp-` Wigner covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
            Returns:
            array: the :math:`Q` matrix.
        """
        # number of modes
        N = len(cov) // 2
        I = np.identity(N)
        x = cov[:N, :N] * (2.0 / hbar)
        xp = cov[:N, N:] * (2.0 / hbar)
        p = cov[N:, N:] * (2.0 / hbar)
        # the (Hermitian) matrix elements <a_i^\dagger a_j>
        aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
        # the (symmetric) matrix elements <a_i a_j>
        aiaj = (x - p + 1j * (xp + xp.T)) / 4
        # calculate the covariance matrix sigma_Q appearing in the Q function:
        Q = nb_block(((aidaj, aiaj.conj()), (aiaj, aidaj.conj()))) + np.identity(2 * N)
        return Q

    @numba.jit(nopython=True)
    def powerset(parent_set):  # pragma: no cover
        """Generates the powerset, the set of all the subsets, of its input. Does not include the empty set.

        Args:
        parent_set (Sequence) : sequence to take powerset from

        Return:
        subset (tuple) : subset of parent_set
        """
        n = len(parent_set)
        for i in range(n + 1):
            for subset in combinations(parent_set, i):
                yield subset

    @numba.jit(nopython=True)
    def threshold_detection_prob_displacement(mu, cov, det_pattern, hbar=2):

        det_pattern = np.asarray(det_pattern).astype(np.int8)

        m = len(cov)
        assert cov.shape == (m, m)
        assert m % 2 == 0
        n = m // 2
        means_x = mu[:n]
        means_p = mu[n:]
        avec = np.concatenate((means_x + 1j * means_p, means_x - 1j * means_p), axis=0) / np.sqrt(
            2 * hbar
        )

        Q = Qmat_numba(cov, hbar=hbar)

        if max(det_pattern) > 1:
            raise ValueError(
                "When using threshold detectors, the detection pattern can contain only 1s or 0s."
            )

        nonzero_idxs = np.where(det_pattern == 1)[0]
        zero_idxs = np.where(det_pattern == 0)[0]

        ii1 = np.concatenate((nonzero_idxs, nonzero_idxs + n), axis=0)
        ii0 = np.concatenate((zero_idxs, zero_idxs + n), axis=0)

        Qaa = numba_ix(Q, ii0, ii0)
        Qab = numba_ix(Q, ii0, ii1)
        Qba = numba_ix(Q, ii1, ii0)
        Qbb = numba_ix(Q, ii1, ii1)

        Qaa_inv = np.linalg.inv(Qaa)
        Qcond = Qbb - Qba @ Qaa_inv @ Qab

        avec_a = avec[ii0]
        avec_b = avec[ii1]
        avec_cond = avec_b - Qba @ Qaa_inv @ avec_a

        p0a_fact_exp = np.exp(avec_a @ Qaa_inv @ avec_a.conj() * (-0.5)).real
        p0a_fact_det = np.sqrt(np.linalg.det(Qaa).real)
        p0a = p0a_fact_exp / p0a_fact_det

        n_det = len(nonzero_idxs)
        p_sum = 1.0  # empty set is not included in the powerset function so we start at 1
        for z in powerset(np.arange(n_det)):
            Z = np.asarray(z)
            ZZ = np.concatenate((Z, Z + n_det), axis=0)

            avec0 = avec_cond[ZZ]
            Q0 = numba_ix(Qcond, ZZ, ZZ)
            Q0inv = np.linalg.inv(Q0)

            fact_exp = np.exp(avec0 @ Q0inv @ avec0.conj() * (-0.5)).real
            fact_det = np.sqrt(np.linalg.det(Q0).real)

            p_sum += ((-1) ** len(Z)) * fact_exp / fact_det
        return p0a * p_sum

    def threshold_detection_prob(mu, cov, det_pattern, hbar=2, atol=1e-10, rtol=1e-10):

        if np.allclose(mu, 0, atol=atol, rtol=rtol):
            # no displacement
            n_modes = cov.shape[0] // 2
            Q = Qmat(cov, hbar)
            O = Xmat(n_modes) @ Amat(cov, hbar=hbar)
            rpt2 = np.concatenate((det_pattern, det_pattern))
            Os = reduction(O, rpt2)
            return tor(Os) / np.sqrt(np.linalg.det(Q))
        det_pattern = np.asarray(det_pattern).astype(np.int8)
        return threshold_detection_prob_displacement(mu, cov, det_pattern, hbar)

    def measure_threshold(self, modes, shots=1, select=None, **kwargs):
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
            return threshold_detection_prob_displacement(mu, cov, det_pattern, hbar)

        x_idxs = array(modes)
        p_idxs = x_idxs + len(mu)
        modes_idxs = concatenate([x_idxs, p_idxs])
        reduced_cov = cov[ix_(modes_idxs, modes_idxs)]
        samples = torontonian_sample_state(reduced_cov, shots)

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
        return BaseGaussianState((means, covmat), len(modes), mode_names=mode_names)

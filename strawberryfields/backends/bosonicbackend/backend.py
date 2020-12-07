# Copyright 2020 Xanadu Quantum Technologies Inc.

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
import warnings

from numpy import (
    empty,
    concatenate,
    arange,
    array,
    identity,
    arctan2,
    angle,
    sqrt,
    vstack,
    zeros_like,
    allclose,
    ix_,
    zeros,
    shape,
    cos,
    sin,
    exp,
    zeros,
    logical_and,
    logical_or,
)
from thewalrus.samples import hafnian_sample_state, torontonian_sample_state
import itertools as it

from strawberryfields.backends import BaseBosonic
from strawberryfields.backends.shared_ops import changebasis
from strawberryfields.backends.states import BaseBosonicState

from .bosoniccircuit import BosonicModes


class BosonicBackend(BaseBosonic):
    r"""The BosonicBackend...

    ..
        .. currentmodule:: strawberryfields.backends.gaussianbackend
        .. autosummary::
            :toctree: api

            ~bosoniccircuit.BosonicModes
            ~ops
    """

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

    def add_mode(self, n=1):
        self.circuit.add_mode(n)

    def del_mode(self, modes):
        self.circuit.del_mode(modes)

    def get_modes(self):
        return self.circuit.get_modes()

    def reset(self, pure=True, **kwargs):
        self.circuit.reset(self._init_modes)

    def prepare_thermal_state(self, nbar, mode):
        # self.circuit.init_thermal(nbar, mode)
        pass

    def prepare_vacuum_state(self, mode):
        # self.circuit.loss(0.0, mode)
        pass

    def prepare_coherent_state(self, r, phi, mode):
        # self.circuit.loss(0.0, mode)
        # self.circuit.displace(r, phi, mode)
        pass

    def prepare_squeezed_state(self, r, phi, mode):
        # self.circuit.loss(0.0, mode)
        # self.circuit.squeeze(r, phi, mode)
        pass

    def prepare_displaced_squeezed_state(self, r_d, phi_d, r_s, phi_s, mode):
        # self.circuit.loss(0.0, mode)
        # self.circuit.squeeze(r_s, phi_s, mode)
        # self.circuit.displace(r_d, phi_d, mode)
        pass

    def prepare_cat(self, alpha, phi, desc):
        """ Prepares the arrays of weights, means and covs for a cat state"""

        norm = np.exp(-np.absolute(alpha) ** 2) / (
            2 * (1 + np.exp(-2 * np.absolute(alpha) ** 2) * np.cos(phi))
        )
        rplus = np.sqrt(2 * self.hbar) * np.array([alpha.real, alpha.imag])
        rminus = -replus
        cov = 0.5 * self.hbar * np.identity(2)

        if desc == "complex":
            cplx_coef = np.exp(-2 * np.absolute(alpha) ** 2 - 1j * phi)
            rcomplex = np.sqrt(2 * self.hbar) * np.array([1j * alpha.imag, -1j * alpha.real])
            weights = norm * np.array([1, 1, cplx_coef, np.conjugate(cplx_coef)])
            means = np.array([rplus, rminus, rcomplex, np.conjugate(rcomplex)])

            return [weights, means, cov]

        elif desc == "real":
            raise ValueError("The real description of Cat state is not implemented")

        else:
            raise ValueError('desc accept only "real" or "complex" arguments')

    def prepare_gkp(self, state, epsilon, cutoff, desc="real", shape="square"):
        """ Prepares the arrays of weights, means and covs for a gkp state """

        theta, phi = state[0], state[1]

        if shape == "square":
            if desc == "real":

                def coef(arr):
                    l, m = arr[:, 0], arr[:, 1]
                    t = np.zeros(arr.shape[0], dtype=float)
                    t += np.logical_and(l % 2 == 0, m % 2 == 0) * (
                        np.cos(0.5 * theta) ** 2 + np.sin(0.5 * theta) ** 2
                    )
                    t += np.logical_and(l % 4 == 0, m % 2 == 1) * (
                        np.cos(0.5 * theta) ** 2 - np.sin(0.5 * theta) ** 2
                    )
                    t += np.logical_and(l % 4 == 2, m % 2 == 1) * (
                        np.sin(0.5 * theta) ** 2 - np.cos(0.5 * theta) ** 2
                    )
                    t += np.logical_and(l % 4 % 2 == 1, m % 4 == 0) * np.sin(theta) * np.cos(phi)
                    t -= np.logical_and(l % 4 % 2 == 1, m % 4 == 2) * np.sin(theta) * np.cos(phi)
                    t -= np.logical_and(l % 4 == 3, m % 4 == 3) * np.sin(theta) * np.sin(phi)
                    t += (
                        np.logical_or(
                            np.logical_and(l % 4 == 3, m % 4 == 1),
                            np.logical_and(l % 4 == 1, m % 4 == 3),
                        )
                        * np.sin(theta)
                        * np.sin(phi)
                    )
                    # if l % 2 == 0  and m % 2 == 0 :
                    # c = np.cos( 0.5 * theta ) ** 2 + np.sin( 0.5 * theta ) ** 2
                    # else if l % 4 == 0  and m % 2 == 1 :
                    # c = np.cos( 0.5 * theta ) ** 2 - np.sin( 0.5 * theta ) ** 2
                    # else if l % 4 == 2 and m % 2 == 1 :
                    # c = np.sin( 0.5 * theta ) ** 2 - np.cos( 0.5 * theta ) ** 2
                    # else if m % 4 == 0 and l % 4 % 2 == 1 :
                    # c = np.sin( theta ) * np.cos( phi )
                    # else if m % 4 == 2 and l % 4 % 2 == 1 :
                    # c = - np.sin( theta ) * np.cos ( phi )
                    # else if ( l % 4 == 3 and m % 4 == 3 ) or ( l % 4 == 1 and m % 4 == 1 ):
                    # c = - np.sin( theta ) * np.sin( phi )
                    # else if ( l % 4 == 3 and m % 4 == 1) or ( l % 4 == 1 and m % 4 == 3 ):
                    # c = np.sin( theta ) * np.sin( phi )

                    return t * np.exp(
                        -np.pi
                        * 0.25
                        * (l ** 2 + m ** 2)
                        * (1 - np.exp(-2 * epsilon))
                        / (1 + np.exp(-2 * epsilon))
                    )

                z_max = np.ceil(
                    np.sqrt(
                        -4
                        * self.hbar
                        * np.log(cutoff)
                        * (1 + np.exp(-2 * epsilon))
                        / (1 - np.exp(-2 * epsilon))
                    )
                )
                damping = 2 * np.exp(-epsilon) / (1 + np.exp(-2 * epsilon))

                means_large_gen = [
                    l + 1j * m for l, m in it.product(range(-z_max, z_max + 1), repeat=2)
                ]
                means_gen = it.tee(
                    it.filterfalse(
                        lambda x: (np.exp(-0.25 * np.pi * np.abs(x) ** 2) < cutoff), means_large_gen
                    ),
                    2,
                )
                means = np.concatenate(
                    np.reshape(np.fromiter(means_gen[0], complex), (-1, 1)).real,
                    np.reshape(np.fromiter(a[1], complex), (-1, 1).imag),
                    axis=1,
                )
                weights = coef(means)
                means *= 0.5 * damping * np.sqrt(np.pi)
                cov = 2 * (1 + np.exp(-2 * epsilon)) / (1 - np.exp(-2 * epsilon)) * np.identity(2)

                return [weights, means, cov]

            elif desc == "complex":
                raise ValueError("The complex description of GKP is not implemented")
        else:
            raise ValueError("Only square GKP are implemented for now")

    def prepare_fock(self, n, r=0.0001):
        """ Prepares the arrays of weights, means and covs of a Fock state"""

        pass

        ## To be completed here

    def prepare_comb(self, n, d, r, cutoff):
        """ Prepares the arrays of weights, means and covs of a squeezed comb state"""

        pass

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
            raise NotImplementedError(
                "Threshold measurement is only supported for " "Gaussian states with zero mean"
            )
        x_idxs = array(modes)
        p_idxs = x_idxs + len(mu)
        modes_idxs = concatenate([x_idxs, p_idxs])
        reduced_cov = cov[ix_(modes_idxs, modes_idxs)]
        samples = torontonian_sample_state(reduced_cov, shots)

        return samples

    def state(self, modes=None, peaks=None, **kwargs):
        """Returns the state of the quantum simulation.

        See :meth:`.BaseBackend.state`.

        Returns:
            BosonicState: state description
        """
        if modes is None:
            modes = list(range(len(self.get_modes())))

        w = self.circuit.weights

        # Generate dictionary between tuples of the form (peek_0, ... peek_i)
        # where the subscript denotes the mode, and the corresponding index
        # in the cov object.
        # if peaks is None:
        #     peaks = tuple(zeros(len(modes)))
        # g_list = [arange(len(w)) for i in range(len(modes))]
        # combs = it.product(*g_list)
        # covs_dict = {tuple: index for (index, tuple) in enumerate(combs)}

        listmodes = list(concatenate((2 * array(modes), 2 * array(modes) + 1)))
        covmat = self.circuit.covs
        means = self.circuit.means
        if len(w) == 1:
            m = covmat[0]
            r = means[0]

            covmat = empty((2 * len(modes), 2 * len(modes)))
            means = r[listmodes]

            for i, ii in enumerate(listmodes):
                for j, jj in enumerate(listmodes):
                    covmat[i, j] = m[ii, jj]

            means *= sqrt(2 * self.circuit.hbar) / 2
            covmat *= self.circuit.hbar / 2

        mode_names = ["q[{}]".format(i) for i in array(self.get_modes())[modes]]
        num_w = int(len(w) ** (1 / len(modes)))
        return BaseBosonicState((means, covmat, w), len(modes), num_w, mode_names=mode_names)

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

"""Module containing Gaussian backend specific extensions to BaseGaussianState"""

import numpy as np

from ..states import BaseGaussianState
from .ops import fock_amplitudes_one_mode, fock_prob, sm_fidelity


class GaussianState(BaseGaussianState):
    r"""Class for the representation of quantum states using Gaussian representation.

    Args:
        state_data (tuple(mu, cov)): A tuple containing the vector of means ``mu`` and the
            covariance matrix ``cov``.
        num_modes (int): the number of modes in the state
        qmat (array): The covariance matrix for the Q function
        Amat (array): The A matrix from Hamilton's paper
        mode_names (Sequence): (optional) this argument contains a list providing mode names
            for each mode in the state
    """

    def __init__(self, state_data, num_modes, qmat, Amat, mode_names=None):
        # pylint: disable=too-many-arguments
        super().__init__(state_data, num_modes, mode_names)
        self.qmat = qmat
        self.Amat = Amat

    def reduced_dm(self, modes, **kwargs):
        r"""Returns the reduced density matrix in the Fock basis for a particular mode.

        Args:
            modes (int): specifies the mode. For the Gaussian backend, only a **single** mode
                reduced density matrix can be returned.
            **kwargs:

                  * **cutoff** (*int*): (default 10) specifies where to truncate the returned density matrix.
                    Note that the cutoff argument only applies for Gaussian representation;
                    states represented in the Fock basis will use their own internal cutoff dimension.

        Returns:
            array: the reduced density matrix for the specified modes
        """
        cutoff = kwargs.get("cutoff", 10)
        mu, cov = self.reduced_gaussian([modes])  # pylint: disable=unused-variable
        cov = cov * 2 / self._hbar

        return fock_amplitudes_one_mode(self._alpha[modes], cov, cutoff)

    # ==========================================
    # The methods below inherit their docstring
    # from BaseState

    def fock_prob(self, n, **kwargs):
        if len(n) != self._modes:
            raise ValueError("Fock state must be same length as the number of modes")

        cutoff = kwargs.get("cutoff", 10)
        if sum(n) >= cutoff:
            raise ValueError("Cutoff argument must be larger than the sum of photon numbers")

        ocp = np.uint8(np.array(n))

        # get the vector and means and covariance
        # matrix, and normalize so that hbar=2
        mu = self.means() * np.sqrt(2 / self.hbar)
        cov = self.cov() * 2 / self.hbar

        return fock_prob(mu, cov, ocp)

    def mean_photon(self, mode, **kwargs):
        mu, cov = self.reduced_gaussian([mode])
        mean = (np.trace(cov) + mu.T @ mu) / (2 * self._hbar) - 1 / 2
        var = (np.trace(cov @ cov) + 2 * mu.T @ cov @ mu) / (2 * self._hbar ** 2) - 1 / 4
        return mean, var

    def fidelity(self, other_state, mode, **kwargs):
        mu1 = other_state[0] * 2 / np.sqrt(2 * self._hbar)
        cov1 = other_state[1] * 2 / self._hbar

        mu2, cov2 = self.reduced_gaussian([mode])
        mu2 *= 2 / np.sqrt(2 * self._hbar)
        cov2 /= self._hbar / 2

        return sm_fidelity(mu1, mu2, cov1, cov2)

    def fidelity_vacuum(self, **kwargs):
        alpha = np.zeros(len(self._alpha))
        return self.fidelity_coherent(alpha)

    def fidelity_coherent(self, alpha_list, **kwargs):
        if len(alpha_list) != self._modes:
            raise ValueError("alpha_list must be same length as the number of modes")

        Qi = np.linalg.inv(self.qmat)
        delta = self._alpha - alpha_list

        delta = np.concatenate((delta, delta.conj()))
        fac = np.sqrt(np.linalg.det(Qi).real)
        exp = np.exp(-0.5 * np.dot(delta, np.dot(Qi, delta.conj())).real)
        return fac * exp

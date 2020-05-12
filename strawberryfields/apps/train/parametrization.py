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
r"""
Contains the :class:`~.VGBS` (variational GBS) class which provides a trainable parametrization
of GBS.
"""
from typing import Optional

import numpy as np
from thewalrus.quantum import Qmat, Xmat, photon_number_mean_vector
from thewalrus.quantum import find_scaling_adjacency_matrix as rescale
from thewalrus.quantum import find_scaling_adjacency_matrix_torontonian as rescale_tor
import thewalrus.samples


def rescale_adjacency(A: np.ndarray, n_mean: float, threshold: bool) -> np.ndarray:
    """TODO"""
    scale = rescale_tor(A, n_mean) if threshold else rescale(A, n_mean)
    return A * scale


def A_to_cov(A: np.ndarray) -> np.ndarray:
    """TODO"""
    n = len(A)
    A_big = np.block([[A, 0 * A], [0 * A, np.conj(A)]])
    I = np.identity(2 * n)
    X = Xmat(n)
    return np.linalg.inv(I - X @ A_big) - I / 2


# TODO: update embedding class in type hint
class VGBS:
    """TODO"""

    def __init__(
        self,
        A: np.ndarray,
        n_mean: float,
        embedding,
        threshold: bool,
        samples: Optional[np.ndarray] = None,
    ):
        if not np.allclose(A, A.T):
            raise ValueError("Input must be a NumPy array corresponding to a symmetric matrix")
        self.A_init = A
        self.embedding = embedding
        self.threshold = threshold
        self.A_init_scaled = rescale_adjacency(A, n_mean, threshold)
        self.n_modes = len(A)
        self._A_init_samples = None
        if samples:
            self._add_A_init_samples(samples)

    def W(self, params: np.ndarray) -> np.ndarray:
        """TODO"""
        return np.diag(self.embedding(params))

    @staticmethod
    def WAW(A: np.ndarray, W: np.ndarray) -> np.ndarray:
        """TODO"""
        return W @ A @ W

    def A(self, params: np.ndarray) -> np.ndarray:
        """TODO"""
        return self.WAW(self.A_init, self.W(params))

    def _A_scaled(self, params: np.ndarray) -> np.ndarray:
        """TODO"""
        return self.WAW(self.A_init_scaled, self.W(params))

    def _generate_samples(self, A: np.ndarray, samples: int, **kwargs) -> np.ndarray:
        """TODO"""
        cov = A_to_cov(A)

        if self.threshold:
            samples = thewalrus.samples.torontonian_sample_state(cov, samples, hbar=1, **kwargs)
        else:
            samples = thewalrus.samples.hafnian_sample_state(cov, samples, hbar=1, **kwargs)
        return samples

    def generate_samples(self, params: np.ndarray, samples: int, **kwargs) -> np.ndarray:
        """TODO"""
        return self._generate_samples(self._A_scaled(params), samples, **kwargs)

    def _generate_A_init_samples(self, samples, **kwargs) -> np.ndarray:
        """TODO"""
        samples = self._generate_samples(self.A_init_scaled, samples, **kwargs)
        self.add_A_samples(samples)
        return samples

    def _add_A_init_samples(self, samples: np.ndarray):
        """TODO"""
        shape = samples.shape
        if shape[1] != self.n_modes:
            raise ValueError("Must input samples of shape (number, {})".format(self.n_modes))

        if self._A_init_samples is None:
            self._A_init_samples = samples
        else:
            self._A_init_samples = np.vstack([self._A_init_samples, samples])

    def mean_photons_by_mode(self, params: np.ndarray) -> np.ndarray:
        """TODO"""
        disp = np.zeros(2 * self.n_modes)
        cov = A_to_cov(self._A_scaled(params))
        return photon_number_mean_vector(disp, cov, hbar=1)  # TODO: consider hbar=2

    def mean_clicks_by_mode(self, params: np.ndarray) -> np.ndarray:
        """TODO"""
        cov = A_to_cov(self._A_scaled(params))
        Q = Qmat(cov, hbar=1)
        m = self.n_modes
        Qks = [[[Q[k, k], Q[k, k + m]], [Q[k + m, k], Q[k + m, k + m]]] for k in range(m)]
        cbar = [1 - np.linalg.det(Qks[k]) ** (-0.5) for k in range(m)]
        return np.real(np.array(cbar))

    def n_mean(self, params: np.ndarray) -> float:
        """TODO"""
        if self.threshold:
            return np.sum(self.mean_clicks_by_mode(params))
        else:
            return np.sum(self.mean_photons_by_mode(params))

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
TODO
"""
from typing import Callable

import numpy as np
from strawberryfields.apps.train.param import _Omat, VGBS


class Stochastic:
    """TODO"""

    def __init__(self, h: Callable, vgbs: VGBS):
        self.h = h
        self.vgbs = vgbs

    def evaluate(self, params: np.ndarray, n_samples: int = 1000) -> float:
        """Evaluates the cost function.

        The cost function is evaluated by finding its average over a number ``n_samples`` of
        samples generated from the VGBS system using the trainable parameters :math:`\theta`.

        **Example usage:**

        >>> TODO

        Args:
            params (array): the trainable parameters :math:`\theta`
            n_samples (int): the number of GBS samples used to average the cost function over

        Returns:
            float: the value of the stochastic cost function
        """
        samples = self.vgbs.get_A_init_samples(n_samples)
        return np.mean([self.h_reparametrized(s, params) for s in samples])

    def h_reparametrized(self, sample: np.ndarray, params: np.ndarray) -> float:
        """Include trainable parameters in the cost function to allow for sampling from the
        initial adjacency matrix.

        Args:
            sample (array): the sample
            params (array): the trainable parameters :math:`\theta`

        Returns:
            float: the cost function with respect to a given sample and set of trainable parameters
        """
        h = self.h(sample)
        A = self.vgbs.A(params)
        w = self.vgbs.embedding(params)
        Id = np.eye(2 * self.vgbs.n_modes)

        dets_numerator = np.linalg.det(Id - _Omat(A))
        dets_denominator = np.linalg.det(Id - _Omat(self.vgbs.A_init))
        dets = np.sqrt(dets_numerator / dets_denominator)

        prod = np.prod(np.power(w, sample))

        return h * dets * prod

    def _sample_difference_from_mean(self, sample: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Calculates the difference between an input sample and the vector of mean clicks or
        photons by mode.

        Args:
            sample (array): the sample
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: The difference
        """
        if self.vgbs.threshold:
            n_diff = sample - self.vgbs.mean_clicks_by_mode(params)
        else:
            n_diff = sample - self.vgbs.mean_photons_by_mode(params)

        return n_diff

    def _gradient_one_sample(self, sample: np.ndarray, params: np.ndarray) -> np.ndarray:
        """TODO"""
        w = self.vgbs.embedding(params)
        jac = self.vgbs.embedding.jacobian(params)

        h = self.h_reparametrized(sample, params)
        diff = self._sample_difference_from_mean(sample, params)
        return h * (diff / w) @ jac

    def gradient(self, params: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """TODO"""
        samples = self.vgbs.get_A_init_samples(n_samples)
        return np.mean([self._gradient_one_sample(s, params) for s in samples], axis=0)

    def __call__(self, params: np.ndarray, n_samples: int = 1000) -> float:
        return self.evaluate(params, n_samples)

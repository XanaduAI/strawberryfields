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
from math import factorial
from typing import Callable

import numpy as np


class Stochastic:
    """TODO"""

    # TODO update type hints
    def __init__(self, h: Callable, vgbs):
        self.h = h
        self.vgbs = vgbs

    def evaluate(self, weights: np.ndarray, n_samples: int) -> float:
        """TODO"""
        samples = self.vgbs.get_A_init_samples(n_samples)
        return np.mean([self._h_reparametrized(s, weights) for s in samples])

    def _h_reparametrized(self, sample: np.ndarray, params: np.ndarray) -> float:
        """TODO"""
        h = self.h(sample)
        A = self.vgbs.A(params)
        w = self.vgbs.embedding(params)
        Id = np.eye(self.vgbs.n_modes)

        dets_numerator = np.linalg.det(Id - A ** 2)
        dets_denominator = np.linalg.det(Id - self.vgbs.self.A_init ** 2)
        dets = np.sqrt(dets_numerator / dets_denominator)

        prod_numerator = np.power(w, sample)
        prod_denominator = np.array([factorial(n) for n in sample])
        prod = np.prod(prod_numerator / prod_denominator)

        return h * dets * prod

    def _sample_difference_from_mean(self, sample: np.ndarray, params: np.ndarray):
        """TODO"""
        if self.vgbs.threshold:
            n_diff = sample - self.vgbs.mean_clicks_by_mode(params)
        else:
            n_diff = sample - self.vgbs.mean_photons_by_mode(params)

        return n_diff

    def gradient(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        """TODO"""
        samples = self.vgbs.get_A_init_samples(n_samples)

        w = self.vgbs.embedding(params)
        jac = self.vgbs.embedding.jacobian(params)

        def _gradient_one_sample(sample):
            h = self._h_reparametrized(sample)
            diff = self._sample_difference_from_mean(sample, params)
            return h * (diff / w) @ jac

        return np.mean([_gradient_one_sample(s) for s in samples])

    def __call__(self, params: np.ndarray, n_samples: int = 1000) -> float:
        return self.evaluate(params)

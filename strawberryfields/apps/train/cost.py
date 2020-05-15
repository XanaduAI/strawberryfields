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
Submodule for computing gradients, evaluating cost functions, and optimizing GBS
circuits.

In the context of stochastic optimization, cost functions are expressed as expectation values
over the GBS distribution. Within the WAW parametrization, gradients of such cost functions can be
expressed as expectation values over the GBS distribution. This module contains methods for
calculating these gradients and for using gradient-based methods to optimize GBS circuits. In the
case of optimization with respect to a Kullback-Leibler divergence or log-likelihood cost
function, gradients can be computed efficiently, leading to fast training. This module also
includes the ability to optimize GBS circuits using these efficient methods.
"""

import numpy as np


class KL:

    def __init__(self, data, vgbs):
        self.data = data
        self.vgbs = vgbs
        self.nr_samples = len(data)
        self.nr_modes = len(data[0])

    def mean_n_data(self):
        return np.sum(self.data, axis=1)/self.nr_samples

    def grad(self, params):
        weights = self.vgbs.embedding(params)
        if vgbs.threshold():
            n_diff = self.vgbs.mean_clicks_by_mode(params) - self.mean_n_data()
        else:
            n_diff = self.vgbs.mean_photons_by_mode(params) - self.mean_n_data()
        return (n_diff/weights) @ self.vgbs.embedding.jacobian(params)

    # TODO: add evaluate cost function

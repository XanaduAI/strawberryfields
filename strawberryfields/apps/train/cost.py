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

    def __init__(self, data, gbs):
        self.data = data
        self.gbs = gbs
        self.nr_samples = len(data)
        self.nr_modes = len(data[0])

    def f_data(self, data, features):
        """Computes the constant F_D appearing in the gradient formula."""
        f = np.zeros(len(features))
        for t in range(self.nr_samples):
            for i in range(self.nr_modes):  # more elegant way of doing this?
                f += data[t][i] * features[i]
        return f/self.nr_samples

    def f_model(self, params, features):
        mean_photons = self.gbs.mean_photons_by_mode(params).T
        return mean_photons @ features

    def grad(self, params, data, features):
        return f_model(params, features) - f_data(data, features)

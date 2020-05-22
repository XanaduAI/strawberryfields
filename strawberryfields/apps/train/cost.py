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
Submodule for computing gradients and evaluating cost functions with respect to GBS circuits

In the context of stochastic optimization, cost functions are expressed as expectation values
over the GBS distribution. Within the WAW parametrization, gradients of cost functions can be
expressed as expectation values over the GBS distribution. This module contains methods for
calculating these gradients and for using gradient-based methods to optimize GBS circuits. In the
case of optimization with respect to a Kullback-Leibler divergence or log-likelihood cost
function, gradients can be computed efficiently, leading to fast training.
"""

import numpy as np


class KL:
    r"""Optimization of GBS distributions based on a Kullback-Liebler divergence cost function.

    In a standard unsupervised learning scenario, data are assumed to be sampled from an unknown
    distribution and a common goal is to learn that distribution. Training of a model
    distribution can be performed by minimizing the Kullback-Leibler (KL) divergence:
    .. math::

        KL(P, Q) = -\frac{1}{T}\sum_t \log[P(S^{(t)})]-\log(T),

    where :math:`S^{(t)}` is the *t*-th element in the data, :math:`P(S^{(t)})` is the probability
    of observing that element when sampling from the model distribution, and :math:`T` is the
    total number of elements in the data. For the GBS distribution in the WAW parametrization,
    the gradient of the KL divergence can be written as
    .. math::

        \partial_\theta KL(\theta) = - \sum_{k=1}^m(\langle n_k\rangle_{\text{data}-
        \langle n_k\rangle)_{\text{GBS}}\partial_\theta w_k,

    where :math:`\langle n_k\rangle)` denotes the average photon numbers in mode *k*. This class
    provides methods to compute gradients and evaluate the cost function.

    **Example usage**

    >>> embedding = embed.Exp(4)
    >>> A = np.ones((4, 4))
    >>> vgbs = param.VGBS(A, 3, embedding, threshold=True)
    >>> params = np.array([0.05, 0.1, 0.02, 0.01])
    >>> data = np.zeros((4, 4))
    >>> kl = cost.KL(data, vgbs)
    >>> kl.evaluate(params)
    -0.2866830267216749
    >>> kl.grad(params)
    array([-0.52812574, -0.5201932 , -0.53282312, -0.53437824])

    Args:
        data (np.array): Array of samples representing the training data
        vgbs: Variational GBS class

    """

    def __init__(self, data: np.ndarray, vgbs):
        self.data = data
        self.vgbs = vgbs
        self.nr_samples = len(data)
        self.nr_modes = len(data[0])

    def mean_n_data(self):
        r"""Computes the average number of photons :math:`\langle n_k\rangle_{\text{data}` from
        the data. When the data are samples from threshold detectors, this gives the average number
        of clicks.

        **Example usage**

        >>> kl.mean_n_data()
        array([0., 0., 0., 0.])

        Returns:
            array: vector of mean photon numbers per mode
        """
        return np.sum(self.data, axis=0) / self.nr_samples

    def grad(self, params: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the Kullback-Liebler cost function with respect to the
        trainable parameters

        **Example usage**

        >>> kl.grad(params)
        array([-0.52812574, -0.5201932 , -0.53282312, -0.53437824])

        Args:
            params (array): the trainable parameters :math:`\theta`
        Returns:
            array: the gradient of the K-L cost function with respect to :math:`\theta`
        """
        weights = self.vgbs.embedding(params)
        if self.vgbs.threshold:
            n_diff = self.vgbs.mean_clicks_by_mode(params) - self.mean_n_data()
        else:
            n_diff = self.vgbs.mean_photons_by_mode(params) - self.mean_n_data()
        return (n_diff / weights) @ self.vgbs.embedding.jacobian(params)

    def evaluate(self, params: np.ndarray) -> float:
        """Computes the value of the Kullback-Liebler divergence cost function.

        **Example usage**

        >>> kl.evaluate(params)
        -0.2866830267216749

        Args:
            params (array): the trainable parameters :math:`\theta`
        Returns:
            float: the value of the cost function
        """
        kl = 0
        for sample in self.data:
            kl += np.log(self.vgbs.prob_sample(params, sample))
        return -kl / self.nr_samples - np.log(self.nr_samples)

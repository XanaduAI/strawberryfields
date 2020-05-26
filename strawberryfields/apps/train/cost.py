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
TODO - this is provided in another PR
"""
from typing import Callable

import numpy as np
from strawberryfields.apps.train.param import _Omat, VGBS


class Stochastic:
    r"""Stochastic cost function given by averaging over samples from a trainable GBS distribution.

    A stochastic optimization problem is defined with respect to a function :math:`h(\bar{n})` that
    assigns a cost to an input sample :math:`\bar{n}`. The cost function is the
    average of :math:`h(\bar{n})` over samples generated from a parametrized distribution
    :math:`P_{\theta}(\bar{n})`:

    .. math::

        C (\theta) = \sum_{\bar{n}} h(\bar{n}) P_{\theta}(\bar{n})

    The cost function :math:`C (\theta)` can then be optimized by varying
    :math:`P_{\theta}(\bar{n})`.

    In this setting, :math:`P_{\theta}(\bar{n})` is the variational GBS distribution and is input by
    instantiating :class:`~.VGBS`.

    **Example usage:**

    The function :math:`h(\bar{n})` in this example is decreased when clicks are present in
    odd-numbered modes and is increased when clicks are present in even numbered modes.

    >>> embedding = train.embed.Exp(4)
    >>> A = np.ones((4, 4))
    >>> vgbs = train.VGBS(A, 3, embedding, threshold=True)
    >>> h = lambda x: sum([x[i] * (-1) ** (i + 1) for i in range(4)])
    >>> cost = Stochastic(h, vgbs)
    >>> params = np.array([0.05, 0.1, 0.02, 0.01])
    >>> cost.evaluate(params, 100)
    0.03005489236683591
    >>> cost.gradient(params, 100)
    array([ 0.10880756, -0.1247146 ,  0.12426481, -0.13783342])

    Args:
        h (callable): a function that assigns a cost to an input sample
        vgbs (train.VGBS): the trainable GBS distribution, which must be an instance of
            :class:`~.VGBS`
    """

    def __init__(self, h: Callable, vgbs: VGBS):
        self.h = h
        self.vgbs = vgbs

    def evaluate(self, params: np.ndarray, n_samples: int = 100) -> float:
        r"""Evaluates the cost function.

        The cost function is evaluated by finding its average over a number ``n_samples`` of
        samples generated from the VGBS system using the trainable parameters :math:`\theta`.

        **Example usage:**

        >>> cost.evaluate(params, 100)
        0.03005489236683591

        Args:
            params (array): the trainable parameters :math:`\theta`
            n_samples (int): the number of GBS samples used to average the cost function

        Returns:
            float: the value of the stochastic cost function
        """
        samples = self.vgbs.get_A_init_samples(n_samples)
        return np.mean([self.h_reparametrized(s, params) for s in samples])

    def h_reparametrized(self, sample: np.ndarray, params: np.ndarray) -> float:
        r"""Include trainable parameters in the :math:`h(\bar{n})` function to allow sampling
        from the initial adjacency matrix.

        **Example usage:**

        >>> sample = [1, 1, 0, 0]
        >>> cost.h_reparametrized(sample, params)
        -1.6688383062813434

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

    def sample_difference_from_mean(self, sample: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Calculates the difference between an input sample and the vector of mean clicks or
        photons by mode.

        **Example usage:**

        >>> sample = [1, 0, 0, 0]
        >>> cost.sample_difference_from_mean(sample, params)
        array([ 0.47187426,  0.4798068 ,  0.46717688, -0.53437824])

        Args:
            sample (array): the sample
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: the difference
        """
        if self.vgbs.threshold:
            n_diff = sample - self.vgbs.mean_clicks_by_mode(params)
        else:
            n_diff = sample - self.vgbs.mean_photons_by_mode(params)

        return n_diff

    def _gradient_one_sample(self, sample: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Evaluates the gradient equation on a single sample.

        Args:
            sample (array): the sample
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: the one-shot gradient
        """
        w = self.vgbs.embedding(params)
        jac = self.vgbs.embedding.jacobian(params)

        h = self.h_reparametrized(sample, params)
        diff = self.sample_difference_from_mean(sample, params)
        return h * (diff / w) @ jac

    def gradient(self, params: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Evaluates the gradient of the cost function.

        The gradient is evaluated by finding an average of a function over a number ``n_samples`` of
        samples generated from the VGBS system using the trainable parameters :math:`\theta`.

        The averaged function is specified in `this paper <https://arxiv.org/abs/2004.04770>`__.

        **Example usage:**

        >>> cost.gradient(params, 100)
        array([ 0.10880756, -0.1247146 ,  0.12426481, -0.13783342])

        Args:
            params (array): the trainable parameters :math:`\theta`
            n_samples (int): the number of GBS samples used in the gradient estimation

        Returns:
            array: the gradient vector
        """
        samples = self.vgbs.get_A_init_samples(n_samples)
        return np.mean([self._gradient_one_sample(s, params) for s in samples], axis=0)

    def __call__(self, params: np.ndarray, n_samples: int = 100) -> float:
        return self.evaluate(params, n_samples)

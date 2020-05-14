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
of the GBS probability distribution.
"""
from typing import Optional

import numpy as np
import thewalrus.samples
from thewalrus.quantum import Qmat, Xmat
from thewalrus.quantum import find_scaling_adjacency_matrix as rescale
from thewalrus.quantum import find_scaling_adjacency_matrix_torontonian as rescale_tor
from thewalrus.quantum import photon_number_mean_vector

import strawberryfields as sf


def rescale_adjacency(A: np.ndarray, n_mean: float, threshold: bool) -> np.ndarray:
    """Rescale an adjacency matrix so that it can be mapped to GBS.

    An adjacency matrix must have singular values not exceeding one if it can be mapped to GBS.
    Arbitrary adjacency matrices must first be rescaled to satisfy this condition.

    This function rescales an input adjacency matrix :math:`A` so that the corresponding gaussian
    state has:

    - a mean number of *clicks* equal to ``n_mean`` when ``threshold=True``;
    - a mean number of *photons* equal to ``n_mean`` when ``threshold=False``.

    **Example usage:**

    >>> a = np.ones((3, 3))
    >>> rescale_adjacency(a, 2, True)
    array([[0.32232919, 0.32232919, 0.32232919],
           [0.32232919, 0.32232919, 0.32232919],
           [0.32232919, 0.32232919, 0.32232919]])

    Args:
        A (array): the adjacency matrix to rescale
        n_mean (float): the target mean number of clicks or mean number of photons
        threshold (bool): determines whether rescaling is for a target mean number of clicks or
            photons

    Returns:
        array: the rescaled adjacency matrix
    """
    scale = rescale_tor(A, n_mean) if threshold else rescale(A, n_mean)
    return A * scale


def A_to_cov(A: np.ndarray) -> np.ndarray:
    """Convert an adjacency matrix to a covariance matrix of a GBS device.

    The input adjacency matrix must have singular values not exceeding one. This can be achieved
    by rescaling an arbitrary adjacency matrix using :func:`rescale_adjacency`.

    **Example usage:**

    >>> a = np.ones((3, 3))
    >>> a = rescale_adjacency(a, 2, True)
    >>> cov = A_to_cov(a)

    Args:
        A (array): the adjacency matrix

    Returns:
        array: the covariance matrix of :math:`A`
    """
    n = len(A)
    A_big = np.block([[A, 0 * A], [0 * A, np.conj(A)]])
    I = np.identity(2 * n)
    X = Xmat(n)
    return sf.hbar * (np.linalg.inv(I - X @ A_big) - I / 2)


class VGBS:
    r"""Create a variational GBS model for optimization and machine learning.

    An input adjacency matrix :math:`A` can be varied using:

    .. math::

        A(\theta) = W(\theta) A W(\theta),

    with :math:`W` a diagonal matrix of weights that depend on a set of parameters :math:`\theta`.

    By varying :math:`\theta`, the distribution of samples from GBS can be trained to solve
    stochastic optimization and unsupervised machine learning problems.

    The above variational model can be used in both the threshold and PNR modes of GBS, which
    is specified using the ``threshold`` flag. An initial value for the mean number of clicks
    (if ``threshold=True``) or mean number of photons (if ``threshold=False``) is also required.
    The initial value ``n_mean`` should be set high since varying :math:`\theta` can only lower
    the mean number of clicks or photons.

    The mapping from :math:`\theta` to :math:`W(\theta)` is specified using the ``embedding``
    argument. TODO mention embedding module

    **Example usage:**

    >>> g = nx.erdos_renyi_graph(4, 0.7, seed=1967)
    >>> A = nx.to_numpy_array(g)
    >>> embedding = Exp(4)  # TODO does this need updating?
    >>> vgbs = VGBS(A, 3, embedding, threshold=True)
    >>> params = np.array([0.05, 0.1, 0.02, 0.01])
    >>> vgbs.A(params)
    array([[0.        , 0.30298161, 0.31534653, 0.31692721],
           [0.30298161, 0.        , 0.30756059, 0.30910225],
           [0.31534653, 0.30756059, 0.        , 0.32171695],
           [0.31692721, 0.30910225, 0.32171695, 0.        ]])
    >>> vgbs.n_mean(params)
    2.299036355948707
    >>> vgbs.generate_samples(vgbs.A(params), 2)
    array([[1, 1, 1, 0],
           [0, 1, 0, 1]])

    Args:
        A (array): the input adjacency matrix :math:`A`
        n_mean (float): the initial mean number of clicks or photons
        embedding: the method of converting from trainable parameters :math:`\theta` to
            :math:`W(\theta)` #TODO: any update from other PR?
        threshold (bool): determines whether to use GBS in threshold or PNR mode
        samples (array): an optional array of samples from :math:`A` used to speed up gradient
            calculations #TODO: more info
    """

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
        self.A_init = rescale_adjacency(A, n_mean, threshold)
        self._A_init_samples = None
        self.embedding = embedding
        self.threshold = threshold
        self.n_modes = len(A)
        if samples:
            self.add_A_init_samples(samples)

    def W(self, params: np.ndarray) -> np.ndarray:
        r"""Calculate the diagonal matrix of weights :math:`W` that depends on the trainable
        parameters :math:`\theta`.

        **Example usage:**

        >>> vgbs.W(params)
        array([[0.97530991, 0.        , 0.        , 0.        ],
               [0.        , 0.95122942, 0.        , 0.        ],
               [0.        , 0.        , 0.99004983, 0.        ],
               [0.        , 0.        , 0.        , 0.99501248]])

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: the diagonal matrix of weights
        """
        return np.sqrt(np.diag(self.embedding(params)))

    def A(self, params: np.ndarray) -> np.ndarray:
        r"""Calculate the trained adjacency matrix :math:`A(\theta)`.

        **Example usage:**

        >>> vgbs.A(params)
        array([[0.        , 0.30298161, 0.31534653, 0.31692721],
               [0.30298161, 0.        , 0.30756059, 0.30910225],
               [0.31534653, 0.30756059, 0.        , 0.32171695],
               [0.31692721, 0.30910225, 0.32171695, 0.        ]])

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: the trained adjacency matrix
        """
        return self.W(params) @ self.A_init @ self.W(params)

    def generate_samples(self, A: np.ndarray, n_samples: int, **kwargs) -> np.ndarray:
        """Generate GBS samples from an input adjacency matrix.

        **Example usage:**

        >>> vgbs.generate_samples(vgbs.A(params), 2)
        array([[1, 1, 1, 0],
               [0, 1, 0, 1]])

        Args:
            A (array): the adjacency matrix
            n_samples (int): the number of GBS samples to generate
            **kwargs: additional arguments to pass to the sampler from
                `The Walrus <https://the-walrus.readthedocs.io/en/stable/>`__

        Returns:
            array:  the generated samples
        """
        cov = A_to_cov(A)

        if self.threshold:
            samples = thewalrus.samples.torontonian_sample_state(
                cov, n_samples, hbar=sf.hbar, **kwargs
            )
        else:
            samples = thewalrus.samples.hafnian_sample_state(cov, n_samples, hbar=sf.hbar, **kwargs)
        return samples

    def add_A_init_samples(self, samples: np.ndarray):
        r"""Add samples of the initial adjacency matrix.
        # TODO consider a get_A_init_samples

        .. warning::

            The added samples must be from the *input* adjacency matrix and not the trained one
            :math:`A(\theta)`.

        **Example usage:**

        >>> samples = np.array([[0, 1, 0, 0], [0, 1, 1, 1]])
        >>> vgbs.add_A_init_samples(samples)

        Args:
            samples (array): samples from the initial adjacency matrix
        """
        shape = samples.shape
        if shape[1] != self.n_modes:
            raise ValueError("Must input samples of shape (number, {})".format(self.n_modes))

        if self._A_init_samples is None:
            self._A_init_samples = samples
        else:
            self._A_init_samples = np.vstack([self._A_init_samples, samples])

    def mean_photons_by_mode(self, params: np.ndarray) -> np.ndarray:
        r"""Calculate the mean number of photons in each mode when using the trainable parameters
        :math:`\theta`.

        **Example usage:**

        >>> vgbs.mean_photons_by_mode(params)
        array([1.87217857, 1.8217392 , 1.90226515, 1.91225543])

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: a vector giving the mean number of photons in each mode
        """
        disp = np.zeros(2 * self.n_modes)
        cov = A_to_cov(self.A(params))
        return photon_number_mean_vector(disp, cov, hbar=sf.hbar)

    def mean_clicks_by_mode(self, params: np.ndarray) -> np.ndarray:
        r"""Calculate the mean number of clicks in each mode when using the trainable parameters
        :math:`\theta`.

        **Example usage:**

        >>> vgbs.mean_clicks_by_mode(params)
        array([0.57419812, 0.5680168 , 0.57781579, 0.57900564])

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: a vector giving the mean number of clicks in each mode
        """
        cov = A_to_cov(self.A(params))
        Q = Qmat(cov, hbar=sf.hbar)
        m = self.n_modes
        Qks = [[[Q[k, k], Q[k, k + m]], [Q[k + m, k], Q[k + m, k + m]]] for k in range(m)]
        cbar = [1 - np.linalg.det(Qks[k]) ** (-0.5) for k in range(m)]
        return np.real(np.array(cbar))

    def n_mean(self, params: np.ndarray) -> float:
        r"""Calculates the mean number of clicks or photons.

        Evaluates the mean number of clicks or photons of the VGBS system when using the
        trainable parameters :math:`\theta`. The mean number of clicks is returned when
        :attr:`threshold` is ``True``, otherwise the mean number of photons is returned.

        **Example usage:**

        >>> vgbs.n_mean(params)
        2.299036355948707

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            float: the mean number of clicks or photons
        """
        if self.threshold:
            return np.sum(self.mean_clicks_by_mode(params))

        return np.sum(self.mean_photons_by_mode(params))

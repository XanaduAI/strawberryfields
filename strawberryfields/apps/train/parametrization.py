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
    return np.linalg.inv(I - X @ A_big) - I / 2


# TODO: update embedding class in type hint
class VGBS:
    r"""Create a variational GBS model for optimization and machine learning.

    An input adjacency matrix :math:`A` can be varied using:

    .. math::

        A(\theta) = W(\theta) A W(\theta)

    with :math:`W` a diagonal matrix of weights that depend on a set of parameters :math:`\theta`.

    By varying :math:`\theta`, the distribution of samples from GBS can be preferentially biased
    towards solving stochastic optimization and unsupervised machine learning problems.

    The above variational model can be used in both the threshold and PNR modes of GBS, which
    is specified using the ``threshold`` flag. An initial value for the mean number of clicks
    (if ``threshold=True``) or mean number of photons (if ``threshold=False``) is also required.
    The initial value ``n_mean`` should be set high since varying :math:`\theta` can only lower
    the mean number of clicks or photons.

    The mapping from :math:`\theta` to :math:`W(\theta)` is specified using the ``embedding``
    argument.

    **Example usage:**

    >>> g = nx.erdos_renyi_graph(4, 0.7, seed=1967)
    >>> a = nx.to_numpy_array(g)
    >>> embedding = Exp()  # TODO does this need updating?
    >>> vgbs = VGBS(a, 3, embedding, True)
    >>> params = np.array([0.05, 0.1, 0.02, 0.01])
    >>> vgbs.A(params)
    array([[0.        , 0.86070798, 0.93239382, 0.94176453],
           [0.86070798, 0.        , 0.88692044, 0.89583414],
           [0.93239382, 0.88692044, 0.        , 0.97044553],
           [0.94176453, 0.89583414, 0.97044553, 0.        ]])
    >>> vgbs.n_mean(params)
    1.8906000271819803
    >>> vgbs.generate_samples(params, 2)
    array([[0, 0, 1, 1],
           [1, 1, 0, 0]])

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
        self.A_init = A
        self.embedding = embedding
        self.threshold = threshold
        self.A_init_scaled = rescale_adjacency(A, n_mean, threshold)
        self.n_modes = len(A)
        self._A_init_samples = None
        if samples:
            self._add_A_init_samples(samples)

    def W(self, params: np.ndarray) -> np.ndarray:
        """Calculate the diagonal matrix of weights :math:`W` that depends on the trainable
        parameters :math:`\theta`.

        **Example usage:**

        >>> vgbs.W(params)
        array([[0.95122942, 0.        , 0.        , 0.        ],
               [0.        , 0.90483742, 0.        , 0.        ],
               [0.        , 0.        , 0.98019867, 0.        ],
               [0.        , 0.        , 0.        , 0.99004983]])

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: the diagonal matrix of weights
        """
        return np.diag(self.embedding(params))

    @staticmethod
    def _WAW(A: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Calculate the :math:`_WAW` parametrization.

        Args:
            A (array): the adjacency matrix
            W (array): the diagonal matrix of weights

        Returns:
            array: the :math:`_WAW` matrix
        """
        return W @ A @ W

    def A(self, params: np.ndarray) -> np.ndarray:
        """Calculate the trained adjacency matrix :math:`A(\theta)`.

        **Example usage:**

        >>> vgbs.A(params)
        array([[0.        , 0.86070798, 0.93239382, 0.94176453],
               [0.86070798, 0.        , 0.88692044, 0.89583414],
               [0.93239382, 0.88692044, 0.        , 0.97044553],
               [0.94176453, 0.89583414, 0.97044553, 0.        ]])

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: the trained adjacency matrix
        """
        return self._WAW(self.A_init, self.W(params))

    def _A_scaled(self, params: np.ndarray) -> np.ndarray:
        """Calculate the trained scaled adjacency matrix :math:`A_{\rm scale}(\theta)`.

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: the trained adjacency matrix
        """
        return self._WAW(self.A_init_scaled, self.W(params))

    def _generate_samples(self, A: np.ndarray, n_samples: int, **kwargs) -> np.ndarray:
        """Generate GBS samples from a chosen adjacency matrix.

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
            samples = thewalrus.samples.torontonian_sample_state(cov, n_samples, hbar=1, **kwargs)
        else:
            samples = thewalrus.samples.hafnian_sample_state(cov, n_samples, hbar=1, **kwargs)
        return samples

    def generate_samples(self, params: np.ndarray, n_samples: int, **kwargs) -> np.ndarray:
        """Generate GBS samples from the trained adjacency matrix :math:`A(\theta)`.

        **Example usage:**

        >>> vgbs.generate_samples(params, 2)
        array([[0, 0, 1, 1],
               [1, 1, 0, 0]])

        Args:
            params (array): the trainable parameters :math:`\theta`
            n_samples (int): the number of GBS samples to generate
            **kwargs: additional arguments to pass to the sampler from
                `The Walrus <https://the-walrus.readthedocs.io/en/stable/>`__

        Returns:
            array: the generated samples
        """
        return self._generate_samples(self._A_scaled(params), n_samples, **kwargs)

    def _generate_A_init_samples(self, n_samples: int, **kwargs) -> np.ndarray:
        """Generate GBS samples from the initial adjacency matrix.

        The resulting samples are added to the to the internal :attr:`_A_init_samples` attribute.

        .. warning::

            Samples are generate from the *input* adjacency matrix and not the trained one
            :math:`A(\theta)`.

        Args:
            n_samples (int): the number of GBS samples to generate
            **kwargs: additional arguments to pass to the sampler from
                `The Walrus <https://the-walrus.readthedocs.io/en/stable/>`__

        Returns:
            array: the generated samples
        """
        samples = self._generate_samples(self.A_init_scaled, n_samples, **kwargs)
        self.add_A_samples(samples)
        return samples

    def _add_A_init_samples(self, samples: np.ndarray):
        """Add samples of the initial adjacency matrix to the internal :attr:`_A_init_samples`
        attribute.

        .. warning::

            The added samples must be from the *input* adjacency matrix and not the trained one
            :math:`A(\theta)`.

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
        """Calculate the mean number of photons in each mode when using the trainable parameters
        :math:`\theta`.

        **Example usage:**

        >>> vgbs.mean_photons_by_mode(params)
        array([1.09176888, 1.03067234, 1.12816899, 1.14021221])

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: a vector giving the mean number of photons in each mode
        """
        disp = np.zeros(2 * self.n_modes)
        cov = A_to_cov(self._A_scaled(params))
        return photon_number_mean_vector(disp, cov, hbar=1)  # TODO: consider hbar=2

    def mean_clicks_by_mode(self, params: np.ndarray) -> np.ndarray:
        """Calculate the mean number of clicks in each mode when using the trainable parameters
        :math:`\theta`.

        **Example usage:**

        >>> vgbs.mean_clicks_by_mode(params)
        array([0.47149519, 0.4578502 , 0.47935017, 0.48190447])

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            array: a vector giving the mean number of clicks in each mode
        """
        cov = A_to_cov(self._A_scaled(params))
        Q = Qmat(cov, hbar=1)
        m = self.n_modes
        Qks = [[[Q[k, k], Q[k, k + m]], [Q[k + m, k], Q[k + m, k + m]]] for k in range(m)]
        cbar = [1 - np.linalg.det(Qks[k]) ** (-0.5) for k in range(m)]
        return np.real(np.array(cbar))

    def n_mean(self, params: np.ndarray) -> float:
        """Calculates the mean number of clicks or photons.

        Evaluates the mean number of clicks or photons of the VGBS system when using the
        trainable parameters :math:`\theta`. The mean number of clicks is returned when
        :attr:`threshold` is ``True``, otherwise the mean number of photons is returned.

        **Example usage:**

        >>> vgbs.n_mean(params)
        1.8906000271819803

        Args:
            params (array): the trainable parameters :math:`\theta`

        Returns:
            float: the mean number of clicks or photons
        """
        if self.threshold:
            return np.sum(self.mean_clicks_by_mode(params))
        else:
            return np.sum(self.mean_photons_by_mode(params))

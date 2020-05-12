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
Submodule for embedding trainable parameters into the GBS distribution.

Training algorithms for GBS distributions rely on the :math:`WAW` parametrization, where :math:`W`
is a diagonal matrix of weights and :math:`A` is a symmetric matrix. Trainable parameters are 
embedded into the GBS distribution by expressing the weights as functions of the parameters.

This submodule contains methods to implement such embeddings. It also provides derivatives
of the weights with respect to the trainable parameters. There are two main classes, each
corresponding to a different embedding. The :class:`Exp` class is a simple embedding where the
weights are exponentials of the trainable parameters. The :class:`ExpFeatures` class is a more
general embedding that makes use of user-defined feature vectors, which potentially provide more
flexibility in training strategies."""

import numpy as np


class ExpFeatures:
    r"""Exponential embedding with feature vectors.

    Weights of the :math:`W` matrix in the :math:`WAW` parametrization are expressed as exponentials
    of the inner product between user-specified feature vectors and trainable parameters:
    :math:`w_i = \exp(-f^{(i)}\cdot\theta)`. The Jacobian, which encapsulates the derivatives of
    the weights with respect to the parameters can be computed straightforwardly as: :math:`
    \frac{d w_i}{d\theta_k} = -f^{(i)}_k w_i`.

    **Example usage:**

    >>> features = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
    >>> embed = ExpFeatures(features)
    >>> params = np.array([0.1, 0.2, 0.3])
    >>> embed(params)
    [0.94176453 0.88692044 0.83527021]

    Args:
        features (np.array): Matrix of feature vectors where the i-th row is the i-th feature vector
    """

    def __init__(self, features):
        """Initializes the class for an input matrix whose rows are feature vectors"""
        self.features = features
        self.m, self.d = np.shape(features)

    def __call__(self, params):
        """Makes the class callable, so it can be used as a function"""
        return self.weights(params)

    def weights(self, params):
        r"""Computes weights as a function of input parameters.
        
        Args:
            params (np.array): trainable parameters
            
        Returns:
            np.array: weights
        """
        d = self.d
        if d != len(params):
            raise ValueError(
                "Dimension of parameter vector must be equal to dimension of feature vectors"
            )
        return np.exp(-self.features @ params)

    def jacobian(self, params):
        r"""Computes the Jacobian matrix of weights with respect to input parameters :math:`J_{
        ij} = \frac{\partial w_i}{\partial \theta_j}`.

        Args:
            params (np.array): trainable parameters

        Returns:
            np.array: Jacobian matrix of weights with respect to parameters
        """
        m, d = self.m, self.d
        if d != len(params):
            raise ValueError(
                "Dimension of parameter vector must be equal to dimension of feature vectors"
            )
        w = self.weights(params)
        return - self.features * w[:, np.newaxis]


class Exp(ExpFeatures):
    r"""Simple exponential embedding.

    Weights of the :math:`W` matrix in the :math:`WAW` parametrization are expressed as exponentials
    of trainable parameters: :math:`w_i = \exp(-\theta_i)`. The Jacobian, which encapsulates the
    derivatives of the weights with respect to the parameters can be computed straightforwardly as:
    :math:`\frac{d w_i}{d\theta_k} = -w_i\delta_{i,k}`.

    **Example usage:**

    >>> dim = 3
    >>> embed = Exp(dim)
    >>> params = np.array([0.1, 0.2, 0.3])
    >>> embed(params)
    [0.90483742 0.81873075 0.74081822]

    Args:
        dim (int): Dimension of the vector of parameters, which is equal to the number of modes
            in the GBS distribution
    """

    def __init__(self, dim):
        """The simple exponential mapping is a special case where the matrix of feature vectors
        is the identity"""
        super().__init__(np.eye(dim))

dim = 4
params = np.zeros(dim)
exp = Exp(dim)
g = exp.jacobian(params)
print(g)
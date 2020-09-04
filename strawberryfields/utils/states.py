# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module defines and implements several utility functions allowing the
calculation of various quantum states in either the Fock basis (a
one-dimensional array indexed by Fock state) or the Gaussian basis (returning a
vector of means and covariance matrix). These state calculations are NOT done
in the simulators, but rather in NumPy.

These are useful for generating states for use in calculating the fidelity of
simulations.
"""
import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.special import factorial as fac

__all__ = [
    "squeezed_cov",
    "vacuum_state",
    "coherent_state",
    "squeezed_state",
    "displaced_squeezed_state",
    "fock_state",
    "cat_state",
]

# ------------------------------------------------------------------------
# State functions - Fock basis and Gaussian basis                |
# ------------------------------------------------------------------------


def squeezed_cov(r, phi, hbar=2):
    r"""Returns the squeezed covariance matrix of a squeezed state

    Args:
        r (complex): the squeezing magnitude
        p (float): the squeezing phase :math:`\phi`
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
    Returns:
        array: the squeezed state
    """
    cov = np.array([[np.exp(-2 * r), 0], [0, np.exp(2 * r)]]) * hbar / 2

    R = np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])

    return np.dot(np.dot(R, cov), R.T)


def vacuum_state(basis="fock", fock_dim=5, hbar=2.0):
    r"""Returns the vacuum state

    Args:
        basis (str): If 'fock', calculates the initial state
            in the Fock basis. If 'gaussian', returns the
            vector of means and the covariance matrix.
        fock_dim (int): the size of the truncated Fock basis if
            using the Fock basis representation
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
    Returns:
        array: the vacuum state
    """
    if basis == "fock":
        state = np.zeros((fock_dim))
        state[0] = 1.0

    elif basis == "gaussian":
        means = np.zeros((2))
        cov = np.identity(2) * hbar / 2
        state = [means, cov]

    return state


def coherent_state(r, phi, basis="fock", fock_dim=5, hbar=2.0):
    r"""Returns the coherent state

    This can be returned either in the Fock basis,

    .. math::

        |\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^\infty
        \frac{\alpha^n}{\sqrt{n!}}|n\rangle

    or as a Gaussian:

    .. math::

        \mu = (\text{Re}(\alpha),\text{Im}(\alpha)),~~~\sigma = I

    where :math:`\alpha` is the displacement.

    Args:
        r (float) : displacement magnitude
        phi (float) : displacement phase
        basis (str): If 'fock', calculates the initial state
            in the Fock basis. If 'gaussian', returns the
            vector of means and the covariance matrix.
        fock_dim (int): the size of the truncated Fock basis if
            using the Fock basis representation
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
    Returns:
        array: the coherent state
    """
    a = r * np.exp(1j * phi)

    if basis == "fock":
        state = np.array(
            [np.exp(-0.5 * r ** 2) * a ** n / np.sqrt(fac(n)) for n in range(fock_dim)]
        )

    elif basis == "gaussian":
        means = np.array([a.real, a.imag]) * np.sqrt(2 * hbar)
        cov = np.identity(2) * hbar / 2
        state = [means, cov]

    return state


def squeezed_state(r, p, basis="fock", fock_dim=5, hbar=2.0):
    r"""Returns the squeezed state

    This can be returned either in the Fock basis,

    .. math::

        |z\rangle = \frac{1}{\sqrt{\cosh(r)}}\sum_{n=0}^\infty
        \frac{\sqrt{(2n)!}}{2^n n!}(-e^{i\phi}\tanh(r))^n|2n\rangle

    or as a Gaussian:

    .. math:: \mu = (0,0)

    .. math::
        :nowrap:

        \begin{align*}
            \sigma = R(\phi/2)\begin{bmatrix}e^{-2r} & 0 \\0 & e^{2r} \\\end{bmatrix}R(\phi/2)^T
        \end{align*}

    where :math:`z = re^{i\phi}` is the squeezing factor.

    Args:
        r (complex): the squeezing magnitude
        p (float): the squeezing phase :math:`\phi`
        basis (str): If 'fock', calculates the initial state
            in the Fock basis. If 'gaussian', returns the
            vector of means and the covariance matrix.
        fock_dim (int): the size of the truncated Fock basis if
            using the Fock basis representation
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
    Returns:
        array: the squeezed state
    """
    phi = p

    if basis == "fock":

        def ket(n):
            """Squeezed state kets"""
            return (np.sqrt(fac(2 * n)) / (2 ** n * fac(n))) * (-np.exp(1j * phi) * np.tanh(r)) ** n

        state = np.array([ket(n // 2) if n % 2 == 0 else 0.0 for n in range(fock_dim)])
        state *= np.sqrt(1 / np.cosh(r))

    elif basis == "gaussian":
        means = np.zeros((2))
        state = [means, squeezed_cov(r, phi, hbar)]

    return state


def displaced_squeezed_state(r_d, phi_d, r_s, phi_s, basis="fock", fock_dim=5, hbar=2.0):
    r"""Returns the squeezed coherent state

    This can be returned either in the Fock basis,

    .. math::

        |\alpha,z\rangle = e^{-\frac{1}{2}|\alpha|^2-\frac{1}{2}{\alpha^*}^2 e^{i\phi}\tanh{(r)}}
        \sum_{n=0}^\infty\frac{\left[\frac{1}{2}e^{i\phi}\tanh(r)\right]^{n/2}}{\sqrt{n!\cosh(r)}}
        H_n\left[ \frac{\alpha\cosh(r)+\alpha^*e^{i\phi}\sinh(r)}{\sqrt{e^{i\phi}\sinh(2r)}} \right]|n\rangle

    where :math:`H_n(x)` is the Hermite polynomial, or as a Gaussian:

    .. math:: \mu = (\text{Re}(\alpha),\text{Im}(\alpha))

    .. math::
        :nowrap:

        \begin{align*}
            \sigma = R(\phi/2)\begin{bmatrix}e^{-2r} & 0 \\0 & e^{2r} \\\end{bmatrix}R(\phi/2)^T
        \end{align*}

    where :math:`z = re^{i\phi}` is the squeezing factor
    and :math:`\alpha` is the displacement.

    Args:
        r_d (float): displacement magnitude
        phi_d (float): displacement phase
        r_s (float): the squeezing magnitude
        phi_s (float): the squeezing phase :math:`\phi`
        basis (str): If 'fock', calculates the initial state
            in the Fock basis. If 'gaussian', returns the
            vector of means and the covariance matrix.
        fock_dim (int): the size of the truncated Fock basis if
            using the Fock basis representation
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
    Returns:
        array: the squeezed coherent state
    """
    # pylint: disable=too-many-arguments
    a = r_d * np.exp(1j * phi_d)

    if basis == "fock":
        if r_s != 0:
            phase_factor = np.exp(1j * phi_s)
            ch = np.cosh(r_s)
            sh = np.sinh(r_s)
            th = np.tanh(r_s)

            gamma = a * ch + np.conj(a) * phase_factor * sh
            N = np.exp(-0.5 * np.abs(a) ** 2 - 0.5 * np.conj(a) ** 2 * phase_factor * th)

            coeff = np.diag(
                [
                    (0.5 * phase_factor * th) ** (n / 2) / np.sqrt(fac(n) * ch)
                    for n in range(fock_dim)
                ]
            )

            vec = [hermval(gamma / np.sqrt(phase_factor * np.sinh(2 * r_s)), row) for row in coeff]

            state = N * np.array(vec)

        else:
            state = coherent_state(r_d, phi_d, basis="fock", fock_dim=fock_dim)  # pragma: no cover

    elif basis == "gaussian":
        means = np.array([a.real, a.imag]) * np.sqrt(2 * hbar)
        state = [means, squeezed_cov(r_s, phi_s, hbar)]

    return state


# ------------------------------------------------------------------------
# State functions - Fock basis only                              |
# ------------------------------------------------------------------------


def fock_state(n, fock_dim=5):
    r"""Returns the Fock state

    Args:
        n (int): the occupation number
        fock_dim (int): the size of the truncated Fock basis
    Returns:
        array: the Fock state
    """
    ket = np.zeros((fock_dim))
    ket[n] = 1.0
    return ket


def cat_state(a, p=0, fock_dim=5):
    r"""Returns the cat state

    .. math::

        |cat\rangle = \frac{1}{\sqrt{2(1+e^{-2|\alpha|^2}\cos(\phi))}}
        \left(|\alpha\rangle +e^{i\phi}|-\alpha\rangle\right)

    with the even cat state given for :math:`\phi=0`, and the odd
    cat state given for :math:`\phi=\pi`.

    Args:
        a (complex): the displacement
        p (float): parity, where :math:`\phi=p\pi`. ``p=0`` corresponds to an even
            cat state, and ``p=1`` an odd cat state
        fock_dim (int): the size of the truncated Fock basis
    Returns:
        array: the cat state
    """
    # p=0 if even, p=pi if odd
    phi = np.pi * p

    # normalisation constant
    temp = np.exp(-0.5 * np.abs(a) ** 2)
    N = temp / np.sqrt(2 * (1 + np.cos(phi) * temp ** 4))

    # coherent states
    k = np.arange(fock_dim)
    c1 = (a ** k) / np.sqrt(fac(k))
    c2 = ((-a) ** k) / np.sqrt(fac(k))

    # add them up with a relative phase
    ket = (c1 + np.exp(1j * phi) * c2) * N

    return ket

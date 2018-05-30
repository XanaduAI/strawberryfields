# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Utilities
=========

**Module name:**  :mod:`strawberryfields.utils`

.. currentmodule:: strawberryfields.utils

This module defines and implements several utility functions and language extensions that complement
StrawberryFields.


Classical processing functions
------------------------------

These functions provide common mathematical operations that may be required for
classical processing of measured modes input to other gates. They may be used
as follows:

.. code-block:: python

    MeasureX | q[0]
    Xgate(scale(q[0], sqrt(0.5))) | q[1]

Available classical processing functions include:

.. autosummary::
    neg
    mag
    phase
    scale
    shift
    scale_shift
    power

If more advanced classical processing is required, custom classical processing
functions can be created using the :func:`strawberryfields.convert` decorator.


NumPy state functions
-----------------------------

These functions allow the calculation of various quantum states in either the Fock
basis (a one-dimensional array indexed by Fock state) or the Gaussian basis (returning
a vector of means and covariance matrix). These state calculations are NOT done in the
simulators, but rather in NumPy.

These are useful for generating states for use in calculating the fidelity of simulations.

.. autosummary::
   squeezed_cov
   vacuum_state
   coherent_state
   squeezed_state
   displaced_squeezed_state
   fock_state
   cat_state


Random functions
------------------------

These functions generate random numbers and matrices corresponding to various
quantum states and operations.

.. autosummary::
   randnc
   random_covariance
   random_symplectic
   random_interferometer

Code details
~~~~~~~~~~~~

"""
import numpy as np
from numpy.random import randn
from numpy.polynomial.hermite import hermval as H

import scipy as sp
from scipy.special import factorial as fac

from .engine import _convert

# pylint: disable=abstract-method,ungrouped-imports,

#------------------------------------------------------------------------
# RegRef convert functions                                              |
#------------------------------------------------------------------------

@_convert
def neg(x):
    r"""Negates a measured mode value.

    Args:
        x (RegRef): mode that has been previously measured
    """
    return -x


@_convert
def mag(x):
    r"""Returns the magnitude :math:`|z|` of a measured mode value.

    Args:
        x (RegRef): mode that has been previously measured
    """
    return np.abs(x)


@_convert
def phase(x):
    r"""Returns the phase :math:`\phi` of a measured mode value :math:`z=re^{i\phi}`.

    Args:
        x (RegRef): mode that has been previously measured
    """
    return np.angle(x)


def scale(x, a):
    r"""Scales the value of a measured mode by factor ``a``.

    Args:
        x (RegRef): mode that has been previously measured
        a (float): scaling factor
    """
    @_convert
    def rrt(x):
        """RegRefTransform function"""
        return a*x
    return rrt(x)


def shift(x, b):
    r"""Shifts the value of a measured mode by factor ``b``.

    Args:
        x (RegRef): mode that has been previously measured
        b (float): shifting factor
    """
    @_convert
    def rrt(x):
        """RegRefTransform function"""
        return b+x
    return rrt(x)


def scale_shift(x, a, b):
    r"""Scales the value of a measured mode by factor ``a`` then shifts the result by ``b``.

    .. math:: u' = au + b

    Args:
        x (RegRef): mode that has been previously measured
        a (float): scaling factor
        b (float): shifting factor
    """
    @_convert
    def rrt(x):
        """RegRefTransform function"""
        return a*x + b
    return rrt(x)


def power(x, a):
    r"""Raises the value of a measured mode to power a.

    Args:
        x (RegRef): mode that has been previously measured
        a (float): the exponent of x. Note that a can be
            negative and fractional.
    """
    if a < 0:
        tmp = float(a)
    else:
        tmp = a

    @_convert
    def rrt(x):
        """RegRefTransform function"""
        return np.power(x, tmp)
    return rrt(x)

#------------------------------------------------------------------------
# State functions - Fock basis and Gaussian basis                |
#------------------------------------------------------------------------

def squeezed_cov(r, phi, hbar=2):
    r"""Returns the squeezed covariance matrix of a squeezed state

    Args:
        r (complex): the squeezing magnitude
        p (float): the squeezing phase :math:`\phi`
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the squeezed state
    """
    cov = np.array([[np.exp(-2*r), 0],
                    [0, np.exp(2*r)]]) * hbar/2

    R = np.array([[np.cos(phi/2), -np.sin(phi/2)],
                  [np.sin(phi/2), np.cos(phi/2)]])

    return np.dot(np.dot(R, cov), R.T)


def vacuum_state(basis='fock', fock_dim=5, hbar=2.):
    r""" Returns the vacuum state

    Args:
        basis (str): if 'fock', calculates the initial state
            in the Fock basis. If 'gaussian', returns the
            vector of means and the covariance matrix.
        fock_dim (int): the size of the truncated Fock basis if
            using the Fock basis representation.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the vacuum state
    """
    if basis == 'fock':
        state = np.zeros((fock_dim))
        state[0] = 1.

    elif basis == 'gaussian':
        means = np.zeros((2))
        cov = np.identity(2) * hbar/2
        state = [means, cov]

    return state


def coherent_state(a, basis='fock', fock_dim=5, hbar=2.):
    r""" Returns the coherent state

    This can be returned either in the Fock basis,

    .. math::

        |\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^\infty
        \frac{\alpha^n}{\sqrt{n!}}|n\rangle

    or as a Gaussian:

    .. math::

        \mu = (\text{Re}(\alpha),\text{Im}(\alpha)),~~~\sigma = I

    where :math:`\alpha` is the displacement.

    Args:
        a (complex) : the displacement
        basis (str): if 'fock', calculates the initial state
            in the Fock basis. If 'gaussian', returns the
            vector of means and the covariance matrix.
        fock_dim (int): the size of the truncated Fock basis if
            using the Fock basis representation.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the coherent state
    """
    if basis == 'fock':
        state = np.array([
            np.exp(-0.5*np.abs(a)**2)*a**n/np.sqrt(fac(n))
            for n in range(fock_dim)])

    elif basis == 'gaussian':
        means = np.array([a.real, a.imag]) * np.sqrt(2*hbar)
        cov = np.identity(2) * hbar/2
        state = [means, cov]

    return state


def squeezed_state(r, p, basis='fock', fock_dim=5, hbar=2.):
    r""" Returns the squeezed state

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
        basis (str): if 'fock', calculates the initial state
            in the Fock basis. If 'gaussian', returns the
            vector of means and the covariance matrix.
        fock_dim (int): the size of the truncated Fock basis if
            using the Fock basis representation.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the squeezed state
    """
    phi = p

    if basis == 'fock':
        ket = lambda n: (np.sqrt(fac(2*n))/(2**n*fac(n))) * \
                        (-np.exp(1j*phi)*np.tanh(r))**n
        state = np.array([ket(n//2) if n %
                          2 == 0 else 0. for n in range(fock_dim)])
        state *= np.sqrt(1/np.cosh(r))

    elif basis == 'gaussian':
        means = np.zeros((2))
        state = [means, squeezed_cov(r, phi, hbar)]

    return state


def displaced_squeezed_state(a, r, phi, basis='fock', fock_dim=5, hbar=2.):
    r""" Returns the squeezed coherent state

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
        a (complex): the displacement
        r (complex): the squeezing magnitude
        phi (float): the squeezing phase :math:`\phi`
        basis (str): if 'fock', calculates the initial state
            in the Fock basis. If 'gaussian', returns the
            vector of means and the covariance matrix.
        fock_dim (int): the size of the truncated Fock basis if
            using the Fock basis representation.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the squeezed coherent state
    """
    # pylint: disable=too-many-arguments
    if basis == 'fock':

        if r != 0:
            phase_factor = np.exp(1j*phi)
            ch = np.cosh(r)
            sh = np.sinh(r)
            th = np.tanh(r)

            gamma = a*ch+np.conj(a)*phase_factor*sh
            N = np.exp(-0.5*np.abs(a)**2-0.5*np.conj(a)**2*phase_factor*th)

            coeff = np.diag(
                [(0.5*phase_factor*th)**(n/2)/np.sqrt(fac(n)*ch)
                 for n in range(fock_dim)]
            )

            vec = [H(gamma/np.sqrt(phase_factor*np.sinh(2*r)), row) for row in coeff]

            state = N*np.array(vec)

        else:
            state = coherent_state(a, basis='fock', fock_dim=fock_dim) # pragma: no cover

    elif basis == 'gaussian':
        means = np.array([a.real, a.imag]) * np.sqrt(2*hbar)
        state = [means, squeezed_cov(r, phi, hbar)]

    return state


#------------------------------------------------------------------------
# State functions - Fock basis only                              |
#------------------------------------------------------------------------


def fock_state(n, fock_dim=5):
    r""" Returns the Fock state

    Args:
        n (int): the occupation number
        fock_dim (int): the size of the truncated Fock basis
    Returns:
        array: the Fock state
    """
    ket = np.zeros((fock_dim))
    ket[n] = 1.
    return ket


def cat_state(a, p=0, fock_dim=5):
    r""" Returns the cat state

    .. math::

        |cat\rangle = \frac{e^{-|\alpha|^2/2}}{\sqrt{2(1+e^{-2|\alpha|^2}\cos(\phi))}}
        \left(|\alpha\rangle +e^{i\phi}|-\alpha\rangle\right)

    with the even cat state given for :math:`\phi=0`, and the odd
    cat state given for :math:`\phi=\pi`.

    Args:
        a (complex): the displacement
        p (float): parity, where :math:`\phi=p\pi`. ``p=0`` corresponds to an even
            cat state, and ``p=1`` an odd cat state.
        fock_dim (int): the size of the truncated Fock basis
    Returns:
        array: the cat state
    """
    # p=0 if even, p=pi if odd
    phi = np.pi*p

    # normalisation constant
    temp = np.exp(-0.5 * np.abs(a)**2)
    N = temp / np.sqrt(2*(1 + np.cos(phi) * temp**4))

    # coherent states
    k = np.arange(fock_dim)
    c1 = (a**k) / np.sqrt(fac(k))
    c2 = ((-a)**k) / np.sqrt(fac(k))

    # add them up with a relative phase
    ket = (c1 + np.exp(1j*phi) * c2) * N

    return ket


#------------------------------------------------------------------------
# Random numbers and matrices                                           |
#------------------------------------------------------------------------


def randnc(*arg):
    """Normally distributed array of random complex numbers."""
    return randn(*arg) + 1j*randn(*arg)


def random_covariance(N, hbar=2, pure=False):
    r"""Returns a random covariance matrix.

    Args:
        N (int): number of modes
        hbar (float): the value of :math:`\hbar` to use in the definition
            of the quadrature operators :math:`\x` and :math:`\p`
        pure (bool): if True, a random covariance matrix corresponding
            to a pure state is returned
    Returns:
        array: random :math:`2N\times 2N` covariance matrix
    """
    S = random_symplectic(N)

    if pure:
        return (hbar/2) * S @ S.T

    nbar = np.abs(np.random.random(N))
    Vth = (hbar/2) * np.diag(np.concatenate([nbar, nbar]))

    return S @ Vth @ S.T


def random_symplectic(N, passive=False):
    r"""Returns a random symplectic matrix representing a Gaussian transformation.

    The squeezing parameters :math:`r` for active transformations are randomly
    sampled from the standard normal distribution, while passive transformations
    are randomly sampled from the Haar measure.

    Args:
        N (int): number of modes
        passive (bool): if True, returns a passive Gaussian transformation (i.e.,
            one that preserves photon number). If False (default), returns an active
            transformation.
    Returns:
        array: random :math:`2N\times 2N` symplectic matrix
    """
    U = random_interferometer(N)
    O = np.vstack([np.hstack([U.real, -U.imag]), np.hstack([U.imag, U.real])])

    if passive:
        return O

    U = random_interferometer(N)
    P = np.vstack([np.hstack([U.real, -U.imag]), np.hstack([U.imag, U.real])])

    r = np.abs(randnc(N))
    Sq = np.diag(np.concatenate([np.exp(-r), np.exp(r)]))

    return O @ Sq @ P


def random_interferometer(N):
    r"""Returns a random unitary matrix representing an interferometer.

    For more details, see :cite:`mezzadri2006`.

    Args:
        N (int): number of modes
        passive (bool): if True, returns a passive Gaussian transformation (i.e.,
            one that preserves photon number). If False (default), returns an active
            transformation.
    Returns:
        array: random :math:`N\times N` unitary distributed with the Haar measure
    """
    z = randnc(N, N)/np.sqrt(2.0)
    q, r = sp.linalg.qr(z)
    d = sp.diagonal(r)
    ph = d/np.abs(d)
    U = np.multiply(q, ph, q)
    return U

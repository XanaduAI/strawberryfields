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
"""Miscellaneous Fock backend operations"""
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


import functools
import string
from itertools import product

import numpy as np
from numpy import sqrt, sinh, cosh, tanh, array, exp
from numpy.polynomial.hermite import hermval as H

from scipy.special import factorial as fac
from scipy.linalg import expm as matrixExp

from thewalrus.fock_gradients import (
    displacement as displacement_tw,
    squeezing as squeezing_tw,
    two_mode_squeezing as two_mode_squeezing_tw,
    beamsplitter as beamsplitter_tw,
)

def_type = np.complex128
indices = string.ascii_lowercase


def indexRange(lst, trunc):
    """
    Returns a generator ranging over the possible values for unspecified
    indices in `lst`.

    Example:
        .. code-block:: python

                >>> for i in indexRange([0, None, 1, None], 3): print i
                (0, 0, 1, 0)
                (0, 0, 1, 1)
                (0, 0, 1, 2)
                (0, 1, 1, 0)
                (0, 1, 1, 1)
                (0, 1, 1, 2)
                (0, 2, 1, 0)
                (0, 2, 1, 1)
                (0, 2, 1, 2)

    Args:
        lst (list<int or None>): a list of (possible unspecified) integers
        trunc (int): the number to range unspecified values up to

    Returns:
        Generator of ints
    """

    for vals in product(*([range(trunc) for x in lst if x is None])):
        gen = (v for v in vals)
        yield [next(gen) if v is None else v for v in lst]  # pylint: disable=stop-iteration-return


def index(lst, trunc):
    """
    Converts an n-ary index to a 1-dimensional index.
    """
    return sum([lst[i] * trunc ** (len(lst) - i - 1) for i in range(len(lst))])


def unIndex(i, n, trunc):
    """
    Converts a 1-dimensional index ``i`` with truncation ``trunc`` and
    number of modes ``n`` to a n-ary index.
    """
    return [i // trunc ** (n - 1 - m) % trunc for m in range(n)]


def sliceExp(axes, ind, n):
    """
    Generates a slice expression for a list of pairs of axes (modes) and indices.
    """
    return [ind[i] if i in axes else slice(None, None, None) for i in range(n)]


def abssqr(z):
    r"""
    Given :math:`z` returns :math:`|z|^2`.
    """
    return z.real ** 2 + z.imag ** 2


def dagger(mat):
    r"""
    Given :math:`U` returns :math:`U^\dagger`.
    """
    return mat.conj().T


def mix(state, n):
    """
    Transforms a pure state into a mixed state. Does not do any checks on the
    shape of the input state.
    """

    left_str = [indices[i] for i in range(0, 2 * n, 2)]
    right_str = [indices[i] for i in range(1, 2 * n, 2)]
    out_str = [indices[: 2 * n]]
    einstr = "".join(left_str + [","] + right_str + ["->"] + out_str)
    return np.einsum(einstr, state, state.conj())


def diagonal(state, n):
    """
    Computes the diagonal of a density matrix.
    """

    left_str = [indices[i] + indices[i] for i in range(n)]
    out_str = [indices[:n]]
    einstr = "".join(left_str + ["->"] + out_str)
    return np.einsum(einstr, state)


def trace(state, n):
    """
    Computes the trace of a density matrix.
    """

    left_str = [indices[i] + indices[i] for i in range(n)]
    einstr = "".join(left_str)
    return np.einsum(einstr, state)


def partial_trace(state, n, modes):
    """
    Computes the partial trace of a state over the modes in `modes`.

    Expects state to be in mixed state form.
    """
    left_str = [
        indices[2 * i] + indices[2 * i] if i in modes else indices[2 * i : 2 * i + 2]
        for i in range(n)
    ]
    out_str = ["" if i in modes else indices[2 * i : 2 * i + 2] for i in range(n)]
    einstr = "".join(left_str + ["->"] + out_str)

    return np.einsum(einstr, state)


def tensor(u, v, n, pure, pos=None):
    """
    Returns the tensor product of `u` and `v`, optionally spliced into a
    at location `pos`.
    """

    w = np.tensordot(u, v, axes=0)

    if pos is not None:
        if pure:
            scale = 1
        else:
            scale = 2
        for i in range(v.ndim):
            w = np.rollaxis(w, scale * n + i, scale * pos + i)

    return w


def project_reset(modes, x, state, pure, n, trunc):
    r"""
    Applies the operator :math:`\ket{00\dots 0}\bra{\mathbf{x}}` to the
    modes in `modes`.
    """
    inSlice = tuple(sliceExp(modes, dict(zip(modes, x)), n))
    outSlice = tuple(sliceExp(modes, dict(zip(modes, [0] * len(modes))), n))

    def intersperse(lst):
        # pylint: disable=missing-docstring
        return tuple([lst[i // 2] for i in range(len(lst) * 2)])

    if pure:
        ret = np.zeros([trunc for i in range(n)], dtype=def_type)
        ret[outSlice] = state[inSlice]
    else:
        ret = np.zeros([trunc for i in range(n * 2)], dtype=def_type)
        ret[intersperse(outSlice)] = state[intersperse(inSlice)]

    return ret


# ============================================
#
# Gates
#
# ============================================


@functools.lru_cache()
def a(trunc):
    r"""
    The annihilation operator :math:`a`.
    """
    ret = np.zeros((trunc, trunc), dtype=def_type)
    for i in range(1, trunc):
        ret[i - 1][i] = sqrt(i)
    return ret


@functools.lru_cache()
def displacement(r, phi, trunc):
    r"""The displacement operator :math:`D(\alpha)`.

    Uses the `displacement operation from The Walrus`_ to calculate the displacement.

    .. _`displacement operation from The Walrus`: https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.displacement.html

    Args:
            r (float): the displacement amplitude
            phi (float): the displacement angle
            trunc (int): the Fock cutoff
    """

    ret = displacement_tw(r, phi, cutoff=trunc)

    return ret


@functools.lru_cache()
def squeezing(r, theta, trunc):
    r"""The squeezing operator :math:`S(re^{i\theta})`.

    Uses the `squeezing operation from The Walrus`_ to calculate the squeezing.

    .. _`squeezing operation from The Walrus`: https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.squeezing.html

    Args:
            r (float): the magnitude of the squeezing
            theta (float): the squeezing angle
            trunc (int): the Fock cutoff
    """

    ret = squeezing_tw(r, theta, cutoff=trunc)

    return ret


@functools.lru_cache()
def two_mode_squeeze(r, theta, trunc):
    r"""The two-mode squeezing operator :math:`S_2(re^{i\theta})`.

    Args:
        r (float): two-mode squeezing magnitude
        theta (float): two-mode squeezing phase
        trunc (int): Fock ladder cutoff
    """
    ret = two_mode_squeezing_tw(r, theta, cutoff=trunc)

    # Transpose needed because of different conventions in SF and The Walrus.
    ret = np.transpose(ret, [0, 2, 1, 3])

    return ret


@functools.lru_cache()
def kerr(kappa, trunc):
    r"""
    The Kerr interaction :math:`K(\kappa)`.
    """
    n = np.arange(trunc)
    ret = np.diag(np.exp(1j * kappa * n ** 2))
    return ret


@functools.lru_cache()
def cross_kerr(kappa, trunc):
    r"""
    The cross-Kerr interaction :math:`CK(\kappa)`.
    """
    n1 = np.arange(trunc)[None, :]
    n2 = np.arange(trunc)[:, None]
    n1n2 = np.ravel(n1 * n2)
    ret = np.diag(np.exp(1j * kappa * n1n2)).reshape([trunc] * 4).swapaxes(1, 2)
    return ret


@functools.lru_cache()
def cubicPhase(gamma, hbar, trunc):
    r"""
    The cubic phase gate :math:`\exp{(i\frac{\gamma}{3\hbar}\hat{x}^3)}`.
    """
    a_ = a(trunc)
    x = (a_ + np.conj(a_).T) * np.sqrt(hbar / 2)
    x3 = x @ x @ x
    ret = matrixExp(1j * gamma / (3 * hbar) * x3)

    return ret


@functools.lru_cache()
def phase(theta, trunc):
    r"""
    The phase gate :math:`R(\theta)`
    """
    return np.array(np.diag([exp(1j * n * theta) for n in range(trunc)]), dtype=def_type)


# pylint: disable=unused-argument
@functools.lru_cache()
def beamsplitter(theta, phi, trunc):
    r"""The beamsplitter :math:`B(\theta, \phi)`.

    Uses the `beamsplitter operation from The Walrus`_ to calculate the beamsplitter.

    .. _`beamsplitter operation from The Walrus`: https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.beamsplitter.html
    """
    # pylint: disable=bad-whitespace

    BS_tw = beamsplitter_tw(theta, phi, cutoff=trunc)

    # Transpose needed because of different conventions in SF and The Walrus.
    return BS_tw.transpose((0, 2, 1, 3))


@functools.lru_cache()
def proj(i, j, trunc):
    r"""
    The projector :math:`P = \ket{j}\bra{i}`.
    """
    P = np.zeros((trunc, trunc), dtype=def_type)
    P[j][i] = 1.0 + 0.0j
    return P


# ============================================
#
# State vectors
#
# ============================================


def vacuumState(n, trunc):
    r"""
    The `n`-mode vacuum state :math:`\ket{00\dots 0}`
    """

    state = np.zeros([trunc for i in range(n)], dtype=def_type)
    state.ravel()[0] = 1.0 + 0.0j
    return state


def vacuumStateMixed(n, trunc):
    r"""
    The `n`-mode mixed vacuum state :math:`\ket{00\dots 0}\bra{00\dots 0}`
    """

    state = np.zeros([trunc for i in range(n * 2)], dtype=def_type)
    state.ravel()[0] = 1.0 + 0.0j
    return state


@functools.lru_cache()
def fockState(n, trunc):
    r"""
    The Fock state :math:`\ket{n}`.
    """
    return array([1.0 + 0.0j if i == n else 0.0 + 0.0j for i in range(trunc)])


@functools.lru_cache()
def coherentState(r, phi, trunc):
    r"""
    The coherent state :math:`D(\alpha)\ket{0}` where `alpha = r * np.exp(1j * phi)`.
    """
    alpha = r * np.exp(1j * phi)

    def entry(n):
        """coherent summation term"""
        return alpha ** n / sqrt(fac(n))

    return exp(-abssqr(alpha) / 2) * array([entry(n) for n in range(trunc)])


@functools.lru_cache()
def squeezedState(r, theta, trunc):
    r"""
    The squeezed state :math:`S(re^{i\theta})`.
    """

    def entry(n):
        """squeezed summation term"""
        return (sqrt(fac(2 * n)) / (2 ** n * fac(n))) * (-exp(1j * theta) * tanh(r)) ** n

    vec = array([entry(n // 2) if n % 2 == 0 else 0.0 + 0.0j for n in range(trunc)])
    return sqrt(1 / cosh(r)) * vec


@functools.lru_cache()
def displacedSqueezed(r_d, phi_d, r_s, phi_s, trunc):
    r"""
    The displaced squeezed state :math:`\ket{\alpha,\zeta} = D(\alpha)S(r\exp{(i\phi)})\ket{0}`  where `alpha = r_d * np.exp(1j * phi_d)` and `zeta = r_s * np.exp(1j * phi_s)`.
    """
    if np.allclose(r_s, 0.0):
        return coherentState(r_d, phi_d, trunc)

    if np.allclose(r_d, 0.0):
        return squeezedState(r_s, phi_s, trunc)

    ph = np.exp(1j * phi_s)
    ch = cosh(r_s)
    sh = sinh(r_s)
    th = tanh(r_s)
    alpha = r_d * np.exp(1j * phi_d)

    gamma = alpha * ch + np.conj(alpha) * ph * sh
    hermite_arg = gamma / np.sqrt(ph * np.sinh(2 * r_s) + 1e-10)

    # normalization constant
    N = np.exp(-0.5 * np.abs(alpha) ** 2 - 0.5 * np.conj(alpha) ** 2 * ph * th)

    coeff = np.array([(0.5 * ph * th) ** (n / 2) / np.sqrt(fac(n) * ch) for n in range(trunc)])
    vec = np.array([H(hermite_arg, row) for row in np.diag(coeff)])
    state = N * vec

    return state


@functools.lru_cache()
def thermalState(nbar, trunc):
    r"""
    The thermal state :math:`\rho(\overline{nbar})`.
    """
    if nbar == 0:
        st = fockState(0, trunc)
        state = np.outer(st, st.conjugate())
    else:
        coeff = np.array([nbar ** n / (nbar + 1) ** (n + 1) for n in range(trunc)])
        state = np.diag(coeff)

    return state


# ============================================
#
# Channels (Kraus operators)
#
# ============================================


@functools.lru_cache()
def lossChannel(T, trunc):
    r"""
    The Kraus operators for the loss channel :math:`\mathcal{N}(T)`.
    """

    TToAdaggerA = np.array(np.diag([T ** (i / 2) for i in range(trunc)]), dtype=def_type)

    def aToN(n):
        """the nth matrix power of the annihilation operator matrix a"""
        return np.linalg.matrix_power(a(trunc), n)

    def E(n):
        """the loss channel amplitudes in the Fock basis"""
        return ((1 - T) / T) ** (n / 2) * np.dot(aToN(n) / sqrt(fac(n)), TToAdaggerA)

    if T == 0:
        return [proj(i, 0, trunc) for i in range(trunc)]

    return [E(n) for n in range(trunc)]


# ============================================
#
# Misc
#
# ============================================


@functools.lru_cache()
def hermiteVals(q_mag, num_bins, m_omega_over_hbar, trunc):
    """
    Helper function for homodyne measurements. Computes a range of (physicist's)
    Hermite polynomials at a particular vector.
    """
    q_tensor = np.linspace(-q_mag, q_mag, num_bins)
    x = np.sqrt(m_omega_over_hbar) * q_tensor

    Hvals = [None] * trunc
    Hvals[0] = 1
    Hvals[1] = 2 * x
    for i in range(2, trunc):
        Hvals[i] = 2 * x * Hvals[i - 1] - 2 * (i - 1) * Hvals[i - 2]

    return q_tensor, Hvals

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
Miscellaneous Fock backend operations
======================================

.. currentmodule:: strawberryfields.backends.fockbackend.ops

Conventions
----------------------
Density matrices and unitaries are both stored as stacked matrices --
that is, the kets and bras of subsystems are grouped together.

Utilities
----------------------

.. autosummary::
     genOfRange
     genOfTuple
     indexRange
     index
     unIndex
     sliceExp
     abssqr
     dagger
     mix
     diagonal
     trace
     partial_trace
     tensor
     project_reset

Matrix multiplication
----------------------

.. autosummary::
     apply_gate_BLAS
     apply_gate_einsum

Gates
----------------------

.. autosummary::
     a
     displacement
     squeezing
     phase
     beamsplitter
     addition
     controlledPhase
     projector
     proj
     multimode_projector
     multimode_proj

State vectors
----------------------

.. autosummary::
     vacuumState
     vacuumStateMixed
     fockState
     coherentState
     squeezedState
     squeezedCoherentState

Channels
----------------------

.. autosummary::
     lossChanel

"""
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


import functools
import string
from itertools import product

import numpy as np
from numpy import (pi, sqrt, sinh, cosh, tanh, array, exp)
from numpy.polynomial.hermite import hermval as H

from scipy.special import factorial as fac
from scipy.linalg import expm as matrixExp

from .. import shared_ops as so


def_type = np.complex128
indices = string.ascii_lowercase


def genOfRange(size):
    """
    Converts a range into a generator.
    """
    for i in range(size):
        yield i


def genOfTuple(t):
    """
    Converts a tuple into a generator
    """
    for val in t:
        yield val


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
        gen = genOfTuple(vals)
        yield [next(gen) if v is None else v for v in lst] #pylint: disable=stop-iteration-return


def index(lst, trunc):
    """
    Converts an n-ary index to a 1-dimensional index.
    """
    return sum([lst[i] * trunc**(len(lst)-i-1) for i in range(len(lst))])


def unIndex(i, n, trunc):
    """
    Converts a 1-dimensional index ``i`` with truncation ``trunc`` and
    number of modes ``n`` to a n-ary index.
    """
    return [i // trunc**(n - 1 - m) % trunc for m in range(n)]


def sliceExp(axes, ind, n):
    """
    Generates a slice expression for a list of pairs of axes (modes) and indices.
    """
    return [ind[i] if i in axes else slice(None, None, None) for i in range(n)]


def abssqr(z):
    r"""
    Given :math:`z` returns :math:`|z|^2`.
    """
    return z.real**2 + z.imag**2


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

    left_str = [indices[i] for i in range(0, 2*n, 2)]
    right_str = [indices[i] for i in range(1, 2*n, 2)]
    out_str = [indices[:2*n]]
    einstr = ''.join(left_str + [','] + right_str + ['->'] + out_str)
    return np.einsum(einstr, state, state.conj())


def diagonal(state, n):
    """
    Computes the diagonal of a density matrix.
    """

    left_str = [indices[i] + indices[i] for i in range(n)]
    out_str = [indices[:n]]
    einstr = ''.join(left_str + ['->'] + out_str)
    return np.einsum(einstr, state)


def trace(state, n):
    """
    Computes the trace of a density matrix.
    """

    left_str = [indices[i] + indices[i] for i in range(n)]
    einstr = ''.join(left_str)
    return np.einsum(einstr, state)


def partial_trace(state, n, modes):
    """
    Computes the partial trace of a state over the modes in `modes`.

    Expects state to be in mixed state form.
    """
    left_str = [indices[2*i] + indices[2*i] if i in modes else indices[2*i:2*i+2] for i in range(n)]
    out_str = ['' if i in modes else indices[2*i:2*i+2] for i in range(n)]
    einstr = ''.join(left_str + ['->'] + out_str)

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
            w = np.rollaxis(w, scale*n + i, scale*pos + i)

    return w


def project_reset(modes, x, state, pure, n, trunc):
    r"""
    Applies the operator :math:`\ket{00\dots 0}\bra{\mathbf{x}}` to the
    modes in `modes`.
    """
    inSlice = sliceExp(modes, dict(zip(modes, x)), n)
    outSlice = sliceExp(modes, dict(zip(modes, [0] * len(modes))), n)

    def intersperse(lst):
        # pylint: disable=missing-docstring
        return tuple([lst[i//2] for i in range(len(lst)*2)])

    if pure:
        ret = np.zeros([trunc for i in range(n)], dtype=def_type)
        ret[outSlice] = state[inSlice]
    else:
        ret = np.zeros([trunc for i in range(n*2)], dtype=def_type)
        ret[intersperse(outSlice)] = state[intersperse(inSlice)]

    return ret


# ============================================
#
# Matrix multiplication
#
# ============================================

def apply_gate_BLAS(mat, state, pure, modes, n, trunc):
    """
    Gate application based on custom indexing and matrix multiplication.
    Assumes the input matrix has shape (out1, in1, ...).

    This implementation uses indexing and BLAS. As per stack overflow,
    einsum doesn't actually use BLAS but rather a c implementation. In theory
    if reshaping is efficient this should be faster.
    """

    size = len(modes)
    dim = trunc**size
    stshape = [trunc for i in range(size)]

    # Apply the following matrix transposition:
    # |m1><m1| |m2><m2| ... |mn><mn| -> |m1>|m2>...|mn><m1|<m2|...<mn|
    transpose_list = [2*i for i in range(size)] + [2*i + 1 for i in range(size)]
    matview = np.transpose(mat, transpose_list).reshape((dim, dim))

    if pure:
        if n == 1:
            return np.dot(mat, state)

        # Transpose the state into the following form:
        # |psi> |mode[0]> |mode[1]> ... |mode[n]>
        transpose_list = [i for i in range(n) if not i in modes] + modes
        view = np.transpose(state, transpose_list)

        # Apply matrix to each substate
        ret = np.zeros([trunc for i in range(n)], dtype=def_type)
        for i in product(*([range(trunc) for j in range(n - size)])):
            ret[i] = np.dot(matview, view[i].ravel()).reshape(stshape)

        # "untranspose" the return matrix ret
        untranspose_list = [0] * len(transpose_list)
        for i in range(len(transpose_list)): # pylint: disable=consider-using-enumerate
            untranspose_list[transpose_list[i]] = i

        return np.transpose(ret, untranspose_list)
    else:
        if n == 1:
            return np.dot(mat, np.dot(state, dagger(mat)))

        # Transpose the state into the following form:
        # |psi><psi||mode[0]>|mode[1]>...|mode[n]><mode[0]|<mode[1]|...<mode[n]|
        transpose_list = [i for i in range(n*2) if not i//2 in modes]
        transpose_list = transpose_list + [2*i for i in modes] + [2*i + 1 for i in modes]
        view = np.transpose(state, transpose_list)

        # Apply matrix to each substate
        ret = np.zeros([trunc for i in range(n*2)], dtype=def_type)
        for i in product(*([range(trunc) for j in range((n - size)*2)])):
            ret[i] = np.dot(matview, np.dot(view[i].reshape((dim, dim)), dagger(matview))).reshape(stshape + stshape)

        # "untranspose" the return matrix ret
        untranspose_list = [0] * len(transpose_list)
        for i in range(len(transpose_list)): # pylint: disable=consider-using-enumerate
            untranspose_list[transpose_list[i]] = i

        return np.transpose(ret, untranspose_list)


def apply_gate_einsum(mat, state, pure, modes, n, trunc):
    """
    Gate application based on einsum.
    Assumes the input matrix has shape (out1, in1, ...)
    """
    # pylint: disable=unused-argument

    size = len(modes)

    if pure:
        if n == 1:
            return np.dot(mat, state)

        left_str = [indices[:size*2]]

        j = genOfRange(size)
        right_str = [indices[2*next(j) + 1] if i in modes else indices[size*2 + i] \
            for i in range(n)]

        j = genOfRange(size)
        out_str = [indices[2*next(j)] if i in modes else indices[size*2 + i] \
            for i in range(n)]

        einstring = ''.join(left_str + [','] + right_str + ['->'] + out_str)
        return np.einsum(einstring, mat, state)
    else:

        if n == 1:
            return np.dot(mat, np.dot(state, dagger(mat)))

        in_str = indices[:n*2]

        j = genOfRange(n*2)
        out_str = ''.join([indices[n*2 + next(j)] if i//2 in modes else indices[i] for i in range(n*2)])

        j = genOfRange(size*2)
        left_str = ''.join([out_str[modes[i//2]*2] if (i%2) == 0 else in_str[modes[i//2]*2] for i in range(size*2)])
        right_str = ''.join([out_str[modes[i//2]*2 + 1] if (i%2) == 0 else in_str[modes[i//2]*2 + 1] for i in range(size*2)])

        einstring = ''.join([left_str, ',', in_str, ',', right_str, '->', out_str])
        return np.einsum(einstring, mat, state, mat.conj())


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
        ret[i-1][i] = sqrt(i)
    return ret


@functools.lru_cache()
def displacement(alpha, trunc):
    r"""The displacement operator :math:`D(\alpha)`.

    Args:
            alpha (complex): the displacement
            trunc (int): the Fock cutoff
    """
    # pylint: disable=duplicate-code
    if alpha == 0:
        # return the identity
        ret = np.eye(trunc, dtype=def_type)
    else:
        # generate the broadcasted index arrays
        dim_array = np.arange(trunc)
        N = dim_array.reshape((-1, 1, 1))
        n = dim_array.reshape((1, -1, 1))
        i = dim_array.reshape((1, 1, -1))

        j = n-i
        k = N-i

        # the prefactors are only calculated if
        # i<=min(n,N). This is equivalent to k>=0, j>=0
        mask = np.logical_and(k >= 0, j >= 0)
        denom = fac(j)*fac(k)*fac(n-j)
        num = sqrt(fac(N)*fac(n))

        # the numpy.divide is to avoid division by 0 errors
        # (these are removed by the following mask anyway)
        prefactor = np.divide(num, denom, where=denom != 0)
        prefactor = np.where(mask, prefactor, 0)

        # sum over i
        ret = exp(-0.5 * abssqr(alpha)) * np.sum(alpha**k * (np.conj(-alpha)**j) * prefactor, axis=-1)

    return ret


@functools.lru_cache()
def squeezing(r, theta, trunc, save=False, directory=None):
    r"""The squeezing operator :math:`S(re^{i\theta})`.

    Args:
            r (float): the magnitude of the squeezing in the
                    x direction
            theta (float): the squeezing angle
            trunc (int): the Fock cutoff
    """
    # pylint: disable=duplicate-code
    if r == 0:
        # return the identity
        ret = np.eye(trunc, dtype=def_type)
    else:
        # broadcast the index arrays
        dim_array = np.arange(trunc)
        N = dim_array.reshape((-1, 1, 1))
        n = dim_array.reshape((1, -1, 1))
        k = dim_array.reshape((1, 1, -1))

        try:
            prefac = so.load_squeeze_factors(trunc, directory)
        except FileNotFoundError:
            prefac = so.generate_squeeze_factors(trunc)
            if save:
                so.save_bs_factors(prefac, directory)

        # we only perform the sum when n+N is divisible by 2
        # in which case we sum 0 <= k <= min(N,n)
        # mask = np.logical_and((n+N)%2 == 0, k <= np.minimum(N, n))
        mask = np.logical_and((n+N)%2 == 0, k <= np.minimum(N, n))

        # perform the summation over k
        scale = mask * np.power(sinh(r)/2, mask*(N+n-2*k)/2) / (cosh(r)**((N+n+1)/2))
        ph = exp(1j*theta*(N-n)/2)
        ret = np.sum(scale*ph*prefac, axis=-1)

    return ret


@functools.lru_cache()
def kerr(kappa, trunc):
    r"""
    The Kerr interaction :math:`K(\kappa)`.
    """
    n = np.arange(trunc)
    ret = np.diag(np.exp(1j*kappa*n**2))
    return ret


@functools.lru_cache()
def cross_kerr(kappa, trunc):
    r"""
    The cross-Kerr interaction :math:`CK(\kappa)`.
    """
    n1 = np.arange(trunc)[None, :]
    n2 = np.arange(trunc)[:, None]
    n1n2 = np.ravel(n1*n2)
    ret = np.diag(np.exp(1j*kappa*n1n2)).reshape([trunc]*4).swapaxes(1, 2)
    return ret


@functools.lru_cache()
def cubicPhase(gamma, hbar, trunc):
    r"""
    The cubic phase gate :math:`\exp{(i\frac{\gamma}{3\hbar}\hat{x}^3)}`.
    """
    a_ = a(trunc)
    x = (a_ + np.conj(a_).T) * np.sqrt(hbar/2)
    x3 = x @ x @ x
    ret = matrixExp(1j * gamma / (3*hbar) * x3)

    return ret


@functools.lru_cache()
def phase(theta, trunc):
    r"""
    The phase gate :math:`R(\theta)`
    """
    return np.array(np.diag([exp(1j*n*theta) for n in range(trunc)]), dtype=def_type)


@functools.lru_cache()
def beamsplitter(t, r, phi, trunc, save=False, directory=None):
    r"""
    The beamsplitter :math:`B(cos^{-1} t, phi)`.
    """
    # pylint: disable=bad-whitespace
    try:
        prefac = so.load_bs_factors(trunc, directory)
    except FileNotFoundError:
        prefac = so.generate_bs_factors(trunc)
        if save:
            so.save_bs_factors(prefac, directory)

    dim_array = np.arange(trunc)
    N = dim_array.reshape((-1, 1, 1, 1, 1))
    n = dim_array.reshape((1, -1, 1, 1, 1))
    M = dim_array.reshape((1, 1, -1, 1, 1))
    k = dim_array.reshape((1, 1, 1, 1, -1))

    tpwr = M-n+2*k
    rpwr = n+N-2*k

    T = np.power(t, tpwr) if t != 0 else np.where(tpwr != 0, 0, 1)
    R = np.power(r, rpwr) if r != 0 else np.where(rpwr != 0, 0, 1)

    BS = np.sum(exp(-1j*(pi+phi)*(n-N)) * T * R * prefac[:trunc,:trunc,:trunc,:trunc,:trunc], axis=-1)
    BS = BS.swapaxes(0, 1).swapaxes(2, 3)

    return BS


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

    state = np.zeros([trunc for i in range(n*2)], dtype=def_type)
    state.ravel()[0] = 1.0 + 0.0j
    return state


@functools.lru_cache()
def fockState(n, trunc):
    r"""
    The Fock state :math:`\ket{n}`.
    """
    return array([1.0 + 0.0j if i == n else 0.0 + 0.0j for i in range(trunc)])


@functools.lru_cache()
def coherentState(alpha, trunc):
    r"""
    The coherent state :math:`D(\alpha)\ket{0}`.
    """
    def entry(n):
        """coherent summation term"""
        return alpha**n / sqrt(fac(n))
    return exp(-abssqr(alpha) / 2) * array([entry(n) for n in range(trunc)])


@functools.lru_cache()
def squeezedState(r, theta, trunc):
    r"""
    The squeezed state :math:`S(re^{i\theta})`.
    """
    def entry(n):
        """squeezed summation term"""
        return (sqrt(fac(2*n))/(2**n*fac(n))) * (-exp(1j*theta)*tanh(r))**n

    vec = array([entry(n//2) if n % 2 == 0 else 0.0 + 0.0j for n in range(trunc)])
    return sqrt(1/cosh(r)) * vec


@functools.lru_cache()
def displacedSqueezed(alpha, r, phi, trunc):
    r"""
    The displaced squeezed state :math:`\ket{\alpha,\zeta} = D(\alpha)S(r\exp{(i\phi)})\ket{0}`.
    """
    if r == 0:
        return coherentState(alpha, trunc)

    if alpha == 0:
        return squeezedState(r, phi, trunc)

    ph = np.exp(1j * phi)
    ch = cosh(r)
    sh = sinh(r)
    th = tanh(r)

    gamma = alpha * ch + np.conj(alpha) * ph * sh
    hermite_arg = gamma / np.sqrt(ph * np.sinh(2 * r) + 1e-10)

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

    TToAdaggerA = np.array(np.diag([T**(i/2) for i in range(trunc)]), dtype=def_type)

    def aToN(n):
        """the nth matrix power of the annihilation operator matrix a"""
        return np.linalg.matrix_power(a(trunc), n)

    def E(n):
        """the loss channel amplitudes in the Fock basis"""
        return ((1-T)/T)**(n/2) * np.dot(aToN(n)/sqrt(fac(n)), TToAdaggerA)

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
    Hvals[1] = 2*x
    for i in range(2, trunc):
        Hvals[i] = 2*x*Hvals[i-1] - 2*(i-1)*Hvals[i-2]

    return q_tensor, Hvals

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

Decorators
----------

The :class:`~.strawberryfields.utils.operation` decorator allows functions containing quantum operations
acting on a qumode to be used as an operation itself within an engine context.

.. autosummary::
   operation

Code details
~~~~~~~~~~~~

"""
import collections
from inspect import signature

import numpy as np
import tensorflow as tf
from numpy.random import randn
from numpy.polynomial.hermite import hermval as H

import scipy as sp
from scipy.special import factorial as fac

from .engine import _convert
from .backends import load_backend
from .ops import Command, Gate, Channel, Ket

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

        |cat\rangle = \frac{1}{\sqrt{2(1+e^{-2|\alpha|^2}\cos(\phi))}}
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

    Returns:
        array: random :math:`N\times N` unitary distributed with the Haar measure
    """
    z = randnc(N, N)/np.sqrt(2.0)
    q, r = sp.linalg.qr(z)
    d = sp.diagonal(r)
    ph = d/np.abs(d)
    U = np.multiply(q, ph, q)
    return U


# ------------------------------------------------------------------------
# Decorators                                                            |
# ------------------------------------------------------------------------

class operation:
    """Groups a sequence of gates into a single operation to be used
    within an engine context.

    For example:

    .. code-block:: python

        @sf.operation(3)
        def custom_operation(v1, v2, q):
            CZgate(v1) | (q[0], q[1])
            Vgate(v2) | q[2]

    Here, the ``operation`` decorator must recieve an argument
    detailing the number of subsystems the resulting custom
    operation acts on.

    The function it acts on can contain arbitrary
    Python and blackbird code that may normally be placed within an
    engine context. Note that it must always accept the qumode register
    ``q`` it acts on as the *last* argument of the function.

    Once defined, it can be used like any other quantum operation:

    .. code-block:: python

        eng, q = sf.Engine(3)
        with eng:
            custom_operation(0.5719, 2.0603) | (q[0], q[1], q[3])

    Note that here, we do not pass the qumode register ``q`` directly
    to the function - instead, it is defined on the right hand side
    of the ``|`` operation, like all other blackbird code.

    Args:
        ns (int): number of registers required by the operation
    """

    def __init__(self, ns):
        self.ns = ns
        self.func = None
        self.args = None

    def __or__(self, reg):
        """Apply the operation to a part of a quantum register.

        Redirects the execution flow to the wrapped function.

        Args:
            reg (RegRef, Sequence[RegRef]): subsystem(s) the operation is acting on

        Returns:
            list[RegRef]: subsystem list as RegRefs
        """
        if (not reg) or (not self.ns):
            raise ValueError("Wrong number of subsystems")

        reg_len = 1
        if isinstance(reg, collections.Sized):
            reg_len = len(reg)

        if reg_len != self.ns:
            raise ValueError("Wrong number of subsystems")

        return self._call_function(reg)

    def _call_function(self, reg):
        """Executes the wrapped function and passes the quantum registers.

        Args:
            reg (RegRef, Sequence[RegRef]): subsystem(s) the operation is acting on

        Returns:
            list[RegRef]: subsystem list as RegRefs
        """
        func_sig = signature(self.func)
        num_params = len(func_sig.parameters)

        if num_params == 0:
            raise ValueError("Operation must receive the qumode register as an argument.")

        if num_params != len(self.args) + 1:
            raise ValueError("Mismatch in the number of arguments")

        # pass parameters and subsystems to the function
        if num_params == 1:
            self.func(reg)
        else:
            self.func(*self.args, reg)

        return reg

    def __call__(self, func):
        self.func = func

        def f_proxy(*args):
            """
            Proxy for function execution. Function will actually execute in __or__
            """
            self.args = args
            return self

        return f_proxy



#=================================================
# extract_xxx methods
#=================================================


def is_unitary(engine):
    "Returns true if every command in the queue is of type Gate"
    return all(isinstance(cmd.op, Gate) for cmd in engine.cmd_queue)

def is_channel(engine):
    "Returns true if every command in the queue is either of type Gate or type Channel"
    return all(isinstance(cmd.op, (Channel, Gate)) for cmd in engine.cmd_queue)

def vectorize_dm(tensor):
    """Given a tensor with 4N indices each of dimension D each,
    it returns the vectorized tensor with 4 indices of dimension D^N each.
    
    Args:
        tensor (array): a tensor with 4N indices of dimension D each
    Returns:
        array: a tensor with 4 indices of dimension D^N each
    Raises:
        ValueError: if the input tensor's dimensions are not all equal or if the number
            of its indices is not a multiple of 4.

    Example:
    For N=2
         ____
    0 --|    |-- 1
    2 --|    |-- 3
    4 --|    |-- 5
    6 --|____|-- 7

    goes to:
             ____
    (0,2) --|    |-- (1,3)
    (4,6) --|____|-- (5,7)

    """
    dims = tensor.ndim
    if dims % 4 != 0:
        raise ValueError(f'tensor must have a number of indices that is a multiple of 4, but it has {dims} indices')
    shape = tensor.shape
    if len(set(shape)) != 1:
        raise ValueError(f'tensor indices must have all the same dimension, but tensor has shape {shape}')

    transposed = np.einsum(tensor, [int(n) for n in np.arange(dims).reshape((2, dims//2)).T.reshape([-1])])
    vectorized = np.reshape(transposed, [shape[0]**(dims//4)]*4)
    transposed_back = np.einsum('abcd -> acbd', vectorized)

    return transposed_back

def unvectorize_dm(tensor, num_subsystems):
    """Given a tensor with 4 indices, each of dimension D^N,
    return the unvectorized tensor with 4N indices of dimension D each.
    (inverse of the procedure given by vectorize_dm)
    """
    dims = tensor.ndim
    if dims != 4:
        raise ValueError(f'tensor must have 4 indices, but it has {dims} indices')
    shape = tensor.shape
    if len(set(shape)) != 1:
        raise ValueError(f'tensor indices must have all the same dimension, but tensor has shape {shape}')

    transposed = np.einsum('abcd -> acbd', tensor)
    unvectorized = np.reshape(transposed, [int(shape[0]**(1/num_subsystems))]*(4*num_subsystems))
    transposed_back = np.einsum(unvectorized, [int(n) for n in np.arange(4*num_subsystems).reshape((2*num_subsystems, 2)).T.reshape([-1])])

    return transposed_back

def _interleaved_identities(num_subsystems: int, cutoff_dim: int):
    """Returns the tensor product of `num_subsystems` copies of the identity matrix,
    with the indices interleaved.
    Example for num_subsystems = 3: make the tensor product of three identity matrices I_ijklmn
    where ij, kl and mn are the three sets of indices, and return I_ikmjln.
    """
    I = np.identity(cutoff_dim)
    for _ in range(1, num_subsystems):
        I = np.tensordot(I, np.identity(cutoff_dim), axes=0)
    return np.einsum(I, [int(n) for n in np.arange(2*num_subsystems).reshape((2, num_subsystems)).T.reshape([-1])])

def _engine_with_CJ_cmd_queue(engine, cutoff_dim: int):
    """Doubles the number of modes of an engine objects and prepends to its command queue
    the operation that creates the interleaved identities ket. CJ is from Choi-Jamiolkowski, as 
    under the hood we are expliting the CJ isomorphism.
    """
    engine._add_subsystems(engine.init_num_subsystems)
    I = _interleaved_identities(num_subsystems=engine.init_num_subsystems, cutoff_dim=cutoff_dim)
    engine.cmd_queue = [Command(Ket(I), list(engine.reg_refs.values()))] + engine.cmd_queue
    return engine

def extract_unitary(engine, cutoff_dim: int, vectorize_modes: bool = False, backend: str = 'fock'):
    """Returns the array representation of a unitary circuit as an ndarray ('fock' backend)
    or as a TensorFlow Tensor ('tf' backend).

    The circuit must only include operations of the Gate class.
    If `vectorize_modes` = True, it returns a matrix.
    If `vectorize_modes` = False, it returns an operator with 2N indices,
    where N is the number of modes that the engine is created with. Adjacent
    indices correspond to output-input pairs of the same mode.

    Args:
        engine (Engine): the engine containing the circuit
        cutoff_dim (int): dimension of each index.
        vectorize_modes (bool): if true, reshape input and output modes in order to return a matrix.
        backend (str): the backend to build the unitary. 'fock' (default) and 'tf' are supported.
    Returns:
        array: the numerical array of the unitary circuit
    Raises:
        TypeError: if the operations used to construct the circuit are not all unitary.
    """
    if not is_unitary(engine):
        raise TypeError(f"The circuit definition contains elements that are not of type Gate")
    if backend not in ('fock', 'tf'):
        raise ValueError("Only 'fock' and 'tf' backends are supported")

    from copy import deepcopy
    _engine = _engine_with_CJ_cmd_queue(deepcopy(engine), cutoff_dim=cutoff_dim)

    _backend = load_backend(backend) # (!) _backend is the object, backend is just its name
    _backend.begin_circuit(num_subsystems=_engine.num_subsystems, cutoff_dim=cutoff_dim, hbar=_engine.hbar, pure=True)
    result = _engine.run(_backend, cutoff_dim=cutoff_dim).ket()

    if vectorize_modes:
        if backend == 'fock':
            return np.reshape(result, [cutoff_dim**_engine.init_num_subsystems, cutoff_dim**_engine.init_num_subsystems])
        else:
            return tf.reshape(result, [cutoff_dim**_engine.init_num_subsystems, cutoff_dim**_engine.init_num_subsystems])
    else:
        if backend == 'fock':
            return np.transpose(result, [int(n) for n in np.arange(2*_engine.init_num_subsystems).reshape((2, _engine.init_num_subsystems)).T.reshape([-1])])
        else:
            return tf.transpose(result, [int(n) for n in np.arange(2*_engine.init_num_subsystems).reshape((2, _engine.init_num_subsystems)).T.reshape([-1])])

def extract_channel(engine, cutoff_dim: int, representation: str = 'choi', vectorize_modes: bool = False):
    """Returns a numerical array representation of a channel.
       The choices are among the Choi state representation, the Liouville representation and
       the Kraus representation.

    # Tensor shapes
    If `vectorize_modes=True`, `representation='choi'` and `representation='liouville'` return an array
    with 4 indices, while `representation='kraus'` returns an array of Kraus operators in matrix form.
    If `vectorize_modes=False`, `representation='choi'` and `representation='liouville'` return an array
    with 4N indices, while `representation='kraus'` returns an array of Kraus operators with 2N indices each,
    where N is the number of modes that the engine is created with.
    The Kraus representation automatically returns only the non-zero Kraus operators. One can reduce the number
    of operators by discarding Kraus operators with small norm (thus approximating the channel).
    
    # Choi representation
    The indices of the non-vectorized Choi operator match exactly those of the state, so that the action
    of the channel can be computed as (e.g. for one mode or for `vectorize_modes = True`):
    `rho_out = np.einsum('ab,abcd', rho_in, choi)`
    or for two modes:
    `rho_out = np.einsum('abcd,abcdefgh', rho_in, choi)`
    Combining consecutive channels (in the order 1,2,3,...) is also straightforward with the Choi operator:
    `choi_combined = np.einsum('abcd,cdef,efgh', choi_1, choi_2, choi_3)`

    # Liouville operator
    The Liouville operator is a partial transpose of the Choi operator, such that the first half of
    consecutive index pairs are the output-input right modes (i.e. acting on the "bra" part of the state)
    and the second half are the output-input left modes (i.e. acting on the "ket" part of the state).
    Therefore, the action of the Liouville operator (e.g. for one mode or for `vectorize_modes = True`) is
    `rho_out = np.einsum('abcd,bd->ca', liouville, rho_in)`
    notice that the state contracts with the second index of each pair and that we output the ket
    on the left (`c`) and the bra on the right (`a`).
    For two modes we have:
    `rho_out = np.einsum('abcdefgh,fbhd->eagc', liouville, rho_in)`
    The Liouville representation has the property that if the channel is unitary, the operator is separable.
    Whereas even if the channel were the identity, the Choi operator would correspond to a maximally entangled state.

    The choi and liouville operators in _matrix_ form (i.e. with two indices) can be found as follows, where
    D is the dimension of each vectorized index (i.e. for N modes D=cutoff_dim**N):
    `choi_matrix = liouville.reshape(D**2, D**2).T`
    `liouville_matrix = choi.reshape(D**2, D**2).T`

    # Kraus representation
    Adjacent indices of each Kraus operator correspond to output-input pairs of the same mode, so the action
    of the channel can be written as (here for one mode or for vectorize_modes = True):
    `rho_out = np.einsum('abc,cd,aed->be', kraus, rho_in, np.conj(kraus))`
    Notice the transpose on the third index string (`aed` rather than `ade`), as the last operator should be the
    conjugate transpose of the first, and we cannot just do `np.conj(kraus).T` because kraus has 3 indices and we 
    just need to transpose the last two.

    Args:
        engine (Engine): the engine containing the circuit
        cutoff_dim (int): dimension of each index
        representation (str): choice between 'choi', 'liouville' or 'kraus'
        vectorize_modes (bool): if True, reshapes the result into rank-4 tensor,
            otherwise it returns a rank-4N tensor, where N is the number of modes.
    Returns:
        The numerical array of the channel, according to the specified representation
        and vectorization options.
    Raises:
        TypeError: if the gates used to construct the circuit are not all unitary or channels.

    """
    if is_unitary(engine):
        #raise Warning(f"This circuit is unitary and you could use extract_unitary for a more compact representation")
        pass

    if not is_channel(engine):
        raise TypeError("The circuit definition contains elements that are neither of type Gate nor of type Channel")

    from copy import deepcopy
    _engine = _engine_with_CJ_cmd_queue(deepcopy(engine), cutoff_dim=cutoff_dim)
    N = _engine.init_num_subsystems

    backend = load_backend('fock')
    backend.begin_circuit(num_subsystems=_engine.num_subsystems, cutoff_dim=cutoff_dim, hbar=_engine.hbar, pure=True)
    choi = _engine.run(backend, cutoff_dim=cutoff_dim).dm()
    choi = np.einsum('abcd->cdab',vectorize_dm(choi))

    if representation.lower() == 'choi':
        result = choi
        if vectorize_modes == False:
            result = unvectorize_dm(result, N)

    elif representation.lower() == 'liouville':
        result = np.einsum('abcd -> dbca', choi)
        if vectorize_modes == False:
            result = unvectorize_dm(result, N)
            pass

    elif representation.lower() == 'kraus':
        eigvals, eigvecs = np.linalg.eig(np.einsum('abcd -> cadb', choi).reshape([cutoff_dim**(2*N), cutoff_dim**(2*N)]))
        eigvecs = eigvecs[:, ~np.isclose(abs(eigvals), 0)]
        eigvals = eigvals[~np.isclose(abs(eigvals), 0)]
        result = np.einsum('abc->cab', np.einsum('b,ab->ab', np.sqrt(eigvals), eigvecs).reshape([cutoff_dim**N, cutoff_dim**N, -1]))
        if vectorize_modes == False:
            result = np.einsum(np.reshape(result, [-1]+[cutoff_dim]*(2*N)), range(1+2*N), [0]+[2*n+1 for n in range(N)]+[2*n+2 for n in range(N)])
    else:
        raise ValueError(f'representation {representation} not supported')

    return result

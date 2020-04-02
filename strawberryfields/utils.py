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
This module defines and implements several utility functions and language extensions that complement
StrawberryFields. These include:


* **NumPy state functions**

  These functions allow the calculation of various quantum states in either the Fock
  basis (a one-dimensional array indexed by Fock state) or the Gaussian basis (returning
  a vector of means and covariance matrix). These state calculations are NOT done in the
  simulators, but rather in NumPy.

  These are useful for generating states for use in calculating the fidelity of simulations.


* **Random functions**

  These functions generate random numbers and matrices corresponding to various
  quantum states and operations.

* **Decorators**

  The :class:`~.strawberryfields.utils.operation` decorator allows functions
  containing quantum operations acting on a qumode to be used as an
  operation itself within a :class:`.Program` context.

* **Program functions**

  These functions act on :class:`.Program` instances, returning
  or extracting information from the quantum circuit.
"""
import collections
import copy
from inspect import signature

try:
    import tensorflow as tf
except ImportError:
    tf_available = False

import numpy as np
from numpy.polynomial.hermite import hermval
import scipy as sp
from scipy.special import factorial as fac

from .engine import LocalEngine
from .program_utils import Command
from .ops import Gate, Channel, Ket


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
    r""" Returns the vacuum state

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


def coherent_state(a, basis="fock", fock_dim=5, hbar=2.0):
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
    if basis == "fock":
        state = np.array(
            [np.exp(-0.5 * np.abs(a) ** 2) * a ** n / np.sqrt(fac(n)) for n in range(fock_dim)]
        )

    elif basis == "gaussian":
        means = np.array([a.real, a.imag]) * np.sqrt(2 * hbar)
        cov = np.identity(2) * hbar / 2
        state = [means, cov]

    return state


def squeezed_state(r, p, basis="fock", fock_dim=5, hbar=2.0):
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


def displaced_squeezed_state(a, r, phi, basis="fock", fock_dim=5, hbar=2.0):
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
    if basis == "fock":

        if r != 0:
            phase_factor = np.exp(1j * phi)
            ch = np.cosh(r)
            sh = np.sinh(r)
            th = np.tanh(r)

            gamma = a * ch + np.conj(a) * phase_factor * sh
            N = np.exp(-0.5 * np.abs(a) ** 2 - 0.5 * np.conj(a) ** 2 * phase_factor * th)

            coeff = np.diag(
                [
                    (0.5 * phase_factor * th) ** (n / 2) / np.sqrt(fac(n) * ch)
                    for n in range(fock_dim)
                ]
            )

            vec = [hermval(gamma / np.sqrt(phase_factor * np.sinh(2 * r)), row) for row in coeff]

            state = N * np.array(vec)

        else:
            state = coherent_state(a, basis="fock", fock_dim=fock_dim)  # pragma: no cover

    elif basis == "gaussian":
        means = np.array([a.real, a.imag]) * np.sqrt(2 * hbar)
        state = [means, squeezed_cov(r, phi, hbar)]

    return state


# ------------------------------------------------------------------------
# State functions - Fock basis only                              |
# ------------------------------------------------------------------------


def fock_state(n, fock_dim=5):
    r""" Returns the Fock state

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
    r""" Returns the cat state

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


# ------------------------------------------------------------------------
# Random numbers and matrices                                           |
# ------------------------------------------------------------------------


def randnc(*arg):
    """Normally distributed array of random complex numbers."""
    return np.random.randn(*arg) + 1j * np.random.randn(*arg)


def random_covariance(N, hbar=2, pure=False, block_diag=False):
    r"""Random covariance matrix.

    Args:
        N (int): number of modes
        hbar (float): the value of :math:`\hbar` to use in the definition
            of the quadrature operators :math:`\x` and :math:`\p`
        pure (bool): If True, a random covariance matrix corresponding
            to a pure state is returned.
        block_diag (bool): If True, uses passive Gaussian transformations that are orthogonal
            instead of unitary. This implies that the positions :math:`q` do not mix with
            the momenta :math:`p` and thus the covariance matrix is block diagonal.
    Returns:
        array: random :math:`2N\times 2N` covariance matrix
    """
    S = random_symplectic(N, block_diag=block_diag)

    if pure:
        return (hbar / 2) * S @ S.T

    nbar = 2 * np.abs(np.random.random(N)) + 1
    Vth = (hbar / 2) * np.diag(np.concatenate([nbar, nbar]))

    return S @ Vth @ S.T


def random_symplectic(N, passive=False, block_diag=False, scale=1.0):
    r"""Random symplectic matrix representing a Gaussian transformation.

    The squeezing parameters :math:`r` for active transformations are randomly
    sampled from the standard normal distribution, while passive transformations
    are randomly sampled from the Haar measure. Note that for the Symplectic
    group there is no notion of Haar measure since this is group is not compact.

    Args:
        N (int): number of modes
        passive (bool): If True, returns a passive Gaussian transformation (i.e.,
            one that preserves photon number). If False (default), returns an active
            transformation.
        block_diag (bool): If True, uses passive Gaussian transformations that are orthogonal
            instead of unitary. This implies that the positions :math:`q` do not mix with
            the momenta :math:`p` and thus the symplectic operator is block diagonal
        scale (float): Sets the scale of the random values used as squeezing parameters.
            They will range from 0 to :math:`\sqrt{2}\texttt{scale}`

    Returns:
        array: random :math:`2N\times 2N` symplectic matrix
    """
    U = random_interferometer(N, real=block_diag)
    O = np.vstack([np.hstack([U.real, -U.imag]), np.hstack([U.imag, U.real])])

    if passive:
        return O

    U = random_interferometer(N, real=block_diag)
    P = np.vstack([np.hstack([U.real, -U.imag]), np.hstack([U.imag, U.real])])

    r = scale * np.abs(randnc(N))
    Sq = np.diag(np.concatenate([np.exp(-r), np.exp(r)]))

    return O @ Sq @ P


def random_interferometer(N, real=False):
    r"""Random unitary matrix representing an interferometer.

    For more details, see :cite:`mezzadri2006`.

    Args:
        N (int): number of modes
        real (bool): return a random real orthogonal matrix

    Returns:
        array: random :math:`N\times N` unitary distributed with the Haar measure
    """
    if real:
        z = np.random.randn(N, N)
    else:
        z = randnc(N, N) / np.sqrt(2.0)
    q, r = sp.linalg.qr(z)
    d = sp.diagonal(r)
    ph = d / np.abs(d)
    U = np.multiply(q, ph, q)
    return U


# ------------------------------------------------------------------------
# Decorators                                                            |
# ------------------------------------------------------------------------


class operation:
    """Groups a sequence of gates into a single operation to be used
    within a Program context.

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
    Python and Blackbird code that may normally be placed within a
    Program context. Note that it must always accept the register
    ``q`` it acts on as the *last* argument of the function.

    Once defined, it can be used like any other quantum operation:

    .. code-block:: python

        prog = sf.Program(3)
        with prog.context as q:
            custom_operation(0.5719, 2.0603) | (q[0], q[1], q[3])

    Note that here, we do not pass the register ``q`` directly
    to the function - instead, it is defined on the right hand side
    of the ``|`` operation, like all other Blackbird code.

    Args:
        ns (int): number of subsystems required by the operation
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
        if isinstance(reg, collections.abc.Sized):
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


# =================================================
# Program functions
# =================================================


def is_unitary(prog):
    """True iff all the operations in the program are unitary.

    Args:
        prog (Program): quantum program
    Returns:
        bool: True iff all operations in the program are of type :class:`strawberryfields.ops.Gate`
    """
    return all(isinstance(cmd.op, Gate) for cmd in prog.circuit)


def is_channel(prog):
    """True iff all the operations in the program can be represented as quantum channels.

    Args:
        prog (Program): quantum program
    Returns:
        bool: True if all operations in the program are of types :class:`strawberryfields.ops.Gate` and :class:`strawberryfields.ops.Channel`
    """
    # FIXME isn't a preparation also a quantum channel?
    return all(isinstance(cmd.op, (Channel, Gate)) for cmd in prog.circuit)


def _vectorize(tensor):
    """Given a tensor with 4N indices of dimension :math:`D` each, it returns the vectorized
    tensor with 4 indices of dimension :math:`D^N` each. This is the inverse of the procedure
    given by :func:`_unvectorize`.
    Caution: this private method is intended to be used only for Choi and Liouville operators.

    For example, :math:`N=2`,
    ::
        0 --|‾‾‾‾|-- 1
        2 --|    |-- 3
        4 --|    |-- 5
        6 --|____|-- 7

    goes to
    ::
        (0,2) --|‾‾‾‾|-- (1,3)
        (4,6) --|____|-- (5,7)

    Args:
        tensor (array): a tensor with :math:`4N` indices of dimension :math:`D` each

    Returns:
        array: a tensor with 4 indices of dimension :math:`D^N` each

    Raises:
        ValueError: if the input tensor's dimensions are not all equal or if the number
            of its indices is not a multiple of 4
    """
    dims = tensor.ndim

    if dims % 4 != 0:
        raise ValueError(
            "Tensor must have a number of indices that is a multiple of 4, but it has {dims} indices".format(
                dims=dims
            )
        )

    shape = tensor.shape

    if len(set(shape)) != 1:
        raise ValueError(
            "Tensor indices must have all the same dimension, but tensor has shape {shape}".format(
                shape=shape
            )
        )

    transposed = np.einsum(
        tensor, [int(n) for n in np.arange(dims).reshape((2, dims // 2)).T.reshape([-1])]
    )
    vectorized = np.reshape(transposed, [shape[0] ** (dims // 4)] * 4)
    transposed_back = np.einsum("abcd -> acbd", vectorized)

    return transposed_back


def _unvectorize(tensor, num_subsystems):
    """Given a tensor with 4 indices, each of dimension :math:`D^N`, return the unvectorized
    tensor with 4N indices of dimension D each. This is the inverse of the procedure
    given by :func:`_vectorize`.
    Caution: this private method is intended to be used only for Choi and Liouville operators.

    Args:
        tensor (array): a tensor with :math:`4` indices of dimension :math:`D^N`

    Returns:
        array: a tensor with :math:`4N` indices of dimension :math:`D` each

    Raises:
        ValueError: if the input tensor's dimensions are not all equal or if the number
            of its indices is not 4
    """
    dims = tensor.ndim

    if dims != 4:
        raise ValueError("tensor must have 4 indices, but it has {dims} indices".format(dims=dims))

    shape = tensor.shape

    if len(set(shape)) != 1:
        raise ValueError(
            "tensor indices must have all the same dimension, but tensor has shape {shape}".format(
                shape=shape
            )
        )

    transposed = np.einsum("abcd -> acbd", tensor)
    unvectorized = np.reshape(
        transposed, [int(shape[0] ** (1 / num_subsystems))] * (4 * num_subsystems)
    )
    transposed_back = np.einsum(
        unvectorized,
        [
            int(n)
            for n in np.arange(4 * num_subsystems).reshape((2 * num_subsystems, 2)).T.reshape([-1])
        ],
    )

    return transposed_back


def _interleaved_identities(n: int, cutoff_dim: int):
    r"""Maximally entangled state of `n` modes.

    Returns the tensor :math:`\sum_{abc\ldots} \ket{abc\ldots}\bra{abc\ldots}`
    representing an unnormalized, maximally entangled state of `n` subsystems.

    Args:
        n (int): number of subsystems
        cutoff_dim (int): Fock basis truncation dimension

    Returns:
        array: unnormalized maximally entangled state, shape == (cutoff_dim,) * (2*n)
    """
    I = np.identity(cutoff_dim)
    temp = I
    for _ in range(1, n):
        temp = np.tensordot(temp, I, axes=0)

    # use einsum to permute the indices such that |a><a|*|b><b|*|c><c|*... becomes |abc...><abc...|
    sublist = [int(n) for n in np.arange(2 * n).reshape((2, n)).T.reshape([-1])]
    return np.einsum(temp, sublist)


def _program_in_CJ_rep(prog, cutoff_dim: int):
    """Convert a Program object to Choi-Jamiolkowski representation.

    Doubles the number of modes of a Program object and prepends to its circuit
    the preparation of the maximally entangled ket state.

    The core idea is that when we apply any quantum channel (e.g. a unitary gate)
    to the density matrix of the maximally entangled state, we obtain the Choi matrix
    of the channel as the result.

    If the channel is unitary, applying it on the maximally entangled ket yields
    the corresponding unitary matrix, reshaped.

    Args:
        prog (Program): quantum program
        cutoff_dim (int): the Fock basis truncation

    Returns:
        Program: modified program
    """
    prog = copy.deepcopy(prog)
    prog.locked = False  # unlock the copy so we can modify it
    N = prog.init_num_subsystems
    prog._add_subsystems(N)  # pylint: disable=protected-access
    prog.init_num_subsystems = 2 * N
    I = _interleaved_identities(N, cutoff_dim)
    # prepend the circuit with the I ket preparation
    prog.circuit.insert(0, Command(Ket(I), list(prog.reg_refs.values())))
    return prog


def extract_unitary(prog, cutoff_dim: int, vectorize_modes: bool = False, backend: str = "fock"):
    r"""Numerical array representation of a unitary quantum circuit.

    Note that the circuit must only include operations of the :class:`strawberryfields.ops.Gate` class.

    * If ``vectorize_modes=True``, it returns a matrix.
    * If ``vectorize_modes=False``, it returns an operator with :math:`2N` indices,
      where N is the number of modes that the Program is created with. Adjacent
      indices correspond to output-input pairs of the same mode.


    **Example:**

    This shows the Hong-Ou-Mandel effect by extracting the unitary of a 50/50 beamsplitter, and then
    computing the output given by one photon at each input (notice the order of the indices: :math:`[out_1, in_1, out_2, in_2,\dots]`).
    The result tells us that the two photons always emerge together from a random output port and never one per port.

    >>> prog = sf.Program(num_subsystems=2)
    >>> with prog.context as q:
    >>>     BSgate(np.pi/4) | q
    >>> U = extract_unitary(prog, cutoff_dim=3)
    >>> print(abs(U[:,1,:,1])**2)
    [[0.  0.  0.5]
     [0.  0.  0. ]
     [0.5 0.  0. ]])

    Args:
        prog (Program): quantum program
        cutoff_dim (int): dimension of each index
        vectorize_modes (bool): If True, reshape input and output modes in order to return a matrix.
        backend (str): the backend to build the unitary; ``'fock'`` (default) and ``'tf'`` are supported

    Returns:
        array, tf.Tensor: numerical array of the unitary circuit
            as a NumPy ndarray (``'fock'`` backend) or as a TensorFlow Tensor (``'tf'`` backend)

    Raises:
        TypeError: if the operations used to construct the circuit are not all unitary
    """

    if not is_unitary(prog):
        raise TypeError("The circuit definition contains elements that are not of type Gate")

    if backend not in ("fock", "tf"):
        raise ValueError("Only 'fock' and 'tf' backends are supported")

    N = prog.init_num_subsystems
    # extract the unitary matrix by running a modified version of the Program
    p = _program_in_CJ_rep(prog, cutoff_dim)
    eng = LocalEngine(backend, backend_options={"cutoff_dim": cutoff_dim, "pure": True})
    result = eng.run(p).state.ket()

    if vectorize_modes:
        if backend == "fock":
            reshape = np.reshape
        else:
            reshape = tf.reshape
        return reshape(result, [cutoff_dim ** N, cutoff_dim ** N])

    # here we rearrange the indices to go back to the order [in1, out1, in2, out2, etc...]
    if backend == "fock":
        tp = np.transpose
    else:
        tp = tf.transpose
    return tp(result, [int(n) for n in np.arange(2 * N).reshape((2, N)).T.reshape([-1])])


def extract_channel(
    prog, cutoff_dim: int, representation: str = "choi", vectorize_modes: bool = False
):
    r"""Numerical array representation of the channel corresponding to a quantum circuit.

    The representation choices include the Choi state representation, the Liouville representation, and
    the Kraus representation.

    .. note:: Channel extraction can currently only be performed using the ``'fock'`` backend.

    **Tensor shapes**

    * If ``vectorize_modes=True``:

      - ``representation='choi'`` and ``representation='liouville'`` return an array
        with 4 indices
      - ``representation='kraus'`` returns an array of Kraus operators in matrix form


    * If ``vectorize_modes=False``:

      - ``representation='choi'`` and ``representation='liouville'`` return an array
        with :math:`4N` indices
      - ``representation='kraus'`` returns an array of Kraus operators with :math:`2N` indices each,
        where :math:`N` is the number of modes that the Program is created with

    Note that the Kraus representation automatically returns only the non-zero Kraus operators.
    One can reduce the number of operators by discarding Kraus operators with small norm (thus approximating the channel).

    **Choi representation**

    Mathematically, the Choi representation of a channel is a bipartite state :math:`\Lambda_{AB}`
    which contains a complete description of the channel. The way we use it to compute the action
    of the channel :math:`\mathcal{C}` on an input state :math:`\mathcal{\rho}` is as follows:

    .. math::

            \mathcal{C}(\rho) = \mathrm{Tr}[(\rho_A^T\otimes\mathbb{1}_B)\Lambda_{AB}]

    The indices of the non-vectorized Choi operator match exactly those of the state, so that the action
    of the channel can be computed as (e.g., for one mode or for ``vectorize_modes=True``):

    >>> rho_out = np.einsum('ab,abcd', rho_in, choi)

    Notice that this respects the transpose operation.

    For two modes:

    >>> rho_out = np.einsum('abcd,abcdefgh', rho_in, choi)

    Combining consecutive channels (in the order :math:`1,2,3,\dots`) is also straightforward with the Choi operator:

    >>> choi_combined = np.einsum('abcd,cdef,efgh', choi_1, choi_2, choi_3)

    **Liouville operator**

    The Liouville operator is a partial transpose of the Choi operator, such that the first half of
    consecutive index pairs are the output-input right modes (i.e., acting on the "bra" part of the state)
    and the second half are the output-input left modes (i.e., acting on the "ket" part of the state).

    Therefore, the action of the Liouville operator (e.g., for one mode or for ``vectorize_modes=True``) is

    .. math::

            \mathcal{C}(\rho) = \mathrm{unvec}[\mathcal{L}\mathrm{vec}(\rho)]

    where vec() and unvec() are the operations that stack the columns of a matrix to form
    a vector and vice versa.
    In code:

    >>> rho_out = np.einsum('abcd,bd->ca', liouville, rho_in)

    Notice that the state contracts with the second index of each pair and that we output the ket
    on the left (``c``) and the bra on the right (``a``).

    For two modes we have:

    >>> rho_out = np.einsum('abcdefgh,fbhd->eagc', liouville, rho_in)

    The Liouville representation has the property that if the channel is unitary, the operator is separable.
    On the other hand, even if the channel were the identity, the Choi operator would correspond to a maximally entangled state.

    The choi and liouville operators in matrix form (i.e., with two indices) can be found as follows, where
    ``D`` is the dimension of each vectorized index (i.e., for :math:`N` modes, ``D=cutoff_dim**N``):

    >>> choi_matrix = liouville.reshape(D**2, D**2).T
    >>> liouville_matrix = choi.reshape(D**2, D**2).T

    **Kraus representation**

    The Kraus representation is perhaps the most well known:

    .. math::

            \mathcal{C}(\rho) = \sum_k A_k\rho A_k^\dagger

    So to define a channel in the Kraus representation one needs to supply a list of Kraus operators :math:`\{A_k\}`.
    In fact, the result of ``extract_channel`` in the Kraus representation is a rank-3 tensor, where the first
    index is the one indexing the list of operators.

    Adjacent indices of each Kraus operator correspond to output-input pairs of the same mode, so the action
    of the channel can be written as (here for one mode or for ``vectorize_modes=True``):

    >>> rho_out = np.einsum('abc,cd,aed->be', kraus, rho_in, np.conj(kraus))

    Notice the transpose on the third index string (``aed`` rather than ``ade``), as the last operator should be the
    conjugate transpose of the first, and we cannot just do ``np.conj(kraus).T`` because ``kraus`` has 3 indices and we
    just need to transpose the last two.


    Example:
        Here we show that the Choi operator of the identity channel is proportional to
        a maximally entangled Bell :math:`\ket{\phi^+}` state:

    >>> prog = sf.Program(num_subsystems=1)
    >>> C = extract_channel(prog, cutoff_dim=2, representation='choi')
    >>> print(abs(C).reshape((4,4)))
    [[1. 0. 0. 1.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [1. 0. 0. 1.]]

    Args:
        prog (Program): program containing the circuit
        cutoff_dim (int): dimension of each index
        representation (str): choice between ``'choi'``, ``'liouville'`` or ``'kraus'``
        vectorize_modes (bool): if True, reshapes the result into rank-4 tensor,
            otherwise it returns a rank-4N tensor, where N is the number of modes

    Returns:
        array: channel, according to the specified options

    Raises:
        TypeError: if the gates used to construct the circuit are not all unitary or channels
    """
    if not is_channel(prog):
        raise TypeError(
            "The circuit definition contains elements that are neither of type Gate nor of type Channel"
        )

    N = prog.init_num_subsystems
    p = _program_in_CJ_rep(prog, cutoff_dim)

    eng = LocalEngine("fock", backend_options={"cutoff_dim": cutoff_dim, "pure": True})
    choi = eng.run(p).state.dm()
    choi = np.einsum("abcd->cdab", _vectorize(choi))

    if representation.lower() == "choi":
        result = choi
        if not vectorize_modes:
            result = _unvectorize(result, N)

    elif representation.lower() == "liouville":
        result = np.einsum("abcd -> dbca", choi)
        if not vectorize_modes:
            result = _unvectorize(result, N)

    elif representation.lower() == "kraus":
        # The liouville operator is the sum of a bipartite product of kraus matrices, so if we vectorize them we obtain
        # a matrix whose eigenvectors are proportional to the vectorized kraus operators
        vectorized_liouville = np.einsum("abcd -> cadb", choi).reshape(
            [cutoff_dim ** (2 * N), cutoff_dim ** (2 * N)]
        )
        eigvals, eigvecs = np.linalg.eig(vectorized_liouville)

        # We keep only those eigenvectors that correspond to non-zero eigenvalues
        eigvecs = eigvecs[:, ~np.isclose(abs(eigvals), 0)]
        eigvals = eigvals[~np.isclose(abs(eigvals), 0)]

        # We rescale the eigenvectors with the sqrt of the eigenvalues (the other sqrt would rescale the right eigenvectors)
        rescaled_eigenvectors = np.einsum("b,ab->ab", np.sqrt(eigvals), eigvecs)

        # Finally we reshape the eigenvectors to form matrices, i.e., the Kraus operators and we make the first index
        # be the one that indexes the list of Kraus operators.
        result = np.einsum(
            "abc->cab", rescaled_eigenvectors.reshape([cutoff_dim ** N, cutoff_dim ** N, -1])
        )

        if not vectorize_modes:
            result = np.einsum(
                np.reshape(result, [-1] + [cutoff_dim] * (2 * N)),
                range(1 + 2 * N),
                [0] + [2 * n + 1 for n in range(N)] + [2 * n + 2 for n in range(N)],
            )
    else:
        raise ValueError("representation {} not supported".format(representation))

    return result

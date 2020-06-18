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
This module defines and implements several utility functions that act on
:class:`.Program` instances, returning or extracting information from the
quantum circuit.
"""
import copy

try:
    import tensorflow as tf
except ImportError:
    tf_available = False

import numpy as np

from strawberryfields.engine import LocalEngine
from strawberryfields.program_utils import Command
from strawberryfields.ops import Gate, Channel, Ket

__all__ = [
    "is_unitary",
    "is_channel",
    "extract_unitary",
    "extract_channel",
]


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

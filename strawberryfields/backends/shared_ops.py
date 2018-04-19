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
"""Common shared operations that can be used by backends"""

import os
import functools
import re
import itertools
from bisect import bisect
import pkg_resources

import numpy as np
import scipy as sp
from scipy.special import binom
from scipy.special import gammaln as lg
from scipy.linalg import qr

DATA_PATH = pkg_resources.resource_filename('strawberryfields', 'backends/data')
def_type = np.complex128


#================================+
#   Fock space shared operations |
#================================+

@functools.lru_cache()
def find_dim_files(regex, D, directory=None, name=""):
    r"""Return files matching a certain regex for specified dimension D.

    If no such file satisfying this condition exists, then an Exception
    is returned.

    Args:
        regex (str): regex matching the allowed file names. Should contain
        "(\d+)", this represents the Fock dimension in the file name.
        D (int): the dimension D
        directory (str): location to load the precomputed beamsplitter
            factors from. By default, this will be the Strawberry Fields data directory.
    """
    if directory is None:
        directory = DATA_PATH
    else:
        check_dir = os.path.isdir(directory)
        if not check_dir: # pragma: no cover
            raise NotADirectoryError("Directory {} does not exist!".format(directory))

    files = [f for f in os.listdir(directory) if re.match(regex, f)]
    avail_dims = sorted([int(re.findall(r'\d+', f)[0]) for f in files])

    idx = bisect(avail_dims, D-1)
    if idx+1 > len(avail_dims):
        raise FileNotFoundError("File containing {} factors does not exist "
                                "for dimension {} in directory {}".format(name, D, directory))

    return avail_dims[idx], directory


@functools.lru_cache()
def generate_bs_factors(D):
    r"""Generate beamsplitter factors in the Fock basis.

    This function generates the beamsplitter prefactors,

        .. math::
            prefac_{N,n,M,m,k} = (-1)^{N-k}\sqrt{\binom{n,k}\binom{m,N-k}\binom{N,k}\binom{M,n-k}}

    for a specific cutoff dimension :math:`D`.

    Note that the last dimension will only contain non-zero values
    for indices ``0`` to ``n``.

    Args:
        D (int): generate prefactors for :math:`D` dimensions.
    """
    prefac = np.zeros([D]*5, dtype=def_type)

    for (N, M, n) in itertools.product(*([range(D)]*3)):
        m = N+M-n
        k = np.arange(n+1)
        if 0 <= m < D:
            # pylint: disable=bad-whitespace
            prefac[N,n,M,m,:n+1] = (-1.0)**(N-k) \
                * np.sqrt(binom(n, k)*binom(m, N-k)*binom(N, k)*binom(M, n-k))

    return prefac


@functools.lru_cache()
def load_bs_factors(D, directory=None):
    r"""Load precomputed beamsplitter factors in the Fock basis.

    This function searches the data directory for a BS prefactor file
    containing for cutoff dimension higher or equal to that requested
    (``D``). It then reshapes the rank-2 sparse array to a
    :math:`D\times D\times D\times D\times D` dense array.

    If no such file satisfying this condition exists, then an Exception
    is returned.

    Args:
        D (int): load prefactors containing at least ``D`` dimensions.
        directory (str): location to load the precomputed beamsplitter
            factors from. By default, this will be the Strawberry Fields data directory.
    """

    regex = r"fock_beamsplitter_factors_(\d+)\.npz"
    load_dim, location = find_dim_files(regex, D, directory=directory, name="beamsplitter")
    filename = "fock_beamsplitter_factors_{}.npz".format(load_dim)
    prefac = sp.sparse.load_npz(os.path.join(location, filename))
    return np.reshape(prefac.toarray(), [load_dim]*5)


def save_bs_factors(prefac, directory=None):
    r"""Saves precomputed beamsplitter factors in the Fock basis to a file.

    This function reshapes the rank-5 array with dimension
    :math:`D\times D\times D\times D\times D` to a rank-2 array of dimension
    :math:`D^4\times D`, before converting it to a sparse array, and saving
    it to a file in the specified directory.

    Args:
        prefac (numpy.array): the Numpy array containing the precomputed beamsplitter
            prefactors in the Fock basis. Must be of size [D,D,D,D,D] for some integer D
        directory (str): location to save the precomputed beamsplitter factors. By default,
            this will be the Strawberry Fields data directory.
    """
    if directory is None:
        directory = DATA_PATH
    else:
        check_dir = os.path.isdir(directory)
        if not check_dir: # pragma: no cover
            raise NotADirectoryError("Directory {} does not exist!".format(directory))

    D = prefac.shape[0]
    filename = 'fock_beamsplitter_factors_{}.npz'.format(D)

    prefac_rank2 = np.reshape(prefac, ((D)**4, D), order='C')
    prefac_sparse = sp.sparse.csc_matrix(prefac_rank2)

    sp.sparse.save_npz(os.path.join(directory, filename), prefac_sparse)


@functools.lru_cache()
def squeeze_parity(D):
    r"""Creates the parity prefactor needed for squeezing in the Fock basis.

    .. math::
        \text{\sigma}_{N,k} = \begin{cases}
            (N-k)/2, & \text{mod}(N-k,2) \neq 0\\
            0, &\text{otherwise}
        \end{cases}

    Args:
        D (numpy.array): generate the prefactors for a Fock truncation of :math:`D`.
    """
    k = np.int(np.ceil(D/4) * 4)
    v = np.full(k, 1)
    v[1::2] = 0
    v[2::4] = -1
    v = np.vstack([np.roll(v, i) for i in range(k)])
    return v[:D, :D]


@functools.lru_cache()
def generate_squeeze_factors(D):
    r"""Generate squeezing factors in the Fock basis.

    This function generates the squeezing prefactors,

        .. math::
            prefac_{N,n,k} = \frac{\sigma_{N,k}\sqrt{n!N!}}
            {k!\left(\frac{n-k}{2}\right)!\left(\frac{N-k}{2}\right)!}

    where :math:`\sigma_{N,k}` is the parity, given by :func:`~.squeeze_parity`.

    Args:
        D (int): generate prefactors for :math:`D` dimensions.
    """
    dim_array = np.arange(D)
    N = dim_array.reshape((-1, 1, 1))
    n = dim_array.reshape((1, -1, 1))
    k = dim_array.reshape((1, 1, -1))

    # we only perform the sum when n+N is divisible by 2
    # in which case we sum 0 <= k <= min(N,n)
    mask = np.logical_and((n+N)%2 == 0, k <= np.minimum(N, n))

    # need to use np.power to avoid taking the root of a negative
    # in the numerator (these are discarded by the mask anyway)
    signs = squeeze_parity(D).reshape([D, 1, D])
    logfac = np.where(mask, 0.5*(lg(n+1) + lg(N+1)) - lg(k+1) - lg((n-k)/2+1) - lg((N-k)/2+1), 0)

    if D <= 600:
        prefactor = np.exp(logfac, dtype=np.float64)*signs*mask
    else:
        prefactor = np.exp(logfac, dtype=np.float128)*signs*mask

    return prefactor


def save_squeeze_factors(prefac, directory=None):
    r"""Saves precomputed squeeze factors in the Fock basis to a file.

    This function reshapes the rank-3 array with dimension
    :math:`D\times D\times D` to a rank-2 array of dimension
    :math:`D^2\times D`, before converting it to a sparse array, and saving
    it to a file in the specified directory.

    Args:
        prefac (numpy.array): the Numpy array containing the precomputed squeeze
            prefactors in the Fock basis. Must be of size [D,D,D] for some integer D
        directory (str): location to save the precomputed beamsplitter factors. By default,
            this will be the Strawberry Fields data directory.
    """
    if directory is None:
        directory = DATA_PATH
    else:
        check_dir = os.path.isdir(directory)
        if not check_dir:
            raise NotADirectoryError("Directory {} does not exist!".format(directory))

    D = prefac.shape[0]
    filename = 'fock_squeeze_factors_{}.npz'.format(D)

    prefac_rank2 = np.reshape(prefac, ((D)**2, D), order='C')
    prefac_sparse = sp.sparse.csc_matrix(prefac_rank2)

    sp.sparse.save_npz(os.path.join(directory, filename), prefac_sparse)


@functools.lru_cache()
def load_squeeze_factors(D, directory=None):
    r"""Load precomputed squeeze factors in the Fock basis.

    This function searches the data directory for a squeeze prefactor file
    containing for cutoff dimension higher or equal to that requested
    (``D``). It then reshapes the rank-2 sparse array to a
    :math:`D\times D\times D` dense array.

    If no such file satisfying this condition exists, then an Exception
    is returned.

    Args:
        D (int): load prefactors containing at least ``D`` dimensions.
        directory (str): location to load the precomputed squeeze
            factors from. By default, this will be the Strawberry Fields data directory.
    """
    regex = r"fock_squeeze_factors_(\d+)\.npz"
    load_dim, location = find_dim_files(regex, D, directory=directory, name="squeeze")

    filename = "fock_squeeze_factors_{}.npz".format(load_dim)
    prefac = sp.sparse.load_npz(os.path.join(location, filename))

    return np.reshape(prefac.toarray(), [load_dim]*3)


#================================+
# Phase space shared operations  |
#================================+

@functools.lru_cache()
def rotation_matrix(phi):
    r"""Rotation matrix.

    Args:
        phi (float): rotation angle
    Returns:
        array: :math:`2\times 2` rotation matrix
    """
    return np.array([[np.cos(phi), -np.sin(phi)],
                     [np.sin(phi), np.cos(phi)]])


@functools.lru_cache()
def sympmat(n):
    r""" Returns the symplectic matrix of order n

    Args:
        n (int): order
        hbar (float): the value of hbar used in the definition
            of the quadrature operators
    Returns:
        array: symplectic matrix
    """
    idm = np.identity(n)
    omega = np.concatenate((np.concatenate((0*idm, idm), axis=1),
                            np.concatenate((-idm, 0*idm), axis=1)), axis=0)
    return omega


@functools.lru_cache()
def changebasis(n):
    r"""Change of basis matrix between the two Gaussian representation orderings.

    This is the matrix necessary to transform covariances matrices written
    in the (x_1,...,x_n,p_1,...,p_n) to the (x_1,p_1,...,x_n,p_n) ordering

    Args:
        n (int): number of modes
    Returns:
        array: :math:`2n\times 2n` matrix
    """
    m = np.zeros((2*n, 2*n))
    for i in range(n):
        m[2*i, i] = 1
        m[2*i+1, i+n] = 1
    return m


def haar_measure(n):
    """A Random matrix distributed with the Haar measure.

    For more details, see :cite:`mezzadri2006`.

    Args:
        n (int): matrix size
    Returns:
        array: an nxn random matrix
    """
    z = (sp.randn(n, n) + 1j*sp.randn(n, n))/np.sqrt(2.0)
    q, r = qr(z)
    d = sp.diagonal(r)
    ph = d/np.abs(d)
    q = np.multiply(q, ph, q)
    return q

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
"""Gaussian operations"""

import numpy as np


def chop_in_blocks_multi(m, idtodelete):
    """
    Splits an array of (symmetric) matrices each into 3 blocks, A, B, C
    Blocks A and C are diagonal blocks and B is the offdiagonal block
    idtodelete specifies which indices go into C.
    """
    A = np.copy(m)
    A = np.delete(A, idtodelete, axis=1)
    A = np.delete(A, idtodelete, axis=2)
    B = np.delete(m[:, :, idtodelete], idtodelete, axis=1)
    C = m[:, idtodelete, :][:, :, idtodelete]
    return (A, B, C)


def chop_in_blocks_vector_multi(v, idtodelete):
    """
    Splits an array of vectors into two arrays of vectors, where
    idtodelete specifies which elements of the vectors go into vb
    """
    idtokeep = np.sort(list(set(np.arange(len(v[0]))) - set(idtodelete)))
    va = v[:, idtokeep]
    vb = v[:, idtodelete]
    return (va, vb)


def reassemble_multi(A, idtodelete):
    """
    Puts the matrices A inside larger matrices of dimensions
    dim(A)+len(idtodelete)
    The empty space are filled with zeros (offdiagonal) and ones (diagonals)
    """
    nweights = len(A[:, 0, 0])
    ntot = len(A[0]) + len(idtodelete)
    ind = np.sort(list(set(np.arange(ntot)) - set(idtodelete)))
    newmat = np.tile(np.eye(ntot, dtype=complex), (nweights, 1, 1))
    newmat[np.ix_(np.arange(newmat.shape[0], dtype=int), ind, ind)] = A
    return newmat


def reassemble_vector_multi(va, idtodelete):
    r"""Creates an array of vectors with zeros at indices idtodelete
    and everywhere else it puts the entries of va
    """
    nweights = len(va[:, 0])
    ntot = len(va[0]) + len(idtodelete)
    ind = np.sort(list(set(np.arange(ntot)) - set(idtodelete)))
    newv = np.zeros((nweights, ntot), dtype=complex)
    newv[:, ind] = va
    return newv

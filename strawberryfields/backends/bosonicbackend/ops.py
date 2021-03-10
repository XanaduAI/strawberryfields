# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Gaussian operations vectorizing commonly used operation on covariance matrices and vectors of means"""

import numpy as np


def chop_in_blocks_multi(m, id_to_delete):
    r"""
    Splits an array of (symmetric) matrices each into 3 blocks (``A``, ``B``, ``C``).

    Blocks ``A`` and ``C`` are diagonal blocks and ``B`` is the offdiagonal block.

    Args:
        m (ndarray): array of matrices
        id_to_delete (ndarray): array for the indices that go into ``C``

    Returns:
        tuple: tuple of the ``A``, ``B`` and ``C`` matrices
    """
    A = np.delete(m, id_to_delete, axis=1)
    A = np.delete(A, id_to_delete, axis=2)
    B = np.delete(m[:, :, id_to_delete], id_to_delete, axis=1)
    C = m[:, id_to_delete, :][:, :, id_to_delete]
    return (A, B, C)


def chop_in_blocks_vector_multi(v, id_to_delete):
    r"""
    For an array of vectors ``v``, splits ``v`` into two arrays of vectors,
    ``va`` and ``vb``. ``vb`` contains the components of ``v`` specified by
    ``id_to_delete``, and ``va`` contains the remaining components.

    Args:
        v (ndarray): array of vectors
        id_to_delete (ndarray): array for the indices that go into vb

    Returns:
        tuple: tuple of ``(va,vb)`` vectors
    """
    id_to_keep = np.sort(list(set(np.arange(len(v[0]))) - set(id_to_delete)))
    va = v[:, id_to_keep]
    vb = v[:, id_to_delete]
    return (va, vb)


def reassemble_multi(A, id_to_delete):
    r"""
    For an array of matrices ``A``, creates a new array of matrices, each with
    dimension ``dim(A)+len(id_to_delete)``. The subspace of each new matrix
    specified by indices ``id_to_delete`` is set to the identity matrix, while
    the rest of each new matrix is filled with the matrices from ``A``.

    Args:
        m (ndarray): array of matrices
        id_to_delete (ndarray): array of indices in the new matrices that will
            be set to the identity

    Returns:
        array: array of new matrices, each filled with ``A`` and identity
    """
    num_weights = len(A[:, 0, 0])
    new_mat_dim = len(A[0]) + len(id_to_delete)
    ind = np.sort(list(set(np.arange(new_mat_dim)) - set(id_to_delete)))
    new_mat = np.tile(np.eye(new_mat_dim, dtype=complex), (num_weights, 1, 1))
    new_mat[np.ix_(np.arange(new_mat.shape[0], dtype=int), ind, ind)] = A
    return new_mat


def reassemble_vector_multi(va, id_to_delete):
    r"""
    For an array of vectors ``va``, creates a new array of vectors, each with
    dimension ``dim(va)+len(id_to_delete)``. The subspace of each new vector
    specified by indices ``id_to_delete`` is set to 0, while the rest of each
    new vector is filled with the vectors from ``va``.

    Args:
        va (ndarray): array of vectors
        id_to_delete (ndarray): array of indices in the new vectors that will
            be set to 0

    Returns:
        array: array of new vectors, each filled with ``va`` and 0
    """
    num_weights = len(va[:, 0])
    new_vec_dim = len(va[0]) + len(id_to_delete)
    ind = np.sort(list(set(np.arange(new_vec_dim)) - set(id_to_delete)))
    new_vec = np.zeros((num_weights, new_vec_dim), dtype=complex)
    new_vec[:, ind] = va
    return new_vec

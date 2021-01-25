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

r"""
Unit tests for backends.bosonicbackend.ops file.
"""

import numpy as np
import strawberryfields.backends.bosonicbackend.ops as ops


def test_chop_in_blocks_multi():
    # Create submatrices
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 3)
    C = np.random.rand(3, 3)

    # Repeat them in an array
    reps = np.random.randint(1, 10)
    Atile = np.tile(A, [reps, 1, 1])
    Btile = np.tile(B, [reps, 1, 1])
    Ctile = np.tile(C, [reps, 1, 1])

    # Make a new block matrix out of them and repeat it
    m = np.block([[A, B], [B.T, C]])
    m = np.tile(m, [reps, 1, 1])

    # Choose to delete the indices corresponding to C
    idtodelete = np.arange(2, 5, dtype=int)

    A2, B2, C2 = ops.chop_in_blocks_multi(m, idtodelete)

    assert np.allclose(A2, Atile)
    assert np.allclose(B2, Btile)
    assert np.allclose(C2, Ctile)


def test_chop_in_blocks_vector_multi():
    # Create vectors
    va = np.random.rand(6)
    vb = np.random.rand(4)

    # Repeat them in an array
    reps = np.random.randint(1, 10)
    vatile = np.tile(va, [reps, 1])
    vbtile = np.tile(vb, [reps, 1])

    # Make a new vector out of them and repeat it
    v = np.append(va, vb)
    v = np.tile(v, [reps, 1])

    # Choose to delete the indices corresponding to C
    idtodelete = np.arange(6, 10, dtype=int)

    va2, vb2 = ops.chop_in_blocks_vector_multi(v, idtodelete)

    assert np.allclose(va2, vatile)
    assert np.allclose(vb2, vbtile)


def test_reassemble_multi():
    # Create matrix
    A = np.random.rand(2, 2)
    reps = np.random.randint(1, 10)
    Atile = np.tile(A, [reps, 1, 1])
    # Create indices
    idtodelete = [0, 2, 4, 5]
    m = ops.reassemble_multi(Atile, idtodelete)
    dim = len(A) + len(idtodelete)

    assert m.shape == (reps, dim, dim)

    A2, B2, C2 = ops.chop_in_blocks_multi(m, [1, 3])
    assert np.allclose(C2, Atile)
    assert np.allclose(B2, 0)
    assert np.allclose(A2, np.tile(np.eye(4), [reps, 1, 1]))


def test_reassemble_vector_multi():
    # Create matrix
    v = np.random.rand(4)
    reps = np.random.randint(1, 10)
    vtile = np.tile(v, [reps, 1])
    # Create indices
    idtodelete = [0, 2, 3, 5, 7, 9, 10]
    m = ops.reassemble_vector_multi(vtile, idtodelete)
    dim = len(v) + len(idtodelete)

    assert m.shape == (reps, dim)

    va2, vb2 = ops.chop_in_blocks_vector_multi(m, [1, 4, 6, 8])
    assert np.allclose(vb2, vtile)
    assert np.allclose(va2, 0)

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
Unit tests for backends.bosonicbackend.ops.py .
"""

import pytest

import numpy as np
import strawberryfields.backends.bosonicbackend.ops as ops

pytestmark = pytest.mark.bosonic


class TestOpsFunctions:
    r"""Tests all the functions inside backends.bosonicbackend.ops.py"""

    @pytest.mark.parametrize("reps", [1, 2, 5, 10])
    def test_chop_in_blocks_multi(self, reps):
        r"""Checks that ops.chop_in_block_multi partitions arrays of matrices correctly"""
        # Create submatrices
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 3)
        C = np.random.rand(3, 3)

        # Repeat them in an array
        Atile = np.tile(A, [reps, 1, 1])
        Btile = np.tile(B, [reps, 1, 1])
        Ctile = np.tile(C, [reps, 1, 1])

        # Make a new block matrix out of them and repeat it
        m = np.block([[A, B], [B.T, C]])
        m = np.tile(m, [reps, 1, 1])

        # Choose to delete the indices corresponding to C
        id_to_delete = np.arange(2, 5, dtype=int)

        A2, B2, C2 = ops.chop_in_blocks_multi(m, id_to_delete)

        assert np.allclose(A2, Atile)
        assert np.allclose(B2, Btile)
        assert np.allclose(C2, Ctile)

    @pytest.mark.parametrize("reps", [1, 2, 5, 10])
    def test_chop_in_blocks_vector_multi(self, reps):
        r"""Checks that ops.chop_in_block_vector_multi partitions arrays of vectors correctly"""

        # Create vectors
        va = np.random.rand(6)
        vb = np.random.rand(4)

        # Repeat them in an array
        vatile = np.tile(va, [reps, 1])
        vbtile = np.tile(vb, [reps, 1])

        # Make a new vector out of them and repeat it
        v = np.append(va, vb)
        v = np.tile(v, [reps, 1])

        # Choose to delete the indices corresponding to C
        id_to_delete = np.arange(6, 10, dtype=int)

        va2, vb2 = ops.chop_in_blocks_vector_multi(v, id_to_delete)

        assert np.allclose(va2, vatile)
        assert np.allclose(vb2, vbtile)

    @pytest.mark.parametrize("id_to_delete", [[0], [0, 1], [2, 0, 1], [0, 2, 4, 5]])
    def test_reassemble_multi(self, id_to_delete):
        r"""Checks that ops.reassemble_multi generates the correct output"""

        # Create matrix
        A = np.random.rand(2, 2)
        reps = np.random.randint(1, 10)
        Atile = np.tile(A, [reps, 1, 1])
        # Create indices
        m = ops.reassemble_multi(Atile, id_to_delete)
        dim = len(A) + len(id_to_delete)
        id_to_keep = list(set(range(dim)) - set(id_to_delete))
        id_to_keep.sort()
        assert m.shape == (reps, dim, dim)

        A2, B2, C2 = ops.chop_in_blocks_multi(m, id_to_keep)
        assert np.allclose(C2, Atile)
        assert np.allclose(B2, 0)
        assert np.allclose(A2, np.tile(np.eye(len(id_to_delete)), [reps, 1, 1]))

    @pytest.mark.parametrize("id_to_delete", [[0], [0, 1], [2, 0, 1], [0, 2, 4, 5]])
    def test_reassemble_vector_multi(self, id_to_delete):
        r"""Checks that ops.reassemble_vector_multi generates the correct output"""
        # Create matrix
        v = np.random.rand(4)
        reps = np.random.randint(1, 10)
        vtile = np.tile(v, [reps, 1])
        # Create indices
        m = ops.reassemble_vector_multi(vtile, id_to_delete)
        dim = len(v) + len(id_to_delete)
        id_to_keep = list(set(range(dim)) - set(id_to_delete))
        id_to_keep.sort()
        assert m.shape == (reps, dim)

        va2, vb2 = ops.chop_in_blocks_vector_multi(m, id_to_keep)
        assert np.allclose(vb2, vtile)
        assert np.allclose(va2, 0)

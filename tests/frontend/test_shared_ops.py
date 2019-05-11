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
r"""Unit tests for the Strawberry Fields shared_ops module"""
import os
import pytest

pytestmark = pytest.mark.frontend

import numpy as np

import strawberryfields.backends.shared_ops as so

# fmt: off
BS_4_VAL = np.array([ 1.00000000+0.j,  1.00000000+0.j,  1.00000000+0.j,  1.00000000+0.j,
        1.00000000+0.j,  1.41421356+0.j,  1.73205081+0.j,  1.00000000+0.j,
        1.73205081+0.j,  1.00000000+0.j, -1.00000000+0.j, -1.41421356+0.j,
       -1.73205081+0.j,  1.00000000+0.j, -1.00000000+0.j,  1.00000000+0.j,
       -2.00000000+0.j,  1.00000000+0.j, -3.00000000+0.j,  1.00000000+0.j,
        1.41421356+0.j, -1.00000000+0.j,  2.00000000+0.j, -2.44948974+0.j,
        2.44948974+0.j,  1.73205081+0.j, -1.00000000+0.j,  3.00000000+0.j,
        1.00000000+0.j,  1.73205081+0.j, -1.41421356+0.j,  1.00000000+0.j,
       -2.00000000+0.j,  2.44948974+0.j, -2.44948974+0.j,  1.00000000+0.j,
       -2.00000000+0.j,  1.00000000+0.j,  1.00000000+0.j, -4.00000000+0.j,
        1.00000000+0.j,  3.00000000+0.j, -6.00000000+0.j,  1.00000000+0.j,
        1.73205081+0.j, -2.44948974+0.j,  2.44948974+0.j,  1.00000000+0.j,
       -6.00000000+0.j,  3.00000000+0.j, -1.00000000+0.j,  1.73205081+0.j,
       -1.00000000+0.j,  3.00000000+0.j, -1.73205081+0.j,  2.44948974+0.j,
       -2.44948974+0.j, -1.00000000+0.j,  6.00000000+0.j, -3.00000000+0.j,
        1.00000000+0.j, -3.00000000+0.j,  1.00000000+0.j,  3.00000000+0.j,
       -6.00000000+0.j,  1.00000000+0.j, -1.00000000+0.j,  9.00000000+0.j,
       -9.00000000+0.j,  1.00000000+0.j])

BS_4_IDX = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3],
       [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2,
        2, 2, 2, 3, 3, 3, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3],
       [0, 1, 2, 3, 1, 2, 3, 2, 3, 3, 0, 1, 2, 0, 1, 1, 2, 2, 3, 3, 1, 2,
        2, 3, 3, 2, 3, 3, 0, 1, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2, 3, 3, 3,
        1, 2, 2, 3, 3, 3, 0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2,
        3, 3, 3, 3],
       [0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0, 1,
        1, 2, 2, 0, 1, 1, 2, 3, 1, 2, 2, 3, 3, 0, 1, 1, 2, 2, 2, 3, 3, 3,
        0, 1, 1, 2, 2, 2, 3, 2, 3, 3, 1, 2, 2, 3, 3, 3, 0, 1, 1, 2, 2, 2,
        3, 3, 3, 3],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0,
        1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2,
        2, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 1, 2, 0, 1, 2, 3, 2, 3, 1, 2, 3,
        0, 1, 2, 3]])

# fmt: on

SQUEEZE_PARITY_8 = np.array(
    [
        [1, 0, -1, 0, 1, 0, -1, 0],
        [0, 1, 0, -1, 0, 1, 0, -1],
        [-1, 0, 1, 0, -1, 0, 1, 0],
        [0, -1, 0, 1, 0, -1, 0, 1],
        [1, 0, -1, 0, 1, 0, -1, 0],
        [0, 1, 0, -1, 0, 1, 0, -1],
        [-1, 0, 1, 0, -1, 0, 1, 0],
        [0, -1, 0, 1, 0, -1, 0, 1],
    ]
)


SQUEEZE_FACTOR_4 = np.array(
    [
        [
            [1.0, 0.0, -0.0, 0.0],
            [0.0, 0.0, -0.0, 0.0],
            [1.41421356, 0.0, -0.0, 0.0],
            [0.0, 0.0, -0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, -0.0],
            [0.0, 1.0, 0.0, -0.0],
            [0.0, 0.0, 0.0, -0.0],
            [0.0, 2.44948974, 0.0, -0.0],
        ],
        [
            [-1.41421356, 0.0, 0.0, 0.0],
            [-0.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 1.0, 0.0],
            [-0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, -0.0, 0.0, 0.0],
            [0.0, -2.44948974, 0.0, 0.0],
            [0.0, -0.0, 0.0, 0.0],
            [0.0, -6.0, 0.0, 1.0],
        ],
    ]
)


# TODO: write unit tests for find_dim_files function


class TestBeamsplitterFactors:
    """Tests for the beamsplitter prefactors"""

    def test_generate(self, tol):
        """test generating beamsplitter factors gives expected results"""
        factors = so.generate_bs_factors(4)
        factors_val = factors[factors != 0.0]
        factors_idx = np.array(np.nonzero(factors))

        assert np.allclose(factors_val, BS_4_VAL, atol=tol, rtol=0)
        assert np.allclose(factors_idx, BS_4_IDX, atol=tol, rtol=0)

    def test_save_load(self, tmpdir, tol):
        """test saving and loading of beamsplitter factors"""
        factors = so.generate_bs_factors(4)
        so.save_bs_factors(factors, directory=str(tmpdir))

        factors = so.load_bs_factors(4, directory=str(tmpdir))
        factors_val = factors[factors != 0.0]
        factors_idx = np.array(np.nonzero(factors))

        assert np.allclose(factors_val, BS_4_VAL, atol=tol, rtol=0)
        assert np.allclose(factors_idx, BS_4_IDX, atol=tol, rtol=0)


class TestSqueezingFactors:
    """Tests for the squeezing prefactors"""

    def test_squeeze_parity(self, tol):
        """Test the squeeze parity function returns the correct result"""
        parity = so.squeeze_parity(8)
        assert np.allclose(parity, SQUEEZE_PARITY_8, atol=tol, rtol=0)

    def test_generate(self, tol):
        """test generating squeezoing factors gives expected results"""
        factors = so.generate_squeeze_factors(4)
        assert np.allclose(factors, SQUEEZE_FACTOR_4, atol=tol, rtol=0)

    def test_save_load(self, tmpdir, tol):
        """test saving and loading of squeezoing factors"""
        factors_in = so.generate_squeeze_factors(4)
        so.save_squeeze_factors(factors_in, directory=str(tmpdir))
        factors_out = so.load_squeeze_factors(4, directory=str(tmpdir))

        assert np.allclose(factors_out, SQUEEZE_FACTOR_4, atol=tol, rtol=0)


class TestPhaseSpaceFunctions:
    """Tests for the shared phase space operations"""

    @pytest.mark.parametrize("phi", np.linspace(0, np.pi, 4))
    def test_rotation_matrix(self, phi):
        """Test the function rotation_matrix"""
        res = so.rotation_matrix(phi)
        expected = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        assert np.all(res == expected)

    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_sympmat(self, n):
        """Test the symplectic matrix function"""
        res = so.sympmat(n)
        O = np.zeros([n, n])
        I = np.identity(n)
        expected = np.block([[O, I], [-I, O]])

        assert np.all(res == expected)

    def test_means_changebasis(self):
        """Test the change of basis function applied to vectors. This function
        converts from xp to symmetric ordering, and vice versa."""
        C = so.changebasis(3)
        means_xp = [1, 2, 3, 4, 5, 6]
        means_symmetric = [1, 4, 2, 5, 3, 6]

        assert np.all(C @ means_xp == means_symmetric)
        assert np.all(C.T @ means_symmetric == means_xp)

    def test_cov_changebasis(self):
        """Test the change of basis function applied to matrices. This function
        converts from xp to symmetric ordering, and vice versa."""
        C = so.changebasis(2)
        cov_xp = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )

        cov_symmetric = np.array(
            [[0, 2, 1, 3], [8, 10, 9, 11], [4, 6, 5, 7], [12, 14, 13, 15]]
        )

        assert np.all(C @ cov_xp @ C.T == cov_symmetric)
        assert np.all(C.T @ cov_symmetric @ C == cov_xp)

    @pytest.mark.parametrize("n", [1, 2, 4, 10])
    def test_haar_measure(self, n, tol):
        """test that the haar measure function returns unitary matrices"""
        U = so.haar_measure(n)
        assert np.allclose(U @ U.conj().T, np.identity(n), atol=tol, rtol=0)

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
r""" Tests for various multi mode state preparation operations"""
import itertools

import pytest

import numpy as np

from strawberryfields import ops
from strawberryfields.backends.gaussianbackend.states import GaussianState
from strawberryfields.utils import random_covariance, displaced_squeezed_state

# make test deterministic
np.random.seed(42)


MAG_ALPHAS = np.linspace(0, 0.8, 3)
PHASE_ALPHAS = np.linspace(0, 2 * np.pi, 3, endpoint=False)
NBARS = np.linspace(0, 5)


@pytest.mark.backends("tf", "fock")
class TestFockBasisMultimode:
    """Testing preparing multimode states on the Fock backends"""

    def test_multimode_ket_mode_permutations(self, setup_backend, pure, cutoff, tol):
        """Test multimode ket preparation when modes are permuted"""
        backend = setup_backend(4)

        random_ket0 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(
            -1, 1, cutoff
        )
        random_ket0 = random_ket0 / np.linalg.norm(random_ket0)

        random_ket1 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(
            -1, 1, cutoff
        )
        random_ket1 = random_ket1 / np.linalg.norm(random_ket1)

        random_ket = np.outer(random_ket0, random_ket1)
        rho = np.einsum("ij,kl->ikjl", random_ket, random_ket.conj())

        backend.prepare_ket_state(random_ket, modes=[3, 1])
        state = backend.state([3, 1])
        multi_mode_preparation_dm = state.dm()

        assert np.allclose(multi_mode_preparation_dm, rho, atol=tol, rtol=0)

    def test_compare_single_mode_and_multimode_ket_preparation(
        self, setup_backend, batch_size, pure, cutoff, tol
    ):
        """Test single and multimode ket preparation"""
        backend = setup_backend(4)

        random_ket0 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(
            -1, 1, cutoff
        )
        random_ket0 = random_ket0 / np.linalg.norm(random_ket0)

        random_ket1 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(
            -1, 1, cutoff
        )
        random_ket1 = random_ket1 / np.linalg.norm(random_ket1)

        random_ket = np.outer(random_ket0, random_ket1)

        backend.prepare_ket_state(random_ket0, 0)
        backend.prepare_ket_state(random_ket1, 1)
        state = backend.state([0, 1])
        single_mode_preparation_dm = state.dm()
        single_mode_preparation_probs = np.array(state.all_fock_probs())

        backend.reset(pure=pure)
        backend.prepare_ket_state(random_ket, [0, 1])
        state = backend.state([0, 1])
        multi_mode_preparation_dm = state.dm()
        multi_mode_preparation_probs = np.array(state.all_fock_probs())

        assert np.allclose(
            single_mode_preparation_dm, multi_mode_preparation_dm, atol=tol, rtol=0
        )
        assert np.allclose(
            single_mode_preparation_probs,
            multi_mode_preparation_probs,
            atol=tol,
            rtol=0,
        )

        if batch_size is not None:
            single_mode_preparation_dm = single_mode_preparation_dm[0]
            multi_mode_preparation_dm = multi_mode_preparation_dm[0]

        assert np.all(
            single_mode_preparation_dm.shape == multi_mode_preparation_dm.shape
        )

    def test_compare_single_mode_and_multimode_dm_preparation(
        self, setup_backend, batch_size, pure, cutoff, tol
    ):
        """Compare the results of a successive single mode preparations
        and a multi mode preparation of a product state."""
        backend = setup_backend(4)
        random_rho0 = np.random.normal(size=[cutoff] * 2) + 1j * np.random.normal(
            size=[cutoff] * 2
        )
        random_rho0 = np.dot(random_rho0.conj().T, random_rho0)
        random_rho0 = random_rho0 / random_rho0.trace()

        random_rho1 = np.random.normal(size=[cutoff] * 2) + 1j * np.random.normal(
            size=[cutoff] * 2
        )
        random_rho1 = np.dot(random_rho1.conj().T, random_rho1)
        random_rho1 = random_rho1 / random_rho1.trace()
        random_dm = np.outer(random_rho0, random_rho1)
        random_dm = random_dm.reshape([cutoff] * 4)

        backend.prepare_dm_state(random_rho0, 0)
        backend.prepare_dm_state(random_rho1, 1)
        state = backend.state([0, 1])
        single_mode_preparation_dm = state.dm()
        single_mode_preparation_probs = np.array(state.all_fock_probs())

        # first we do a preparation from random_dm, with shape [cutoff]*4
        backend.reset(pure=pure)
        backend.prepare_dm_state(random_dm, [0, 1])
        state = backend.state(modes=[0, 1])
        multi_mode_preparation_dm = state.dm()
        multi_mode_preparation_probs = np.array(state.all_fock_probs())

        # second we do a preparation from the corresponding matrix with shape [cutoff**2]*2
        backend.reset(pure=pure)
        backend.prepare_dm_state(random_dm.reshape([cutoff ** 2] * 2), [0, 1])
        state = backend.state(modes=[0, 1])
        multi_mode_preparation_from_matrix_dm = state.dm()
        multi_mode_preparation_from_matrix_probs = np.array(state.all_fock_probs())

        # third we do a preparation from random_dm on modes 3 and 1 (in that order!) and test if the states end up in the correct modes
        backend.reset(pure=pure)
        backend.prepare_dm_state(random_dm, [3, 1])

        multi_mode_preparation_31_mode_0 = backend.state(modes=0).dm()
        multi_mode_preparation_31_mode_1 = backend.state(modes=1).dm()
        multi_mode_preparation_31_mode_2 = backend.state(modes=2).dm()
        multi_mode_preparation_31_mode_3 = backend.state(modes=3).dm()
        multi_mode_preparation_31_probs = np.array(
            backend.state(modes=[3, 1]).all_fock_probs()
        )

        single_mode_vac = np.zeros((cutoff, cutoff), dtype=np.complex128)
        single_mode_vac.itemset(0, 1.0 + 0.0j)

        assert np.allclose(random_dm, single_mode_preparation_dm, atol=tol, rtol=0)
        assert np.allclose(
            multi_mode_preparation_dm, single_mode_preparation_dm, atol=tol, rtol=0
        )
        assert np.allclose(
            multi_mode_preparation_from_matrix_dm,
            single_mode_preparation_dm,
            atol=tol,
            rtol=0,
        )

        assert np.allclose(
            multi_mode_preparation_31_mode_0, single_mode_vac, atol=tol, rtol=0
        )
        assert np.allclose(
            multi_mode_preparation_31_mode_1, random_rho1, atol=tol, rtol=0
        )
        assert np.allclose(
            multi_mode_preparation_31_mode_2, single_mode_vac, atol=tol, rtol=0
        )
        assert np.allclose(
            multi_mode_preparation_31_mode_3, random_rho0, atol=tol, rtol=0
        )

        # also check the fock probabilities to catch errors in both the preparation and state() the would cancel each other out
        assert np.allclose(
            single_mode_preparation_probs,
            multi_mode_preparation_probs,
            atol=tol,
            rtol=0,
        )
        assert np.allclose(
            single_mode_preparation_probs,
            multi_mode_preparation_from_matrix_probs,
            atol=tol,
            rtol=0,
        )
        assert np.allclose(
            single_mode_preparation_probs,
            multi_mode_preparation_31_probs,
            atol=tol,
            rtol=0,
        )

        if batch_size is not None:
            single_mode_preparation_dm = single_mode_preparation_dm[0]
            multi_mode_preparation_dm = multi_mode_preparation_dm[0]
            multi_mode_preparation_from_matrix_dm = multi_mode_preparation_from_matrix_dm[
                0
            ]

        assert np.all(random_dm.shape == single_mode_preparation_dm.shape)
        assert np.all(random_dm.shape == multi_mode_preparation_dm.shape)
        assert np.all(random_dm.shape == multi_mode_preparation_from_matrix_dm.shape)

    def test_prepare_multimode_random_product_dm_state_on_different_modes(
        self, setup_backend, batch_size, pure, cutoff, tol
    ):
        """Tests if a random multi mode dm state is correctly prepared on different modes."""
        backend = setup_backend(4)
        N = 4

        random_rho = np.random.normal(size=[cutoff ** 2] * 2) + 1j * np.random.normal(
            size=[cutoff ** 2] * 2
        )  # two mode random state
        random_rho = np.dot(random_rho.conj().T, random_rho)
        random_rho = random_rho / random_rho.trace()
        random_rho = random_rho.reshape(
            [cutoff] * 4
        )  # reshape for easier comparison later

        # test the state preparation on the first two modes
        backend.prepare_dm_state(random_rho, [0, 1])
        multi_mode_preparation_with_modes_ordered = backend.state([0, 1]).dm()
        assert np.allclose(
            multi_mode_preparation_with_modes_ordered, random_rho, atol=tol, rtol=0
        )

        # test the state preparation on two other modes that are not in order
        backend.reset(pure=pure)
        backend.prepare_dm_state(random_rho, [1, 2])
        multi_mode_preparation_with_modes_inverted = backend.state([1, 2]).dm()
        assert np.allclose(
            multi_mode_preparation_with_modes_inverted, random_rho, atol=tol, rtol=0
        )

        # test various subsets of subsystems in various orders
        for subsystems in list(itertools.permutations(range(N), 2)):
            subsystems = list(subsystems)
            backend.reset(pure=pure)
            backend.prepare_dm_state(random_rho, subsystems)
            dm = backend.state(modes=subsystems).dm()

            assert np.allclose(random_rho, dm, atol=tol, rtol=0)
            if batch_size is not None:
                dm = dm[0]

            assert np.all(random_rho.shape == dm.shape)

    def test_fast_state_prep_on_all_modes(
        self, setup_backend, batch_size, pure, cutoff, tol
    ):
        """Tests if a random multi mode ket state is correctly prepared with
        the shortcut method on all modes."""
        backend = setup_backend(4)
        N = 4
        random_ket = np.random.normal(size=[cutoff] * N) + 1j * np.random.normal(
            size=[cutoff] * N
        )
        random_ket = random_ket / np.linalg.norm(random_ket)

        backend.prepare_dm_state(random_ket, modes=range(N))
        all_mode_preparation_ket = (
            backend.state().ket()
        )  # Returns None if the state if mixed

        assert np.allclose(all_mode_preparation_ket, random_ket, atol=tol, rtol=0)

        if batch_size is not None:
            all_mode_preparation_ket = all_mode_preparation_ket[0]

        assert np.all(all_mode_preparation_ket.shape == random_ket.shape)


@pytest.mark.backends("gaussian")
class TestGaussianMultimode:
    """Tests for simulators that use the Gaussian representation."""

    def test_singlemode_gaussian_state(self, setup_backend, batch_size, pure, tol, hbar):
        """Test single mode Gaussian state preparation"""
        N = 4
        backend = setup_backend(N)

        means = 2 * np.random.random(size=[2]) - 1
        cov = random_covariance(1, pure=pure)

        a = 0.2 + 0.4j
        r = 1
        phi = 0

        # circuit is initially in displaced squeezed state
        for i in range(N):
            backend.prepare_displaced_squeezed_state(a, r, phi, mode=i)

        # prepare Gaussian state in mode 1
        backend.prepare_gaussian_state(means, cov, modes=1)

        # test Gaussian state is correct
        state = backend.state([1])
        assert np.allclose(state.means(), means*np.sqrt(hbar/2), atol=tol, rtol=0)
        assert np.allclose(state.cov(), cov*hbar/2, atol=tol, rtol=0)

        # test that displaced squeezed states are unchanged
        ex_means, ex_V = displaced_squeezed_state(a, r, phi, basis="gaussian", hbar=hbar)
        for i in [0, 2, 3]:
            state = backend.state([i])
            assert np.allclose(state.means(), ex_means, atol=tol, rtol=0)
            assert np.allclose(state.cov(), ex_V, atol=tol, rtol=0)

    def test_multimode_gaussian_state(self, setup_backend, batch_size, pure, tol, hbar):
        """Test multimode Gaussian state preparation"""
        N = 4
        backend = setup_backend(N)

        cov = np.diag(np.exp(2 * np.array([-1, -1, 1, 1])))
        means = np.zeros([4])

        # prepare displaced squeezed states in all modes
        a = 0.2 + 0.4j
        r = 0.5
        phi = 0.12
        for i in range(N):
            backend.prepare_displaced_squeezed_state(a, r, phi, i)

        # prepare new squeezed displaced state in mode 1 and 3
        backend.prepare_gaussian_state(means, cov, modes=[1, 3])
        state = backend.state([1, 3])

        # test Gaussian state is correct
        state = backend.state([1, 3])
        assert np.allclose(state.means(), means*np.sqrt(hbar/2), atol=tol, rtol=0)
        assert np.allclose(state.cov(), cov*hbar/2, atol=tol, rtol=0)

        # test that displaced squeezed states are unchanged
        ex_means, ex_V = displaced_squeezed_state(a, r, phi, basis="gaussian", hbar=hbar)
        for i in [0, 2]:
            state = backend.state([i])
            assert np.allclose(state.means(), ex_means, atol=tol, rtol=0)
            assert np.allclose(state.cov(), ex_V, atol=tol, rtol=0)

    def test_full_mode_squeezed_state(self, setup_backend, batch_size, pure, tol, hbar):
        """Test full register Gaussian state preparation"""
        N = 4
        backend = setup_backend(N)

        cov = np.diag(np.exp(2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])))
        means = np.zeros([8])

        backend.reset(pure=pure)
        backend.prepare_gaussian_state(means, cov, modes=range(N))
        state = backend.state()

        # test Gaussian state is correct
        state = backend.state()
        assert np.allclose(state.means(), means*np.sqrt(hbar/2), atol=tol, rtol=0)
        assert np.allclose(state.cov(), cov*hbar/2, atol=tol, rtol=0)

    def test_multimode_gaussian_random_state(
        self, setup_backend, batch_size, pure, tol, hbar
    ):
        """Test multimode Gaussian state preparation on a random state"""
        N = 4
        backend = setup_backend(N)

        means = 2 * np.random.random(size=[2 * N]) - 1
        cov = random_covariance(N, pure=pure)

        backend.reset(pure=pure)

        # circuit is initially in a random state
        backend.prepare_gaussian_state(means, cov, modes=range(N))

        # test Gaussian state is correct
        state = backend.state()
        assert np.allclose(state.means(), means*np.sqrt(hbar/2), atol=tol, rtol=0)
        assert np.allclose(state.cov(), cov*hbar/2, atol=tol, rtol=0)

        # prepare Gaussian state in mode 2 and 1
        means2 = 2 * np.random.random(size=[4]) - 1
        cov2 = random_covariance(2, pure=pure)
        backend.prepare_gaussian_state(means2, cov2, modes=[2, 1])

        # test resulting Gaussian state is correct
        state = backend.state()

        # in the new means vector, the modes 0 and 3 remain unchanged
        # Modes 1 and 2, however, now have values given from elements
        # means2[1] and means2[0].
        ex_means = np.array(
            [
                means[0],
                means2[1],
                means2[0],
                means[3],  # position
                means[4],
                means2[3],
                means2[2],
                means[7],
            ]
        )  # momentum

        ex_cov = np.zeros([8, 8])

        # in the new covariance matrix, modes 0 and 3 remain unchanged
        idx = np.array([0, 3, 4, 7])
        rows = idx.reshape(-1, 1)
        cols = idx.reshape(1, -1)
        ex_cov[rows, cols] = cov[rows, cols]

        # in the new covariance matrix, modes 1 and 2 have values given by
        # rows 1 and 0 respectively from cov2
        idx = np.array([1, 2, 5, 6])
        rows = idx.reshape(-1, 1)
        cols = idx.reshape(1, -1)

        idx = np.array([1, 0, 3, 2])
        rows2 = idx.reshape(-1, 1)
        cols2 = idx.reshape(1, -1)

        ex_cov[rows, cols] = cov2[rows2, cols2]

        assert np.allclose(state.means(), ex_means*np.sqrt(hbar/2), atol=tol, rtol=0)
        assert np.allclose(state.cov(), ex_cov*hbar/2, atol=tol, rtol=0)

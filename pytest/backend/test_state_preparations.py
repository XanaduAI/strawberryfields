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

r"""Unit tests for various state preparation operations"""

import pytest

import numpy as np


MAG_ALPHAS = np.linspace(0, 0.8, 4)
PHASE_ALPHAS = np.linspace(0, 2 * np.pi, 7, endpoint=False)
NBARS = np.linspace(0, 5, 7)
SEED = 143


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    def test_prepare_vac(self, setup_backend, tol):
        """Tests the ability to prepare vacuum states."""
        backend = setup_backend(1)
        backend.prepare_vacuum_state(0)
        assert np.all(backend.is_vacuum(tol))

    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    def test_fidelity_coherent(self, setup_backend, mag_alpha, phase_alpha, tol):
        """Tests if a range of coherent states have the correct fidelity."""

        backend = setup_backend(1)

        alpha = mag_alpha * np.exp(1j * phase_alpha)
        backend.prepare_coherent_state(alpha, 0)
        state = backend.state()
        assert np.allclose(state.fidelity_coherent([alpha]), 1.0, atol=tol, rtol=0.0)

    @pytest.mark.parametrize("nbar", NBARS)
    def test_prepare_thermal_state(self, setup_backend, nbar, cutoff, tol):
        """Tests if thermal states with different nbar values
        give the correct fock probabilities"""

        backend = setup_backend(1)

        backend.prepare_thermal_state(nbar, 0)
        ref_probs = np.array([nbar ** n / (nbar + 1) ** (n + 1) for n in range(cutoff)])
        state = backend.state()
        state_probs = np.array(
            [state.fock_prob([n]) for n in range(cutoff)]
        ).T  # transpose needed for array broadcasting to work in batch mode (data is unaffected in non-batched mode)
        assert np.allclose(ref_probs, state_probs, atol=tol, rtol=0.0)


@pytest.mark.backends("fock", "tf")
class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    def test_normalized_prepare_vac(self, setup_backend, tol):
        """Tests the ability to prepare vacuum states."""

        backend = setup_backend(1)

        backend.prepare_vacuum_state(0)
        state = backend.state()
        tr = state.trace()
        assert np.allclose(tr, 1.0, atol=tol, rtol=0.0)

    @pytest.mark.parametrize("mag_alpha", MAG_ALPHAS)
    @pytest.mark.parametrize("phase_alpha", PHASE_ALPHAS)
    def test_normalized_coherent_state(
        self, setup_backend, mag_alpha, phase_alpha, tol
    ):
        """Tests if a range of coherent states are normalized."""

        alpha = mag_alpha * np.exp(1j * phase_alpha)
        backend = setup_backend(1)

        backend.prepare_coherent_state(alpha, 0)
        state = backend.state()
        tr = state.trace()
        assert np.allclose(tr, 1.0, atol=tol, rtol=0.0)

    def test_prepare_ket_state(self, setup_backend, cutoff, tol):
        """Tests if a ket state with arbitrary parameters is correctly prepared."""
        np.random.seed(SEED)
        random_ket = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(
            -1, 1, cutoff
        )
        random_ket = random_ket / np.linalg.norm(random_ket)
        backend = setup_backend(1)

        backend.prepare_ket_state(random_ket, 0)
        state = backend.state()
        assert np.allclose(state.fidelity(random_ket, 0), 1.0, atol=tol, rtol=0.0)

    def test_prepare_batched_ket_state(
        self, setup_backend, pure, batch_size, cutoff, tol
    ):
        """Tests if a batch of ket states with arbitrary parameters is correctly
        prepared by comparing the fock probabilities of the batched case with
        individual runs with non batched input states."""

        if batch_size is None:
            pytest.skip("Test skipped if no batching")

        np.random.seed(SEED)
        random_kets = np.array(
            [
                (lambda ket: ket / np.linalg.norm(ket))(
                    np.random.uniform(-1, 1, cutoff)
                    + 1j * np.random.uniform(-1, 1, cutoff)
                )
                for _ in range(batch_size)
            ]
        )
        backend = setup_backend(1)

        backend.prepare_ket_state(random_kets, 0)
        state = backend.state()
        batched_probs = np.array(state.all_fock_probs())

        individual_probs = []
        for random_ket in random_kets:
            backend.reset(pure=pure)
            backend.prepare_ket_state(random_ket, 0)
            state = backend.state()
            probs_for_this_ket = np.array(state.all_fock_probs())
            individual_probs.append(probs_for_this_ket[0])

        individual_probs = np.array(individual_probs)
        assert np.allclose(batched_probs, individual_probs, atol=tol, rtol=0.0)

    def test_prepare_rank_two_dm_state(self, setup_backend, cutoff, tol):
        """Tests if rank two dm states with arbitrary parameters are correctly prepared."""

        np.random.seed(SEED)
        random_ket1 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(
            -1, 1, cutoff
        )
        random_ket1 = random_ket1 / np.linalg.norm(random_ket1)
        random_ket2 = np.random.uniform(-1, 1, cutoff) + 1j * np.random.uniform(
            -1, 1, cutoff
        )
        random_ket2 = random_ket2 / np.linalg.norm(random_ket2)

        backend = setup_backend(1)
        backend.prepare_ket_state(random_ket1, 0)
        state = backend.state()
        ket_probs1 = np.array([state.fock_prob([n]) for n in range(cutoff)])

        backend = setup_backend(1)
        backend.prepare_ket_state(random_ket2, 0)
        state = backend.state()
        ket_probs2 = np.array([state.fock_prob([n]) for n in range(cutoff)])

        ket_probs = 0.2 * ket_probs1 + 0.8 * ket_probs2

        random_rho = 0.2 * np.outer(np.conj(random_ket1), random_ket1) + 0.8 * np.outer(
            np.conj(random_ket2), random_ket2
        )

        backend = setup_backend(1)
        backend.prepare_dm_state(random_rho, 0)
        state = backend.state()
        rho_probs = np.array([state.fock_prob([n]) for n in range(cutoff)])

        assert np.allclose(state.trace(), 1.0, atol=tol, rtol=0.0)
        assert np.allclose(rho_probs, ket_probs, atol=tol, rtol=0.0)

    def test_prepare_random_dm_state(
        self, setup_backend, batch_size, pure, cutoff, tol
    ):
        """Tests if a random dm state is correctly prepared."""

        np.random.seed(SEED)
        random_rho = np.random.normal(size=[cutoff, cutoff]) + 1j * np.random.normal(
            size=[cutoff, cutoff]
        )
        random_rho = np.dot(random_rho.conj().T, random_rho)
        random_rho = random_rho / random_rho.trace()

        backend = setup_backend(1)
        backend.prepare_dm_state(random_rho, 0)
        state = backend.state()
        rho_probs = np.array(state.all_fock_probs())

        es, vs = np.linalg.eig(random_rho)

        if batch_size is not None:
            kets_mixed_probs = np.zeros([batch_size, len(es)], dtype=complex)
        else:
            kets_mixed_probs = np.zeros([len(es)], dtype=complex)

        for e, v in zip(es, vs.T.conj()):
            backend.reset(pure=pure)
            backend.prepare_ket_state(v, 0)
            state = backend.state()
            probs_for_this_v = np.array(state.all_fock_probs())
            kets_mixed_probs += e * probs_for_this_v

        assert np.allclose(rho_probs, kets_mixed_probs, atol=tol, rtol=0.0)

    @pytest.mark.parametrize("theta", PHASE_ALPHAS)
    def test_rotated_superposition_states(
        self, setup_backend, theta, pure, cutoff, tol
    ):
        r"""Tests if a range of phase-shifted superposition states are equal to the form of
        \sum_n exp(i * theta * n)|n>"""

        backend = setup_backend(1)

        ref_state = np.array([np.exp(1j * theta * k) for k in range(cutoff)]) / np.sqrt(
            cutoff
        )

        if not pure:
            ref_state = np.outer(ref_state, np.conj(ref_state))

        backend.prepare_ket_state(ref_state, 0)
        s = backend.state()

        if s.is_pure:
            numer_state = s.ket()
        else:
            numer_state = s.dm()

        assert np.allclose(numer_state, ref_state, atol=tol, rtol=0.0)

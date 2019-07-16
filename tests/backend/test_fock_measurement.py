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

r"""Unit tests for measurements in the Fock basis"""
import pytest

import numpy as np


NUM_REPEATS = 50


@pytest.mark.backends("gaussian")
class TestGaussianRepresentation:
    """Tests that make use of the Fock basis representation."""

    def test_measure_fock_gaussian_warning(self, setup_backend):
        """Tests that Fock measurements are not implemented when shots != 1.
        Should be deleted when this functionality is implemented."""

        backend = setup_backend(3)

        with pytest.warns(Warning, match="Gaussian backend cannot simulate non-Gaussian states. Conditional state after "
                                         "Fock measurement has not been updated."):
            backend.measure_fock([0, 1], shots=5)


@pytest.mark.backends("fock", "tf")
class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    def test_normalized_conditional_states(self, setup_backend, cutoff, pure, tol):
        """Tests if the conditional states resulting from Fock measurements in a subset of modes are normalized."""
        state_preps = [n for n in range(cutoff)] + [
            cutoff - n for n in range(cutoff)
        ]  # [0, 1, 2, ..., cutoff-1, cutoff, cutoff-1, ..., 2, 1]
        backend = setup_backend(3)

        for idx in range(NUM_REPEATS):
            backend.reset(pure=pure)

            # cycles through consecutive triples in `state_preps`
            backend.prepare_fock_state(state_preps[idx % cutoff], 0)
            backend.prepare_fock_state(state_preps[(idx + 1) % cutoff], 1)
            backend.prepare_fock_state(state_preps[(idx + 2) % cutoff], 2)

            for mode in range(3):
                backend.measure_fock([mode])
                state = backend.state()
                tr = state.trace()
                assert np.allclose(tr, 1, atol=tol, rtol=0)

    def test_fock_measurements(self, setup_backend, cutoff, batch_size, pure):
        """Tests if Fock measurements results on a variety of multi-mode Fock states are correct."""
        state_preps = [n for n in range(cutoff)] + [
            cutoff - n for n in range(cutoff)
        ]  # [0, 1, 2, ..., cutoff-1, cutoff, cutoff-1, ..., 2, 1]

        singletons = [(0,), (1,), (2,)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        triples = [(0, 1, 2)]
        mode_choices = singletons + pairs + triples

        backend = setup_backend(3)

        for idx in range(NUM_REPEATS):
            backend.reset(pure=pure)

            n = [
                state_preps[idx % cutoff],
                state_preps[(idx + 1) % cutoff],
                state_preps[(idx + 2) % cutoff],
            ]
            n = np.array(n)
            meas_modes = np.array(
                mode_choices[idx % len(mode_choices)]
            )  # cycle through mode choices

            backend.prepare_fock_state(n[0], 0)
            backend.prepare_fock_state(n[1], 1)
            backend.prepare_fock_state(n[2], 2)

            meas_result = backend.measure_fock(meas_modes)  # shape (modes, [batch_size,] shots)
            print("measuring modes {}".format(meas_modes))
            print("measurement results are {}".format(meas_result))

            ref_result = n[meas_modes]  # shape (modes,)
            ref_result = np.expand_dims(ref_result, 1)  # account for shots axis
            if batch_size is not None:
                ref_result = np.expand_dims(ref_result, 1)

            assert np.all(meas_result == ref_result)


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    @pytest.mark.backends("tf")
    def test_shots_not_implemented_fock(self, setup_backend):
        """Tests that Fock measurements are not implemented when shots != 1.
        Should be deleted when this functionality is implemented."""

        backend = setup_backend(3)

        with pytest.raises(NotImplementedError, match="TF backend currently does not support "
                                                      "shots != 1 for Fock measurement"):
            backend.measure_fock([0, 1], shots=5)

    def test_vacuum_fock(self, setup_backend, batch_size):
        """Tests the Fock measurement with shots==1."""

        if batch_size is None:
            shape = (3, 1)
        else:
            shape = (3, batch_size, 1)

        backend = setup_backend(3)
        res = backend.measure_fock([0, 1, 2])
        assert isinstance(res, np.ndarray)
        assert res.shape == shape
        assert np.all(res == 0)

    @pytest.mark.backends('fock', 'gaussian')
    def test_vacuum_multishot_fock(self, setup_backend):
        """Tests the Fock measurement with shots > 1."""
        backend = setup_backend(3)
        res = backend.measure_fock([0, 1], shots=3)
        assert isinstance(res, np.ndarray)
        assert res.shape == (2, 3)
        assert np.all(res == 0)

    def test_two_mode_squeezed_measurements(self, setup_backend, pure):
        """Tests Fock measurement on the two mode squeezed vacuum state."""
        for _ in range(NUM_REPEATS):
            backend = setup_backend(2)
            backend.reset(pure=pure)

            r = 0.25
            # Circuit to prepare two mode squeezed vacuum
            backend.squeeze(-r, 0)
            backend.squeeze(r, 1)
            backend.beamsplitter(np.sqrt(0.5), -np.sqrt(0.5), 0, 1)
            meas_modes = [0, 1]
            meas_results = backend.measure_fock(meas_modes)
            assert np.all(meas_results[0] == meas_results[1])

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

r"""Unit tests for add_mode and del_mode functions"""
import pytest

import numpy as np

NUM_REPEATS = 10


class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    def test_add_mode_vacuum(self, setup_backend, tol):
        """Tests if added modes are initialized to the vacuum state."""
        backend = setup_backend(1)

        for _n in range(4):
            backend.add_mode(1)
            assert np.all(backend.is_vacuum(tol))

    def test_del_mode_vacuum(self, setup_backend, tol):
        """Tests if reduced density matrix is in vacuum after deleting some modes."""
        backend = setup_backend(4)

        for n in range(4):
            backend.del_mode([n])
            assert np.all(backend.is_vacuum(tol))

    def test_get_modes(self, setup_backend):
        """Tests that get modes returns the correct result after deleting modes from the circuit"""
        backend = setup_backend(4)

        backend.squeeze(0.1, 0)
        backend.del_mode([0, 2])

        res = backend.get_modes()
        assert np.all(res == [1, 3])


@pytest.mark.backends("fock", "tf")
class TestFockRepresentation:
    """Tests that make use of the Fock basis representation."""

    def test_normalized_add_mode(self, setup_backend, tol):
        """Tests if a state is normalized after adding modes."""
        backend = setup_backend(1)

        for num_subsystems in range(3):
            backend.add_mode(num_subsystems)
            state = backend.state()
            tr = state.trace()
            assert np.allclose(tr, 1.0, atol=tol, rtol=0.0)

    def test_normalized_del_mode(self, setup_backend, tol):
        """Tests if a state is normalized after deleting modes."""
        backend = setup_backend(4)

        for n in range(4):
            backend.del_mode(n)
            state = backend.state()
            tr = state.trace()
            assert np.allclose(tr, 1.0, atol=tol, rtol=0.0)

    def test_fock_measurements_after_add_mode(self, setup_backend, pure, cutoff):
        """Tests Fock measurements on a system after adding vacuum modes."""
        backend = setup_backend(1)

        for m in range(3):
            meas_results = []

            for _ in range(NUM_REPEATS):
                backend.reset(pure=pure)
                backend.prepare_fock_state(cutoff - 1, 0)
                backend.add_mode(m)

                meas_result = backend.measure_fock([0])
                meas_results.append(meas_result)

            assert np.all(np.array(meas_results) == cutoff - 1)

    def test_fock_measurements_after_del_mode(self, setup_backend, pure, cutoff):
        """Tests Fock measurements on a system after tracing out an unentagled mode."""
        backend = setup_backend(4)

        for m in range(1, 4):
            meas_results = []

            for _ in range(NUM_REPEATS):
                backend.reset(pure=pure)
                backend.prepare_fock_state(cutoff - 1, 0)
                backend.del_mode(m)

                meas_result = backend.measure_fock([0])
                meas_results.append(meas_result)

            assert np.all(np.array(meas_results) == cutoff - 1)

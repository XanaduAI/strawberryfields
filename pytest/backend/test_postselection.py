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
r"""tests for backend postselection"""
import pytest

import numpy as np


def test_homodyne(setup_backend, tol):
    """Test that homodyne detection on a TMS state
    returns post-selected value."""
    x = 0.2
    r = 5

    backend = setup_backend(2)
    backend.squeeze(r, 0)
    backend.squeeze(-r, 1)
    backend.beamsplitter(1 / np.sqrt(2), 1 / np.sqrt(2), 0, 1)

    res = backend.measure_homodyne(0, mode=0, select=0.2)
    assert np.allclose(res, x, atol=tol, rtol=0)


@pytest.mark.backends("gaussian")
def test_heterodyne(setup_backend, tol):
    """Test that heterodyne detection on a TMS state
    returns post-selected value."""
    alpha = 0.43 - 0.12j
    r = 5

    backend = setup_backend(2)
    backend.squeeze(r, 0)
    backend.squeeze(-r, 1)
    backend.beamsplitter(1 / np.sqrt(2), 1 / np.sqrt(2), 0, 1)

    res = backend.measure_heterodyne(mode=0, select=alpha)
    assert np.allclose(res, alpha, atol=tol, rtol=0)


@pytest.mark.backends("fock", "tf")
def test_measure_fock(setup_backend, cutoff, batch_size):
    """Test that Fock post-selection on Fock states
    exiting one arm of a beamsplitter results in conservation
    of photon number in the other. """
    backend = setup_backend(2)

    for n in range(cutoff - 1):
        total_photons = cutoff - 1

        backend.prepare_fock_state(n, 0)
        backend.prepare_fock_state(total_photons - n, 1)
        backend.beamsplitter(1 / np.sqrt(2), 1 / np.sqrt(2), 0, 1)
        res1 = backend.measure_fock([0], select=[cutoff // 2])
        res2 = backend.measure_fock([1])

        photons_out = sum(res1 + res2)

        if batch_size is not None:
            total_photons = np.tile(total_photons, batch_size)

        assert np.all(photons_out == total_photons)

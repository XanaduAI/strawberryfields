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
r"""Integration tests for frontend post-selection"""
import pytest

import numpy as np

from strawberryfields import ops


def test_homodyne(setup_eng, tol):
    """Test that homodyne detection on a TMS state
    returns post-selected value."""
    x = 0.2
    r = 5

    eng, q = setup_eng(2)

    with eng:
        ops.S2gate(r) | q
        ops.MeasureHomodyne(0, select=x) | q[0]

    eng.run()
    assert np.allclose(q[0].val, x, atol=tol, rtol=0)


@pytest.mark.backends("gaussian")
def test_heterodyne(setup_eng, tol):
    """Test that heterodyne detection on a TMS state
    returns post-selected value."""
    alpha = 0.43 - 0.12j
    r = 5

    eng, q = setup_eng(2)

    with eng:
        ops.S2gate(r) | q
        ops.MeasureHeterodyne(select=alpha) | q[0]

    eng.run()
    assert np.allclose(q[0].val, alpha, atol=tol, rtol=0)


@pytest.mark.backends("fock", "tf")
def test_measure_fock(setup_eng, cutoff, batch_size):
    """Test that Fock post-selection on Fock states
    exiting one arm of a beamsplitter results in conservation
    of photon number in the other. """
    eng, q = setup_eng(2)

    for n in range(cutoff - 1):
        total_photons = cutoff-1

        with eng:
            ops.Fock(n) | q[0]
            ops.Fock(total_photons-n) | q[1]
            ops.BSgate() | q
            ops.MeasureFock(select=cutoff//2) | q[0]
            ops.MeasureFock() | q[1]

        eng.run()
        photons_out = sum([i.val for i in q])

        if batch_size is not None:
            total_photons = np.tile(total_photons, batch_size)

        assert np.all(photons_out == total_photons)

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
r"""
Unit tests for two-mode squeezing operation
Convention: The squeezing unitary is fixed to be
U(z) = \exp(0.5 (z^* \hat{a}_1\hat{a}_2 - z (\hat{a^\dagger}_1\hat{a}_2)))
where \hat{a} is the photon annihilation operator.
"""
# pylint: disable=too-many-arguments
import itertools as it

import pytest
import numpy as np


PHASE = np.linspace(0, 2 * np.pi, 3, endpoint=False)
MAG = np.linspace(0.0, 0.2, 5, endpoint=False)
MODES = list(it.combinations(range(4), 2))


def get_amplitude(k, r, p):
    """Get amplitude for two mode squeezing operation

    Args:
        k (tuple): fock state numbers for two modes
        r (float): two-mode squeezing magnitude
        p (float): two-mode squeezing phase

    Returns:
        float: the two-mode squeezed vacuum amplitude
    """
    if k[0] == k[1]:
        tmsv = (np.exp(1j * p) * np.tanh(r)) ** k[0] / np.cosh(r)
        return tmsv
    return 0.0


@pytest.mark.backends("fock")
class TestTwomodeSqueezing:
    """Tests two-mode squeezing operation"""

    @pytest.mark.parametrize("r", MAG)
    @pytest.mark.parametrize("p", PHASE)
    def test_two_mode_squeezing(self, setup_backend, r, p, cutoff, pure, tol):
        r"""Test two-mode squeezing on vacuum-state for both pure states and
        mixed states with the amplitude given by
                :math:`\delta_{kl} \frac{e^{in\phi} \tanh^n{r}}{\cosh{r}}`
        """

        backend = setup_backend(2)
        backend.two_mode_squeeze(r, p, 0, 1)

        state = backend.state()

        if pure:
            for k in it.product(range(cutoff), repeat=2):
                tmsv = get_amplitude(k, r, p)
                assert np.allclose(state.data[k], tmsv, atol=tol, rtol=0)
        else:
            for k in it.product(range(cutoff), repeat=2):
                for l in it.product(range(cutoff), repeat=2):
                    t = (k[0], l[0], k[1], l[1])
                    tmsv2 = get_amplitude(k, r, p) * np.conj(get_amplitude(l, r, p))

                    assert np.allclose(state.data[t], tmsv2, atol=tol, rtol=0)

    @pytest.mark.parametrize("r", MAG)
    @pytest.mark.parametrize("p", PHASE)
    @pytest.mark.parametrize("modes", MODES)
    def test_squeezing_on_mode_subset(self, setup_backend, r, p, modes, cutoff, pure, tol):
        r"""Test two-mode squeezing on vacuum-state for both pure states and
        mixed states on different mode subsets with the amplitude given by
                :math:`\delta_{kl} \frac{e^{in\phi} \tanh^n{r}}{\cosh{r}}`
        """

        backend = setup_backend(4)
        backend.two_mode_squeeze(r, p, *modes)

        state = backend.state()

        state_data = state.reduced_dm(list(modes))

        for k in it.product(range(cutoff), repeat=2):
            for l in it.product(range(cutoff), repeat=2):
                t = (k[0], l[0], k[1], l[1])
                tmsv2 = get_amplitude(k, r, p) * np.conj(get_amplitude(l, r, p))

                assert np.allclose(state_data[t], tmsv2, atol=tol, rtol=0)

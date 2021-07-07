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
r"""Unit tests for space_unroll in tdmprogram.py"""

import pytest
import numpy as np
import strawberryfields as sf
from strawberryfields.tdm.tdmprogram import get_mode_indices
from strawberryfields.ops import Sgate, Rgate, BSgate, LossChannel, MeasureFock


def generate_valid_bs_sequence(delays, modes, random_func=np.random.rand):
    """
    Generate sequences of valied beamsplitter angles for a time-domain program where
    the modes inside the delays loops are in vacuum.

    Args:
        delays (list): list containing the number of modes each loop can support
        modes (list): number of computational modes to be entangled
        random_func (function): function to draw samples from

    Returns:
        array: array containing a valid sequency of beamsplitter that will not entangle the computational modes
            with the mode inhabiting the loops at the beginning of the sequence
    """
    angles = random_func(len(delays), modes + sum(delays))
    sum_delay = 0
    for i, delay in enumerate(delays):
        angles[i, modes + sum_delay :] = 0
        sum_delay += delay
        angles[i, :sum_delay] = 0
    return angles


def generate_valid_r_sequence(delays, modes, random_func=np.random.rand):
    """
    Generate sequences of valid rotation angles for a time-domain program where
    the modes inside the delays loops are in vacuum.

    Args:
        delays (list): list containing the number of modes each loop can support
        modes (list): number of computational modes to be entangled
        random_func (function): function to draw samples from

    Returns:
        array: array containing a valid sequency of rotation angles that will not act like the identity in the modes
            inhabiting the loops at the beginning of the sequence
    """
    angles = random_func(len(delays), modes + sum(delays))
    sum_delay = delays[0]
    angles[0, modes:] = 0
    if len(delays) > 1:
        for i, delay in enumerate(delays[1:]):
            angles[i + 1, :sum_delay] = 0
            angles[i + 1, sum_delay + modes :] = 0
            sum_delay += delay
    return angles


def is_permutation_matrix(x):
    """Checks whether a matrix is a permutation matrix"""
    x = np.asanyarray(x)
    return (
        x.ndim == 2
        and x.shape[0] == x.shape[1]
        and (x.sum(axis=0) == 1).all()
        and (x.sum(axis=1) == 1).all()
        and ((x == 1) | (x == 0)).all()
    )


@pytest.mark.parametrize("delays", [np.random.randint(low=1, high=10, size=i) for i in range(2, 6)])
@pytest.mark.parametrize("modes", [10, 50, 100])
def test_lossless_no_mixing_no_rotation_U(delays, modes):
    """Test that the U matrix is the identity if there is no beamsplitter mixing and no rotations"""
    delays = list(delays)
    angles = np.zeros([2 * len(delays), modes + sum(delays)])
    d = len(delays)
    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram([N], delays=delays)
    vac_modes = sum(delays)
    with prog.context(*angles) as (p, q):
        for i in range(d):
            Rgate(p[i + d]) | q[n[i] + vac_modes]
            BSgate(p[i], np.pi / 2) | (q[n[i + 1] + vac_modes], q[n[i] + vac_modes])
    prog.space_unroll(1)  # Not sure what is this
    compiled = prog.compile(compiler="passive")
    passive_elem = compiled.circuit[0]
    U = passive_elem.op.p[0]
    # Check that it is indeed the identity
    assert np.allclose(U, np.identity(len(U)))


@pytest.mark.parametrize("delays", [np.array([1,10])])#[np.random.randint(low=1, high=10, size=i) for i in range(2, 6)])
@pytest.mark.parametrize("modes", [100])#[70, 80, 100])
def test_no_entanglement_between_padding_and_computational_modes(delays, modes):
    """Test that the U matrix is the identity if there is no beamsplitter mixing and no rotations"""
    delays = list(delays)
    angles = np.concatenate([generate_valid_bs_sequence(delays,modes), generate_valid_r_sequence(delays,modes)])
    print(angles.shape)
    d = len(delays)
    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram([N], delays=delays)
    vac_modes = sum(delays)
    print("N=",N)
    with prog.context(*angles) as (p, q):
        for i in range(d):
            print(i, i+d, n[i], vac_modes)
            Rgate(p[i + d]) | q[n[i] + vac_modes]
            BSgate(p[i], np.pi / 2) | (q[n[i + 1] + vac_modes], q[n[i] + vac_modes])
    prog.space_unroll(1)  # Not sure what is this
    compiled = prog.compile(compiler="passive")
    passive_elem = compiled.circuit[0]
    U = passive_elem.op.p[0]
    print(U.shape, vac_modes + modes)
    # Check that it is indeed the identity
    U_AA = U[:vac_modes,:vac_modes]
    U_AB = U[vac_modes:,:vac_modes]
    U_BA = U[:vac_modes,vac_modes:]
    U_BB = U[vac_modes:,vac_modes:]
    #assert np.allclose(U @ U.T.conj(), np.identity(len(U)))
    assert np.allclose(U_AA, np.identity(vac_modes))
    assert np.allclose(U_AB, 0)
    assert np.allclose(U_BA, 0)
    assert np.allclose(U_BB @ U_BB.T.conj(), np.identity(len(U_BB)))
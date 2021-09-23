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
from thewalrus.symplectic import reduced_state
from thewalrus.quantum import is_pure_cov

pytestmark = pytest.mark.frontend


def generate_valid_bs_sequence(delays, modes, random_func=np.random.rand):
    """
    Generate sequences of valid beamsplitter angles for a time-domain program where
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


@pytest.mark.parametrize("delays", [np.random.randint(low=1, high=10, size=i) for i in range(2, 6)])
@pytest.mark.parametrize("modes", [10, 50, 100])
def test_lossless_no_mixing_no_rotation_U(delays, modes):
    """Test that the U matrix is the identity if there is no beamsplitter mixing and no rotations"""
    delays = list(delays)
    angles = np.zeros([2 * len(delays), modes + sum(delays)])
    d = len(delays)
    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram([N])

    with prog.context(*angles) as (p, q):
        for i in range(d):
            Rgate(p[i + d]) | q[n[i]]
            BSgate(p[i], np.pi / 2) | (q[n[i + 1]], q[n[i]])

    prog.space_unroll()

    compiled = prog.compile(compiler="passive")
    passive_elem = compiled.circuit[0]
    U = passive_elem.op.p[0]
    # Check that it is indeed the identity
    assert np.allclose(U, np.identity(len(U)))


@pytest.mark.parametrize("delays", [np.random.randint(low=1, high=10, size=i) for i in range(2, 6)])
@pytest.mark.parametrize("modes", [70, 80, 100])
def test_no_entanglement_between_padding_and_computational_modes(delays, modes):
    """Test that the U matrix is the identity if there is no beamsplitter mixing and no rotations"""
    delays = list(delays)
    angles = np.concatenate(
        [generate_valid_bs_sequence(delays, modes), generate_valid_r_sequence(delays, modes)]
    )
    d = len(delays)
    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram([N])
    vac_modes = sum(delays)

    with prog.context(*angles) as (p, q):
        for i in range(d):
            Rgate(p[i + d]) | q[n[i]]
            BSgate(p[i], np.pi / 2) | (q[n[i + 1]], q[n[i]])

    prog.space_unroll()

    compiled = prog.compile(compiler="passive")
    passive_elem = compiled.circuit[0]
    U = passive_elem.op.p[0]
    # Check that it is indeed the identity
    U_AA = U[:vac_modes, :vac_modes]
    U_AB = U[vac_modes:, :vac_modes]
    U_BA = U[:vac_modes, vac_modes:]
    U_BB = U[vac_modes:, vac_modes:]

    assert np.allclose(U_AA, np.identity(vac_modes))
    assert np.allclose(U_AB, 0)
    assert np.allclose(U_BA, 0)
    assert np.allclose(U_BB @ U_BB.T.conj(), np.identity(len(U_BB)))


@pytest.mark.parametrize("delays", [np.random.randint(low=1, high=10, size=i) for i in range(2, 3)])
@pytest.mark.parametrize("modes", [20])
def test_is_permutation_when_angle_pi_on_two(delays, modes):
    """Checks that if all the beamsplitters are cross then the absolute value output matrix is a permutation matrix"""
    delays = list(delays)
    net = modes + sum(delays)
    angles = np.concatenate(
        [generate_valid_bs_sequence(delays, modes), generate_valid_r_sequence(delays, modes)]
    )
    angles[0] = np.pi / 2 * np.random.randint(2, size=net)
    angles[1] = np.pi / 2 * np.random.randint(2, size=net)
    angles[2] = np.pi / 2 * np.random.randint(2, size=net)
    d = len(delays)
    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram([N])
    vac_modes = sum(delays)

    with prog.context(*angles) as (p, q):
        for i in range(d):
            Rgate(p[i + d]) | q[n[i]]
            BSgate(p[i], np.pi / 2) | (q[n[i + 1]], q[n[i]])

    prog.space_unroll()

    compiled = prog.compile(compiler="passive")
    passive_elem = compiled.circuit[0]
    U = passive_elem.op.p[0]
    assert np.allclose(U @ U.T.conj(), np.identity(len(U)))
    assert np.allclose(list(map(max, np.abs(U))), 1.0)


def test_cov_is_pure():
    """Tests space unrolling when going into the Gaussian backend"""
    delays = [1, 6, 36]
    modes = 216
    angles = np.concatenate(
        [generate_valid_bs_sequence(delays, modes), generate_valid_r_sequence(delays, modes)]
    )
    net = modes + sum(delays)
    d = len(delays)
    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram([N])
    vac_modes = sum(delays)

    with prog.context(*angles) as (p, q):
        Sgate(0.8) | q[n[0]]
        for i in range(d):
            Rgate(p[i + d]) | q[n[i]]
            BSgate(p[i], np.pi / 2) | (q[n[i + 1]], q[n[i]])

    prog.space_unroll()

    eng = sf.Engine(backend="gaussian")
    results = eng.run(prog)
    cov = results.state.cov()
    mu = np.zeros(len(cov))
    mu_vac, cov_vac = reduced_state(mu, cov, list(range(vac_modes)))
    mu_comp, cov_comp = reduced_state(mu, cov, list(range(vac_modes, net)))
    assert np.allclose(cov_vac, 0.5 * (sf.hbar) * np.identity(2 * vac_modes))
    assert is_pure_cov(cov_comp, hbar=sf.hbar)


def test_space_unrolling():
    """Tests that space-unrolling works and that it can be done twice"""
    delays = [1, 6, 36]
    modes = 216
    angles = np.concatenate(
        [generate_valid_bs_sequence(delays, modes), generate_valid_r_sequence(delays, modes)]
    )

    d = len(delays)
    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram([N])

    with prog.context(*angles) as (p, q):
        Sgate(0.8) | q[n[0]]
        for i in range(d):
            Rgate(p[i + d]) | q[n[i]]
            BSgate(p[i], np.pi / 2) | (q[n[i + 1]], q[n[i]])

    assert prog._is_space_unrolled == False

    prog.space_unroll()

    assert prog.timebins == 259
    assert prog.num_subsystems == 259

    # check that the number of gates are correct.
    assert sum([isinstance(cmd.op, Sgate) for cmd in prog.circuit]) == 216
    assert (
        sum([isinstance(cmd.op, Rgate) for cmd in prog.circuit]) == 259 - 43 + 259 - 42 + 259 - 36
    )
    assert (
        sum([isinstance(cmd.op, BSgate) for cmd in prog.circuit]) == 259 - 43 + 259 - 42 + 259 - 36
    )

    prog.space_unroll()

    # space-unroll the program twice to check that it works
    assert prog._is_space_unrolled == True


def test_rolling_space_unrolled():
    """Tests that rolling a space-unrolled circuit works"""
    delays = [1, 6, 36]
    modes = 216
    angles = np.concatenate(
        [generate_valid_bs_sequence(delays, modes), generate_valid_r_sequence(delays, modes)]
    )

    d = len(delays)
    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram([N])

    with prog.context(*angles) as (p, q):
        Sgate(0.8) | q[n[0]]
        for i in range(d):
            Rgate(p[i + d]) | q[n[i]]
            BSgate(p[i], np.pi / 2) | (q[n[i + 1]], q[n[i]])

    rolled_circuit = prog.circuit.copy()
    num_subsystems_pre_roll = prog.num_subsystems
    init_num_subsystems_pre_roll = prog.init_num_subsystems

    assert prog._is_space_unrolled == False

    # space-unroll the program
    prog.space_unroll()

    assert prog._is_space_unrolled == True

    # roll the program back up
    prog.roll()

    assert prog._is_space_unrolled == False
    assert prog.num_subsystems == num_subsystems_pre_roll
    assert prog.init_num_subsystems == init_num_subsystems_pre_roll

    assert len(prog.circuit) == len(rolled_circuit)
    assert prog.circuit == rolled_circuit

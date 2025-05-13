# Copyright 2020-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Unit tests for tdm/program.py"""
from collections.abc import Iterable

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.tdm import (TDMProgram, get_mode_indices, move_vac_modes,
                                  random_bs, random_r, to_args_list,
                                  vacuum_padding)

pytestmark = pytest.mark.frontend


def singleloop(r, alpha, phi, theta, shots, shift="default"):
    """Single-loop program.

    Args:
        r (float): squeezing parameter
        alpha (Sequence[float]): beamsplitter angles
        phi (Sequence[float]): rotation angles
        theta (Sequence[float]): homodyne measurement angles
        shots (int): number of shots
        shift (string): type of shift used in the program
    Returns:
        (list): homodyne samples from the single loop simulation
    """
    prog = TDMProgram(N=2)
    with prog.context(alpha, phi, theta, shift=shift) as (p, q):
        ops.Sgate(r, 0) | q[1]
        ops.BSgate(p[0]) | (q[0], q[1])
        ops.Rgate(p[1]) | q[1]
        ops.MeasureHomodyne(p[2]) | q[0]
    eng = sf.Engine("gaussian")
    result = eng.run(prog, shots=shots)

    return result.samples


def multiple_loops_program(delays, gate_args):
    """Multiple loops program. This function automatically creates
    a multiple loop program given a list of delays and the corresponding
    gate arguments.

    Args:
        delays (list): a list with the given delay for each loop in the program
        gate_args (list[array]): a list containing the parameters of the gates
    """
    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram(N)
    with prog.context(*gate_args) as (p, q):
        ops.Sgate(p[0]) | q[n[0]]
        for i in range(len(delays)):
            ops.Rgate(p[2 * i + 1]) | q[n[i]]
            ops.BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i + 1]], q[n[i]])
        ops.MeasureX | q[0]
    return prog


def random_gate_args(delays, num_modes):
    """Helper function to return random gate args with padded with 0 for vacuum modes."""
    random_gate_args = {
        "Sgate": np.random.uniform(low=0, high=1, size=num_modes).tolist(),
        "loops": {},
    }
    for idx in range(len(delays)):
        random_gate_args["loops"][idx] = {
            "Rgate": random_r(length=num_modes),
            "BSgate": random_bs(length=num_modes),
        }

    # set the initial cross states of each BS
    for idx in range(len(delays)):
        random_gate_args["loops"][idx]["BSgate"][: delays[idx]] = [0] * delays[idx]

    padded_gate_args = vacuum_padding(gate_args=random_gate_args, delays=delays)
    return to_args_list(padded_gate_args), padded_gate_args["crop"]


class TestTDMErrorRaising:
    """Test that the correct error messages are raised when a TDMProgram is created"""

    def test_gates_equal_length(self):
        """Checks gate list parameters have same length"""
        sq_r = 1.0
        c = 4
        shots = 10
        alpha = [0, np.pi / 4] * c
        phi = [np.pi / 2, 0] * c
        theta = [0, 0] + [0, np.pi / 2] + [np.pi / 2, 0] + [np.pi / 2]
        with pytest.raises(ValueError, match="Gate-parameter lists must be of equal length."):
            singleloop(sq_r, alpha, phi, theta, shots)

    def test_passing_list_of_tdmprograms(self):
        """Test that error is raised when passing a list containing TDM programs"""
        prog = TDMProgram(N=2)
        with prog.context([1, 1], [1, 1], [1, 1]) as (p, q):
            ops.Sgate(0, 0) | q[1]
            ops.BSgate(p[0]) | (q[0], q[1])
            ops.Rgate(p[1]) | q[1]
            ops.MeasureHomodyne(p[2]) | q[0]

        eng = sf.Engine("gaussian")

        with pytest.raises(
            NotImplementedError, match="Lists of TDM programs are not currently supported"
        ):
            eng.run([prog, prog])

    def test_cropping_tdmprogram_with_several_spatial_modes(self):
        """Test that error is raised when cropping samples from a TDM program with more than
        one spatial mode"""

        prog = TDMProgram(N=[1, 2])
        with prog.context([0, np.pi / 2], [0, np.pi / 2]) as (p, q):
            ops.Sgate(4) | q[0]
            ops.Sgate(4) | q[2]
            ops.MeasureHomodyne(p[0]) | q[0]
            ops.MeasureHomodyne(p[1]) | q[1]

        with pytest.raises(
            NotImplementedError,
            match="Cropping vacuum modes for programs with more than one spatial mode is not implemented.",
        ):
            prog.get_crop_value()

    def test_delays_tdmprogram_with_nested_loops(self):
        """Test that error is raised when calculating the delays of a TDM program with nested loops"""

        gate_args = [random_bs(8), random_bs(8)]

        # a program with nested delays
        prog = TDMProgram(N=4)
        with prog.context(*gate_args) as (p, q):
            ops.Sgate(0.4, 0) | q[3]
            ops.BSgate(p[0]) | (q[0], q[3])
            ops.BSgate(p[1]) | (q[2], q[1])
            ops.MeasureX | q[0]

        with pytest.raises(
            NotImplementedError,
            match="Calculating delays for programs with nested loops is not implemented.",
        ):
            prog.get_delays()

    def test_delays_tdmprogram_with_several_spatial_modes(self):
        """Test that error is raised when calculating the delays of a TDM program with more than
        one spatial mode"""

        prog = TDMProgram(N=[1, 2])
        with prog.context([0, np.pi / 2], [0, np.pi / 2]) as (p, q):
            ops.Sgate(4) | q[0]
            ops.Sgate(4) | q[2]
            ops.MeasureHomodyne(p[0]) | q[0]
            ops.MeasureHomodyne(p[1]) | q[1]

        with pytest.raises(
            NotImplementedError,
            match="Calculating delays for programs with more than one spatial mode is not implemented.",
        ):
            prog.get_delays()


def test_shift_by_specified_amount():
    """Checks that shifting by 1 is equivalent to shift='end' for a program
    with one spatial mode"""
    sq_r = 1.0
    shots = 1
    alpha = [0] * 4
    phi = [0] * 4
    theta = [0] * 4
    np.random.seed(42)
    x = singleloop(sq_r, alpha, phi, theta, shots)
    np.random.seed(42)
    y = singleloop(sq_r, alpha, phi, theta, shots, shift=1)
    assert np.allclose(x, y)


def test_str_tdm_method():
    """Testing the string method"""
    prog = TDMProgram(N=1)
    assert prog.__str__() == "<TDMProgram: concurrent modes=1, time bins=0, spatial modes=0>"


def test_single_parameter_list_program():
    """Test that a TDMProgram with a single parameter list works."""
    prog = sf.TDMProgram(2)
    eng = sf.Engine("gaussian")

    with prog.context([1, 2]) as (p, q):
        ops.Sgate(p[0]) | q[0]
        ops.MeasureHomodyne(p[0]) | q[0]

    eng.run(prog)

    assert isinstance(prog.loop_vars, Iterable)
    assert prog.parameters == {"p0": [1, 2]}


@pytest.mark.parametrize("delays", ([3], [2, 4], [8, 5], [1, 6, 36], [6, 1, 4]))
def test_delay_loops(delays):
    """Test that program calculates correctly the delays present in the circuit for a
    TDM program."""

    gate_args, _ = random_gate_args(delays, num_modes=216)

    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram(N=N)
    with prog.context(*gate_args) as (p, q):
        ops.Sgate(p[0]) | q[n[0]]
        for i in range(len(delays)):
            ops.Rgate(p[2 * i + 1]) | q[n[i]]
            ops.BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i + 1]], q[n[i]])
        ops.MeasureFock() | q[0]

    assert prog.get_delays() == delays


@pytest.mark.parametrize("delays", ([3], [2, 4], [8, 5], [1, 6, 36], [6, 1, 4]))
def test_crop_value(delays):
    """Test that program calculates correctly the the number of vacuum modes arriving at the
    detector before the first computational mode"""

    gate_args, expected_crop_value = random_gate_args(delays, num_modes=216)

    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram(N)
    with prog.context(*gate_args) as (p, q):
        ops.Sgate(p[0]) | q[n[0]]
        for i in range(len(delays)):
            ops.Rgate(p[2 * i + 1]) | q[n[i]]
            ops.BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i + 1]], q[n[i]])
        ops.MeasureFock() | q[0]

    assert prog.get_crop_value() == expected_crop_value


class TestSingleLoopNullifier:
    """Groups tests where a nullifier associated with a state generated by a oneloop setup is checked."""

    def test_epr(self):
        """Generates an EPR state and checks that the correct correlations (noise reductions) are observed
        from the samples"""
        vac_modes = 1
        sq_r = 5.0
        c = 2
        shots = 100

        # This will generate c EPRstates per copy. I chose c = 4 because it allows us to make 4 EPR pairs per copy that can each be measured in different basis permutations.
        alpha = [0, np.pi / 4] * c
        phi = [np.pi / 2, 0] * c

        # Measurement of 2 subsequent EPR states in XX, PP to investigate nearest-neighbour correlations in all basis permutations
        theta = [np.pi / 2, 0, 0, np.pi / 2]

        timebins_per_shot = len(alpha)

        samples = singleloop(sq_r, alpha, phi, theta, shots)
        reshaped_samples = move_vac_modes(samples, 2, crop=True)

        X0 = reshaped_samples[:, 0, 0]
        X1 = reshaped_samples[:, 0, 1]
        P2 = reshaped_samples[:, 0, 2]
        P3 = reshaped_samples[:, 0, 3]

        rtol = 5 / np.sqrt(shots)
        minusstdX1X0 = (X1 - X0).var()
        plusstdX1X0 = (X1 + X0).var()
        squeezed_std = np.exp(-2 * sq_r)
        expected_minus = sf.hbar * squeezed_std
        expected_plus = sf.hbar / squeezed_std
        assert np.allclose(minusstdX1X0, expected_minus, rtol=rtol)
        assert np.allclose(plusstdX1X0, expected_plus, rtol=rtol)

        minusstdP2P3 = (P2 - P3).var()
        plusstdP2P3 = (P2 + P3).var()
        assert np.allclose(minusstdP2P3, expected_plus, rtol=rtol)
        assert np.allclose(plusstdP2P3, expected_minus, rtol=rtol)

    def test_ghz(self):
        """Generates a GHZ state and checks that the correct correlations (noise reductions) are observed
        from the samples
        See Eq. 5 of https://advances.sciencemag.org/content/5/5/eaaw4530
        """
        # Set up the circuit
        vac_modes = 1
        n = 4
        shots = 100
        sq_r = 5
        alpha = [np.arccos(np.sqrt(1 / (n - i + 1))) if i != n + 1 else 0 for i in range(n)]
        alpha[0] = 0.0
        phi = [0] * n
        phi[0] = np.pi / 2
        timebins_per_shot = len(alpha)

        # Measuring X nullifier
        theta = [0] * n
        samples_X = singleloop(sq_r, alpha, phi, theta, shots)
        reshaped_samples_X = move_vac_modes(samples_X, 2, crop=True)

        # Measuring P nullifier
        theta = [np.pi / 2] * n
        samples_P = singleloop(sq_r, alpha, phi, theta, shots)
        reshaped_samples_P = move_vac_modes(samples_P, 2, crop=True)

        # We will check that the x of all the modes equal the x of the last one
        nullifier_X = lambda sample: (sample - sample[-1])[:-1]
        val_nullifier_X = np.var([nullifier_X(x[0]) for x in reshaped_samples_X], axis=0)
        assert np.allclose(val_nullifier_X, sf.hbar * np.exp(-2 * sq_r), rtol=5 / np.sqrt(shots))

        # We will check that the sum of all the p is equal to zero
        val_nullifier_P = np.var([np.sum(p[0]) for p in reshaped_samples_P], axis=0)
        assert np.allclose(
            val_nullifier_P, 0.5 * sf.hbar * n * np.exp(-2 * sq_r), rtol=5 / np.sqrt(shots)
        )

    def test_one_dimensional_cluster(self):
        """Test that the nullifier have the correct value in the experiment described in
        See Eq. 10 of https://advances.sciencemag.org/content/5/5/eaaw4530
        """
        vac_modes = 1
        n = 20
        shots = 100
        sq_r = 3
        alpha_c = np.arccos(np.sqrt((np.sqrt(5) - 1) / 2))
        alpha = [alpha_c] * n
        alpha[0] = 0.0
        phi = [np.pi / 2] * n
        theta = [0, np.pi / 2] * (
            n // 2
        )  # Note that we measure x for mode i and the p for mode i+1.
        timebins_per_shot = len(alpha)

        reshaped_samples = singleloop(sq_r, alpha, phi, theta, shots)

        nullifier = lambda x: np.array(
            [-x[i - 2] + x[i - 1] - x[i] for i in range(2, len(x) - 2, 2)]
        )[1:]
        nullifier_samples = np.array([nullifier(y[0]) for y in reshaped_samples])
        delta = np.var(nullifier_samples, axis=0)
        assert np.allclose(delta, 1.5 * sf.hbar * np.exp(-2 * sq_r), rtol=5 / np.sqrt(shots))


def test_one_dimensional_cluster_tokyo():
    """
    One-dimensional temporal-mode cluster state as demonstrated in
    https://aip.scitation.org/doi/pdf/10.1063/1.4962732
    """
    sq_r = 5

    n = 10  # for an n-mode cluster state
    shots = 3

    # first half of cluster state measured in X, second half in P
    theta1 = [0] * int(n / 2) + [np.pi / 2] * int(n / 2)  # measurement angles for detector A
    theta2 = theta1  # measurement angles for detector B

    prog = TDMProgram(N=[1, 2])
    with prog.context(theta1, theta2, shift="default") as (p, q):
        ops.Sgate(sq_r, 0) | q[0]
        ops.Sgate(sq_r, 0) | q[2]
        ops.Rgate(np.pi / 2) | q[0]
        ops.BSgate(np.pi / 4) | (q[0], q[2])
        ops.BSgate(np.pi / 4) | (q[0], q[1])
        ops.MeasureHomodyne(p[0]) | q[0]
        ops.MeasureHomodyne(p[1]) | q[1]
    eng = sf.Engine("gaussian")

    result = eng.run(prog, shots=shots)
    reshaped_samples = result.samples

    for sh in range(shots):
        X_A = reshaped_samples[sh][0][: n // 2]  # X samples from detector A
        P_A = reshaped_samples[sh][0][n // 2 :]  # P samples from detector A
        X_B = reshaped_samples[sh][1][: n // 2]  # X samples from detector B
        P_B = reshaped_samples[sh][1][n // 2 :]  # P samples from detector B

        # nullifiers defined in https://aip.scitation.org/doi/pdf/10.1063/1.4962732, Eqs. (1a) and (1b)
        ntot = len(X_A) - 1
        nX = np.array([X_A[i] + X_B[i] + X_A[i + 1] - X_B[i + 1] for i in range(ntot)])
        nP = np.array([P_A[i] + P_B[i] - P_A[i + 1] + P_B[i + 1] for i in range(ntot)])

        nXvar = np.var(nX)
        nPvar = np.var(nP)

        assert np.allclose(nXvar, 2 * sf.hbar * np.exp(-2 * sq_r), rtol=5 / np.sqrt(n))
        assert np.allclose(nPvar, 2 * sf.hbar * np.exp(-2 * sq_r), rtol=5 / np.sqrt(n))


def test_two_dimensional_cluster_denmark():
    """
    Two-dimensional temporal-mode cluster state as demonstrated in https://arxiv.org/pdf/1906.08709
    """
    sq_r = 3
    delay1 = 1  # number of timebins in the short delay line
    delay2 = 12  # number of timebins in the long delay line
    n = 200  # number of timebins
    shots = 10
    # first half of cluster state measured in X, second half in P

    theta_A = [0] * int(n / 2) + [np.pi / 2] * int(n / 2)  # measurement angles for detector A
    theta_B = theta_A  # measurement angles for detector B

    # 2D cluster
    prog = TDMProgram([1, delay2 + delay1 + 1])
    with prog.context(theta_A, theta_B, shift="default") as (p, q):
        ops.Sgate(sq_r, 0) | q[0]
        ops.Sgate(sq_r, 0) | q[delay2 + delay1 + 1]
        ops.Rgate(np.pi / 2) | q[delay2 + delay1 + 1]
        ops.BSgate(np.pi / 4, np.pi) | (q[delay2 + delay1 + 1], q[0])
        ops.BSgate(np.pi / 4, np.pi) | (q[delay2 + delay1], q[0])
        ops.BSgate(np.pi / 4, np.pi) | (q[delay1], q[0])
        ops.MeasureHomodyne(p[1]) | q[0]
        ops.MeasureHomodyne(p[0]) | q[delay1]
    eng = sf.Engine("gaussian")
    result = eng.run(prog, shots=shots)
    reshaped_samples = result.samples

    for sh in range(shots):
        X_A = reshaped_samples[sh][0][: n // 2]  # X samples from detector A
        P_A = reshaped_samples[sh][0][n // 2 :]  # P samples from detector A
        X_B = reshaped_samples[sh][1][: n // 2]  # X samples from detector B
        P_B = reshaped_samples[sh][1][n // 2 :]  # P samples from detector B

        # nullifiers defined in https://arxiv.org/pdf/1906.08709.pdf, Eqs. (1) and (2)
        N = delay2
        ntot = len(X_A) - delay2 - 1
        nX = np.array(
            [
                X_A[k]
                + X_B[k]
                - X_A[k + 1]
                - X_B[k + 1]
                - X_A[k + N]
                + X_B[k + N]
                - X_A[k + N + 1]
                + X_B[k + N + 1]
                for k in range(ntot)
            ]
        )
        nP = np.array(
            [
                P_A[k]
                + P_B[k]
                + P_A[k + 1]
                + P_B[k + 1]
                - P_A[k + N]
                + P_B[k + N]
                + P_A[k + N + 1]
                - P_B[k + N + 1]
                for k in range(ntot)
            ]
        )
        nXvar = np.var(nX)
        nPvar = np.var(nP)

        assert np.allclose(nXvar, 4 * sf.hbar * np.exp(-2 * sq_r), rtol=5 / np.sqrt(ntot))
        assert np.allclose(nPvar, 4 * sf.hbar * np.exp(-2 * sq_r), rtol=5 / np.sqrt(ntot))


def test_two_dimensional_cluster_tokyo():
    """
    Two-dimensional temporal-mode cluster state as demonstrated by Universtiy of Tokyo. See: https://arxiv.org/pdf/1903.03918.pdf
    """
    # temporal delay in timebins for each spatial mode
    delayA = 0
    delayB = 1
    delayC = 5
    delayD = 0

    # concurrent modes in each spatial mode
    concurrA = 1 + delayA
    concurrB = 1 + delayB
    concurrC = 1 + delayC
    concurrD = 1 + delayD

    N = [concurrA, concurrB, concurrC, concurrD]

    sq_r = 5

    # first half of cluster state measured in X, second half in P
    n = 400  # number of timebins
    theta_A = [0] * int(n / 2) + [np.pi / 2] * int(n / 2)  # measurement angles for detector A
    theta_B = theta_A  # measurement angles for detector B
    theta_C = theta_A
    theta_D = theta_A

    shots = 10

    # 2D cluster
    prog = TDMProgram(N)
    with prog.context(theta_A, theta_B, theta_C, theta_D, shift="default") as (p, q):

        ops.Sgate(sq_r, 0) | q[0]
        ops.Sgate(sq_r, 0) | q[2]
        ops.Sgate(sq_r, 0) | q[8]
        ops.Sgate(sq_r, 0) | q[9]

        ops.Rgate(np.pi / 2) | q[0]
        ops.Rgate(np.pi / 2) | q[8]

        ops.BSgate(np.pi / 4) | (q[0], q[2])
        ops.BSgate(np.pi / 4) | (q[8], q[9])
        ops.BSgate(np.pi / 4) | (q[2], q[8])
        ops.BSgate(np.pi / 4) | (q[0], q[1])
        ops.BSgate(np.pi / 4) | (q[3], q[9])

        ops.MeasureHomodyne(p[0]) | q[0]
        ops.MeasureHomodyne(p[1]) | q[1]
        ops.MeasureHomodyne(p[2]) | q[3]
        ops.MeasureHomodyne(p[3]) | q[9]

    eng = sf.Engine("gaussian")
    result = eng.run(prog, shots=shots)
    reshaped_samples = result.samples

    for sh in range(shots):

        X_A = reshaped_samples[sh][0][: n // 2]  # X samples from detector A
        P_A = reshaped_samples[sh][0][n // 2 :]  # P samples from detector A
        X_B = reshaped_samples[sh][1][: n // 2]  # X samples from detector B
        P_B = reshaped_samples[sh][1][n // 2 :]  # P samples from detector B
        X_C = reshaped_samples[sh][2][: n // 2]  # X samples from detector C
        P_C = reshaped_samples[sh][2][n // 2 :]  # P samples from detector C
        X_D = reshaped_samples[sh][3][: n // 2]  # X samples from detector D
        P_D = reshaped_samples[sh][3][n // 2 :]  # P samples from detector D

        N = delayC
        # nullifiers defined in https://arxiv.org/pdf/1903.03918.pdf, Fig. S5
        ntot = len(X_A) - N - 1
        nX1 = np.array(
            [
                X_A[k]
                + X_B[k]
                - np.sqrt(1 / 2) * (-X_A[k + 1] + X_B[k + 1] + X_C[k + N] + X_D[k + N])
                for k in range(ntot)
            ]
        )
        nX2 = np.array(
            [
                X_C[k]
                - X_D[k]
                - np.sqrt(1 / 2) * (-X_A[k + 1] + X_B[k + 1] - X_C[k + N] - X_D[k + N])
                for k in range(ntot)
            ]
        )
        nP1 = np.array(
            [
                P_A[k]
                + P_B[k]
                + np.sqrt(1 / 2) * (-P_A[k + 1] + P_B[k + 1] + P_C[k + N] + P_D[k + N])
                for k in range(ntot)
            ]
        )
        nP2 = np.array(
            [
                P_C[k]
                - P_D[k]
                + np.sqrt(1 / 2) * (-P_A[k + 1] + P_B[k + 1] - P_C[k + N] - P_D[k + N])
                for k in range(ntot)
            ]
        )

        nX1var = np.var(nX1)
        nX2var = np.var(nX2)
        nP1var = np.var(nP1)
        nP2var = np.var(nP2)

        assert np.allclose(nX1var, 2 * sf.hbar * np.exp(-2 * sq_r), rtol=5 / np.sqrt(ntot))
        assert np.allclose(nX2var, 2 * sf.hbar * np.exp(-2 * sq_r), rtol=5 / np.sqrt(ntot))
        assert np.allclose(nP1var, 2 * sf.hbar * np.exp(-2 * sq_r), rtol=5 / np.sqrt(ntot))
        assert np.allclose(nP2var, 2 * sf.hbar * np.exp(-2 * sq_r), rtol=5 / np.sqrt(ntot))


class TestTDMProgramFunctions:
    """Test functions in the ``tdm.program`` module"""

    @pytest.mark.parametrize(
        "N, crop, expected",
        [
            (
                1,
                False,
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
            ),
            (
                1,
                True,
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
            ),
            (3, False, [[[3, 4], [5, 6]], [[7, 8], [9, 10]], [[11, 12], [0, 0]]]),
            ([1, 4], False, [[[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 0], [0, 0]]]),
            ([4], True, [[[4, 5], [6, 7]], [[8, 9], [10, 11]]]),
        ],
    )
    def test_move_vac_modes(self, N, crop, expected):
        """Test the `move_vac_modes` function"""
        samples = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
        res = move_vac_modes(samples, N, crop=crop)

        assert np.all(res == expected)


class TestEngineTDMProgramInteraction:
    """Test the Engine class and its interaction with TDMProgram instances."""

    def test_shots_default(self):
        """Test that default shots (1) is used"""
        prog = sf.TDMProgram(2)
        eng = sf.Engine("gaussian")

        with prog.context([1, 2], [3, 4]) as (p, q):
            ops.Sgate(p[0]) | q[0]
            ops.MeasureHomodyne(p[1]) | q[0]

        results = eng.run(prog)
        assert len(results.samples) == 1

    @pytest.mark.parametrize("shots,len_samples", [(None, 0), (5, 5)])
    def test_shots_run_options(self, shots, len_samples):
        """Test that run_options takes precedence over default"""
        prog = sf.TDMProgram(2)
        eng = sf.Engine("gaussian")

        with prog.context([1, 2], [3, 4]) as (p, q):
            ops.Sgate(p[0]) | q[0]
            ops.MeasureHomodyne(p[1]) | q[0]

        prog.run_options = {"shots": shots}
        results = eng.run(prog)
        assert len(results.samples) == len_samples

    @pytest.mark.parametrize("shots,len_samples", [(None, 0), (2, 2)])
    def test_shots_passed(self, shots, len_samples):
        """Test that shots supplied via eng.run takes precedence over
        run_options and that run_options isn't changed"""
        prog = sf.TDMProgram(2)
        eng = sf.Engine("gaussian")

        with prog.context([1, 2], [3, 4]) as (p, q):
            ops.Sgate(p[0]) | q[0]
            ops.MeasureHomodyne(p[1]) | q[0]

        prog.run_options = {"shots": 5}
        results = eng.run(prog, shots=shots)
        assert len(results.samples) == len_samples
        assert prog.run_options["shots"] == 5

    def test_shots_with_timebins_non_multiple_of_concurrent_modes(self):
        """Test that multiple shots work when having the number of timebins be
        a non-multiple of the number of concurrent modes"""
        theta = [0] * 3
        shots = 2

        prog = sf.TDMProgram(N=2)
        with prog.context(theta) as (p, q):
            ops.Xgate(50) | q[1]
            ops.MeasureHomodyne(p[0]) | q[0]
        eng = sf.Engine("gaussian")
        res = eng.run(prog, shots=shots)
        samples = res.samples

        expected = np.array([[[0, 50, 50]], [[50, 50, 50]]])

        assert np.allclose(samples, expected, atol=4)

    @pytest.mark.parametrize("delays", ([3], [2, 4], [8, 5], [1, 6, 36], [6, 1, 4]))
    def test_crop_default(self, delays):
        """Test that ``crop=False`` by default implies all samples from the program are returned (including vacuum modes)"""

        computational_modes = 100
        gate_args, vacuum_modes = random_gate_args(delays, computational_modes)
        prog = multiple_loops_program(delays, gate_args)

        eng = sf.Engine("gaussian")
        results = eng.run(prog, shots=1)
        samples = results.samples
        samples_dict = results.samples_dict

        num_modes = computational_modes + vacuum_modes
        assert samples.shape == (1, 1, num_modes)  # `(shots, spatial modes, timebins)`
        assert samples_dict[0].shape == (1, num_modes)

    @pytest.mark.parametrize("delays", ([3], [2, 4], [8, 5], [1, 6, 36], [6, 1, 4]))
    def test_crop_run_options(self, delays):
        """Test that run_options takes precedence over default, meaning only samples from
        computational modes are returned"""

        computational_modes = 100
        gate_args, _ = random_gate_args(delays, computational_modes)
        prog = multiple_loops_program(delays, gate_args)

        eng = sf.Engine("gaussian")
        prog.run_options = {"crop": True}
        results = eng.run(prog)
        samples = results.samples
        samples_dict = results.samples_dict

        assert samples.shape == (1, 1, computational_modes)  # `(shots, spatial modes, timebins)`
        assert samples_dict[0].shape == (1, computational_modes)

    @pytest.mark.parametrize("delays", ([3], [2, 4], [8, 5], [1, 6, 36], [6, 1, 4]))
    def test_crop_passed(self, delays):
        """Test that eng.run takes precedence over run_options, meaning only samples from
        computational modes are returned"""

        computational_modes = 100
        gate_args, _ = random_gate_args(delays, computational_modes)
        prog = multiple_loops_program(delays, gate_args)

        eng = sf.Engine("gaussian")
        prog.run_options = {"crop": False}
        results = eng.run(prog, crop=True)
        samples = results.samples
        samples_dict = results.samples_dict

        assert samples.shape == (1, 1, computational_modes)  # `(shots, spatial modes, timebins)`
        assert samples_dict[0].shape == (1, computational_modes)

    @pytest.mark.parametrize("delays", ([3], [2, 4], [8, 5], [1, 6, 36], [6, 1, 4]))
    @pytest.mark.parametrize("shots", [None, 1, 2, 3])
    @pytest.mark.parametrize("crop", [False, True])
    def test_crop_and_shots_passed(self, delays, shots, crop):
        """Test that shots and crop parameters produce correct results."""

        computational_modes = 40
        gate_args, vacuum_modes = random_gate_args(delays, computational_modes)
        prog = multiple_loops_program(delays, gate_args)

        eng = sf.Engine("gaussian")
        results = eng.run(prog, crop=crop, shots=shots)
        samples = results.samples
        samples_dict = results.samples_dict

        if shots:
            num_modes = computational_modes if crop else computational_modes + vacuum_modes
            assert samples.shape == (shots, 1, num_modes)  # `(shots, spatial modes, timebins)`
            assert samples_dict[0].shape == (shots, num_modes)
        else:
            assert samples.size == 0  # if shots is None samples array should be empty
            assert samples_dict == {}

    @pytest.mark.parametrize("delays", ([3], [2, 4], [8, 5], [1, 6, 36], [6, 1, 4]))
    @pytest.mark.parametrize("crop", [False, True])
    @pytest.mark.parametrize("space_unroll", [False, True])
    def test_space_unroll_crops_state(self, delays, crop, space_unroll):
        """Tests vacuum modes are cropped from the state when the program is space unrolled."""

        computational_modes = 40
        gate_args, vacuum_modes = random_gate_args(delays, computational_modes)
        prog = multiple_loops_program(delays, gate_args)
        if space_unroll:
            prog.space_unroll()

        eng = sf.Engine("gaussian")
        results = eng.run(prog, crop=crop, shots=None)
        state = results.state

        if space_unroll and crop:
            # when the circuit is space unrolled and cropped the state
            # should contain computational modes only
            assert state.num_modes == computational_modes
        elif space_unroll:
            # when the circuit is space unrolled only the state should contain all modes
            assert state.num_modes == computational_modes + vacuum_modes
        else:
            # when the circuit is not space unrolled the state should contain only concurrent modes
            assert state.num_modes == prog.concurr_modes

    def test_crop_warning(self):
        """Tests warning is raised when cropping is requested for a circuit that has not been space unrolled."""

        crop = True
        space_unroll = False

        delays = [1, 6, 36]
        gate_args, vacuum_modes = random_gate_args(delays, num_modes=40)
        prog = multiple_loops_program(delays, gate_args)
        if space_unroll:
            prog.space_unroll()

        eng = sf.Engine("gaussian")

        with pytest.warns(
            UserWarning, match=r"Cropping the state is only possible for space unrolled circuits *"
        ):
            eng.run(prog, crop=crop, shots=None)


class TestUnrolling:
    """Test that (un)rolling programs work as expected."""

    def test_unroll(self):
        """Test unrolling program."""
        n = 2

        prog = sf.TDMProgram(N=2)
        with prog.context([0] * n, [0] * n, [0] * n) as (p, q):
            ops.Sgate(0.5643, 0) | q[1]
            ops.BSgate(p[0]) | (q[1], q[0])
            ops.Rgate(p[1]) | q[1]
            ops.MeasureHomodyne(p[2]) | q[0]

        prog_length = len(prog.circuit)
        assert prog_length == 4

        prog.unroll()
        assert len(prog.circuit) == n * prog_length

        prog.roll()
        assert len(prog.circuit) == prog_length

    def test_unroll_shots(self):
        """Test unrolling program several times using different number of shots."""
        n = 2
        shots = 2

        prog = sf.TDMProgram(N=2)
        with prog.context([0] * n, [0] * n, [0] * n) as (p, q):
            ops.Sgate(0.5643, 0) | q[1]
            ops.BSgate(p[0]) | (q[1], q[0])
            ops.Rgate(p[1]) | q[1]
            ops.MeasureHomodyne(p[2]) | q[0]

        prog_length = len(prog.circuit)
        assert prog_length == 4

        prog.unroll(shots=shots)
        assert len(prog.circuit) == n * shots * prog_length

        # unroll once more with the same number of shots to cover caching
        prog.unroll(shots=shots)
        assert len(prog.circuit) == n * shots * prog_length

        # unroll once more with a different number of shots
        shots = 3
        prog.unroll(shots=shots)
        assert len(prog.circuit) == n * shots * prog_length

        prog.roll()
        assert len(prog.circuit) == prog_length

    @pytest.mark.parametrize(
        "space_unrolled, start_locked", [[True, True], [True, False], [False, True], [False, False]]
    )
    def test_locking_when_unrolling(self, space_unrolled, start_locked):
        """Test that a locked program can be (un)rolled with the locking intact."""
        prog = sf.TDMProgram(N=2)

        with prog.context([0, 0], [0, 0], [0, 0]) as (p, q):
            ops.Sgate(0.5643, 0) | q[1]
            ops.BSgate(p[0]) | (q[1], q[0])
            ops.Rgate(p[1]) | q[1]
            ops.MeasureHomodyne(p[2]) | q[0]

        if start_locked:
            prog.lock()

        assert prog.locked == start_locked

        if space_unrolled:
            prog.space_unroll()
        else:
            prog.unroll()

        assert prog.locked == start_locked

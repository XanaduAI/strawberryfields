# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Unit tests for tdmprogram.py"""
import pytest
import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.tdm import tdmprogram

# feel free to remove after Fabian's test:
import time

# make test deterministic
np.random.seed(42)


def singleloop(r, alpha, phi, theta, copies, shift="end"):
    """Single delay loop with program.

    Args:
        r (float): squeezing parameter
        alpha (Sequence[float]): beamsplitter angles
        phi (Sequence[float]): rotation angles
        theta (Sequence[float]): homodyne measurement angles

    Returns:
        (list): homodyne samples from the single loop simulation
    """

    prog = tdmprogram.TDMProgram(N=2)
    with prog.context(alpha, phi, theta, copies=copies, shift=shift) as (p, q):
        ops.Sgate(r, 0) | q[1]
        ops.BSgate(p[0]) | (q[0], q[1])
        ops.Rgate(p[1]) | q[1]
        ops.MeasureHomodyne(p[2]) | q[0]
    eng = sf.Engine("gaussian")
    result = eng.run(prog)

    return result.samples[0]


def test_number_of_copies_must_be_integer():
    """Checks number of copies is integer"""
    sq_r = 1.0
    N = 3
    c = 4
    copies = 1 / 137
    alpha = [0, np.pi / 4] * c
    phi = [np.pi / 2, 0] * c
    theta = [0, 0] + [0, np.pi / 2] + [np.pi / 2, 0] + [np.pi / 2, np.pi / 2]
    with pytest.raises(TypeError, match="Number of copies must be a positive integer"):
        singleloop(sq_r, alpha, phi, theta, copies)


def test_gates_equal_length():
    """Checks gate list parameters have same length"""
    sq_r = 1.0
    N = 3
    c = 4
    copies = 10
    alpha = [0, np.pi / 4] * c
    phi = [np.pi / 2, 0] * c
    theta = [0, 0] + [0, np.pi / 2] + [np.pi / 2, 0] + [np.pi / 2]
    with pytest.raises(ValueError, match="Gate-parameter lists must be of equal length."):
        singleloop(sq_r, alpha, phi, theta, copies)


def test_at_least_one_measurement():
    """Checks circuit has at least one measurement operator"""
    sq_r = 1.0
    N = 3
    copies = 1
    alpha = [0] * 4
    phi = [0] * 4
    prog = tdmprogram.TDMProgram(N=3)
    with pytest.raises(ValueError, match="Must be at least one measurement."):
        with prog.context(alpha, phi, copies=copies, shift="end") as (p, q):
            ops.Sgate(sq_r, 0) | q[2]
            ops.BSgate(p[0]) | (q[1], q[2])
            ops.Rgate(p[1]) | q[2]
        eng = sf.Engine("gaussian")
        result = eng.run(prog)


def test_spatial_modes_number_of_measurements_match():
    """Checks number of spatial modes matches number of measurements"""
    sq_r = 1.0
    N = 3
    copies = 1
    alpha = [0] * 4
    phi = [0] * 4
    theta = [0] * 4
    with pytest.raises(
        ValueError, match="Number of measurements operators must match number of spatial_modes."
    ):
        prog = tdmprogram.TDMProgram(N=[3, 3])
        with prog.context(alpha, phi, theta, copies=copies) as (p, q):
            ops.Sgate(sq_r, 0) | q[2]
            ops.BSgate(p[0]) | (q[1], q[2])
            ops.Rgate(p[1]) | q[2]
            ops.MeasureHomodyne(p[2]) | q[0]
        eng = sf.Engine("gaussian")
        result = eng.run(prog)

def test_epr():
    """Generates an EPR state and checks that the correct correlations (noise reductions) are observed
    from the samples"""

    sq_r = 1.0
    N = 3
    c = 4
    copies = 200

    # This will generate c EPRstates per copy. I chose c = 4 because it allows us to make 4 EPR pairs per copy that can each be measured in different basis permutations.
    alpha = [np.pi / 4, 0] * c
    phi = [0, np.pi / 2] * c

    # Measurement of 4 subsequent EPR states in XX, XP, PX, PP to investigate nearest-neighbour correlations in all basis permutations
    theta = [0, 0] + [0, np.pi / 2] + [np.pi / 2, 0] + [np.pi / 2, np.pi / 2]  #

    x = singleloop(sq_r, alpha, phi, theta, copies)

    X0 = x[0::8]
    X1 = x[1::8]
    X2 = x[2::8]
    P0 = x[3::8]
    P1 = x[4::8]
    X3 = x[5::8]
    P2 = x[6::8]
    P3 = x[7::8]
    atol = 5 / np.sqrt(copies)
    minusstdX1X0 = (X1 - X0).std() / np.sqrt(2)
    plusstdX1X0 = (X1 + X0).std() / np.sqrt(2)
    squeezed_std = np.exp(-sq_r)
    assert np.allclose(minusstdX1X0, squeezed_std, atol=atol)
    assert np.allclose(plusstdX1X0, 1 / squeezed_std, atol=atol)
    minusstdP2P3 = (P2 - P3).std() / np.sqrt(2)
    plusstdP2P3 = (P2 + P3).std() / np.sqrt(2)
    assert np.allclose(minusstdP2P3, 1 / squeezed_std, atol=atol)
    assert np.allclose(plusstdP2P3, squeezed_std, atol=atol)
    minusstdP0X2 = (P0 - X2).std()
    plusstdP0X2 = (P0 + X2).std()
    expected = 2 * np.sinh(sq_r) ** 2
    assert np.allclose(minusstdP0X2, expected, atol=atol)
    assert np.allclose(plusstdP0X2, expected, atol=atol)
    minusstdX3P1 = (X3 - P1).std()
    plusstdX3P1 = (X3 + P1).std()
    assert np.allclose(minusstdX3P1, expected, atol=atol)
    assert np.allclose(plusstdX3P1, expected, atol=atol)


def test_ghz():
    """Generates an GHZ state and checks that the correct correlations (noise reductions) are observed
    from the samples
    See paragraph above Eq. 5 of this reference

    https://advances.sciencemag.org/content/5/5/eaaw4530
    """

    sq_r = 2
    copies = 10
    vac_modes = 1  # number of vacuum padding modes in the initial setup

    n = 100  # number of modes
    c = 2  # number of n-mode GHZ states per copy

    alpha = [np.arccos(np.sqrt(1/(n-i+1))) if i !=n+1 else 0 for i in range(n+vac_modes)]
    alpha[0] = 0.0
    phi = [0]*(n+vac_modes)
    phi[0] = np.pi/2
    # # This will generate c GHZ states per copy. I chose c = 2 because it allows us to make 2 GHZ states per copy: one measured in X and one measured in P.
    alpha = alpha * c
    phi = phi * c

    # Measurement of 2 subsequent GHZ states: one for investigation of X-correlations and one for P-correlations
    theta = [0] * (n + vac_modes) + [np.pi / 2] * (n + vac_modes)

    x = singleloop(sq_r, alpha, phi, theta, copies)

    X = []
    P = []
    nXP = []
    for i in range(0, copies):
        # This slicing will be a lot simpler once we output the samples ordered by copy.
        startX = i * (n + vac_modes) * c + vac_modes
        startP = i * (n + vac_modes) * c + vac_modes + n + vac_modes
        X_i = x[
            startX : startX + n
        ]  # <-- If the GHZ-state generation was successful, all elements in X_i should be the same (up to squeezing, loss and noise).
        P_i = x[
            startP : startP + n
        ]  # <-- If the GHZ-state generation was successful, all elements in P_i should add up to zero (up to squeezing, loss and noise).
        X.append(X_i)
        P.append(P_i)

        insepX = np.var(
            X_i
        )  # <-- this should be as close as possible to zero, see https://advances.sciencemag.org/content/5/5/eaaw4530, Eq (5)
        # ^ Computing the variance here is kind of a cheat. Actually, we would need to build the difference between all possible pairs within X_i which becomes very bulky for large n.
        print("insepX",insepX)
        insepP = sum(
            P_i
        )  # <-- this should be as close as possible to zero, see https://advances.sciencemag.org/content/5/5/eaaw4530, Eq (5)
        print("insepP",insepP)
        nullif = insepX + insepP
        nXP.append(nullif)

    nXPvar = np.var(np.array(nXP))
    print("\nNullifier variance:", nXPvar, np.sqrt(0.5)*n*np.exp(-2*sq_r))


def test_1Dcluster_cloud():
    """This example generates a 1D-clusterstate in the 'singleloop' architecture and evaluates the quadrature correlations."""

    sq_r = 5
    N = 3
    copies = 1

    n = 100  # for an n-mode cluster state

    # reference for gate parameters alpha and phi: https://advances.sciencemag.org/content/5/5/eaaw4530, paragraph above Eq (1)
    alpha = []
    for i in range(n):
        if i == 0:
            T = 1
        else:
            T = (np.sqrt(5) - 1) / 2
        alpha.append(np.arccos(np.sqrt(T)))

    phi = [np.pi / 2] * n

    # alternating measurement basis because nullifiers are defined by -X_(k-2)+P_(k-1)-X_(k)
    theta = [0, np.pi / 2] * int(n / 2)

    x = singleloop(sq_r, alpha, phi, theta, copies)

    strip = 4  # first four modes will not satisfy inseparability criteria

    X = x[0 + strip :: 2]
    P = x[1 + strip :: 2]

    # nullifier_k = -X_(k-2)+P_(k-1)-X_(k)
    # see: https://advances.sciencemag.org/content/5/5/eaaw4530, Eq (1)
    nXP = []
    for i in range(strip, n - strip, 2):
        # x[i-2] was measured in X basis, see theta
        # x[i-1] was measured in P basis, see theta
        # x[i]   was measured in X basis, see theta
        nullif = (
            -x[i - 2] + x[i - 1] - x[i]
        )  # <-- clusterstate generation successful when this close to zero
        nXP.append(nullif)
        print(nullif)

    # # this is the same thing, only expressed in terms of X and P
    # for i in range(2,int((n-strip)/2)):
    #     nullif = -X[i-1]+P[i-1]-X[i] # <-- clusterstate generation successful when this close to zero
    #     nXP.append(nullif)
    #     print(nullif)

    nXPvar = np.var(np.array(nXP))
    print("\nNullifier variance:", nXPvar)


####################################
### OTHER EXPERIMENTAL ARCHITECTURES
####################################


def engtest_millionmodes():
    """
    One-dimensional temporal-mode cluster state as demonstrated by University of Tokyo. See https://aip.scitation.org/doi/pdf/10.1063/1.4962732
    """

    sq_r = 5
    N = 3  # concurrent modes

    n = 500  # for an n-mode cluster state
    copies = 1

    # first half of cluster state measured in X, second half in P
    theta1 = [0] * int(n / 2) + [np.pi / 2] * int(n / 2)  # measurement angles for detector A
    theta2 = theta1  # measurement angles for detector B

    prog = tdmprogram.TDMProgram(N=[1, 2])
    with prog.context(theta1, theta2, copies=copies, shift="end") as (p, q):
        ops.Sgate(sq_r, 0) | q[0]
        ops.Sgate(sq_r, 0) | q[2]
        ops.Rgate(np.pi / 2) | q[0]
        ops.BSgate(np.pi / 4) | (q[0], q[2])
        ops.BSgate(np.pi / 4) | (q[0], q[1])
        ops.MeasureHomodyne(p[0]) | q[0]
        ops.MeasureHomodyne(p[1]) | q[1]
    eng = sf.Engine("gaussian")
    result = eng.run(prog)

    xA = result.all_samples[0]
    xB = result.all_samples[1]

    X_A = xA[: n // 2]  # X samples from detector A
    P_A = xA[n // 2 :]  # P samples from detector A
    X_B = xB[: n // 2]  # X samples from detector B
    P_B = xB[n // 2 :]  # P samples from detector B

    # print('\n',x)

    # nullifiers defined in https://aip.scitation.org/doi/pdf/10.1063/1.4962732, Eqs. (1a) and (1b)
    nX = []
    nP = []
    # print('nullif_X,                  nullif_P')
    for i in range(len(X_A) - 1):
        nullif_X = X_A[i] + X_B[i] + X_A[i + 1] - X_B[i + 1]
        nullif_P = P_A[i] + P_B[i] - P_A[i + 1] + P_B[i + 1]
        nX.append(nullif_X)
        nP.append(nullif_P)
        # print(nullif_X, '     ', nullif_P)

    nXvar = np.var(np.array(nX))
    nPvar = np.var(np.array(nP))

    print("\nnullif_X variance:", nXvar)
    print("nullif_P variance:", nPvar)


def test_DTU2D():
    """
    Two-dimensional temporal-mode cluster state as demonstrated by Technical University of Denmark. See: https://arxiv.org/pdf/1906.08709
    """

    start = time.time()

    sq_r = 5

    """
    Settings to generate a cluster state of dimension (delay2 x n). In the DTU-paper https://arxiv.org/pdf/1906.08709 they used:
    delay2=12
    n=15000
    """
    delay1 = 1  # number of timebins in the short delay line
    delay2 = 12  # number of timebins in the long delay line
    n = 400  # number of timebins

    # first half of cluster state measured in X, second half in P
    theta_A = [0] * int(n / 2) + [np.pi / 2] * int(n / 2)  # measurement angles for detector A
    theta_B = theta_A  # measurement angles for detector B

    # 2D cluster
    prog = tdmprogram.TDMProgram([1, delay2 + delay1 + 1])
    with prog.context(theta_A, theta_B, shift="end") as (p, q):
        ops.Sgate(sq_r, 0) | q[0]
        ops.Sgate(sq_r, 0) | q[delay2 + delay1 + 1]
        ops.Rgate(np.pi / 2) | q[delay2 + delay1 + 1]
        ops.BSgate(np.pi / 4, np.pi) | (q[delay2 + delay1 + 1], q[0])
        ops.BSgate(np.pi / 4, np.pi) | (q[delay2 + delay1], q[0])
        ops.BSgate(np.pi / 4, np.pi) | (q[delay1], q[0])
        ops.MeasureHomodyne(p[1]) | q[0]
        ops.MeasureHomodyne(p[0]) | q[delay1]
    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    samples = result.all_samples

    xA = result.all_samples[0]
    xB = result.all_samples[1]

    X_A = xA[: n // 2]  # X samples from detector A
    P_A = xA[n // 2 :]  # P samples from detector A
    X_B = xB[: n // 2]  # X samples from detector B
    P_B = xB[n // 2 :]  # P samples from detector B

    # nullifiers defined in https://arxiv.org/pdf/1906.08709.pdf, Eqs. (1) and (2)
    N = delay2
    nX = []
    nP = []
    # print('\n')
    # print('nullif_X,                  nullif_P')
    for k in range(0, len(X_A) - delay2 - 1):
        nullif_X = (
            X_A[k]
            + X_B[k]
            - X_A[k + 1]
            - X_B[k + 1]
            - X_A[k + N]
            + X_B[k + N]
            - X_A[k + N + 1]
            + X_B[k + N + 1]
        )
        nullif_P = (
            P_A[k]
            + P_B[k]
            + P_A[k + 1]
            + P_B[k + 1]
            - P_A[k + N]
            + P_B[k + N]
            + P_A[k + N + 1]
            - P_B[k + N + 1]
        )
        nX.append(nullif_X)
        nP.append(nullif_P)
        # print(nullif_X, '     ', nullif_P)

    nXvar = np.var(np.array(nX))
    nPvar = np.var(np.array(nP))

    print("\n")
    print("nullif_X variance:", nXvar)
    print("nullif_P variance:", nPvar)

    if 1e-15 < nXvar < 0.5 and 1e-15 < nPvar < 0.5:
        print("\n########## SUCCESS ###########")

    print("\nElapsed time [s]:", time.time() - start)


def test_tokyo2D():
    """
    Two-dimensional temporal-mode cluster state as demonstrated by Universtiy of Tokyo. See: https://arxiv.org/pdf/1903.03918.pdf
    """

    start = time.time()

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

    # 2D cluster
    prog = tdmprogram.TDMProgram(N)
    with prog.context(theta_A, theta_B, theta_C, theta_D, shift="end") as (p, q):

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
    result = eng.run(prog)
    samples = result.all_samples

    xA = result.all_samples[0]
    xB = result.all_samples[1]
    xC = result.all_samples[3]
    xD = result.all_samples[9]

    X_A = xA[: n // 2]  # X samples from detector A
    P_A = xA[n // 2 :]  # P samples from detector A

    X_B = xB[: n // 2]  # X samples from detector B
    P_B = xB[n // 2 :]  # P samples from detector B

    X_C = xC[: n // 2]  # X samples from detector C
    P_C = xC[n // 2 :]  # P samples from detector C

    X_D = xD[: n // 2]  # X samples from detector D
    P_D = xD[n // 2 :]  # P samples from detector D

    N = delayC
    nX1 = []
    nX2 = []
    nP1 = []
    nP2 = []
    # nullifiers defined in https://arxiv.org/pdf/1903.03918.pdf, Fig. S5
    for k in range(0, len(X_A) - N - 1):
        nullif_X1 = (
            X_A[k] + X_B[k] - np.sqrt(1 / 2) * (-X_A[k + 1] + X_B[k + 1] + X_C[k + N] + X_D[k + N])
        )
        nullif_X2 = (
            X_C[k] - X_D[k] - np.sqrt(1 / 2) * (-X_A[k + 1] + X_B[k + 1] - X_C[k + N] - X_D[k + N])
        )
        nullif_P1 = (
            P_A[k] + P_B[k] + np.sqrt(1 / 2) * (-P_A[k + 1] + P_B[k + 1] + P_C[k + N] + P_D[k + N])
        )
        nullif_P2 = (
            P_C[k] - P_D[k] + np.sqrt(1 / 2) * (-P_A[k + 1] + P_B[k + 1] - P_C[k + N] - P_D[k + N])
        )
        nX1.append(nullif_X1)
        nX2.append(nullif_X2)
        nP1.append(nullif_P1)
        nP2.append(nullif_P2)

    nX1var = np.var(np.array(nX1))
    nX2var = np.var(np.array(nX2))
    nP1var = np.var(np.array(nP1))
    nP2var = np.var(np.array(nP2))

    print("\n")
    print("nullif_X1 variance:", nX1var)
    print("nullif_X2 variance:", nX2var)
    print("nullif_P1 variance:", nP1var)
    print("nullif_P2 variance:", nP2var)

    if (
        1e-15 < nX1var < 0.5
        and 1e-15 < nX2var < 0.5
        and 1e-15 < nP1var < 0.5
        and 1e-15 < nP2var < 0.5
    ):
        print("\n########## SUCCESS ###########")

    print("\nElapsed time [s]:", time.time() - start)


# test_epr()
test_ghz()
# test_1Dcluster_cloud()
# test_millionmodes()
# test_DTU2D()
# test_tokyo2D()

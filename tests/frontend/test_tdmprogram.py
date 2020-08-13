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
from strawberryfields.tdm.tdmprogram import reshape_samples

# remove after Fabian's test:
from matplotlib import pyplot as plt

# make test deterministic
np.random.seed(42)


def test_end_to_end():
    """Check the samples have the correct shape"""
    R = [1 / 2, 1, 0.3, 1.4, 0.4]  # phase angles
    BS = [1, 1 / 3, 1 / 2, 1, 1 / 5]  # BS angles
    M = [1, 2.3, 1.2, 1 / 2, 5 / 3]  # measurement angles
    r = 0.8
    eta = 0.99
    nbar = 0.1
    # Instantiate a TDMProgram by passing the number of concurrent modes required
    # (the number of modes that are alive at the same time).
    N = 3
    prog = tdmprogram.TDMProgram(N=N)
    MM = 1
    # The context manager has the signature prog.context(*args, copies=1),
    # where args contains the gate parameters, and copies determines how
    # many copies of the state should be prepared.
    # It returns the 'zipped' parameters, and the automatically shifted register.
    c = 10
    with prog.context(R, BS, M, copies=c) as (p, q):
        ops.Sgate(r, 0) | q[2]
        ops.ThermalLossChannel(eta, nbar) | q[2]
        ops.BSgate(p[0]) | (q[1], q[2])
        ops.Rgate(p[1]) | q[2]
        ops.MeasureHomodyne(p[2]) | q[0]
    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    samples = reshape_samples(result.all_samples, c=1)
    assert samples.shape == (1, len(R) * c)


def test_epr():
    """Check the correct EPR correlations are generated"""
    sq_r = 1.0
    N = 3
    prog = tdmprogram.TDMProgram(N=N)
    c = 4
    copies = 400

    # This will generate c EPRstates per copy. I chose c = 4 because it allows us to make 4 EPR pairs per copy that can each be measured in different basis permutations.
    alpha = [0, np.pi / 4] * c
    phi = [np.pi / 2, 0] * c

    # Measurement of 4 subsequent EPR states in XX, XP, PX, PP to investigate nearest-neighbour correlations in all basis permutations
    M = [0, 0] + [0, np.pi / 2] + [np.pi / 2, 0] + [np.pi / 2, np.pi / 2]  #

    with prog.context(alpha, phi, M, copies=copies) as (p, q):
        ops.Sgate(sq_r, 0) | q[2]
        ops.BSgate(p[0]) | (q[1], q[2])
        ops.Rgate(p[1]) | q[2]
        ops.MeasureHomodyne(p[2]) | q[0]
    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    samples = result.all_samples
    x = reshape_samples(samples)

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
    assert np.allclose(plusstdX1X0, 1 /  squeezed_std, atol=atol)
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

def test_error_no_measurement():
    """Test error is raised when there is no measurement"""
    R = [1 / 2, 1, 0.3, 1.4, 0.4]  # phase angles
    BS = [1, 1 / 3, 1 / 2, 1, 1 / 5]  # BS angles
    M = [1, 2.3, 1.2, 1 / 2, 5 / 3]  # measurement angles
    r = 0.8
    N = 3
    prog = tdmprogram.TDMProgram(N=N)
    c = 10
    with pytest.raises(ValueError, match="Must be at least one measurement"):
        with prog.context(R, BS, M, copies=c) as (p, q):
            ops.Sgate(r, 0) | q[2]
            ops.ThermalLossChannel(eta, nbar) | q[2]
            ops.BSgate(p[0]) | (q[1], q[2])
            ops.Rgate(p[1]) | q[2]


def test_error_unequal_length():
    """Test error is raised when the parameters don't have matching lengths"""
    R = [1 / 2, 1, 0.3, 1.4, 0.4]  # phase angles
    BS = [1, 1 / 3, 1 / 2, 1, 1 / 5]  # BS angles
    M = [1, 2.3, 1.2, 1 / 2]  # measurement angles
    # Note M has one less element than R and BS
    r = 0.8
    N = 3
    prog = tdmprogram.TDMProgram(N=N)
    c = 10
    with pytest.raises(ValueError, match="Gate-parameter lists must be of equal length"):
        with prog.context(R, BS, M, copies=c) as (p, q):
            ops.Sgate(r, 0) | q[2]
            ops.ThermalLossChannel(eta, nbar) | q[2]
            ops.BSgate(p[0]) | (q[1], q[2])
            ops.Rgate(p[1]) | q[2]
            ops.MeasureHomodyne(p[2]) | q[0]

def test_error_noninteger_number_copies():
    """Test error is raised when the number of copies is not an integer"""
    R = [1 / 2, 1, 0.3, 1.4, 0.4]  # phase angles
    BS = [1, 1 / 3, 1 / 2, 1, 1 / 5]  # BS angles
    M = [1, 2.3, 1.2, 1 / 2, 5 / 3]  # measurement angles
    r = 0.8
    N = 3
    prog = tdmprogram.TDMProgram(N=N)
    c = 10
    with pytest.raises(TypeError, match="Number of copies must be a positive integer"):
        with prog.context(R, BS, M, copies=1/137) as (p, q):
            ops.Sgate(r, 0) | q[2]
            ops.ThermalLossChannel(eta, nbar) | q[2]
            ops.BSgate(p[0]) | (q[1], q[2])
            ops.Rgate(p[1]) | q[2]
            ops.MeasureHomodyne(p[2]) | q[0]

def test_error_wrong_mode_measurement_end():
    """Test error is raised when the wrong mode is measured"""
    R = [1 / 2, 1, 0.3, 1.4, 0.4]  # phase angles
    BS = [1, 1 / 3, 1 / 2, 1, 1 / 5]  # BS angles
    M = [1, 2.3, 1.2, 1 / 2, 5 / 3]  # measurement angles
    r = 0.8
    N = 3
    prog = tdmprogram.TDMProgram(N=N)
    c = 10
    with pytest.raises(ValueError, match="Measurements must be on consecutive modes starting from"):
        with prog.context(R, BS, M, copies=1, shift="end") as (p, q):
            ops.Sgate(r, 0) | q[2]
            ops.BSgate(p[0]) | (q[1], q[2])
            ops.Rgate(p[1]) | q[2]
            ops.MeasureHomodyne(p[2]) | q[1]

def test_error_wrong_mode_measurement_after():
    """Test error is raised when the wrong mode is measured"""
    R = [1 / 2, 1, 0.3, 1.4, 0.4]  # phase angles
    BS = [1, 1 / 3, 1 / 2, 1, 1 / 5]  # BS angles
    M = [1, 2.3, 1.2, 1 / 2, 5 / 3]  # measurement angles
    # Note M has one less element than R and BS
    r = 0.8
    N = 3
    prog = tdmprogram.TDMProgram(N=N)
    c = 10
    with pytest.raises(ValueError, match="Measurements allowed only on"):
        with prog.context(R, BS, M, copies=1, shift="after") as (p, q):
            ops.Sgate(r, 0) | q[2]
            ops.BSgate(p[0]) | (q[1], q[2])
            ops.Rgate(p[1]) | q[2]
            ops.MeasureHomodyne(p[2]) | q[1]


def test_ghz():
    """Check the correct GHZ correlations are generated"""
    sq_r = 5
    N = 3
    prog = tdmprogram.TDMProgram(N=N)
    copies = 1
    vac_modes = 2 # number of vacuum modes in the initial setup

    n = 20 # for an n-mode GHZ state
    c = 2 # number of n-mode GHZ states per copy

    # reference for gate parameters alpha and phi: https://advances.sciencemag.org/content/5/5/eaaw4530, paragraph above Eq (5)

    alpha = []
    for i in range(n+vac_modes):
        if i == 0 or i==n+vac_modes-2 or i==n+vac_modes-1:
            T = 1
        else:
            T = 1/(n-i+1)
        # checking if the BS transmissions match with https://advances.sciencemag.org/content/5/5/eaaw4530    
        print(i+1,':',T)
        alpha.append(np.arccos(np.sqrt(T)))

    phi = list(np.zeros(n+vac_modes))
    phi[0] = np.pi/2

    # # This will generate c GHZ states per copy. I chose c = 4 because it allows us to make 4 EPR pairs per copy that can each be measured in different basis permutations.
    alpha = alpha*c
    phi = phi*c

    # Measurement of 2 subsequent GHZ states: one for investivation of x-correlations and one for p-correlations
    theta = [0]*(n+vac_modes) + [np.pi/2]*(n+vac_modes)

    with prog.context(alpha, phi, theta, copies=copies) as (p, q):
        ops.Sgate(sq_r, 0) | q[2]
        ops.BSgate(p[0]) | (q[1], q[2])
        ops.Rgate(p[1]) | q[2]
        ops.MeasureHomodyne(p[2]) | q[0]
    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    samples = result.all_samples
    x = reshape_samples(samples)

    X = x[vac_modes:n+2]
    P = x[n+2+vac_modes:]

    print('X:', X) # <-- these should all be the same, see https://advances.sciencemag.org/content/5/5/eaaw4530, Eq (5)
    print('P:', P)
    print('sum(P):',np.sum(P)) # <-- this should be as close as possible to zero, see https://advances.sciencemag.org/content/5/5/eaaw4530, Eq (5)


def test_1Dcluster():
    """Check the correct 1-clusterstate correlations are generated"""
    sq_r = 5
    N = 3
    prog = tdmprogram.TDMProgram(N=N)
    copies = 1

    n = 100 # for an n-mode cluster state

    # reference for gate parameters alpha and phi: https://advances.sciencemag.org/content/5/5/eaaw4530, paragraph above Eq (1)
    alpha = []
    for i in range(n):
        if i == 0:
            T = 1
        else:
            T = (np.sqrt(5)-1)/2
        alpha.append(np.arccos(np.sqrt(T)))

    phi = [np.pi/2]*n
    
    # alternating measurement basis because nullifiers are defined by -X_(k-2)+P_(k-1)-X_(k)
    theta = [0,np.pi/2]*int(n/2)

    with prog.context(alpha, phi, theta, copies=copies) as (p, q):
        ops.Sgate(sq_r, 0) | q[2]
        ops.BSgate(p[0]) | (q[1], q[2])
        ops.Rgate(p[1]) | q[2]
        ops.MeasureHomodyne(p[2]) | q[0]
    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    samples = result.all_samples
    x = reshape_samples(samples)

    strip = 2 # first two modes are worthless and will be removed

    X = x[0+strip::2]
    P = x[1+strip::2]

    # nullifier_k = -X_(k-2)+P_(k-1)-X_(k)
    # see: https://advances.sciencemag.org/content/5/5/eaaw4530, Eq (1)
    for i in range(2,n-strip,2):
        # x[i-2] was measured in X basis, see theta
        # x[i-1] was measured in P basis, see theta
        # x[i]   was measured in X basis, see theta
        nullif = -x[i-2]+x[i-1]-x[i] # <-- clusterstate generation successful when this close to zero
        print(nullif)

    # # this is the same thing, only expressed in terms of X and P
    # for i in range(2,int((n-strip)/2)):
    #     nullif = -X[i-1]+P[i-1]-X[i] # <-- clusterstate generation successful when this close to zero
    #     print(nullif)


def test_millionmodes():

    sq_r = 5
    N = 3 # concurrent modes
    prog = tdmprogram.TDMProgram(N=N)

    delay = 1 # number of timebins in the delay line
    n = 100 # for an n-mode cluster state
    copies = 1

    # first half of cluster state measured in X, second half in P
    theta1 = [0]*int(n/2) + [np.pi/2]*int(n/2) # measurement angles for detector A
    theta2 = theta1 # measurement angles for detector B

    with prog.context(theta1, theta2, copies=copies, shift='end') as (p, q):
        ops.Sgate(sq_r, 0) | q[2]
        ops.Rgate(np.pi/2) | q[2]
        ops.Sgate(sq_r, 0) | q[1]
        ops.BSgate(np.pi/4) | (q[1],q[2])
        ops.BSgate(np.pi/4) | (q[1],q[0])
        ops.MeasureHomodyne(p[0]) | q[1]
        ops.MeasureHomodyne(p[1]) | q[0]
    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    samples = result.all_samples

    # x seems to be of shape (n*2), i.e. (timebins*spatial_modes). I suggest a two-dimensional array to clearly distinguish samples from detector A and B
    x = reshape_samples(samples)

    X_A = x[0:n:2] # X samples from detector A
    X_B = x[1:n:2] # X samples from detector B
    P_A = x[n::2] # P samples from detector A
    P_B = x[n+1::2] # P samples from detector B

    # nullifiers defined in https://aip.scitation.org/doi/pdf/10.1063/1.4962732, Eqs. (1a) and (1b)
    nX = []
    nP = []
    print('nullif_X,                  nullif_P')
    for i in range(len(X_A)-1):
        nullif_X = X_A[i] + X_B[i] + X_A[i+1] - X_B[i+1]
        nullif_P = P_A[i] + P_B[i] - P_A[i+1] + P_B[i+1]
        nX.append(nullif_X)
        nP.append(nullif_P)
        print(nullif_X, '     ', nullif_P)

    nXvar=np.var(np.array(nX))
    nPvar=np.var(np.array(nP))

    print('nullif_X variance:', nXvar)
    print('nullif_P variance:', nPvar)

# test_epr()
# test_ghz()
# test_1Dcluster()
# test_millionmodes()
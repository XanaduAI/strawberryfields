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
r"""Unit tests for program.py"""
import pytest
import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.tdm import tdmprogram
from strawberryfields.tdm.tdmprogram import reshape_samples

# feel free to remove after Fabian's test:
import time

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
    # Note M has one less element than R and BS
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
    # Note M has one less element than R and BS
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


def test_epr():
    """Single delay loop with variational coupling as built at Xanadu for cloud access. This example generates EPR states and evaluates the quadrature correlations."""

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


def test_ghz():
    """Single delay loop with variational coupling as built at Xanadu for cloud access. This example generates n-mode GHZ states and evaluates the quadrature correlations."""

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
    """Single delay loop with variational coupling as built at Xanadu for cloud access. This example generates a 1D-clusterstate and evaluates the quadrature correlations."""

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
    '''
    One-dimensional temporal-mode cluster state as demonstrated by University of Tokyo. See https://aip.scitation.org/doi/pdf/10.1063/1.4962732
    '''

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
        ops.Sgate(sq_r, 0) | q[1]
        ops.Rgate(np.pi/2) | q[2]
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


def test_DTU2D():
    '''
    Two-dimensional temporal-mode cluster state as demonstrated by Technical University of Denmark. See: https://arxiv.org/pdf/1906.08709
    '''

    start = time.time()

    sq_r = 5

    '''
    Settings to generate a cluster state of dimension (delay2 x n). In the DTU-paper https://arxiv.org/pdf/1906.08709 they used:
    delay2=12
    n=15000
    '''
    delay1 = 1 # number of timebins in the short delay line
    delay2 = 12 # number of timebins in the long delay line
    n_concurr = delay2 + delay1 + 2 # concurrent modes
    n = 1000 # number of timebins

    # Number of timebins must be a multiple of concurrent modes, otherwise sampli() doesn't work
    n = n//n_concurr*n_concurr

    # first half of cluster state measured in X, second half in P
    theta_A = [0]*int(n/2) + [np.pi/2]*int(n/2) # measurement angles for detector A
    theta_B = theta_A # measurement angles for detector B

    # 2D cluster
    prog = tdmprogram.TDMProgram(n_concurr)
    with prog.context(theta_A, theta_B, shift=1) as (p, q):
        ops.Sgate(sq_r, 0) | q[0]
        ops.Sgate(sq_r, 0) | q[delay2+delay1+1]
        ops.Rgate(np.pi/2) | q[delay2+delay1+1]
        ops.BSgate(np.pi/4, np.pi) | (q[delay2+delay1+1],q[0]) # (B, A)
        ops.BSgate(np.pi/4, np.pi) | (q[delay2+delay1],q[0])
        ops.BSgate(np.pi/4, np.pi) | (q[delay1],q[0])
        ops.MeasureHomodyne(p[0]) | q[delay1]
        ops.MeasureHomodyne(p[1]) | q[0]

    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    samples = result.all_samples

    # x = reshape_samples(samples)
    x = sampli(samples)

    X_A = x[0:n:2] # X samples from detector A
    X_B = x[1:n:2] # X samples from detector B
    P_A = x[n::2] # P samples from detector A
    P_B = x[n+1::2] # P samples from detector B

    # print('X_A\n',X_A)
    # print('X_B\n',X_B)
    # print('P_A\n',P_A)
    # print('P_B\n',P_B)

    # print('var(X_A)',np.var(X_A))
    # print('var(X_B)',np.var(X_B))
    # print('var(P_A)',np.var(P_A))
    # print('var(P_B)',np.var(P_B))

    # print(len(x))
    # print(len(X_A))
    # print(len(X_B))
    # print(len(P_A))
    # print(len(P_B))

    # nullifiers defined in https://arxiv.org/pdf/1906.08709.pdf, Eqs. (1) and (2)
    N=delay2
    nX = []
    nP = []
    # print('\n')
    # print('nullif_X,                  nullif_P')
    for k in range(0, len(X_A)-delay2-1):
        nullif_X = X_A[k] + X_B[k] - X_A[k+1] - X_B[k+1] - X_A[k+N] + X_B[k+N] - X_A[k+N+1] + X_B[k+N+1]
        nullif_P = P_A[k] + P_B[k] + P_A[k+1] + P_B[k+1] - P_A[k+N] + P_B[k+N] + P_A[k+N+1] - P_B[k+N+1]
        nX.append(nullif_X)
        nP.append(nullif_P)
        # print(nullif_X, '     ', nullif_P)

    nXvar=np.var(np.array(nX))
    nPvar=np.var(np.array(nP))

    print('\n')
    print('nullif_X variance:', nXvar)
    print('nullif_P variance:', nPvar)

    if 1e-15 < nXvar < 0.5 and 1e-15 < nPvar < 0.5:
        print('\n########## SUCCESS ###########')

    print('\nElapsed time [s]:', time.time()-start)

def sampli(samples):
    '''
    This function rearranges results.all_samples to a chronologically ordered one-dimensional array -- ONLY IN THE SPECIAL CASE of two spatial modes and a one-step shift at the end of the unit cell. Also, make sure the number of timebins is a multiple of the concurrent modes.
    '''

    num_modes=len(samples)
    len_samples = [len(samples[i]) for i in range(num_modes)]
    max_len = max(len_samples)

    x = []
    for j in range(0,max_len,2):
        x.append(samples[0][j])
        for i in range(1,num_modes):
            x.append(samples[i][j])
            x.append(samples[i][j+1])
        x.append(samples[0][j+1])

    return x

# test_epr()
# test_ghz()
# test_1Dcluster()
# test_millionmodes()
test_DTU2D()

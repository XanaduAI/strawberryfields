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
        ops.BSgate(p[0]) | (q[2], q[1])
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
        ops.BSgate(p[0]) | (q[2], q[1])
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
    assert np.allclose(minusstdX1X0, 1 / squeezed_std, atol=atol)
    assert np.allclose(plusstdX1X0, squeezed_std, atol=atol)
    minusstdP2P3 = (P2 - P3).std() / np.sqrt(2)
    plusstdP2P3 = (P2 + P3).std() / np.sqrt(2)
    assert np.allclose(minusstdP2P3, squeezed_std, atol=atol)
    assert np.allclose(plusstdP2P3, 1 / squeezed_std, atol=atol)
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
            ops.BSgate(p[0]) | (q[2], q[1])
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
            ops.BSgate(p[0]) | (q[2], q[1])
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
            ops.BSgate(p[0]) | (q[2], q[1])
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
            ops.BSgate(p[0]) | (q[2], q[1])
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
            ops.BSgate(p[0]) | (q[2], q[1])
            ops.Rgate(p[1]) | q[2]
            ops.MeasureHomodyne(p[2]) | q[1]

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

# make test deterministic
np.random.seed(42)


def test_end_to_end():
	R =  [1/2, 1, 0.3, 1.4, 0.4] # phase angles
	BS = [1, 1/3, 1/2, 1, 1/5] # BS angles
	M =  [1, 2.3, 1.2, 1/2, 5/3] # measurement angles
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
	eng = sf.Engine('gaussian')
	result = eng.run(prog)
	samples = reshape_samples(result.all_samples, c=1)
	assert samples.shape == (1, len(R)*c)

def test_epr():
	sq_r = 1.0
	N = 3
	prog = tdmprogram.TDMProgram(N=N)
	c = 1
	copies = 1000
	alpha = [0, np.pi/2]*c
	phi = [np.pi/4, 0]*c
	M = [0, 0]*c
	with prog.context(alpha, phi, M, copies=copies) as (p, q):
	    ops.Sgate(sq_r, 0) | q[2]
	    ops.Rgate(p[0]) | q[2]
	    ops.BSgate(p[1]) | (q[2], q[1])
	    ops.MeasureHomodyne(p[2]) | q[0]
	eng = sf.Engine('gaussian')
	result = eng.run(prog)
	samples = result.all_samples
	x = reshape_samples(samples)
	### This test is not quite correct
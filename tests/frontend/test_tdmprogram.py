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
	c = 4
	copies = 400

	 # This will generate c EPRstates per copy. I chose c = 4 because it allows us to make 4 EPR pairs per copy that can each be measured in different basis permutations.
	alpha = [0, np.pi/4]*c
	phi = [np.pi/2, 0]*c

	# Measurement of 4 subsequent EPR states in XX, XP, PX, PP to investigate nearest-neighbour correlations in all basis permutations
	M = [0, 0]+[0, np.pi/2]+[np.pi/2, 0]+[np.pi/2, np.pi/2] # 

	with prog.context(alpha, phi, M, copies=copies) as (p, q):
	    ops.Sgate(sq_r, 0) | q[2]
	    ops.BSgate(p[0]) | (q[2], q[1])
	    ops.Rgate(p[1]) | q[2]
	    ops.MeasureHomodyne(p[2]) | q[0]
	eng = sf.Engine('gaussian')
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
	           
	fig, axs = plt.subplots(2, 2, figsize=(10, 10))
	fig.suptitle('Nearest-neighbour correlations', fontsize=18)

	axs[0,0].scatter(X1, X0)
	axs[0,1].scatter(P0, X2)
	axs[1,0].scatter(X3, P1)
	axs[1,1].scatter(P3, P2)

	axs[0,0].set_ylabel('$X_{i-1}$', fontsize=18)
	axs[1,0].set_xlabel('$X_{i}$', fontsize=18)
	axs[1,0].set_ylabel('$P_{i-1}$', fontsize=18)
	axs[1,1].set_xlabel('$P_{i}$', fontsize=18)

	for ax in axs.flat:
		ax.grid(True)

	plt.show()

	### This test is not quite correct

test_epr()


	    # print('p[0]',p[0])
	    # print('p[1]',p[1])
	    # print('p[2]',p[2])

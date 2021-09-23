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
r"""Unit tests for the GaussianMerge class"""

import pytest
import numpy as np

import strawberryfields as sf
import strawberryfields.ops as ops

pytestmark = pytest.mark.frontend


@pytest.mark.parametrize(
    "init", [(1, 1, 1, 1), (0, 2, 1, 0), (0, 1, 1, 1), (0, 1, 0, 3), (0, 0, 0, 0)]
)
def test_complex(init):
    modes = 4
    cutoff_dim = 6

    initial_state = np.zeros([cutoff_dim] * modes, dtype=complex)
    # The ket below corresponds to a single photon going into each of the modes
    initial_state[init] = 1

    prog = sf.Program(modes)
    s_d_params = 0.01
    with prog.context as q:
        ops.Ket(initial_state) | q  # Initial state preparation
        # Gaussian Layer
        ops.S2gate(s_d_params, s_d_params) | (q[0], q[1])
        ops.BSgate(1.9, 1.7) | (q[1], q[2])
        ops.BSgate(0.9, 0.2) | (q[0], q[1])
        # Non-Gaussian Layer
        ops.Kgate(0.5) | q[3]
        ops.CKgate(0.7) | (q[2], q[3])
        # Gaussian Layer
        ops.BSgate(1.0, 0.4) | (q[0], q[1])
        ops.BSgate(2.0, 1.5) | (q[1], q[2])
        ops.Dgate(s_d_params) | q[0]
        ops.Dgate(s_d_params) | q[0]
        ops.Sgate(s_d_params, s_d_params) | q[1]
        # Non-Gaussian Layer
        ops.Vgate(0.5) | q[2]

    # We run the simulation
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    results_norm = eng.run(prog)
    prog_merged = prog.compile(compiler="gaussian_merge")
    results_merged = eng.run(prog_merged)
    ket = results_norm.state.ket()
    ket_merged = results_merged.state.ket()
    assert np.allclose(np.abs(np.sum(np.conj(ket) * ket_merged)), 1)

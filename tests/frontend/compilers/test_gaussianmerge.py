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


def test_no_displacement():
    """Tests that the compiler is able to compile a number of non-primitive Gaussian gates"""
    modes = 2
    cutoff_dim = 3  # (1+ total number of photons)

    initial_state = np.zeros([cutoff_dim] * modes, dtype=np.complex)
    # The ket below corresponds to a single horizontal photon in each of the modes
    initial_state[1, 1] = 1

    # Here is the main program
    # We create the input state and then send it through a network of beamsplitters, rotations and Kerr gates
    # Since all the gates preserve the number of photons we need not to worry about cutoff issues
    prog = sf.Program(2)
    with prog.context as q:
        ops.Ket(initial_state) | q  # Initial state preparation
        # Gaussian layer
        ops.BSgate(0.5, 0.9) | (q[0], q[1])
        ops.Rgate(0.6) | q[0]
        ops.BSgate(1.9, 0.7) | (q[0], q[1])
        ops.Rgate(0.5) | q[1]
        ops.BSgate(1.5, 2.9) | (q[0], q[1])
        ops.BSgate(1.8, 0.9) | (q[0], q[1])
        ops.BSgate(0.5, 0.9) | (q[0], q[1])
        ops.Rgate(0.6) | q[1]
        ops.BSgate(1.9, 0.7) | (q[0], q[1])
        ops.Rgate(0.5) | q[1]
        ops.BSgate(1.5, 2.9) | (q[0], q[1])
        ops.BSgate(1.8, 0.9) | (q[0], q[1])
        ops.Rgate(0.5) | q[1]
        # Non Gaussian layer
        ops.Kgate(0.5) | q[0]
        ops.Kgate(0.4) | q[1]
        # Gaussian layer
        ops.BSgate(0.5, 0.9) | (q[0], q[1])
        ops.Rgate(0.6) | q[0]
        ops.BSgate(1.9, 0.7) | (q[0], q[1])
        ops.Rgate(0.5) | q[1]
        ops.BSgate(1.5, 2.9) | (q[0], q[1])
        ops.BSgate(1.8, 0.9) | (q[0], q[1])
        ops.Rgate(0.5) | q[1]
        ops.BSgate(0.5, 0.9) | (q[0], q[1])
        ops.Rgate(0.6) | q[1]
        ops.BSgate(1.9, 0.7) | (q[0], q[1])
        ops.Rgate(0.5) | q[1]
        ops.BSgate(1.5, 2.9) | (q[0], q[1])
        ops.BSgate(1.8, 0.9) | (q[0], q[1])
        ops.Rgate(0.5) | q[1]
        # Non Gaussian layer
        ops.CKgate(0.5) | (q[0], q[1])

    # We run the simulation
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    results_norm = eng.run(prog)
    prog_merged = prog.compile(compiler="gaussian_merge")
    results_merged = eng.run(prog_merged)
    ket = results_norm.state.ket()
    ket_merged = results_merged.state.ket()
    assert np.allclose(np.abs(np.sum(np.conj(ket) * ket_merged)), 1)


def test_complex_displacements():
    modes = 4
    cutoff_dim = 6  # (1+ total number of photons)

    initial_state = np.zeros([cutoff_dim] * modes, dtype=np.complex)
    # The ket below corresponds to a single horizontal photon in each of the modes
    initial_state[1, 1, 1, 1] = 1

    # Here is the main program
    # We create the input state and then send it through a network of beamsplitters, rotations and Kerr gates
    # Since all the gates preserve the number of photons we need not to worry about cutoff issues
    prog = sf.Program(modes)
    with prog.context as q:
        ops.Ket(initial_state) | q  # Initial state preparation
        # Gaussian layer
        ops.Dgate(0.01) | q[2]
        ops.BSgate(0.5, 0.9) | (q[0], q[1])
        ops.Rgate(0.6) | q[0]
        ops.Rgate(0.2) | q[2]
        ops.Rgate(0.1) | q[3]
        ops.BSgate(0.5, 0.9) | (q[1], q[2])
        ops.BSgate(1.9, 0.7) | (q[0], q[1])
        ops.BSgate(1.8, 0.9) | (q[2], q[3])
        ops.Rgate(0.5) | q[1]
        ops.Rgate(0.7) | q[3]
        ops.BSgate(1.5, 2.9) | (q[0], q[1])
        ops.BSgate(1.8, 0.9) | (q[0], q[1])
        ops.BSgate(1.9, 0.7) | (q[1], q[2])
        ops.BSgate(0.5, 0.9) | (q[0], q[1])
        ops.Rgate(0.6) | q[1]
        ops.Rgate(0.9) | q[2]
        ops.Rgate(0.1) | q[3]
        ops.BSgate(1.9, 0.7) | (q[0], q[1])
        ops.BSgate(1.8, 0.9) | (q[1], q[2])
        ops.BSgate(0.5, 0.9) | (q[2], q[3])
        ops.Rgate(0.5) | q[1]
        ops.Rgate(0.3) | q[2]
        Rops.gate(0.9) | q[3]
        BSgate(1.5, 2.9) | (q[0], q[1])
        BSgate(1.8, 0.9) | (q[0], q[1])
        BSgate(0.5, 0.9) | (q[1], q[2])
        Rgate(0.5) | q[1]
        Rgate(0.3) | q[3]
        # Non Gaussian layer
        Kgate(0.5) | q[0]
        Kgate(0.4) | q[1]
        Kgate(0.6) | q[2]
        Kgate(0.2) | q[3]
        # Gaussian layer
        Dgate(0.01) | q[2]
        BSgate(0.5, 0.9) | (q[0], q[1])
        Rgate(0.6) | q[0]
        Rgate(0.2) | q[2]
        Rgate(0.1) | q[3]
        BSgate(0.5, 0.9) | (q[1], q[2])
        BSgate(1.9, 0.7) | (q[0], q[1])
        BSgate(1.8, 0.9) | (q[2], q[3])
        Rgate(0.5) | q[1]
        Rgate(0.7) | q[3]
        BSgate(1.5, 2.9) | (q[0], q[1])
        BSgate(1.8, 0.9) | (q[0], q[1])
        BSgate(1.9, 0.7) | (q[1], q[2])
        BSgate(0.5, 0.9) | (q[0], q[1])
        Rgate(0.6) | q[1]
        Rgate(0.9) | q[2]
        Rgate(0.1) | q[3]
        BSgate(1.9, 0.7) | (q[0], q[1])
        BSgate(1.8, 0.9) | (q[1], q[2])
        BSgate(0.5, 0.9) | (q[2], q[3])
        Rgate(0.5) | q[1]
        Rgate(0.3) | q[2]
        Rgate(0.9) | q[3]
        BSgate(1.5, 2.9) | (q[0], q[1])
        BSgate(1.8, 0.9) | (q[0], q[1])
        BSgate(0.5, 0.9) | (q[1], q[2])
        Rgate(0.5) | q[1]
        Rgate(0.3) | q[3]
        # Non Gaussian layer
        CKgate(0.5) | (q[0], q[1])
        CKgate(0.3) | (q[1], q[2])
        CKgate(0.1) | (q[2], q[3])

    # We run the simulation
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    results_norm = eng.run(prog)
    prog_merged = prog.compile(compiler="gaussian_merge")
    results_merged = eng.run(prog_merged)
    ket = results_norm.state.ket()
    ket_merged = results_merged.state.ket()
    if np.allclose(np.abs(np.sum(np.conj(ket) * ket_merged)), 1):
        print("Wow its working!!!")

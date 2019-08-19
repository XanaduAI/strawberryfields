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
r"""Unit tests for the Strawberry Fields example algorithms"""
import pytest

import numpy as np

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import scale

@pytest.mark.parametrize('cutoff', [10], indirect=True)  # override default cutoff fixture
def test_teleportation_fidelity(setup_eng, pure):
    """Test that teleportation of a coherent state has high fidelity"""
    eng, prog = setup_eng(3)
    s = np.sqrt(2)
    z = 2
    BS = BSgate(np.pi / 4, 0)
    alpha = 0.5 + 0.2j

    with prog.context as q:
        # TODO: at some point, use the blackbird parser to
        # read in the following directly from the examples folder
        Coherent(alpha) | q[0]

        Squeezed(-z) | q[1]
        Squeezed(z) | q[2]
        BS | (q[1], q[2])

        BS | (q[0], q[1])
        MeasureHomodyne(0, select=0.07) | q[0]
        MeasureHomodyne(np.pi / 2, select=0.1) | q[1]
        Xgate(scale(q[0], s)) | q[2]
        Zgate(scale(q[1], s)) | q[2]

    state = eng.run(prog).state
    fidelity = state.fidelity_coherent([0, 0, alpha])
    assert np.allclose(fidelity, 1, atol=0.02, rtol=0)


@pytest.mark.backends("gaussian")
def test_gaussian_gate_teleportation(setup_eng, pure):
    """Test that gaussian states match after gate teleportation"""
    eng, prog = setup_eng(4)
    with prog.context as q:
        # TODO: at some point, use the blackbird parser to
        # read in the following directly from the examples folder
        Squeezed(0.1) | q[0]
        Squeezed(-2) | q[1]
        Squeezed(-2) | q[2]

        Pgate(0.5) | q[1]

        CZgate(1) | (q[0], q[1])
        CZgate(1) | (q[1], q[2])

        Fourier.H | q[0]
        MeasureHomodyne(0, select=0) | q[0]
        Fourier.H | q[1]
        MeasureHomodyne(0, select=0) | q[1]

        Squeezed(0.1) | q[3]
        Fourier | q[3]
        Pgate(0.5) | q[3]
        Fourier | q[3]

    state = eng.run(prog).state
    cov1 = state.reduced_gaussian(2)[1]
    cov2 = state.reduced_gaussian(3)[1]
    assert np.allclose(cov1, cov2, atol=0.05, rtol=0)


def test_gaussian_boson_sampling_fock_probs(setup_eng, batch_size, tol):
    """Test that GBS returns expected Fock probabilities"""
    eng, prog = setup_eng(4)
    with prog.context as q:
        # TODO: at some point, use the blackbird parser to
        # read in the following directly from the examples folder
        S = Sgate(1)
        S | q[0]
        S | q[1]
        S | q[2]
        S | q[3]

        # rotation gates
        Rgate(0.5719) | q[0]
        Rgate(-1.9782) | q[1]
        Rgate(2.0603) | q[2]
        Rgate(0.0644) | q[3]

        # beamsplitter array
        BSgate(0.7804, 0.8578) | (q[0], q[1])
        BSgate(0.06406, 0.5165) | (q[2], q[3])
        BSgate(0.473, 0.1176) | (q[1], q[2])
        BSgate(0.563, 0.1517) | (q[0], q[1])
        BSgate(0.1323, 0.9946) | (q[2], q[3])
        BSgate(0.311, 0.3231) | (q[1], q[2])
        BSgate(0.4348, 0.0798) | (q[0], q[1])
        BSgate(0.4368, 0.6157) | (q[2], q[3])

    measure_states = [
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
        [2, 0, 0, 0],
    ]

    results = [
        0.176378447614135,
        0.0685595637122246,
        0.002056097258977398,
        0.00834294639986785,
        0.01031294525345511,
    ]

    state = eng.run(prog).state
    probs = [state.fock_prob(i) for i in measure_states]
    probs = np.array(probs).T.flatten()

    if batch_size is not None:
        results = np.tile(results, batch_size).flatten()

    assert np.allclose(probs, results, atol=tol, rtol=0)


@pytest.mark.backends("tf", "fock")
def test_boson_sampling_fock_probs(setup_eng, batch_size, tol):
    """Test that boson sampling returns expected Fock probabilities"""
    eng, prog = setup_eng(4)
    with prog.context as q:
        # TODO: at some point, use the blackbird parser to
        # read in the following directly from the examples folder
        Fock(1) | q[0]
        Fock(1) | q[1]
        Vac | q[2]
        Fock(1) | q[3]

        # rotation gates
        Rgate(0.5719) | q[0]
        Rgate(-1.9782) | q[1]
        Rgate(2.0603) | q[2]
        Rgate(0.0644) | q[3]

        # beamsplitter array
        BSgate(0.7804, 0.8578) | (q[0], q[1])
        BSgate(0.06406, 0.5165) | (q[2], q[3])
        BSgate(0.473, 0.1176) | (q[1], q[2])
        BSgate(0.563, 0.1517) | (q[0], q[1])
        BSgate(0.1323, 0.9946) | (q[2], q[3])
        BSgate(0.311, 0.3231) | (q[1], q[2])
        BSgate(0.4348, 0.0798) | (q[0], q[1])
        BSgate(0.4368, 0.6157) | (q[2], q[3])

    measure_states = [[1, 1, 0, 1], [2, 0, 0, 1]]
    results = [0.174689160486, 0.106441927246]

    state = eng.run(prog).state
    probs = [state.fock_prob(i) for i in measure_states]
    probs = np.array(probs).T.flatten()

    if batch_size is not None:
        results = np.tile(results, batch_size).flatten()

    assert np.allclose(probs, results, atol=tol, rtol=0)


@pytest.mark.parametrize('cutoff', [4], indirect=True)  # override default cutoff fixture
@pytest.mark.backends("tf", "fock")
def test_hamiltonian_simulation_fock_probs(setup_eng, pure, batch_size, tol):
    """Test that Hamiltonian simulation returns expected Fock probabilities"""
    eng, prog = setup_eng(2)

    measure_states = [[0, 2], [1, 1], [2, 0]]
    results = [0.52240124572, 0.235652876857, 0.241945877423]

    # set the Hamiltonian parameters
    J = 1  # hopping transition
    U = 1.5  # on-site interaction
    k = 20  # Lie product decomposition terms
    t = 1.086  # timestep
    theta = -J * t / k
    r = -U * t / (2 * k)

    with prog.context as q:
        Fock(2) | q[0]

        # Two node tight-binding
        # Hamiltonian simulation

        for i in range(k):
            BSgate(theta, np.pi / 2) | (q[0], q[1])
            Kgate(r) | q[0]
            Rgate(-r) | q[0]
            Kgate(r) | q[1]
            Rgate(-r) | q[1]

    state = eng.run(prog).state
    probs = [state.fock_prob(i) for i in measure_states]
    probs = np.array(probs).T.flatten()

    if batch_size is not None:
        results = np.tile(results, batch_size).flatten()

    assert np.allclose(probs, results, atol=tol, rtol=0)


@pytest.mark.backends("gaussian")
class TestGaussianCloning:
    """Tests specific to Gaussian cloning"""

    @staticmethod
    def gaussian_cloning_circuit(q):
        # TODO: at some point, use the blackbird parser to
        # read in the following directly from the examples folder
        BSgate() | (q[0], q[1])
        BSgate() | (q[1], q[2])
        MeasureX | q[1]
        MeasureP | q[2]
        Xgate(scale(q[1], np.sqrt(2))) | q[0]
        Zgate(scale(q[2], np.sqrt(2))) | q[0]
        BSgate() | (q[0], q[3])

    def test_identical_output(self, setup_eng, tol):
        """Test that all outputs are identical clones"""
        a = 0.7 + 1.2j

        eng, prog = setup_eng(4)
        with prog.context as q:
            Coherent(a) | q[0]
            self.gaussian_cloning_circuit(q)

        state = eng.run(prog, run_options={'modes': [0, 3]}).state
        coh = np.array([state.is_coherent(i) for i in range(2)])
        disp = state.displacement()

        # check all outputs are coherent states
        assert np.all(coh)
        # check outputs are identical clones
        assert np.allclose(*disp, atol=tol, rtol=0)

    def test_average_fidelity(self, setup_eng):
        """Test that gaussian cloning clones a Gaussian state with average fidelity 2/3"""
        shots = 500
        a = 0.7 + 1.2j

        eng, prog = setup_eng(4)
        with prog.context as q:
            Coherent(a) | q[0]
            self.gaussian_cloning_circuit(q)

        f_list = np.empty([shots])
        a_list = np.empty([shots], dtype=np.complex128)

        for i in range(shots):
            state = eng.run(prog, run_options={'modes': [0]}).state
            eng.reset()
            f_list[i] = state.fidelity_coherent([0.7 + 1.2j])
            a_list[i] = state.displacement()

        assert np.allclose(np.mean(f_list), 2.0 / 3.0, atol=0.1, rtol=0)
        assert np.allclose(np.mean(a_list), a, atol=0.1, rtol=0)
        assert np.allclose(
            np.cov([a_list.real, a_list.imag]), 0.25 * np.identity(2), atol=0.1, rtol=0
        )

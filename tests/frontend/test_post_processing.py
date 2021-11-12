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
"""
Unit tests for strawberryfields.utils.post_processing
"""
import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils.post_processing import (
    samples_expectation,
    samples_variance,
    all_fock_probs_pnr,
    _check_modes,
    _check_samples,
)

# pylint: disable=bad-continuation,no-self-use,pointless-statement

pytestmark = pytest.mark.api


def fock_states_samples():
    """Sample circuit for sampling fock states."""
    prog = sf.Program(4)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})

    with prog.context as q:
        Fock(2) | q[0]
        Fock(3) | q[1]
        Fock(4) | q[2]
        MeasureFock() | q[0]
        MeasureFock() | q[1]
        MeasureFock() | q[2]

    results = eng.run(prog, shots=1)
    return results.samples


def entangled_gaussian_samples():
    """Obtaining multiple samples from a circuit that generates entanglement on
    the gaussian backend."""
    prog = sf.Program(3)
    eng = sf.Engine("gaussian")

    with prog.context as q:
        S2gate(1) | (q[0], q[1])
        MeasureFock() | (q[0], q[1])

    results = eng.run(prog, shots=5)
    return results.samples


class TestNumberExpectation:
    """Tests the samples_expectation function using PNR samples."""

    @pytest.mark.parametrize("expval, modes", [(2, [0]), (2 * 3, [0, 1]), (2 * 3 * 4, [0, 1, 2])])
    def test_fock_backend_integration_expval(self, expval, modes):
        """Checking the expectation values when fock states were sampled with a single shot."""
        samples = fock_states_samples()
        assert samples_expectation(samples, modes) == expval

    def test_fock_backend_integration_modes_implicit(self):
        """Checking the expectation values when modes are specified implicitly."""
        samples = fock_states_samples()
        expval = 2 * 3 * 4
        assert samples_expectation(samples) == expval

    def test_two_mode_squeezed_expval(self):
        """Checking that the expectation value of samples matches if there was
        correlation."""
        mode1 = [0]
        mode2 = [1]
        samples = entangled_gaussian_samples()
        expval1 = samples_expectation(samples, mode1)
        expval2 = samples_expectation(samples, mode2)
        assert expval1 == expval2


class TestNumberVariance:
    """Tests the samples_variance function using PNR samples."""

    @pytest.mark.parametrize("var, modes", [(0, [0]), (0, [0, 1]), (0, [0, 1, 2])])
    def test_fock_backend_integration_var(self, var, modes):
        """Checking the variance when fock states were sampled with a single shot."""
        samples = fock_states_samples()
        assert samples_variance(samples, modes) == var

    def test_fock_backend_integration_modes_implicit(self):
        """Checking the variance when modes are specified implicitly."""
        samples = fock_states_samples()
        var = 0
        assert samples_variance(samples) == var

    def test_two_mode_squeezed_var(self):
        """Checking that the variance of samples matches if there was
        correlation."""
        mode1 = [0]
        mode2 = [1]
        samples = entangled_gaussian_samples()
        var1 = samples_variance(samples, mode1)
        var2 = samples_variance(samples, mode2)
        assert var1 == var2


homodyne_samples = [
    (np.array([[1.23]]), 1.23, 0),
    (np.array([[1.23], [12.32], [0.3222], [0]]), 3.46805, 26.3224074075),
    (np.array([[12.32, 0.32]]), 3.9424, 0),
    (np.array([[1.23, 0], [12.32, 0.32], [0.3222, 6.34], [0, 3.543]]), 1.496287, 2.689959501507),
]


class TestQuadratureExpectation:
    """Tests the samples_expectation function using homodyne
    samples."""

    @pytest.mark.parametrize(
        "samples, expval", [(samples, expval) for samples, expval, _ in homodyne_samples]
    )
    def test_quadrature_expval(self, samples, expval):
        """Checking the expectation value of pre-defined homodyne samples."""
        assert np.isclose(samples_expectation(samples), expval)


class TestQuadratureVariance:
    """Tests the samples_variance function using homodyne
    samples."""

    @pytest.mark.parametrize(
        "samples, var", [(samples, var) for samples, _, var in homodyne_samples]
    )
    def test_quadrature_variance(self, samples, var):
        """Checking the variance of pre-defined homodyne samples."""
        assert np.isclose(samples_variance(samples), var)


def validation_circuit():
    """Returns samples from an example circuit."""
    prog = sf.Program(3)
    eng = sf.Engine("gaussian")

    with prog.context as q:
        Sgate(0.5) | q[0]
        Sgate(0.5) | q[1]
        MeasureFock() | q[0]

    results = eng.run(prog, shots=5)
    return results.samples


class TestFockProb:
    """Tests the all_fock_probs_pnr function using PNR samples."""

    def test_fock_states(self):
        """Check the probabilities of Fock states based on samples from a
        circuit preparing Fock states."""
        samples = fock_states_samples()
        probs = all_fock_probs_pnr(samples)
        assert probs[(2, 3, 4)] == 1

        prob_sum = np.sum(probs)
        assert np.allclose(prob_sum, 1)

    def test_probs_from_correlated_samples(self):
        """Check the probabilities of Fock states based on correlated
        samples."""
        samples = entangled_gaussian_samples()
        probs = all_fock_probs_pnr(samples)

        # Check that each mode has the same outcome (there should be only
        # diagonal terms in the array)
        diagonal_sum = np.sum(probs.diagonal())
        assert np.allclose(diagonal_sum, 1)

        prob_sum = np.sum(probs)
        assert np.allclose(prob_sum, 1)


class TestInputValidation:
    """Tests for the input validation logic for post-processing samples."""

    invalid_samples = [[1, 2], np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])]

    @pytest.mark.parametrize("samples", invalid_samples)
    def test_invalid_samples(self, samples):
        """Tests that an error is raised if either the type or the shape of the
        input samples is incorrect."""
        modes = [0, 1]
        with pytest.raises(
            ValueError, match="Samples needs to be represented as a two dimensional NumPy array."
        ):
            _check_samples(samples)

    # Arbitrary sequences that are considered to have invalid types during the
    # input checks
    invalid_type_modes_sequences = [
        np.array([list([0]), list([1, 2])]),
    ]

    @pytest.mark.parametrize("modes", invalid_type_modes_sequences)
    def test_invalid_modes_sequence(self, modes):
        """Tests that an error is raised if the modes array specified contain
        objects with an invalid type (e.g. if the array contains lists)."""
        samples = validation_circuit()
        with pytest.raises(
            TypeError,
            match="The input modes need to be specified as a flattened sequence of non-negative integers",
        ):
            _check_modes(samples, modes)

    non_index_modes_sequences = [
        np.array([[0, 1], [1, 2]]),
        np.array([0.321]),
    ]

    @pytest.mark.parametrize("modes", non_index_modes_sequences)
    def test_non_index_modes(self, modes):
        """Tests that an error is raised if the modes were not specified as a
        flattened sequence of indices."""
        samples = validation_circuit()
        with pytest.raises(
            ValueError,
            match="The input modes need to be specified as a flattened sequence of non-negative integers",
        ):
            _check_modes(samples, modes)

    def test_invalid_modes_for_system(self):
        """Tests that an error is raised if the modes specified are not
        applicable for the system."""
        samples = validation_circuit()
        modes = [1, 2]

        # Need to escape [ and ] characters due to regular expression pattern matching
        invalid_modes = "\[1 2\]"
        with pytest.raises(
            ValueError,
            match="Cannot specify mode indices {} for a 1 mode system.".format(invalid_modes),
        ):
            _check_modes(samples, modes)

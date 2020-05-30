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
Unit tests for strawberryfields.api.post_processing
"""
import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.api.post_processing import _check_samples_modes

# pylint: disable=bad-continuation,no-self-use,pointless-statement

pytestmark = pytest.mark.api

def sample_circuit():
    """Returns samples from an example circuit."""
    prog = sf.Program(3)
    eng = sf.Engine("gaussian")

    with prog.context as q:
        Fock(2) | q[0]
        Sgate(0.5) | q[0]
        Sgate(0.5) | q[1]
        MeasureFock() | q[0]

    results = eng.run(prog, shots=5)
    return results.samples

def fock_states_samples():
    """TODO"""
    prog = sf.Program(3)
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

print(results.samples)
class TestNumberExpectation:
    """TODO"""

    @parametrize("expval, modes", [(2, [0]), (2*3, [0,1]), (2*3*4, [0,1,2])])
    def test_fock_backend(self, expval, modes):
        samples = fock_states_samples 
        assert number_expectation(samples, modes) == expval
        

class TestInputValidation:
    """Tests for the input validation logic for post-processing samples."""

    invalid_samples = [[1,2], np.array([[[1,2], [1,2]],[[1,2], [1,2]]])]

    @pytest.mark.parametrize("samples", invalid_samples)
    def test_invalid_samples(self, samples):
        """Tests that an error is raised if either the type or the shape of the
        input samples is incorrect."""
        modes = [0,1]
        with pytest.raises(
            Exception, match="Samples needs to be represented as a two dimensional numpy array."
        ):
            _check_samples_modes(samples, modes)

    invalid_modes_sequences = [np.array([list([0]), list([1,2])]),
            np.array([[0,1], [1,2]]), np.array([0.321])]

    @pytest.mark.parametrize("modes", invalid_modes_sequences)
    def test_invalid_modes_sequence(self, modes):
        """Tests that an error is raised if the modes were not specified as a
        flattened sequence of indices."""
        samples = sample_circuit()
        with pytest.raises(
            Exception, match="The input modes need to be specified as a flattened sequence of indices"
        ):
            _check_samples_modes(samples, modes)

    def test_invalid_modes_for_system(self):
        """Tests that an error is raised if the modes specified are not
        applicable for the system."""
        samples = sample_circuit()
        modes = [3, 4]

        # Need to escape [ and ] characters due to regular expression patter matching
        invalid_modes = "\[3 4\]"
        with pytest.raises(
            Exception, match="Cannot specify mode\(s\) {} for a 2 mode system!".format(invalid_modes)
        ):
            _check_samples_modes(samples, modes)

    def test_not_measured_modes_specified(self):
        """Tests that an error is raised if the some modes specified were not
        measured."""
        samples = sample_circuit()
        modes = [0,1,2]

        # Need to escape [ and ] characters due to regular expression patter matching
        not_measured_modes = "\[1 2\]"
        with pytest.raises(
            Exception, match="{} were specified for post-processing, but no samples".format(not_measured_modes)
        ):
            _check_samples_modes(samples, modes)

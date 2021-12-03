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
Unit tests for strawberryfields.result
"""
import numpy as np
import pytest
from strawberryfields.backends.states import BaseGaussianState

from strawberryfields.result import Result

pytestmark = pytest.mark.api


raw_results = {
    "output": [np.ones((2, 3, 4, 5))],
    "meta_array": np.array([1, 0, 1, 2]),
    "meta_matrix": np.array([[1, 2], [3, 4]]),
}


class TestResult:
    """Tests for the ``Result`` class."""

    def test_samples(self):
        """Test that ``samples`` is correctly returned."""
        result = Result(raw_results)
        assert result.samples is not None
        assert np.array_equal(result.samples, np.ones((2, 3, 4, 5)))

    def test_raw(self):
        """Test that raw results are correctly returned."""
        result = Result(raw_results)
        assert result.raw == raw_results

    def test_samples_dict(self):
        """Test that ``samples_dict`` is correctly returned."""
        samples_dict = {0: [1, 2, 3], 1: [4, 5]}
        result = Result(raw_results, samples_dict=samples_dict)
        assert result.samples_dict == samples_dict

    def test_ancillae_samples(self):
        """Test that ancilla samples are correctly returned."""
        ancillae_samples = {0: [0, 1], 2: [1, 3]}
        result = Result(raw_results, ancillae_samples=ancillae_samples)
        assert result.ancillae_samples == ancillae_samples

    def test_state(self):
        """Test that the state can be correctly set and returned, and that the
        correct error is raised when attempting to set it again."""
        state = BaseGaussianState((np.array([0.0, 0.0]), np.identity(2)), 1)
        result = Result(raw_results, samples_dict={0: [0, 1, 0], 2: [0]})
        result.state = state
        assert result.state == state

        with pytest.raises(TypeError, match="State already set and cannot be changed."):
            result.state = state

    def test_state_no_modes(self):
        """Test that the correct error is raised when setting a state on remote job result."""
        state = BaseGaussianState((np.array([0.0, 0.0]), np.identity(2)), 1)
        result = Result(raw_results)

        with pytest.raises(ValueError, match="State can only be set for local simulations."):
            result.state = state

    def test_metadata(self):
        """Test that metadata is correctly returned."""
        result = Result(raw_results)
        expected = {
            "meta_array": np.array([1, 0, 1, 2]),
            "meta_matrix": np.array([[1, 2], [3, 4]]),
        }

        assert result.metadata.keys() == expected.keys()
        for key, val in result.metadata.items():
            assert np.allclose(val, expected[key])


class TestResultPrint:
    """Tests for printing the ``Result`` class."""

    def test_stateless_print(self, capfd):
        """Test that printing a result object with no state provides the correct output."""
        result = Result({"output": [np.array([[1, 2], [3, 4], [5, 6]])]})
        print(result)
        captured = capfd.readouterr()
        assert "modes=2" in captured.out
        assert "shots=3" in captured.out
        assert "contains state=False" in captured.out

    def test_state_print(self, capfd):
        """Test that printing a result object with a state provides the correct output."""
        samples = {"output": [np.array([[1, 2], [3, 4], [5, 6]])]}
        samples_dict = {0: [1, 2, 3, 4, 5, 6]}

        result = Result(samples, samples_dict=samples_dict)
        result.state = BaseGaussianState((np.array([0.0, 0.0]), np.identity(2)), 1)
        print(result)
        captured = capfd.readouterr()
        assert "modes=2" in captured.out
        assert "shots=3" in captured.out
        assert "contains state=True" in captured.out

    def test_tdm_print(self, capfd):
        """Test that printing a result object with TDM samples provides the correct output."""
        samples = np.ones((2, 3, 4))
        result = Result({"output": [samples]})
        print(result)
        captured = capfd.readouterr()
        assert "spatial_modes=3" in captured.out
        assert "shots=2" in captured.out
        assert "timebins=4" in captured.out
        assert "contains state=False" in captured.out

    def test_unknown_shape_print(self, capfd):
        """Test that printing a result object with samples with an unknown shape
        provides the correct output."""
        samples = np.ones((2, 3, 4, 5))
        result = Result({"output": [samples]})
        print(result)
        captured = capfd.readouterr()
        assert "modes" not in captured.out
        assert "shots" not in captured.out
        assert "timebins" not in captured.out
        assert "contains state=False" in captured.out

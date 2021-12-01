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
    0: [1, 2, 3],
    1: [4, 5],
    "other_data": [1, 0, 1, 2],
    "string": "metadata",
}


class TestResult:
    """Tests for the ``Result`` class."""

    def test_samples(self):
        """Test that ``samples`` is correctly returned."""
        result = Result(raw_results)
        assert result.samples is not None
        assert np.equal(result.samples, np.ones((2, 3, 4, 5))).all()

    def test_raw(self):
        """Test that raw results are correctly returned."""
        result = Result(raw_results)
        assert result.raw == raw_results

    def test_samples_dict(self):
        """Test that ``samples_dict`` is correctly returned."""
        result = Result(raw_results)
        assert result.samples_dict == {0: [1, 2, 3], 1: [4, 5]}

    def test_ancilla_samples(self):
        """Test that ancilla samples are correctly returned."""
        ancilla_samples = {0: [0, 1], 2: [1, 3]}
        result = Result(raw_results, ancilla_samples=ancilla_samples)
        assert result.ancilla_samples == ancilla_samples

    def test_state(self):
        """Test that the state can be correctly set and returned."""
        state = BaseGaussianState((np.array([0.0, 0.0]), np.identity(2)), 1)
        result = Result(raw_results)
        result.state = state
        assert result.state == state

        with pytest.raises(TypeError, match="State already set and cannot be changed."):
            result.state = state

    def test_state_no_modes(self):
        """Test that the correct error is raised when setting a state on remote job result."""
        raw_results_without_modes = {
            "output": [np.ones((2, 3, 4, 5))],
            "other_data": [1, 0, 1, 2],
            "string": "metadata",
        }
        state = BaseGaussianState((np.array([0.0, 0.0]), np.identity(2)), 1)
        result = Result(raw_results_without_modes)

        with pytest.raises(ValueError, match="State can only be set for local simulations."):
            result.state = state

    def test_metadata(self):
        """Test that metadata is correctly returned."""
        result = Result(raw_results)
        assert result.metadata == {
            0: [1, 2, 3],
            1: [4, 5],
            "other_data": [1, 0, 1, 2],
            "string": "metadata",
        }


class TestResultPrint:
    """Tests for printing the ``Result`` class."""

    def test_stateless_print(self, capfd):
        """Test that printing a result object with no state provides the correct output."""
        result = Result({"output": [np.array([[1, 2], [3, 4], [5, 6]])]})
        print(result)
        out, _ = capfd.readouterr()
        assert "modes=2" in out
        assert "shots=3" in out
        assert "contains state=False" in out

    def test_state_print(self, capfd):
        """Test that printing a result object with a state provides the correct output."""
        result = Result({"output": [np.array([[1, 2], [3, 4], [5, 6]])], 0: [1, 2, 3, 4, 5, 6]})
        result.state = [[1, 2], [3, 4]]
        print(result)
        out, _ = capfd.readouterr()
        assert "modes=2" in out
        assert "shots=3" in out
        assert "contains state=True" in out

    def test_tdm_print(self, capfd):
        """Test that printing a result object with TDM samples provides the correct output."""
        samples = np.ones((2, 3, 4))
        result = Result({"output": [samples]})
        print(result)
        out, _ = capfd.readouterr()
        assert "spatial_modes=3" in out
        assert "shots=2" in out
        assert "timebins=4" in out
        assert "contains state=False" in out

    def test_unknown_shape_print(self, capfd):
        """Test that printing a result object with samples with an unknown shape
        provides the correct output."""
        samples = np.ones((2, 3, 4, 5))
        result = Result({"output": [samples]})
        print(result)
        out, _ = capfd.readouterr()
        assert "modes" not in out
        assert "shots" not in out
        assert "timebins" not in out
        assert "contains state=False" in out

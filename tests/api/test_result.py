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

from strawberryfields.result import Result

pytestmark = pytest.mark.api


class TestResult:
    """Tests for the ``Result`` class."""

    def test_stateless_print(self, capfd):
        """Test that printing a result object with no state provides the correct output."""
        result = Result({"output": [np.array([[1, 2], [3, 4], [5, 6]])]})
        print(result)
        out, err = capfd.readouterr()
        assert "modes=2" in out
        assert "shots=3" in out
        assert "contains state=False" in out

    def test_state_print(self, capfd):
        """Test that printing a result object with a state provides the correct output."""
        result = Result({"output": [np.array([[1, 2], [3, 4], [5, 6]])]})
        result._state = [[1, 2], [3, 4]]
        print(result)
        out, err = capfd.readouterr()
        assert "modes=2" in out
        assert "shots=3" in out
        assert "contains state=True" in out

    def test_tdm_print(self, capfd):
        """Test that printing a result object with TDM samples provides the correct output."""
        samples = np.ones((2, 3, 4))
        result = Result({"output": [samples]})
        print(result)
        out, err = capfd.readouterr()
        assert "spatial_modes=3" in out
        assert "shots=2" in out
        assert "timebins=4" in out
        assert "contains state=False" in out

    def test_unkown_shape_print(self, capfd):
        """Test that printing a result object with samples with an unknown shape
        provides the correct output."""
        samples = np.ones((2, 3, 4, 5))
        result = Result({"output": [samples]})
        print(result)
        out, err = capfd.readouterr()
        assert "modes" not in out
        assert "shots" not in out
        assert "timebins" not in out
        assert "contains state=False" in out

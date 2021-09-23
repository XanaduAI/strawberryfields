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
Unit tests for strawberryfields.api.result
"""
import numpy as np
import pytest

from strawberryfields.api import Result

# pylint: disable=bad-continuation,no-self-use,pointless-statement

pytestmark = pytest.mark.api


class TestResult:
    """Tests for the ``Result`` class."""

    def test_stateless_result_raises_on_state_access(self):
        """Tests that `result.state` raises an error for a stateless result."""
        result = Result(np.array([[1, 2], [3, 4]]), is_stateful=False)

        with pytest.raises(
            AttributeError, match="The state is undefined for a stateless computation."
        ):
            result.state

    def test_stateless_print(self, capfd):
        """Test that printing a result object with no state provides the correct output."""
        result = Result(np.array([[1, 2], [3, 4], [5, 6]]), is_stateful=False)
        print(result)
        out, err = capfd.readouterr()
        assert "modes=2" in out
        assert "shots=3" in out
        assert "contains state=False" in out

    def test_state_print(self, capfd):
        """Test that printing a result object with a state provides the correct output."""
        result = Result(np.array([[1, 2], [3, 4], [5, 6]]), is_stateful=True)
        print(result)
        out, err = capfd.readouterr()
        assert "modes=2" in out
        assert "shots=3" in out
        assert "contains state=True" in out

    @pytest.mark.parametrize("stateful", [True, False])
    def test_tdm_print(self, stateful, capfd):
        """Test that printing a result object with TDM samples provides the correct output."""
        samples = np.ones((2, 3, 4))
        result = Result(samples, is_stateful=stateful)
        print(result)
        out, err = capfd.readouterr()
        assert "spatial_modes=3" in out
        assert "shots=2" in out
        assert "timebins=4" in out
        assert f"contains state={stateful}" in out

    @pytest.mark.parametrize("stateful", [True, False])
    def test_unkown_shape_print(self, stateful, capfd):
        """Test that printing a result object with samples with an unknown shape
        provides the correct output."""
        samples = np.ones((2, 3, 4, 5))
        result = Result(samples, is_stateful=stateful)
        print(result)
        out, err = capfd.readouterr()
        assert "modes" not in out
        assert "shots" not in out
        assert "timebins" not in out
        assert f"contains state={stateful}" in out

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
"""TODO"""
import pytest

from strawberryfields.api import Result


class TestResult:
    """Tests for the ``Result`` class."""

    def test_stateless_result_raises_on_state_access(self):
        """Tests that `result.state` raises an error for a stateless result.
        """
        result = Result([[1, 2], [3, 4]], is_stateful=False)

        with pytest.raises(AttributeError, match="The state is undefined for a stateless computation."):
            _ = result.state

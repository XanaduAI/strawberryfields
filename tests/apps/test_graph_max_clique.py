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
r"""
Unit tests for strawberryfields.apps.graph.max_clique
"""
# pylint: disable=no-self-use,unused-argument
import pytest

from strawberryfields.apps.graph import max_clique


class TestLocalSearch:
    """Tests for the function ``max_clique.local_search``"""

    def test_bad_iterations(self, graph):
        """Test if function raises a ``ValueError`` when a non-positive number of iterations is
        specified"""
        with pytest.raises(ValueError, match="Number of iterations must be a positive int"):
            max_clique.local_search(clique=[0, 1, 2, 3], graph=graph, iterations=-1)

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
Unit tests for strawberryfields.apps.graph.similarity
"""
# pylint: disable=no-self-use,unused-argument,too-many-arguments
import itertools
import pytest

from strawberryfields.apps.graph import similarity

pytestmark = pytest.mark.apps


@pytest.mark.parametrize("dim", [2, 3])
def test_sample_to_orbit(dim):
    """Test if function ``similarity.sample_to_orbit`` correctly returns the original orbit after
    taking all permutations over the orbit. The starting orbit is a fixed repetition of 2's,
    followed by a repetition of 1's, each repeated ``dim`` times, and in a system of ``3 * dim``
    modes."""
    orbit = [2] * dim + [1] * dim
    sorted_sample = [2] * dim + [1] * dim + [0] * dim
    permutations = itertools.permutations(sorted_sample)
    assert all([similarity.sample_to_orbit(p) == orbit for p in permutations])

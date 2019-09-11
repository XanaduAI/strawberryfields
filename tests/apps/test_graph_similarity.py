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


def test_orbits():
    """Test if function ``similarity.orbits`` correctly returns the integer partitions of 5.
    This test does not require ``similarity.orbits`` to return the orbits in any specified order."""
    partition = [
        [5],
        [4, 1],
        [3, 2],
        [3, 1, 1],
        [2, 1, 1, 1],
        [2, 2, 1],
        [1, 1, 1, 1, 1],
    ]
    orbits = list(similarity.orbits(5))
    try:
        for o in orbits:
            partition.remove(o)
    except ValueError:
        pass
    assert not partition


@pytest.mark.parametrize("dim", [3, 4, 5])
@pytest.mark.parametrize("max_count_per_mode", [1, 2])
def test_sample_to_event(dim, max_count_per_mode):
    """Test if function ``similarity.sample_to_event`` gives the correct set of events when
    applied to all orbits with a fixed number of photons ``dim``. This test ensures: (i) that the
    only events returned are either ``dim`` or ``None``; (ii) that orbits exceeding the
    ``max_count_per_mode`` value are attributed the ``None`` event; and that (iii) that orbits
    not exceeding the ``max_count_per_mode`` are attributed the event ``dim``."""
    orbits = list(similarity.orbits(dim))
    events = [similarity.sample_to_event(o, max_count_per_mode) for o in orbits]

    max_count_correct = True
    photon_number_correct = True

    for i, e in enumerate(events):
        if max(orbits[i]) > max_count_per_mode:
            if e is not None:
                max_count_correct = False
        else:
            if e != dim:
                photon_number_correct = False

    assert set(events) == {dim, None}
    assert max_count_correct
    assert photon_number_correct

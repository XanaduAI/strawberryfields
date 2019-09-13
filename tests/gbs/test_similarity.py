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
Unit tests for strawberryfields.gbs.similarity
"""
# pylint: disable=no-self-use,unused-argument,too-many-arguments
import itertools
from collections import Counter

import pytest

from strawberryfields.gbs import similarity

pytestmark = pytest.mark.apps

all_orbits = {
    3: [[1, 1, 1], [2, 1], [3]],
    4: [[1, 1, 1, 1], [2, 1, 1], [3, 1], [2, 2], [4]],
    5: [[1, 1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1], [2, 2, 1], [4, 1], [3, 2], [5]],
}

all_orbits_cumulative = [o for orbs in all_orbits.values() for o in orbs]

all_events = {
    (3, 1): [3, None, None],
    (4, 1): [4, None, None, None, None],
    (5, 1): [5, None, None, None, None, None, None],
    (3, 2): [3, 3, None],
    (4, 2): [4, 4, None, 4, None],
    (5, 2): [5, 5, None, 5, None, None, None],
}


@pytest.mark.parametrize("dim", [3, 4, 5])
def test_sample_to_orbit(dim):
    """Test if function ``similarity.sample_to_orbit`` correctly returns the original orbit after
    taking all permutations over the orbit. The starting orbits are all orbits for a fixed photon
    number ``dim``."""
    orbits = all_orbits[dim]
    checks = []
    for o in orbits:
        sorted_sample = o.copy()
        sorted_sample_len = len(sorted_sample)
        if sorted_sample_len != dim:
            sorted_sample += [0] * sorted_sample_len
        permutations = itertools.permutations(sorted_sample)
        checks.append(all([similarity.sample_to_orbit(p) == o for p in permutations]))
    assert all(checks)


@pytest.mark.parametrize("dim", [3, 4, 5])
class TestOrbits:
    """Tests for the function ``strawberryfields.gbs.similarity.orbits``"""

    def test_orbit_sum(self, dim):
        """Test if function generates orbits that are lists that sum to ``dim``."""
        assert all([sum(o) == dim for o in similarity.orbits(dim)])

    def test_orbit_sorted(self, dim):
        """Test if function generates orbits that are lists sorted in descending order."""
        assert all([o == sorted(o, reverse=True) for o in similarity.orbits(dim)])

    def test_orbits(self, dim):
        """Test if function returns all the integer partitions of 5. This test does not
        require ``similarity.orbits`` to return the orbits in any specified order."""
        partition = all_orbits[dim]
        orbits = similarity.orbits(dim)

        assert sorted(partition) == sorted(orbits)


@pytest.mark.parametrize("dim", [3, 4, 5])
@pytest.mark.parametrize("max_count_per_mode", [1, 2])
def test_sample_to_event(dim, max_count_per_mode):
    """Test if function ``similarity.sample_to_event`` gives the correct set of events when
    applied to all orbits with a fixed number of photons ``dim``. This test ensures that orbits
    exceeding the ``max_count_per_mode`` value are attributed the ``None`` event and that orbits
    not exceeding the ``max_count_per_mode`` are attributed the event ``dim``."""
    orbits = all_orbits[dim]
    target_events = all_events[(dim, max_count_per_mode)]
    events = [similarity.sample_to_event(o, max_count_per_mode) for o in orbits]

    assert events == target_events


class TestOrbitToSample:
    """Tests for the function ``strawberryfields.apps.graph.similarity.orbit_to_sample``"""

    def test_low_modes(self):
        """Test if function raises a ``ValueError`` if fed an argument for ``modes`` that does
        not exceed the length of the input orbit."""
        with pytest.raises(ValueError, match="Number of modes cannot"):
            similarity.orbit_to_sample([1, 2, 3], 2)

    @pytest.mark.parametrize("orb_dim", [3, 4, 5])
    @pytest.mark.parametrize("modes_dim", [6, 7])
    def test_sample_length(self, orb_dim, modes_dim):
        """Test if function returns a sample that is of correct length ``modes_dim`` when fed a
        collision-free event of ``orb_dim`` photons."""
        samp = similarity.orbit_to_sample(all_orbits[orb_dim][0], modes_dim)
        assert len(samp) == modes_dim

    def test_sample_composition(self):
        """Test if function returns a sample that corresponds to the input orbit. Input orbits
        are orbits from ``all_orbits_cumulative``, i.e., all orbits from 3-5 photons. This test
        checks if a sample corresponds to an orbit by counting the occurrence of elements in the
        sample and comparing to a count of elements in the orbit."""
        modes = 5

        all_orbits_zeros = [
            [1, 1, 1, 0, 0],
            [2, 1, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [2, 1, 1, 0, 0],
            [3, 1, 0, 0, 0],
            [2, 2, 0, 0, 0],
            [4, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 1, 1, 1, 0],
            [3, 1, 1, 0, 0],
            [2, 2, 1, 0, 0],
            [4, 1, 0, 0, 0],
            [3, 2, 0, 0, 0],
            [5, 0, 0, 0, 0],
        ]  # padding orbits with zeros at the end for comparison to samples

        counts = [Counter(similarity.orbit_to_sample(o, modes)) for o in all_orbits_cumulative]
        ideal_counts = [Counter(o) for o in all_orbits_zeros]

        assert counts == ideal_counts


class TestEventToSample:
    """Tests for the function ``strawberryfields.apps.graph.similarity.event_to_sample``"""

    def test_low_count(self):
        """Test if function raises a ``ValueError`` if ``max_count_per_mode`` is not positive."""
        with pytest.raises(ValueError, match="Maximum number of photons"):
            similarity.event_to_sample(2, 0, 5)

    def test_high_photon(self):
        """Test if function raises a ``ValueError`` if ``photon_number`` is so high that it
        cannot correspond to a sample given the constraints of ``max_count_per_mode`` and
        ``modes``"""
        with pytest.raises(ValueError, match="No valid samples can be generated."):
            similarity.event_to_sample(5, 1, 4)

    @pytest.mark.parametrize("photon_num", [5, 6])
    @pytest.mark.parametrize("modes_dim", [10, 11])
    @pytest.mark.parametrize("count", [3, 4])
    def test_sample_length(self, photon_num, modes_dim, count):
        """Test if function returns a sample that is of correct length ``modes_dim``."""
        samp = similarity.event_to_sample(photon_num, count, modes_dim)
        assert len(samp) == modes_dim

    @pytest.mark.parametrize("photon_num", [5, 6])
    @pytest.mark.parametrize("modes_dim", [10, 11])
    @pytest.mark.parametrize("count", [3, 4])
    def test_sample_sum(self, photon_num, modes_dim, count):
        """Test if function returns a sample that has the correct number of photons."""
        samp = similarity.event_to_sample(photon_num, count, modes_dim)
        assert sum(samp) == photon_num

    @pytest.mark.parametrize("photon_num", [5, 6])
    @pytest.mark.parametrize("modes_dim", [10, 11])
    @pytest.mark.parametrize("count", [3, 4])
    def test_sample_max_count(self, photon_num, modes_dim, count):
        """Test if function returns a sample that has maximum element not exceeding ``count``."""
        samp = similarity.event_to_sample(photon_num, count, modes_dim)
        assert max(samp) <= count

    @pytest.mark.parametrize("photon_num", [5, 6])
    @pytest.mark.parametrize("count", [3, 4])
    def test_sample_max_count_deterministic(self, photon_num, count, monkeypatch):
        """Test if function correctly stops adding photons to modes who have reached their
        maximum count. This test ensures that the maximum count is exceeded in each case by
        setting ``photon_num > count`` and monkeypatching the random choice to always pick the
        first element of a list. This should cause the first mode to fill up with ``count``
        photons."""
        modes_dim = 10
        with monkeypatch.context() as m:
            m.setattr("numpy.random.choice", lambda x: x[0])
            samp = similarity.event_to_sample(photon_num, count, modes_dim)
        assert samp[0] == count

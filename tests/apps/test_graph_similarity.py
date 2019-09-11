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
    partition = [[5], [4, 1], [3, 2], [3, 1, 1], [2, 1, 1, 1], [2, 2, 1], [1, 1, 1, 1, 1]]
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


@pytest.mark.parametrize("dim", [3, 4, 5])
class TestUniformOrbit:

    """Tests for the function ``uniform_sample_orbit``"""

    def test_bad_nr_modes(self, dim):
        """Tests if function raises a ``ValueError`` when the number of modes is smaller than the
        length of the orbit. The test creates a collision-free orbit of length `dim` and calls
        the function with `dim-1` modes."""
        with pytest.raises(ValueError, match='Number of modes cannot be smaller than length '
                                             'of orbit'):
            similarity.uniform_sample_orbit([1]*dim, dim-1)

    def test_correct_orbit(self, dim):
        """Tests if the generated sample belongs to the desired orbit. It generates an orbit with
        collisions and creates a uniform sample from that orbit. It then checks that the orbit of
        the sample is the desired one."""

        orbit = [2]*dim + [1]*dim
        sample = similarity.uniform_sample_orbit(orbit, 3*dim)

        assert similarity.sample_to_orbit(sample) == orbit


@pytest.mark.parametrize("dim", [3, 4, 5])
class TestUniformEvent:

    """Tests for the function ``uniform_sample_event``"""

    def test_max_count_per_mode(self, dim):
        """Tests if function raises a ``ValueError`` when the maximum number of photons per
        modes is smaller than 1."""
        with pytest.raises(ValueError, match='Maximum number of photons per mode must be equal or '
                                             'greater than 1'):
            similarity.uniform_sample_event(dim, 0, 2*dim)

    def test_valid_inputs(self, dim):
        """Tests if the function raises a ``ValueError`` when the arguments to the function
        cannot lead to a valid event. The test works by giving as input a number of photons that
        is too large to correspond to a valid event with the given number of modes and maximum
        count per mode."""

        with pytest.raises(ValueError, match="""No valid samples can be generated. 
        Consider increasing the max_count_per_mode or reducing the number of photons."""):
            similarity.uniform_sample_event(dim+1, 1, dim)

    def test_correct_event(self, dim):
        """Tests if the generated sample belongs to the desired event. The test """

        sample = similarity.uniform_sample_event(dim, 1, 2*dim)

        assert similarity.sample_to_event(sample) == dim




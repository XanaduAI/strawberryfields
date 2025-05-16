# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for strawberryfields.apps.similarity
"""
# pylint: disable=no-self-use,unused-argument,too-many-arguments
import itertools
from collections import Counter
from unittest import mock

import networkx as nx
import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields.apps import similarity

pytestmark = pytest.mark.apps

# dict of orbits. accessed using total photon number K
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
    orb = all_orbits[dim]
    checks = []
    for o in orb:
        sorted_sample = o.copy()
        sorted_sample_len = len(sorted_sample)
        if sorted_sample_len != dim:
            sorted_sample += [0] * sorted_sample_len
        permutations = itertools.permutations(sorted_sample)
        checks.append(all([similarity.sample_to_orbit(p) == o for p in permutations]))
    assert all(checks)


@pytest.mark.parametrize("dim", [3, 4, 5])
class TestOrbits:
    """Tests for the function ``strawberryfields.apps.similarity.orbits``"""

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
        orb = similarity.orbits(dim)

        assert sorted(partition) == sorted(orb)


@pytest.mark.parametrize("dim", [3, 4, 5])
@pytest.mark.parametrize("max_count_per_mode", [1, 2])
def test_sample_to_event(dim, max_count_per_mode):
    """Test if function ``similarity.sample_to_event`` gives the correct set of events when
    applied to all orbits with a fixed number of photons ``dim``. This test ensures that orbits
    exceeding the ``max_count_per_mode`` value are attributed the ``None`` event and that orbits
    not exceeding the ``max_count_per_mode`` are attributed the event ``dim``."""
    orb = all_orbits[dim]
    target_events = all_events[(dim, max_count_per_mode)]
    ev = [similarity.sample_to_event(o, max_count_per_mode) for o in orb]

    assert ev == target_events


class TestOrbitToSample:
    """Tests for the function ``strawberryfields.apps.similarity.orbit_to_sample``"""

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
    """Tests for the function ``strawberryfields.apps.similarity.event_to_sample``"""

    def test_low_count(self):
        """Test if function raises a ``ValueError`` if ``max_count_per_mode`` is negative."""
        with pytest.raises(ValueError, match="Maximum number of photons"):
            similarity.event_to_sample(2, -1, 5)

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


orbits = [
    [(1, 1, 2), 4, 12],
    [(1, 1), 4, 6],
    [(1, 2, 3), 4, 24],
    [(1, 1, 1, 1), 5, 5],
    [(1, 1, 2), 5, 30],
    [(1, 2, 3), 5, 60],
    [(2, 1), 171, 29070.0],
]


@pytest.mark.parametrize("orbit, max_photon, expected", orbits)
def test_orbit_cardinality(orbit, max_photon, expected):
    """Test if function ``strawberryfields.apps.similarity.orbit_cardinality`` returns the
    correct number of samples for some hard-coded examples."""

    assert np.allclose(similarity.orbit_cardinality(list(orbit), max_photon), expected)


events = [
    [5, 3, 6, 216],
    [6, 3, 6, 336],
    [5, 2, 6, 126],
    [5, 3, 7, 413],
    [6, 3, 7, 728],
    [5, 2, 7, 266],
]


@pytest.mark.parametrize("photons, max_count, modes, expected", events)
def test_event_cardinality(photons, max_count, modes, expected):
    """Test if function ``strawberryfields.apps.similarity.event_cardinality`` returns the
    correct number of samples for some hard-coded examples."""

    assert similarity.event_cardinality(photons, max_count, modes) == expected


class TestGetState:
    """Tests for the function ``strawberryfields.apps.similarity._get_state``"""

    def test_loss(self, monkeypatch):
        """Test if function correctly creates the SF program for lossy GBS."""
        graph = nx.complete_graph(5)
        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.LocalEngine, "run", mock_eng_run)
            similarity._get_state(graph, loss=0.5)
            p_func = mock_eng_run.call_args[0][0]

        assert isinstance(p_func.circuit[1].op, sf.ops.LossChannel)

    def test_no_loss(self, monkeypatch):
        """Test if function correctly creates the SF program for GBS without loss."""
        graph = nx.complete_graph(5)
        mock_eng_run = mock.MagicMock()

        with monkeypatch.context() as m:
            m.setattr(sf.LocalEngine, "run", mock_eng_run)
            similarity._get_state(graph)
            p_func = mock_eng_run.call_args[0][0]

        assert not all([isinstance(op, sf.ops.LossChannel) for op in p_func.circuit])

    def test_max_loss(self):
        """Test if function samples from the vacuum when maximum loss is applied."""
        dim = 5
        graph = nx.complete_graph(dim)
        state = similarity._get_state(graph, 0, 1)
        cov = state.cov()
        disp = state.displacement()

        assert np.allclose(cov, 0.5 * state.hbar * np.eye(2 * dim))
        assert np.allclose(disp, np.zeros(dim))


class TestProbOrbitExact:
    """Tests for the function ``strawberryfields.apps.similarity.prob_orbit_exact``"""

    def test_invalid_n_mean(self):
        """Test if function raises a ``ValueError`` when the mean photon number is specified to
        be negative."""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Mean photon number must be non-negative"):
            similarity.prob_orbit_exact(g, [1, 1, 1, 1], n_mean=-1)

    def test_invalid_loss(self):
        """Test if function raises a ``ValueError`` when the loss parameter is specified outside
        of range."""
        g = nx.complete_graph(5)
        with pytest.raises(
            ValueError, match="Loss parameter must take a value between zero and one"
        ):
            similarity.prob_orbit_exact(g, [1, 1, 1, 1], loss=2)

    def test_prob_vacuum_orbit(self):
        """Tests if the function gives the right probability for the empty orbit when the GBS
        device has been configured to have zero mean photon number."""
        graph = nx.complete_graph(5)
        assert similarity.prob_orbit_exact(graph, [], 0) == 1.0

    @pytest.mark.parametrize("k", [1, 3, 5, 7, 9])
    def test_odd_photon_numbers(self, k):
        """Test if function returns zero probability for odd number of total photons."""
        graph = nx.complete_graph(10)
        assert similarity.prob_orbit_exact(graph, [k]) == 0.0

    def test_correct_result_returned(self, monkeypatch):
        """Tests if the call to _get_state function is performed correctly and probabilities over all
        permutations of the given orbit are summed correctly. The test monkeypatches the fock_prob
        function so that the probability is the same for each sample permutation and
        is equal to 1/8. For a 4-mode graph, [1, 1] has 6 possible permutations."""
        graph = nx.complete_graph(4)
        with monkeypatch.context() as m:
            m.setattr(
                "strawberryfields.backends.BaseGaussianState.fock_prob",
                lambda *args, **kwargs: 1 / 8,
            )
            assert similarity.prob_orbit_exact(graph, [1, 1]) == 6 / 8

    def test_known_result(self):
        """Tests if the probability for known cases is correctly
        reproduced."""
        g = nx.complete_graph(3)

        p = similarity.prob_orbit_exact(g, [1, 1], 2)
        assert np.allclose(p, 0.24221526825385403)

        s = similarity._get_state(g, 2)
        temp = s.fock_prob([1, 1, 0]) + s.fock_prob([1, 0, 1]) + s.fock_prob([0, 1, 1])
        assert np.allclose(p, temp)

        assert np.allclose(similarity.prob_orbit_exact(g, [4], 1), 0)


class TestProbEventExact:
    """Tests for the function ``strawberryfields.apps.similarity.prob_event_exact``"""

    def test_invalid_n_mean(self):
        """Test if function raises a ``ValueError`` when the mean photon number is specified to
        be negative."""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Mean photon number must be non-negative"):
            similarity.prob_event_exact(g, 2, 2, n_mean=-1)

    def test_invalid_loss(self):
        """Test if function raises a ``ValueError`` when the loss parameter is specified outside
        of range."""
        g = nx.complete_graph(5)
        with pytest.raises(
            ValueError, match="Loss parameter must take a value between zero and one"
        ):
            similarity.prob_event_exact(g, 2, 2, loss=2)

    def test_invalid_photon_number(self):
        """Test if function raises a ``ValueError`` when a photon number below zero is specified"""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Photon number must not be below zero"):
            similarity.prob_event_exact(g, -1, 2)

    def test_low_count(self):
        """Test if function raises a ``ValueError`` if ``max_count_per_mode`` is negative."""
        g = nx.complete_graph(5)
        with pytest.raises(
            ValueError, match="Maximum number of photons per mode must be non-negative"
        ):
            similarity.prob_event_exact(g, 2, max_count_per_mode=-1)

    def test_prob_vacuum_event(self):
        """Tests if the function gives the right probability for an event with zero photons when
        the GBS device has been configured to have zero mean photon number."""
        graph = nx.complete_graph(5)
        assert similarity.prob_event_exact(graph, 0, 0, 0) == 1.0

    @pytest.mark.parametrize("k", [3, 5, 7, 9])
    @pytest.mark.parametrize("nmax", [1, 2])
    def test_odd_photon_numbers(self, k, nmax):
        """Test if function returns zero probability for odd number of total photons."""
        graph = nx.complete_graph(10)
        assert similarity.prob_event_exact(graph, k, nmax) == 0.0

    def test_correct_result_returned(self, monkeypatch):
        """Tests if the call to _get_state function is performed correctly and probabilities over all
        constituent orbits of the given event are summed correctly. The test monkeypatches the fock_prob
        function so that the probability is the same for each sample permutation of all constituent orbits
        and is equal to 1/8. For a 4-mode graph, an event with ``photon_number = 2``, and
        ``max_count_per_mode = 1`` contains orbit [1, 1] which has 6 possible sample permutations."""
        graph = nx.complete_graph(4)
        with monkeypatch.context() as m:
            m.setattr(
                "strawberryfields.backends.BaseGaussianState.fock_prob",
                lambda *args, **kwargs: 1 / 8,
            )
            assert similarity.prob_event_exact(graph, 2, 1) == 6 / 8

    def test_known_result(self):
        """Tests if the probability for known cases is correctly
        reproduced."""
        graph = nx.complete_graph(4)
        p1 = similarity.prob_event_exact(graph, 2, 2, 1)
        p2 = similarity.prob_event_exact(graph, 2, 1, 1)
        p3 = similarity.prob_orbit_exact(graph, [1, 1], 1)
        p4 = similarity.prob_event_exact(graph, 2, 2, 4)
        assert np.allclose(p1 - p2, 0)
        assert np.allclose(p2, p3)
        assert np.allclose(p4, 0.21087781178526066)


class TestProbOrbitMC:
    """Tests for the function ``strawberryfields.apps.similarity.prob_orbit_mc``"""

    def test_invalid_samples(self):
        """Test if function raises a ``ValueError`` when a number of samples less than one is
        requested."""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Number of samples must be at least one"):
            similarity.prob_orbit_mc(g, [1, 1, 1, 1], samples=0)

    def test_invalid_n_mean(self):
        """Test if function raises a ``ValueError`` when the mean photon number is specified to
        be negative."""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Mean photon number must be non-negative"):
            similarity.prob_orbit_mc(g, [1, 1, 1, 1], n_mean=-1)

    def test_invalid_loss(self):
        """Test if function raises a ``ValueError`` when the loss parameter is specified outside
        of range."""
        g = nx.complete_graph(5)
        with pytest.raises(
            ValueError, match="Loss parameter must take a value between zero and one"
        ):
            similarity.prob_orbit_mc(g, [1, 1, 1, 1], loss=2)

    def test_mean_computation_orbit(self, monkeypatch):
        """Tests if the calculation of the sample mean is performed correctly. The test
        monkeypatches the fock_prob function so that the probability is the same for each sample and
        is equal to 1/5, i.e., one over the cardinality of the orbit [1,1,1,1] for 5 modes."""
        graph = nx.complete_graph(5)
        with monkeypatch.context() as m:
            m.setattr(
                "strawberryfields.backends.BaseGaussianState.fock_prob",
                lambda *args, **kwargs: 0.2,
            )
            assert np.allclose(similarity.prob_orbit_mc(graph, [1, 1, 1, 1]), 1.0)

    @pytest.mark.parametrize("k", [1, 3, 5, 7, 9])
    def test_odd_photon_numbers(self, k):
        """Test if function returns zero probability for odd number of total photons."""
        graph = nx.complete_graph(10)
        assert similarity.prob_orbit_mc(graph, [k]) == 0.0

    def test_prob_vacuum_orbit(self):
        """Tests if the function gives the right probability for the empty orbit when the GBS
        device has been configured to have zero mean photon number."""
        graph = nx.complete_graph(5)
        assert similarity.prob_orbit_mc(graph, [], 0) == 1.0

    def test_max_loss(self):
        """Test if function samples from the vacuum when maximum loss is applied."""
        graph = nx.complete_graph(5)
        assert similarity.prob_orbit_mc(graph, [1, 1, 1, 1], samples=1, loss=1) == 0.0

    def test_known_result(self):
        """Tests if the probability for known cases is correctly
        reproduced."""
        g = nx.complete_graph(3)

        assert np.allclose(similarity.prob_orbit_mc(g, [1, 1], 2), 0.2422152682538481)
        assert np.allclose(similarity.prob_orbit_mc(g, [4], 1), 0)


class TestProbEventMC:
    """Tests for the function ``strawberryfields.apps.similarity.prob_event_mc``"""

    def test_invalid_samples(self):
        """Test if function raises a ``ValueError`` when a number of samples less than one is
        requested."""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Number of samples must be at least one"):
            similarity.prob_event_mc(g, 2, 2, samples=0)

    def test_invalid_n_mean(self):
        """Test if function raises a ``ValueError`` when the mean photon number is specified to
        be negative."""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Mean photon number must be non-negative"):
            similarity.prob_event_mc(g, 2, 2, n_mean=-1)

    def test_invalid_loss(self):
        """Test if function raises a ``ValueError`` when the loss parameter is specified outside
        of range."""
        g = nx.complete_graph(5)
        with pytest.raises(
            ValueError, match="Loss parameter must take a value between zero and one"
        ):
            similarity.prob_event_mc(g, 2, 2, loss=2)

    def test_invalid_photon_number(self):
        """Test if function raises a ``ValueError`` when a photon number below zero is specified"""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Photon number must not be below zero"):
            similarity.prob_event_mc(g, -1, 2)

    def test_low_count(self):
        """Test if function raises a ``ValueError`` if ``max_count_per_mode`` is negative."""
        g = nx.complete_graph(5)
        with pytest.raises(
            ValueError, match="Maximum number of photons per mode must be non-negative"
        ):
            similarity.prob_event_mc(g, 2, max_count_per_mode=-1)

    def test_prob_vacuum_event(self):
        """Tests if the function gives the right probability for an event with zero photons when
        the GBS device has been configured to have zero mean photon number."""
        graph = nx.complete_graph(5)
        assert similarity.prob_event_mc(graph, 0, 0, 0) == 1.0

    def test_mean_computation_event(self, monkeypatch):
        """Tests if the calculation of the sample mean is performed correctly. The test
        monkeypatches the fock_prob function so that the probability is the same for each sample
        and is equal to 1/216, i.e., one over the number of samples in the event with 5 modes,
        6 photons, and max 3 photons per mode."""
        graph = nx.complete_graph(6)
        with monkeypatch.context() as m:
            m.setattr(
                "strawberryfields.backends.BaseGaussianState.fock_prob",
                lambda *args, **kwargs: 1.0 / 336,
            )
            assert np.allclose(similarity.prob_event_mc(graph, 6, 3), 1.0)

    @pytest.mark.parametrize("k", [3, 5, 7, 9])
    @pytest.mark.parametrize("nmax", [1, 2])
    def test_odd_photon_numbers(self, k, nmax):
        """Test if function returns zero probability for odd number of total photons."""
        graph = nx.complete_graph(10)
        assert similarity.prob_event_mc(graph, k, nmax) == 0.0

    def test_max_loss(self):
        """Test if function samples from the vacuum when maximum loss is applied."""
        graph = nx.complete_graph(6)
        assert similarity.prob_event_mc(graph, 6, 3, samples=1, loss=1) == 0.0

    def test_known_result(self):
        """Tests if the probability for known cases is correctly
        reproduced."""
        g = nx.complete_graph(4)
        p = similarity.prob_event_mc(g, 2, 1, 4)
        assert np.allclose(p, 0.2108778117852639)

        graph = nx.complete_graph(20)
        assert np.allclose(similarity.prob_event_mc(graph, 20, 1, 1, samples=10), 0)


class TestFeatureVectorOrbits:
    """Tests for the function ``strawberryfields.apps.graph.similarity.feature_vector_orbits``"""

    def test_invalid_orbits_list(self):
        """Test if function raises a ``ValueError`` when the list of orbits is empty."""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="List of orbits must have at least one orbit"):
            similarity.feature_vector_orbits(g, list_of_orbits=[])

    def test_bad_orbit_photon_numbers(self):
        """Test if function raises a ``ValueError`` when input is an orbit with numbers
        below zero."""
        graph = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Cannot request orbits with photon number below zero"):
            similarity.feature_vector_orbits(graph, [[-1, 1]])

    def test_invalid_n_mean(self):
        """Test if function raises a ``ValueError`` when the mean photon number is specified to
        be negative."""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Mean photon number must be non-negative"):
            similarity.feature_vector_orbits(g, [[1, 1], [2]], n_mean=-1)

    def test_invalid_loss(self):
        """Test if function raises a ``ValueError`` when the loss parameter is specified outside
        of range."""
        g = nx.complete_graph(5)
        with pytest.raises(
            ValueError, match="Loss parameter must take a value between zero and one"
        ):
            similarity.feature_vector_orbits(g, [[1, 1], [2]], loss=2)

    def test_calls_exact_for_zero_samples(self):
        """Test if function calls the exact function for zero samples"""
        g = nx.complete_graph(5)
        assert similarity.feature_vector_orbits(
            g, [[1, 1]], samples=0
        ) == similarity.prob_orbit_exact(g, [1, 1])

    def test_correct_vector_returned(self, monkeypatch):
        """Test if function correctly constructs the feature vector. The ``prob_orbit_exact``
        and ``prob_orbit_mc`` function called within ``feature_vector_orbits`` are
        monkeypatched to return hard-coded outputs that depend only on the orbit."""

        with monkeypatch.context() as m:
            m.setattr(
                similarity,
                "prob_orbit_mc",
                lambda _graph, orbit, n_mean, samples, loss: 1.0 / sum(orbit),
            )
            graph = nx.complete_graph(8)
            assert similarity.feature_vector_orbits(graph, [[1, 1], [2, 1, 1]], samples=1) == [
                0.5,
                0.25,
            ]

        with monkeypatch.context() as m:
            m.setattr(
                similarity,
                "prob_orbit_exact",
                lambda _graph, orbit, n_mean, loss: 0.5 * (1.0 / sum(orbit)),
            )
            graph = nx.complete_graph(8)
            assert similarity.feature_vector_orbits(graph, [[1, 1], [2, 1, 1]]) == [0.25, 0.125]

    def test_known_result(self):
        """Tests if the probability for known cases is correctly
        reproduced."""
        graph = nx.complete_graph(4)
        p = similarity.feature_vector_orbits(graph, [[1, 1], [2, 1, 1]], 2)
        assert np.allclose(p, [0.22918531118334962, 0.06509669127495163])
        assert np.allclose(
            similarity.feature_vector_orbits(graph, [[2], [4]], 1, samples=10), [0, 0]
        )


class TestFeatureVectorEvents:
    """Tests for the function ``strawberryfields.apps.graph.similarity.feature_vector_events``"""

    def test_invalid_event_photon_numbers(self):
        """Test if function raises a ``ValueError`` when the list of event photons numbers
        is empty."""
        g = nx.complete_graph(5)
        with pytest.raises(
            ValueError, match="List of photon numbers must have at least one element"
        ):
            similarity.feature_vector_events(g, [], 1)

    def test_invalid_n_mean(self):
        """Test if function raises a ``ValueError`` when the mean photon number is specified to
        be negative."""
        g = nx.complete_graph(5)
        with pytest.raises(ValueError, match="Mean photon number must be non-negative"):
            similarity.feature_vector_events(g, [2, 4], 2, n_mean=-1)

    def test_invalid_loss(self):
        """Test if function raises a ``ValueError`` when the loss parameter is specified outside
        of range."""
        g = nx.complete_graph(5)
        with pytest.raises(
            ValueError, match="Loss parameter must take a value between zero and one"
        ):
            similarity.feature_vector_events(g, [2, 4], 2, loss=2)

    def test_bad_event_photon_numbers(self):
        """Test if function raises a ``ValueError`` when input a minimum photon number that is
        below zero."""
        with pytest.raises(ValueError, match="Cannot request events with photon number below zero"):
            graph = nx.complete_graph(5)
            similarity.feature_vector_events(graph, [-1, 4], 2)

    def test_low_count(self):
        """Test if function raises a ``ValueError`` if ``max_count_per_mode`` is negative."""
        g = nx.complete_graph(5)
        with pytest.raises(
            ValueError, match="Maximum number of photons per mode must be non-negative"
        ):
            similarity.feature_vector_events(g, [2, 4], max_count_per_mode=-1)

    def test_calls_exact_for_zero_samples(self):
        """Test if function calls the exact function for zero samples"""
        g = nx.complete_graph(5)
        assert similarity.feature_vector_events(
            g, [2], 1, samples=0
        ) == similarity.prob_event_exact(g, 2, 1)

    def test_correct_vector_returned(self, monkeypatch):
        """Test if function correctly constructs the feature vector. The ``prob_event_exact``
        and ``prob_event_mc`` function called within ``feature_vector_events`` are
        monkeypatched to return hard-coded outputs that depend only on the orbit."""

        with monkeypatch.context() as m:
            m.setattr(
                similarity,
                "prob_event_mc",
                lambda _graph, photons, max_count, n_mean, samples, loss: 1.0 / photons,
            )
            g = nx.complete_graph(8)
            assert similarity.feature_vector_events(g, [2, 4, 8], 1, samples=1) == [
                0.5,
                0.25,
                0.125,
            ]

        with monkeypatch.context() as m:
            m.setattr(
                similarity,
                "prob_event_exact",
                lambda _graph, photons, max_count, n_mean, loss: 0.5 * (1.0 / photons),
            )
            g = nx.complete_graph(8)
            assert similarity.feature_vector_events(g, [2, 4, 8], 1) == [0.25, 0.125, 0.0625]

    def test_known_result(self):
        """Tests if the probability for known cases is correctly
        reproduced."""
        g = nx.complete_graph(4)
        p = similarity.feature_vector_events(g, [2, 4], 2, 4)
        assert np.allclose(p, [0.21087781178526066, 0.11998024483275889])

        graph = nx.complete_graph(20)
        assert np.allclose(
            similarity.feature_vector_events(graph, [18, 20], 1, 1, samples=10), [0, 0]
        )


class TestFeatureVectorOrbitsSampling:
    """Tests for the function ``strawberryfields.apps.graph.similarity.feature_vector_orbits_sampling``"""

    def test_invalid_orbits_list(self):
        """Test if function raises a ``ValueError`` when the list of orbits is empty."""
        with pytest.raises(ValueError, match="List of orbits must have at least one orbit"):
            similarity.feature_vector_orbits_sampling([[1, 1, 0], [1, 0, 1]], [])

    def test_bad_orbit_photon_numbers(self):
        """Test if function raises a ``ValueError`` when input is an orbit with numbers
        below zero."""
        with pytest.raises(ValueError, match="Cannot request orbits with photon number below zero"):
            similarity.feature_vector_orbits_sampling([[1, 1, 0], [1, 0, 1]], [[-1, 1]])

    def test_correct_vector_returned(self, monkeypatch):
        """Test if function correctly constructs the feature vector corresponding to some hard
        coded samples. This test uses a set of samples, corresponding orbits, and resultant
        feature vector to test against the output of ``feature_vector_orbits_sampling``. The
        ``sample_to_orbit`` function called within ``feature_vector_orbits_sampling`` is
        monkeypatched to return the hard coded events corresponding to the samples."""
        samples_orbits_mapping = {
            (1, 1, 0, 0, 0): [1, 1],
            (1, 1, 1, 0, 0): [1, 1, 1],
            (1, 1, 1, 1, 0): [1, 1, 1, 1],
            (1, 1, 1, 1, 1): [1, 1, 1, 1, 1],
            (2, 0, 0, 0, 0): [2],
            (3, 0, 0, 0, 0): [3],
            (4, 0, 0, 0, 0): [4],
            (5, 0, 0, 0, 0): [5],
            (0, 1, 1, 0, 0): [1, 1],
        }
        samples = list(samples_orbits_mapping.keys()) + [(1, 1, 1, 1, 1)]
        list_of_orbits = [[1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1]]
        fv_true = [0.2, 0.1, 0.2]

        with monkeypatch.context() as m:
            m.setattr(similarity, "sample_to_orbit", lambda x: samples_orbits_mapping[x])
            fv = similarity.feature_vector_orbits_sampling(samples, list_of_orbits)

        assert fv_true == fv


class TestFeatureVectorEventsSampling:
    """Tests for the function ``strawberryfields.apps.graph.similarity.feature_vector_events_sampling``"""

    def test_invalid_event_photon_numbers(self):
        """Test if function raises a ``ValueError`` when the list of event photons numbers
        is empty."""
        with pytest.raises(
            ValueError, match="List of photon numbers must have at least one element"
        ):
            similarity.feature_vector_events_sampling([[1, 1, 0], [1, 0, 1]], [], 1)

    def test_bad_event_photon_numbers(self):
        """Test if function raises a ``ValueError`` when input a minimum photon number that is
        below zero."""
        with pytest.raises(ValueError, match="Cannot request events with photon number below zero"):
            similarity.feature_vector_events_sampling([[1, 1, 0], [1, 0, 1]], [-1, 4], 1)

    def test_low_count(self):
        """Test if function raises a ``ValueError`` if ``max_count_per_mode`` is negative."""
        with pytest.raises(
            ValueError, match="Maximum number of photons per mode must be non-negative"
        ):
            similarity.feature_vector_events_sampling([[1, 1, 0], [1, 0, 1]], [2, 4], -1)

    def test_correct_vector_returned(self, monkeypatch):
        """Test if function correctly constructs the feature vector corresponding to some hard
        coded samples. This test uses a set of samples, corresponding events, and resultant
        feature vector to test against the output of ``feature_vector_events_sampling``. The
        ``sample_to_event`` function called within ``feature_vector_events_sampling`` is
        monkeypatched to return the hard coded events corresponding to the samples."""
        samples_events_mapping = {  # max_count_per_mode = 1
            (1, 1, 0, 0, 0): 2,
            (1, 1, 1, 0, 0): 3,
            (1, 1, 1, 1, 0): 4,
            (1, 1, 1, 1, 1): 5,
            (2, 0, 0, 0, 0): None,
            (3, 0, 0, 0, 0): None,
            (4, 0, 0, 0, 0): None,
            (5, 0, 0, 0, 0): None,
            (0, 1, 1, 0, 0): 2,
        }
        samples = list(samples_events_mapping.keys()) + [(1, 1, 1, 1, 1)]  # add a repetition
        event_photon_numbers = [2, 1, 3, 5]  # test alternative ordering
        fv_true = [0.2, 0, 0.1, 0.2]

        with monkeypatch.context() as m:
            m.setattr(similarity, "sample_to_event", lambda x, _: samples_events_mapping[x])
            fv = similarity.feature_vector_events_sampling(samples, event_photon_numbers, 1)

        assert fv_true == fv

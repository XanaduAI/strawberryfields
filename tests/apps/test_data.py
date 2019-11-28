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
Unit tests for strawberryfields.apps.data
"""
# pylint: disable=no-self-use
import numpy as np
import pytest
import scipy

from strawberryfields.apps import data

pytestmark = pytest.mark.apps

GRAPH_DATASETS_LIST = [
    data.Planted,
    data.TaceAs,
    data.Mutag0,
    data.Mutag1,
    data.Mutag2,
    data.Mutag3,
    data.PHat,
]

MOLECULE_DATASETS_LIST = [data.Formic]

DATASETS_LIST = GRAPH_DATASETS_LIST + MOLECULE_DATASETS_LIST


@pytest.mark.parametrize("datasets", DATASETS_LIST)
class TestDatasets:
    """Tests for the dataset classes in ``data`` module, which should all be listed by the
    ``DATASETS_LIST`` variable above."""

    @pytest.fixture
    def dataset(self, datasets):
        """Fixture for loading each of the datasets in ``DATASETS_LIST``"""
        yield datasets()

    patch_samples = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [2, 1, 1, 1],
        [3, 0, 1, 1],
    ]

    @pytest.fixture
    def dataset_patched(self, monkeypatch, datasets):
        """Fixture for loading each of the datasets in ``DATASETS_LIST`` with the ``__init__``
        monkeypatched to load the above ``patch_samples``."""

        def mock_init(_self):
            """Replacement ``__init__`` for all the datasets in ``DATASETS_LIST``"""
            # pylint: disable=protected-access
            _self.data = scipy.sparse.csr_matrix(self.patch_samples)
            _self.n_samples, _self.modes = 10, 4

        with monkeypatch.context() as m:
            m.setattr(datasets, "__init__", mock_init)
            yield datasets()

    def test_filename(self, dataset):
        """Test if filename is valid string for each dataset"""
        # pylint: disable=protected-access
        assert isinstance(dataset._data_filename, str)

    def test_n_mean(self, dataset):
        """Test if mean photon number is valid float or int for each dataset"""
        assert isinstance(dataset.n_mean, (float, int))
        assert dataset.n_mean >= 0

    def test_threshold(self, dataset):
        """Test if threshold flag is valid bool for each dataset"""
        assert isinstance(dataset.threshold, bool)

    def test_counts_ax0(self, dataset_patched):
        """Test if dataset ``counts`` method returns the number of counts in each mode when
        ``axis==0``. The samples are patched to the fixed ``patch_samples`` list."""
        assert dataset_patched.counts(0) == [5, 5, 6, 6]

    def test_counts_ax1(self, dataset_patched):
        """Test if dataset ``counts`` method returns the number of counts in each sample when
        ``axis==1``. The samples are patched to the fixed ``patch_samples`` list."""
        assert dataset_patched.counts(1) == [0, 1, 1, 2, 1, 2, 2, 3, 5, 5]

    # pylint: disable=unnecessary-comprehension
    def test_iter(self, dataset_patched):
        """Test if dataset class allows correct iteration over itself"""
        assert [i for i in dataset_patched] == self.patch_samples

    # pylint: disable=unnecessary-comprehension
    def test_slice(self, dataset_patched):
        """Test if dataset class allows correct slicing over items"""
        assert [i for i in dataset_patched[1, 4, 2]] == [
            self.patch_samples[1],
            self.patch_samples[3],
        ]
        assert [i for i in dataset_patched[slice(1, 4, 2)]] == [
            self.patch_samples[1],
            self.patch_samples[3],
        ]

    def test_negative_getitem(self, dataset_patched):
        """Test if dataset class allows negative indices"""
        assert dataset_patched[-1] == self.patch_samples[-1]

    def test_len(self, dataset):
        """Test if dataset's ``len`` is equal to the ``n_samples`` attribute."""
        assert len(dataset) == dataset.n_samples

    def test_next(self, dataset_patched):
        """Test if dataset correctly provides ``next`` until looped through whole dataset and
        then raises a StopIteration."""
        counter = 0
        next_vals = []

        while counter < 10:
            next_vals.append(next(dataset_patched))
            counter += 1

        assert next_vals == self.patch_samples

        with pytest.raises(StopIteration):
            next(dataset_patched)


@pytest.mark.parametrize("datasets", GRAPH_DATASETS_LIST)
class TestGraphDatasets:
    """Tests for the ``MoleculeDataset`` class"""

    @pytest.fixture
    def dataset(self, datasets):
        """Fixture for loading each of the datasets in ``GRAPH_DATASETS_LIST``"""
        yield datasets()

    def test_adj_dim(self, dataset):
        """Test if adjacency matrix of dataset is correct dimensions."""
        n, m = dataset.adj.shape

        assert n == m
        assert n == dataset.modes

    def test_adj_valid(self, dataset):
        """Test if adjacency matrix of dataset is symmetric."""
        assert np.allclose(dataset.adj, dataset.adj.T)


@pytest.mark.parametrize("datasets", MOLECULE_DATASETS_LIST)
class TestMoleculeDatasets:
    """Tests for the ``MoleculeDataset`` class"""

    @pytest.fixture
    def dataset(self, datasets):
        """Fixture for loading each of the datasets in ``MOLECULE_DATASETS_LIST``"""
        yield datasets()

    def test_w_dims(self, dataset):
        """Test if w has correct shape"""
        w, _, _, _ = dataset.w, dataset.wp, dataset.Ud, dataset.delta
        assert w.shape == (dataset.modes // 2,)

    def test_wp_dims(self, dataset):
        """Test if wp has correct shape"""
        _, wp, _, _ = dataset.w, dataset.wp, dataset.Ud, dataset.delta
        assert wp.shape == (dataset.modes // 2,)

    def test_Ud_dims(self, dataset):
        """Test if Ud has correct shape"""
        _, _, Ud, _ = dataset.w, dataset.wp, dataset.Ud, dataset.delta
        assert Ud.shape == (dataset.modes // 2, dataset.modes // 2)

    def test_delta_dims(self, dataset):
        """Test if delta has correct shape"""
        _, _, _, delta = dataset.w, dataset.wp, dataset.Ud, dataset.delta
        assert delta.shape == (dataset.modes // 2,)

    def test_uniform_dims(self, dataset):
        """Test if overall dimension is correct"""
        w, wp, Ud, delta = dataset.w, dataset.wp, dataset.Ud, dataset.delta
        dim = w.shape[0]

        assert wp.shape[0] == dim and Ud.shape == (dim, dim) and delta.shape[0] == dim

    def test_w_non_negative(self, dataset):
        """Test if w is non-negative"""
        assert all(dataset.w >= 0)

    def test_wp_non_negative(self, dataset):
        """Test if wp is non-negative"""
        assert all(dataset.wp >= 0)

    def test_T_non_negative(self, dataset):
        """Test if T is non-negative"""
        assert dataset.T >= 0

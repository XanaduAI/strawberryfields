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
Unit tests for strawberryfields.gbs.data
"""
# pylint: disable=no-self-use
import numpy as np
import pytest
import scipy

from strawberryfields.gbs import data

DATASETS_LIST = [data.Planted]


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
            _self.dat = scipy.sparse.csr_matrix(self.patch_samples)
            _self.adj = np.ones((4, 4))
            _self.n_samples, _self.modes = 10, 4

        with monkeypatch.context() as m:
            m.setattr(datasets, "__init__", mock_init)
            yield datasets()

    def test_filename(self, dataset):
        """Test if filename is valid string for each dataset"""
        assert isinstance(dataset._dat_filename, str)

    def test_n_mean(self, dataset):
        """Test if mean photon number is valid float or int for each dataset"""
        assert isinstance(dataset.n_mean, (float, int))
        assert dataset.n_mean >= 0

    def test_n_max(self, dataset):
        """Test if maximum photon number is valid float or int for each dataset"""
        assert isinstance(dataset.n_max, (float, int))
        assert dataset.n_max >= 0

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

    def test_iter(self, dataset_patched):
        """Test if dataset class allows correct iteration over itself"""
        assert [i for i in dataset_patched] == self.patch_samples

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

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
import scipy.sparse

from strawberryfields.gbs import data

datasets_list = [data.Planted]


@pytest.mark.parametrize("datasets", datasets_list)
class TestSampleLoaders:
    @pytest.fixture
    def sample_loader(self, datasets):
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
    def sample_loader_patched(self, monkeypatch, datasets):
        def mock_init(_self):
            _self.dat = scipy.sparse.csr_matrix(self.patch_samples)
            _self.adj = np.ones((4, 4))
            _self.n_samples, _self.modes = 10, 4

        with monkeypatch.context() as m:
            m.setattr(datasets, "__init__", mock_init)
            yield datasets()

    def test_filename(self, sample_loader):
        assert isinstance(sample_loader.dat_filename, str)

    def test_n_mean(self, sample_loader):
        assert isinstance(sample_loader.n_mean, (float, int))
        assert sample_loader.n_mean >= 0

    def test_n_max(self, sample_loader):
        assert isinstance(sample_loader.n_max, (float, int))
        assert sample_loader.n_max >= 0

    def test_threshold(self, sample_loader):
        assert isinstance(sample_loader.threshold, bool)

    def test_counts_ax0(self, sample_loader_patched):
        assert sample_loader_patched.counts(0) == [5, 5, 6, 6]

    def test_counts_ax1(self, sample_loader_patched):
        assert sample_loader_patched.counts(1) == [0, 1, 1, 2, 1, 2, 2, 3, 5, 5]

    def test_iter(self, sample_loader_patched):
        assert [i for i in sample_loader_patched] == self.patch_samples

    def test_slice(self, sample_loader_patched):
        assert [i for i in sample_loader_patched[1, 4, 2]] == [
            self.patch_samples[1],
            self.patch_samples[3],
        ]
        assert [i for i in sample_loader_patched[slice(1, 4, 2)]] == [
            self.patch_samples[1],
            self.patch_samples[3],
        ]

    def test_negative_getitem(self, sample_loader_patched):
        assert sample_loader_patched[-1] == self.patch_samples[-1]

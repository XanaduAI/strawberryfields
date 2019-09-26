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
import pytest

import scipy.sparse

from strawberryfields.gbs import data

datasets_list = [data.Planted]


@pytest.mark.parametrize("datasets", datasets_list)
class TestSampleLoaders:

    @pytest.fixture
    def sample_loader(self, datasets):
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


class TestSampleLoaderBase:

    patch_samples = scipy.sparse.csr_matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
    ])

    @pytest.fixture
    def sample_loader(self, monkeypatch):

        def mock_init(_self):
            _self.dat = self.patch_samples
            _self.n_samples, _self.modes = _self.dat.shape
            _self.dat_filename = _self.n_max = _self.n_mean = _self.threshold = None

        with monkeypatch.context() as m:
            m.setattr(data.SampleLoader, "__init__", mock_init)
            yield data.SampleLoader()

    def test_counts_ax0(self, sample_loader):
        print("hi")
        '''c = sample_loader.counts(0)
        assert len(c) == sample_loader.modes
        assert min(c) >= 0'''

    '''def test_counts_ax1(self, sample_loader):
        c = sample_loader.counts(1)
        assert len(c) == sample_loader.n_samples
        assert min(c) >= 0'''



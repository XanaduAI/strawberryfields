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
Unit tests for strawberryfields.apps.data
"""
# pylint: disable=no-self-use
import numpy as np
import pytest
import scipy

from strawberryfields.apps import data

pytestmark = pytest.mark.apps

GRAPH_DATASETS_LIST = [
    data.sample.Planted,
    data.sample.TaceAs,
    data.sample.Mutag0,
    data.sample.Mutag1,
    data.sample.Mutag2,
    data.sample.Mutag3,
    data.sample.PHat,
]

MOLECULE_DATASETS_LIST = [data.sample.Formic]

WATER_DATASETS_LIST = [data.sample.Water]

PYRROLE_DATASETS_LIST = [data.sample.Pyrrole]

SAMPLE_DATASETS_LIST = GRAPH_DATASETS_LIST + MOLECULE_DATASETS_LIST

FEATURE_DATASETS_LIST = [data.feature.QM9Exact, data.feature.QM9MC, data.feature.MUTAG]


@pytest.mark.parametrize("datasets", SAMPLE_DATASETS_LIST)
class TestSampleDatasets:
    """Tests for the dataset classes in ``data.sample`` module, which should all be listed
    by the ``SAMPLE_DATASETS_LIST`` variable above."""

    @pytest.fixture
    def dataset(self, datasets):
        """Fixture for loading each of the datasets in ``SAMPLE_DATASETS_LIST``"""
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
        """Fixture for loading each of the datasets in ``SAMPLE_DATASETS_LIST`` with the
        ``__init__`` monkeypatched to load the above ``patch_samples``."""

        def mock_init(_self):
            """Replacement ``__init__`` for all the datasets in ``SAMPLE_DATASETS_LIST``"""
            # pylint: disable=protected-access
            _self.data = scipy.sparse.csr_matrix(self.patch_samples)
            _self.n_samples, _self.modes = 10, 4

        with monkeypatch.context() as m:
            m.setattr(datasets, "__init__", mock_init)
            yield datasets()

    def test_filename(self, dataset):
        """Test if filename is a valid string for each dataset"""
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


@pytest.mark.parametrize("datasets", FEATURE_DATASETS_LIST)
class TestFeatureDatasets:
    """Tests for the dataset classes in ``data.feature`` module, which should all be listed
    by the ``FEATURE_DATASETS_LIST`` variable above."""

    @pytest.fixture
    def dataset(self, datasets):
        """Fixture for loading each of the datasets in ``FEATURE_DATASETS_LIST``"""
        yield datasets()

    patch_orbits = [
        [1, 1],
        [2],
        [3],
        [1, 1, 1, 1],
        [2, 2],
    ]

    patch_feature_vectors = np.array(
        [
            [0.1, 0.01, 0, 0.1, 0.05],
            [0.2, 0.02, 0, 0.05, 0.01],
            [0.3, 0.05, 0, 0.05, 0.15],
            [0.4, 0.15, 0, 0.15, 0.15],
        ]
    )

    patch_adjacency_matrices = np.array(
        [
            [
                [1.40801884, 1.37064318, 0.74647764, 0.8656113, 0.79760222],
                [1.37064318, 1.14609425, 1.31463115, 0.48630012, 1.44034118],
                [0.74647764, 1.31463115, 1.03330415, 1.64344102, 1.31083479],
                [0.8656113, 0.48630012, 1.64344102, 0.28176324, 1.43776473],
                [0.79760222, 1.44034118, 1.31083479, 1.43776473, 0.1894344],
            ],
            [
                [0.07507458, 0.99955953, 1.1432025, 0.63240514],
                [0.99955953, 1.71093457, 1.10163201, 0.99179716],
                [1.1432025, 1.10163201, 0.07714951, 0.74115172],
                [0.63240514, 0.99179716, 0.74115172, 0.41063534],
            ],
        ]
    )

    @pytest.fixture
    def dataset_patched(self, monkeypatch, datasets):
        """Fixture for loading each of the datasets in ``FEATURE_DATASETS_LIST`` with the
        ``__init__`` monkeypatched to load the above ``patch_feature_vectors``."""

        def mock_init(_self):
            """Replacement ``__init__`` for all the datasets in ``FEATURE_DATASETS_LIST``"""
            # pylint: disable=protected-access
            _self.vectors = self.patch_feature_vectors
            _self.adjs = self.patch_adjacency_matrices
            _self.n_vectors, _self.n_features = 4, 5

        with monkeypatch.context() as m:
            m.setattr(datasets, "__init__", mock_init)
            m.setattr(datasets, "unit_data", self.patch_orbits)
            yield datasets()

    def test_data_filename(self, dataset):
        """Test if file name is a valid string for each dataset"""
        # pylint: disable=protected-access
        assert isinstance(dataset._data_filename, str)

    def test_n_mean(self, dataset):
        """Test if mean photon number is valid float or int for each dataset"""
        assert isinstance(dataset.n_mean, (float, int))
        assert dataset.n_mean >= 0

    def test_unit(self, dataset):
        """Test if unit is a valid string for each dataset"""
        # pylint: disable=protected-access
        allowed = ["orbits", "events"]
        assert dataset.unit in allowed

    def test_method(self, dataset):
        """Test if method is a valid string for each dataset"""
        # pylint: disable=protected-access
        allowed = ["exact", "mc"]
        assert dataset.method in allowed

    def test_unit_data(self, dataset):
        """Test if unit_data is valid list for each dataset"""
        assert isinstance(dataset.unit_data, list)

    # pylint: disable=unnecessary-comprehension
    def test_iter(self, dataset_patched):
        """Test if dataset class allows correct iteration over itself"""
        assert (dataset_patched[0] == self.patch_feature_vectors[0]).all()
        assert (np.array([i for i in dataset_patched]) == self.patch_feature_vectors).all()
        assert (next(dataset_patched) == self.patch_feature_vectors[0]).all()

    # pylint: disable=unnecessary-comprehension
    def test_slice(self, dataset_patched):
        """Test if dataset class allows correct slicing over itself"""
        a1 = np.array([i for i in dataset_patched[(1, 3)]])
        a2 = np.array([self.patch_feature_vectors[1], self.patch_feature_vectors[2]])
        a3 = np.array(dataset_patched[slice(1, 3, 1)])
        a4 = np.array([self.patch_feature_vectors[1], self.patch_feature_vectors[2]])
        assert (a1 == a2).all()
        assert (a3 == a4).all()

    def test_data_dim_correct(self, dataset_patched):
        """Test if feature, unit and matrix data of dataset have correct dimensions."""
        n, m = self.patch_feature_vectors.shape
        (p,) = self.patch_adjacency_matrices.shape
        q = len(self.patch_orbits)

        assert n == dataset_patched.n_vectors
        assert m == dataset_patched.n_features
        assert p == len(dataset_patched.adjs)
        assert q == len(dataset_patched.unit_data)


@pytest.mark.parametrize("datasets", WATER_DATASETS_LIST)
class TestWaterDatasets:
    """Tests for the ``Water`` class"""

    @pytest.fixture
    def dataset(self, datasets):
        """Fixture for loading each of the datasets in ``WATER_DATASETS_LIST``"""
        yield datasets(t=0)

    def test_w_dims(self, dataset):
        """Test if w has correct shape"""
        w = dataset.w
        assert w.shape == (dataset.modes,)

    def test_u_dims(self, dataset):
        """Test if U has correct shape"""
        U = dataset.U
        assert U.shape == (dataset.modes, dataset.modes)

    def test_u_unitary(self, dataset):
        """Test if U is unitary"""
        U = dataset.U
        assert np.allclose(np.eye(U.shape[0]), U.conj().T @ U)

    def test_sample_dims(self, datasets):
        """Test if sample has correct shape"""
        for t in datasets.available_times:
            dataset = datasets(t)
            assert np.shape(dataset[:]) == (5000, dataset.modes)

    def test_times_water(self, dataset):
        """Test if available times are correct"""
        t = dataset.available_times
        assert np.allclose(t, np.linspace(0, 260, 27))

    def test_dataset_water(self, datasets):
        """Test if function raises a ``ValueError`` when an incorrect time is given"""
        with pytest.raises(ValueError, match="The selected time is not correct"):
            datasets(t=7)


@pytest.mark.parametrize("datasets", PYRROLE_DATASETS_LIST)
class TestPyrroleDatasets:
    """Tests for the ``Pyrrole`` class"""

    @pytest.fixture
    def dataset(self, datasets):
        """Fixture for loading each of the datasets in ``Pyrrole_DATASETS_LIST``"""
        yield datasets(t=0)

    def test_ri_dims(self, dataset):
        """Test if ri has correct shape"""
        ri = dataset.ri
        assert ri.shape == (30,)

    def test_rf_dims(self, dataset):
        """Test if rf has correct shape"""
        rf = dataset.rf
        assert rf.shape == (30,)

    def test_wi_dims(self, dataset):
        """Test if wi has correct shape"""
        wi = dataset.wi
        assert wi.shape == (dataset.modes,)

    def test_wf_dims(self, dataset):
        """Test if wf has correct shape"""
        wf = dataset.wf
        assert wf.shape == (dataset.modes,)

    def test_m_dims(self, dataset):
        """Test if m has correct shape"""
        m = dataset.m
        assert m.shape == (30,)

    def test_Li_dims(self, dataset):
        """Test if Li has correct shape"""
        Li = dataset.Li
        assert Li.shape == (30, dataset.modes)

    def test_Lf_dims(self, dataset):
        """Test if Lf has correct shape"""
        Lf = dataset.Lf
        assert Lf.shape == (30, dataset.modes)

    def test_up_dims(self, dataset):
        """Test if U has correct shape"""
        U = dataset.U
        assert U.shape == (dataset.modes, dataset.modes)

    def test_up_unitary(self, dataset):
        """Test if U is unitary"""
        U = dataset.U
        assert np.allclose(np.eye(U.shape[0]), U.conj().T @ U)

    def test_sample_dims(self, datasets):
        """Test if sample has correct shape"""
        for t in datasets.available_times:
            dataset = datasets(t)
            assert np.shape(dataset[:]) == (dataset.n_samples, dataset.modes)

    def test_times_pyrrole(self, dataset):
        """Test if available times are correct"""
        t = dataset.available_times
        assert np.allclose(t, np.linspace(0, 900, 10))

    def test_dataset_pyrrole(self, datasets):
        """Test if function raises a ``ValueError`` when an incorrect time is given"""
        with pytest.raises(ValueError, match="The selected time is not correct"):
            datasets(t=96)

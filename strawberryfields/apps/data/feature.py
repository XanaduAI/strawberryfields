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
Submodule for feature vector datasets and their base classes.
"""
# pylint: disable=unnecessary-pass
from abc import ABC, abstractmethod
import pkg_resources
import numpy as np

DATA_PATH = pkg_resources.resource_filename("strawberryfields", "apps/data/feature_data") + "/"


class FeatureDataset(ABC):
    """Base class for loading datasets of pre-calculated GBS feature vectors.

    Each dataset contains a collection of feature vectors. The corresponding adjacency matrix
    for each feature vector and the orbits/events used are also provided.

    Attributes:
        n_mean (float): mean number of photons used in the GBS device
        method (str): method used to calculate the feature vectors; ``exact`` for exact calculation
            or ``mc`` for Monte Carlo estimation
        unit (str): signifies the unit of construction of feature vectors; ``orbits`` or ``events``
        unit_data (list[list[int]]): list of orbits/events used to construct the feature vectors. Each
            orbit is a list of integers and each event must be provided as
            ``[total_photon_number, max_photon_per_mode]``
        n_vectors (int): number of feature vectors provided in the dataset
        n_features (int): number of features in each feature vector
        vectors (array): array of feature vectors
        adjs (array): array of adjacency matrices of graphs for which feature vectors
            were calculated
    """

    _count = 0

    @property
    @abstractmethod
    def _data_filename(self) -> str:
        """Base name of files containing the data stored in the ``./feature_data/`` directory.

        For each dataset, feature vectors and the corresponding adjacency matrices are provided
        as NumPy arrays in two separate ``.npy`` format files.

        Given ``_data_filename = example`` and ``method = mc``, feature vectors should be stored
        in ``./feature_data/example_mc_fv.npy`` and the array of corresponding adjacency matrices
        should be saved in ``./feature_data/example_mat.npy``. NumPy functions such as
        ``numpy.save(filename, array, allow_pickle=True)`` and
        ``numpy.load(filename, allow_pickle=True)`` can be used to save and load these NumPy
        arrays."""
        pass

    # pylint: disable=missing-docstring
    @property
    @abstractmethod
    def unit(self) -> str:
        pass

    # pylint: disable=missing-docstring
    @property
    @abstractmethod
    def unit_data(self) -> list:
        pass

    # pylint: disable=missing-docstring
    @property
    @abstractmethod
    def n_mean(self) -> float:
        pass

    # pylint: disable=missing-docstring
    @property
    @abstractmethod
    def method(self) -> str:
        pass

    def __init__(self):
        self.vectors = np.load(
            f"{DATA_PATH}{self._data_filename}_{self.method}_fv.npy", allow_pickle=True
        )
        self.adjs = np.load(DATA_PATH + self._data_filename + "_mat.npy", allow_pickle=True)
        self.n_vectors, self.n_features = self.vectors.shape

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_vectors

    def __getitem__(self, key):
        """If ``key`` is an integer, return the key-th elements of ``vectors``.
        If ``key`` is a tuple (start, stop), return elements of ``vectors`` at index
        ``start`` till ``stop-1``using one step increment.
        It ``key`` is a slice(start, stop, step), return elements of ``vectors`` at index
        ``start`` till ``stop-1`` using the given ``step`` increment."""
        if not isinstance(key, (slice, tuple, int)):
            raise TypeError("Dataset indices must be integers, slices, or tuples")
        if isinstance(key, int):
            return self.vectors[key + self.n_vectors if key < 0 else key]
        if isinstance(key, tuple):
            key = slice(*key)
        return self.vectors[key]

    def __next__(self):
        if self._count < self.n_vectors:
            self._count += 1
            return self.__getitem__(self._count - 1)
        self._count = 0
        raise StopIteration


class QM9Exact(FeatureDataset):
    """Exactly-calculated feature vectors of 1100 randomly-chosen molecules from the
    QM9 dataset.

    The `QM9 dataset <http://quantum-machine.org/datasets/>`__ is widely used in
    benchmarking performance of machine learning models in estimating
    molecular properties :cite:`Ruddigkeit2012`, :cite:`Ramakrishnan2014`.

    Coulomb matrices were used as adjacency matrices to represent molecules in this case.

    The Monte-Carlo estimated feature vectors of certain events of these 1100 molecules are
    also available in the :class:`~.QM9MC` class.

    Attributes:
        method = "exact"
        n_mean = 6
        unit = "orbits"
        unit_data = [[1, 1], [2], [1, 1, 1, 1], [2, 1, 1], [2, 2], [1, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2]]
    """

    _data_filename = "QM9"
    unit = "orbits"
    unit_data = [
        [1, 1],
        [2],
        [1, 1, 1, 1],
        [2, 1, 1],
        [2, 2],
        [1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1],
        [2, 2, 1, 1],
        [2, 2, 2],
    ]
    n_mean = 6
    method = "exact"


class QM9MC(FeatureDataset):
    """Monte-Carlo estimated feature vectors of 1100 randomly-chosen molecules from the
    QM9 dataset.

    The `QM9 dataset <http://quantum-machine.org/datasets/>`__ is widely used in
    benchmarking performance of machine learning models in estimating
    molecular properties :cite:`Ruddigkeit2012`, :cite:`Ramakrishnan2014`.

    Coulomb matrices were used as adjacency matrices to represent molecules in this case.

    The exactly-calculated feature vectors of certain orbits of these 1100 molecules are
    also available in the :class:`~.QM9Exact` class.

    Attributes:
        method = "mc"
        n_mean = 6
        unit = "events"
        unit_data = [[2, 2], [4, 2], [6, 2]]
    """

    _data_filename = "QM9"
    unit = "events"
    unit_data = [[2, 2], [4, 2], [6, 2]]
    n_mean = 6
    method = "mc"


class MUTAG(FeatureDataset):
    """Exactly-calculated feature vectors of the 188 graphs in the MUTAG dataset.

    The `MUTAG dataset <https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets>`__
    is widely used to benchmark performance of graph kernels and graph neural networks
    :cite:`debnath1991structure`, :cite:`kriege2012subgraph`. It contains
    molecular graphs of 188 chemical compounds divided into two classes according
    to their mutagenic effect on a bacterium.

    Attributes:
        method = "exact"
        n_mean = 8
        unit = "orbits"
        unit_data = [[1, 1], [2], [1, 1, 1, 1], [2, 1, 1], [2, 2], [1, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2]]
    """

    _data_filename = "MUTAG"
    unit = "orbits"
    unit_data = [
        [1, 1],
        [2],
        [1, 1, 1, 1],
        [2, 1, 1],
        [2, 2],
        [1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1],
        [2, 2, 1, 1],
        [2, 2, 2],
    ]
    n_mean = 8
    method = "exact"

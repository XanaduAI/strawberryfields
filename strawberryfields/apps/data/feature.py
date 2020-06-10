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
Pre-calculated datasets of simulated GBS kernels.
"""
# pylint: disable=unnecessary-pass
from abc import ABC, abstractmethod

import pkg_resources
import numpy as np
import scipy

DATA_PATH = pkg_resources.resource_filename("strawberryfields", "apps/data/feature_data") + "/"


class FeatureDataset(ABC):
    """Base class for loading datasets of pre-calculated GBS feature vectors.

    Each dataset contains a collection of feature vectors. The corresponding adjacency matrix
    for each feature vector and the orbits/events used are also provided.

    Attributes:
        n_mean (float): mean number of photons used in the GBS device
        threshold (bool): flag to indicate whether feature vectors are calculated with threshold
            detectors or with photon-number-resolving detectors
        method (str): method used to calculate the feature vectors; "exact" for exact calculation
            or "mc" for Monte Carlo estimation
        unit (str): Signifies the unit of construction of feature vectors; "orbits" or "events"
        unitData (list): list of orbits/events used to construct the feature vectors where each
            orbit is a list of integers and each event can be provided as a tuple of
            ``(total_photon_number, max_photon_per_mode)``
        n_vectors (int): number of feature vectors provided in the dataset
        n_features (int): number of features in each vector
        featuresData (array): numpy array of feature vectors
        matData (array): numpy array of adjacency matrices of graphs for which feature vectors
            were calculated
    """

    _count = 0

    @property
    @abstractmethod
    def _data_filename(self) -> str:
        """Base name of files containing the data stored in the ``./feature_data/`` directory.

        For each dataset, feature vectors and the corresponding adjacency matrices are provided
        as numpy arrays in two separate ``.npy`` format files.

        Given ``_data_filename = example`` and ``method = mc``, feature vectors should be stored
        in ``./feature_data/example_mc_fv.npy`` and the array of corresponding adjacency matrices
        should be saved in ``./feature_data/example_mat.npy``. Numpy functions ``numpy.save(filename,
        array, allow_pickle=True) and numpy.load(filename, allow_pickle=True) can be used to
        save and load these numpy arrays."""
        pass

    @property
    @abstractmethod
    def unit(self) -> str:
        pass

    @property
    @abstractmethod
    def unitData(self) -> list:
        pass

    @property
    @abstractmethod
    def n_mean(self) -> float:
        pass

    @property
    @abstractmethod
    def threshold(self) -> bool:
        pass

    @property
    @abstractmethod
    def method(self) -> str:
        pass

    def __init__(self):
        self.featuresData = np.load(
            f"{DATA_PATH}{self._data_filename}_{self.method}_fv.npy", allow_pickle=True
        )
        self.matData = np.load(DATA_PATH + self._data_filename + "_mat.npy", allow_pickle=True)
        self.n_vectors, self.n_features = self.featuresData.shape

    def __iter__(self):
        return iter(self.featuresData)

    # def __len__(self):
    #     return self.n_vectors

    def __getitem__(self, key):

        if not isinstance(key, int):
            raise TypeError("Dataset index must be an integer")

        if key > self.n_vectors:
            raise TypeError("Dataset index must not exceed its length")

        return self.featuresData[key]

    def __next__(self):
        if self._count < self.n_vectors:
            self._count += 1
            return self.__getitem__(self._count - 1)
        self._count = 0
        raise StopIteration

    # def get_feature_vector(self, k):
    #     """Get k-th feature vector of the dataset.
    #     """
    #     if k > self.n_vectors:
    #         raise ValueError("Dataset index must not exceed its length")
    #     return self.featuresData[k]


class QM9Exact(FeatureDataset):
    """Exactly-calculated feature vectors of 1100 randomly-chosen molecules from the
    `QM9 dataset <http://quantum-machine.org/datasets/>`__ used in
    :cite:`Ruddigkeit2012, ramakrishnan2014`, are provided. Coulomb matrices are used
    as adjacency matrices to represent molecules in this case.

    The Monte-Carlo estimated feature vectors of these 1100 molecules are also available
    in the ``QM9MC`` class.
    """

    _data_filename = "QM9"
    unit = "orbits"
    unitData = [
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
    threshold = True
    method = "exact"


class QM9MC(FeatureDataset):
    """Monte-Carlo estimated feature vectors of 1100 randomly-chosen molecules from the
    `QM9 dataset <http://quantum-machine.org/datasets/>`__ used in
    :cite:`Ruddigkeit2012, ramakrishnan2014`, are provided. Coulomb matrices are used
    as adjacency matrices to represent molecules in this case.

    The exactly-calculated feature vectors of these 1100 molecules are also available
    in the ``QM9Exact`` class.
    """

    _data_filename = "QM9"
    unit = "events"
    unitData = [(2, 2), (4, 2), (6, 2)]
    n_mean = 6
    threshold = True
    method = "mc"


class MUTAG(FeatureDataset):
    """Exactly-calculated feature vectors of the 180 graphs in the `MUTAG dataset
    <https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets>`__
    used in :cite:`debnath1991structure, kriege2012subgraph` are provided.
    """

    _data_filename = "MUTAG"
    unit = "orbits"
    unitData = [
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
    threshold = True
    method = "exact"

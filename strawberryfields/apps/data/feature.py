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
Pre-calculated datasets of GBS kernels.

.. seealso::

    :doc:`/introduction/data`
"""
# pylint: disable=unnecessary-pass
from abc import ABC, abstractmethod

import pkg_resources
import numpy as np
import scipy

DATA_PATH = pkg_resources.resource_filename("strawberryfields", "apps/data/FeatureData") + "/"


class FeatureDataset(ABC):
    """Base class for loading datasets of pre-calculated GBS feature vectors.

        For each dataset, feature vectors, information on what orbits/events are used to construct
        these feature vectors and the corresponding adjacency matrices are provided.

    Attributes:
        n_mean (float): mean number of photons used in feature vector calculation
        threshold (bool): flag to indicate whether feature vectors are calculated with threshold
            detection (i.e., detectors of zero or some photons) or with photon-number-resolving
            detectors.
        method (str): method used to calculate the feature vectors; "exact" for exact calculation
         or "MC" for Monte Carlo estimation
            estimation.
        num_fv (int): number of feature vectors provided in the dataset
        dim_fv (int): dimension of all feature vectors
        featuresData (array): array of feature vectors
        unitData (array): array of orbits/events used to construct the feature vectors
        matData (array): array of adjacency matrices of graphs for which feature vectors
            were calculated
    """

    @property
    @abstractmethod
    def _data_filename(self) -> str:
        """Base name of files containing the data stored in the ``./FeatureData/`` directory.

        For each dataset, feature vectors, information on what orbits/events are used to construct
        these feature vectors and the corresponding adjacency matrices are provided in three
        separate files.

        Array of feature vectors should be provided in a text file. For ``_data_filename = "example"``,
        feature vectors should be stored in ``./FeatureData/example_fv.txt``.

        Array of corresponding adjacency matrices should be provided as a ``scipy.sparse.csr_matrix``
        saved in ``.npz`` format. For ``_data_filename = "example"``, adjacency matrices
        are stored in ``./FeatureData/example_mat.npz``."""
        pass

    @property
    @abstractmethod
    def _unit(self) -> str:
        """Signifies the unit of construction of feature vectors; "orbits" or "events". This is
        used with ``_data_filename`` to retrieve data stored in the ``./FeatureData/`` directory.

        If ``_unit = "orbits"``, list of orbits used should be provided in a text file. Hence, for
        ``_data_filename = "example"``, list of orbits should be stored in
        ``./FeatureData/example_orbits.txt``.

        If ``_unit = "events"``, list of events used should be provided in a text file. Hence, for
        ``_data_filename = "example"``, list of events should be stored in
        ``./FeatureData/example_events.txt``."""
        pass

    def __init__(self):
        self.featuresData = np.loadtxt(DATA_PATH + self._data_filename + "_fv.txt")
        self.unitData = np.loadtxt(DATA_PATH + self._data_filename + self._unit + ".txt")
        self.matData = scipy.sparse.load_npz(DATA_PATH + self._data_filename + "_mat.npz").toarray()
        self.num_fv, self.dim_fv = self.featuresData.shape

    def __iter__(self):
        return self

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


class QM9_Exact(FeatureDataset):
    """ Exactly calculated feature vectors of 1000 randomly chosen molecules from the
    ` QM9 dataset <http://quantum-machine.org/datasets/>`__ used in [ref] are provided.
    Coulomb matrices are used as adjacency matrices in this case.
    """

    _data_filename = "QM9"
    _unit = "orbits"
    n_mean = 8
    threshold = True
    method = "exact"


class QM9_MC(FeatureDataset):
    """ Monte-Carlo estimated feature vectors of the same 1000 randomly chosen molecules from the
    ` QM9 dataset <http://quantum-machine.org/datasets/>`__ are provided.
    Coulomb matrices are used as adjacency matrices in this case.
    """

    _data_filename = "QM9"
    _unit = "orbits"
    n_mean = 8
    threshold = True
    method = "MC"


class MUTAG(FeatureDataset):
    """ Exactly calculated feature vectors of 180 graohs in the `MUTAG dataset
    <https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets>`__
    :cite:`debnath1991structure, kriege2012subgraph`.
    """

    _data_filename = "MUTAG"
    _unit = "orbits"
    n_mean = 8
    threshold = True
    method = "exact"



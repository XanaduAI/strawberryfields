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
GBS Datasets
============

**Module name:** :mod:`strawberryfields.gbs.data`

.. currentmodule:: strawberryfields.gbs.data

This module provides access to pre-calculated datasets of GBS samples generated from a range of
input graphs. Each dataset corresponds to a set of samples generated from a fixed graph with
corresponding values for:

- ``n_mean``: mean number of photons in the GBS device
- ``n_max``: maximum number of photons allowed in any sample
-  ``threshold``: flag to indicate whether samples are generated with threshold detection or
   with photon number resolving detectors.

Datasets are available as classes. The following are provided:

- **Planted graph**: A 30-node graph with a dense 10-node subgraph planted inside
  :cite:`arrazola2018using`.


Each dataset has the methods:

.. currentmodule:: strawberryfields.gbs.data.Dataset

.. autosummary::
    n_mean
    n_max
    threshold
    counts
"""
from abc import ABCMeta, abstractmethod

import numpy as np
import pkg_resources
import scipy

DATA_PATH = pkg_resources.resource_filename("strawberryfields", "gbs/data") + "/"


class Dataset(metaclass=ABCMeta):
    """Base class for loading pre-generated samples."""

    _count = 0

    @property
    @abstractmethod
    def _data_filename(self) -> str:
        """Name of files stored in ``./data/``. Samples should be provided as a
        ``scipy.sparse.csr_matrix`` saved in ``.npz`` format and the corresponding adjacency
        matrix should be provided as a ``.npy`` binary. For ``_data_filename = "example"``,
        the corresponding samples should be stored as ``./data/example.npz`` and the adjacency
        matrix as ``./data/example_A.npy``."""
        pass

    def __init__(self):
        """Instantiating ``Dataset`` causes the ``scipy.sparse.csr_matrix`` of samples and
        ``numpy.ndarray`` adjacency matrix to be loaded."""
        self.dat = scipy.sparse.load_npz(DATA_PATH + self._data_filename + ".npz")
        self.adj = np.load(DATA_PATH + self._data_filename + "_A.npy")
        self.n_samples, self.modes = self.dat.shape

    def __iter__(self):
        return self

    def __next__(self):
        if self._count < self.n_samples:
            self._count += 1
            return self.__getitem__(self._count - 1)
        self._count = 0
        raise StopIteration

    def _elem(self, i):
        """Access the i-th element of the sparse array and output as a list."""
        return list(self.dat[i].toarray()[0])

    def __getitem__(self, key):
        if isinstance(key, (slice, tuple)):
            if isinstance(key, tuple):
                key = slice(*key)
            range_tuple = key.indices(self.n_samples)
            return [self._elem(i) for i in range(*range_tuple)]

        if key < 0:
            key += self.n_samples

        return self._elem(key)

    def counts(self, axis: int = 1) -> list:
        """Count number of photons or clicks.

        Counts number of photons/clicks in each sample (``axis==1``) or number of photons/clicks
        in each mode compounded over all samples (``axis=0``).

        Args:
            axis (int): axis to perform count

        Returns:
            list: counts from samples
        """
        return np.array(self.dat.sum(axis)).flatten().tolist()

    @property
    @abstractmethod
    def n_mean(self) -> float:
        """float: Mean number of photons in the GBS device."""
        pass

    @property
    @abstractmethod
    def n_max(self) -> float:
        """float: Maximum number of photons allowed in any sample.

        This number is set to limit the computation time, any sample being simulated that exceeds ``n_max`` will be ignored
        and cause calculation to skip to the next sample."""
        pass

    @property
    @abstractmethod
    def threshold(self) -> bool:
        """bool: flag to indicate whether samples are generated with threshold detection (i.e.,
        detectors of zero or some photons) or with photon number resolving detectors."""
        pass


class Planted(Dataset):

    _data_filename = "planted"
    n_mean = 6
    n_max = 14
    threshold = True

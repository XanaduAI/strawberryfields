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
"""
GBS Datasets
============
"""
from abc import ABCMeta, abstractmethod

import numpy as np
import pkg_resources
import scipy.sparse

DATA_PATH = pkg_resources.resource_filename("strawberryfields", "gbs/data") + '/'


class SampleLoader(metaclass=ABCMeta):
    """Base class for loading pre-generated samples."""

    _count = 0

    @property
    @abstractmethod
    def dat_filename(self) -> str:
        pass

    def __init__(self):
        self.dat = scipy.sparse.load_npz(DATA_PATH + self.dat_filename + ".npz")
        self.adj = np.load(DATA_PATH + self.dat_filename + "_A.npy")
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
        return list(self.dat[i].toarray()[0])

    def __getitem__(self, key):
        if isinstance(key, slice):
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
        pass

    @property
    @abstractmethod
    def n_max(self) -> float:
        pass

    @property
    @abstractmethod
    def threshold(self) -> bool:
        pass


class Planted(SampleLoader):

    dat_filename = "planted"
    n_mean = 6
    n_max = 14
    threshold = True

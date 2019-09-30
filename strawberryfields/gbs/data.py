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

This module provides access to pre-calculated datasets of simulated GBS samples generated from a
range of input graphs. Each dataset corresponds to a set of samples generated from a fixed graph
with corresponding values for:

- ``n_mean``: mean number of photons in the GBS device

- ``n_max``: maximum number of photons allowed in any sample

-  ``threshold``: flag to indicate whether samples are generated with threshold detection or
   with photon-number-resolving detectors.

- ``n_samples``: total number of samples in the dataset

- ``modes``: number of modes in the GBS device or, equivalently, number of nodes in the graph

Datasets are available as classes. The following graphs and datasets are provided:

- **Planted graph**: A 30-node graph with a dense 10-node subgraph planted inside
  :cite:`arrazola2018using`.

  .. image:: ../../_static/planted.png
        :align: center
        :width: 300px

  Provided by the :class:`Planted` Dataset with attributes ``n_mean = 6``, ``n_max = 14``,
  ``threshold = True``, ``n_samples = 50000``, ``modes = 30``.

Loading data
^^^^^^^^^^^^

We use the :class:`Planted` class as an example to show how to interact with the datasets. Datasets
can be loaded by running:

>>> data = Planted()

Simply use indexing and slicing to access samples from the dataset:

>>> sample_3 = data[3]
>>> samples = data[:10]

Datasets also contain metadata relevant to the GBS setup:

>>> data.n_mean
6

>>> len(data)
50000

The number of photons or clicks in each sample is available using the :meth:`Dataset.counts` method:

>>> data.counts()
[2, 0, 8, 11, ... , 6]

For example, we see that the ``data[3]`` sample had 11 clicks.

Datasets
^^^^^^^^

The :class:`Dataset` class provides the base functionality from which all datasets such as
:class:`Planted` inherit.

.. autosummary::
    Dataset
    Planted

Code details
^^^^^^^^^^^^
"""
# pylint: disable=unnecessary-pass
from abc import ABCMeta, abstractmethod

import pkg_resources
import numpy as np
import scipy

DATA_PATH = pkg_resources.resource_filename("strawberryfields", "gbs/data") + "/"


class Dataset(metaclass=ABCMeta):
    """Base class for loading datasets of pre-generated samples.

    Attributes:
        n_mean (float): mean number of photons in the GBS device
        n_max (float): Maximum number of photons allowed in any sample. This number is set to
            limit the computation time: any sample being simulated that exceeds ``n_max`` will be
            ignored and cause the calculation to skip to the next sample.
        threshold (bool): flag to indicate whether samples are generated with threshold detection
            (i.e., detectors of zero or some photons) or with photon-number-resolving detectors.
        adj (array): adjacency matrix of the graph from which samples were generated
        n_samples (int): total number of samples in the dataset
        modes (int): number of modes in the GBS device or, equivalently, number of nodes in graph
        data (sparse): raw data of samples from GBS as a `csr sparse array
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`__.
    """

    _count = 0

    @property
    @abstractmethod
    def _data_filename(self) -> str:
        """Base name of files containing the sample data stored in the ``./data/`` directory.
        Samples should be provided as a ``scipy.sparse.csr_matrix`` saved in ``.npz`` format and
        the corresponding adjacency matrix should be provided as a ``.npy`` binary. For
        ``_data_filename = "example"``, the corresponding samples should be stored as
        ``./data/example.npz`` and the adjacency matrix as ``./data/example_A.npy``."""
        pass

    def __init__(self):
        self.data = scipy.sparse.load_npz(DATA_PATH + self._data_filename + ".npz")
        self.adj = np.load(DATA_PATH + self._data_filename + "_A.npy")
        self.n_samples, self.modes = self.data.shape

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
        return list(self.data[i].toarray()[0])

    def __getitem__(self, key):
        if isinstance(key, (slice, tuple)):
            if isinstance(key, tuple):
                key = slice(*key)
            range_tuple = key.indices(self.n_samples)
            return [self._elem(i) for i in range(*range_tuple)]

        if key < 0:
            key += self.n_samples

        return self._elem(key)

    def __len__(self):
        return self.n_samples

    def counts(self, axis: int = 1) -> list:
        """Count number of photons or clicks.

        Counts number of photons/clicks in each sample (``axis==1``) or number of photons/clicks
        in each mode compounded over all samples (``axis==0``).

        Args:
            axis (int): axis to perform count

        Returns:
            list: counts from samples
        """
        return np.array(self.data.sum(axis)).flatten().tolist()

    # pylint: disable=missing-docstring
    @property
    @abstractmethod
    def n_mean(self) -> float:
        pass

    # pylint: disable=missing-docstring
    @property
    @abstractmethod
    def n_max(self) -> float:
        pass

    # pylint: disable=missing-docstring
    @property
    @abstractmethod
    def threshold(self) -> bool:
        pass


class Planted(Dataset):
    """Dataset of samples generated from GBS with an embedded 30-node graph containing a dense
    10-node subgraph planted inside :cite:`arrazola2018using`.

    Attributes:
        n_mean = 6
        n_max = 14
        threshold = True
        n_samples = 50000
        modes = 30
    """

    _data_filename = "planted"
    n_mean = 6
    n_max = 14
    threshold = True

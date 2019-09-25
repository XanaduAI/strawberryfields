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
import pkg_resources
import scipy.sparse

DATA_PATH = pkg_resources.resource_filename("strawberryfields", "gbs/data")


class SampleLoader:
    """To do"""

    dat = scipy.sparse.spmatrix()
    n_samples = 0
    count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.n_samples:
            self.count += 1
            return self.__getitem__(self.count - 1)
        self.count = 0
        raise StopIteration

    def _elem(self, i):
        return list(self.dat[i].toarray()[0])

    def __getitem__(self, key):
        if isinstance(key, slice):
            range_tuple = key.indices(self.n_samples)
            return [self._elem(i) for i in range(range_tuple)]

        if key < 0:
            key += self.n_samples

        return self._elem(key)

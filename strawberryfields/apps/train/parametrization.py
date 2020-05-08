# Copyright 2020 Xanadu Quantum Technologies Inc.

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
Contains the :class:`~.VGBS` (variational GBS) class which provides a trainable parametrization
of GBS.
"""
import numpy as np


# TODO: update embedding class in type hint
class VGBS:
    """TODO"""

    def __init__(self, a: np.ndarray, n_mean: float, embedding, threshold: bool):
        self.a = a
        self.n_mean = n_mean
        self.embedding = embedding
        self.threshold = threshold

        self.a_scaled = a

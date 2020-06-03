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
"""Data module provide pre-calculated datasets of GBS kernels
 and GBS samples.

.. currentmodule:: strawberryfields.apps.data
.. autosummary::
    :toctree: api

	GBS Sample datasets
	GBS Kernel datasets
"""

import strawberryfields.apps.data.sample
import strawberryfields.apps.data.feature
from strawberryfields.apps.data.sample import Planted, TaceAs, PHat, Mutag0, Mutag1, Mutag2, Mutag3, Formic



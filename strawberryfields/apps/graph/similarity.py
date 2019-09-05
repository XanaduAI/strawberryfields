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
Graph similarity
================

**Module name:** :mod:`strawberryfields.apps.graph.similarity`

.. currentmodule:: strawberryfields.apps.graph.similarity

Functionality for calculating feature vectors of graphs using GBS

Summary
-------

.. autosummary::
    sample_to_orbit


Code details
^^^^^^^^^^^^
"""


def sample_to_orbit(sample: list) -> list:
    """Provides the orbit corresponding to a given sample.

    Args:
        sample (list[int]): a sample from GBS

    Returns:
        list[int]: the orbit of the sample
    """
    return sorted(sample)

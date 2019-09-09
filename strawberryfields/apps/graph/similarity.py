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

Functionality for calculating feature vectors of graphs using GBS.

Summary
-------

.. autosummary::
    sample_to_orbit
    sample_to_event

Code details
^^^^^^^^^^^^
"""


def sample_to_orbit(sample: list) -> list:
    """Provides the orbit corresponding to a given sample.

    Orbits are simply a sorting of samples in non-increasing order.

    Args:
        sample (list[int]): a sample from GBS

    Returns:
        list[int]: the orbit of the sample
    """
    return sorted(sample, reverse=True)


def sample_to_event(sample: list, max_count_per_mode: int) -> list:
    """Provides the event corresponding to a given sample.

    Args:
        sample (list[int]): a sample from GBS
        max_count_per_mode (int): the maximum number of photons counted in any given mode for a
            sample to categorized as an event. Samples with counts exceeding this value are
            attributed the event ``None``.

    Returns:
        list[int]: the orbit of the sample
    """
    if max(sample) <= max_count_per_mode:
        return sum(sample)

    return None

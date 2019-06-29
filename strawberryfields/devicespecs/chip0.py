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
"""Chip0 backend validation data"""
import textwrap

from .device_specs import DeviceSpecs


class Chip0Specs(DeviceSpecs):
    """Validation data for the Chip0 simulator"""

    short_name = 'chip0'
    modes = 4
    remote = True
    local = True
    interactive = True

    primitives = {"S2gate", "Interferometer", "MeasureFock", "Rgate", "BSgate"}

    # TODO: update the below to specify the rectangular_symmetric
    # mapping for the interferometer when #87 is merged
    decompositions = {"Interferometer": {}}

    # TODO: update the below to specify the rectangular_symmetric
    # mapping for the interferometer when #87 is merged.
    # The current topology defined below is just for demonstration
    circuit = textwrap.dedent(
        """\
        name chip0_template
        version 0.0
        target chip0 (shots=1)

        S2gate({sq0}, 0.0) | [0, 2]
        S2gate({sq1}, 0.0) | [1, 3]

        Rgate({phase}) | 0
        BSgate({theta}, {phi}) | [0, 1]
        Rgate({phase}) | 0
        Rgate({phase}) | 1

        Rgate({phase}) | 2
        BSgate({theta}, {phi}) | [2, 3]
        Rgate({phase}) | 2
        Rgate({phase}) | 3

        MeasureFock() | [0]
        MeasureFock() | [1]
        MeasureFock() | [2]
        MeasureFock() | [3]
        """
    )

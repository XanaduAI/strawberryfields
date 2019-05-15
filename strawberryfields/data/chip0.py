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

import networkx as nx
import blackbird
# from strawberryfields.io import to_DiGraph

from .device import DeviceData


class Chip0(DeviceData):
    """Validation data for the Chip0 simulator"""

    modes = 0
    remote = True
    interactive = True

    primitives = {"S2gate", "BSgate", "MeasureFock"}

    decompositions = {"BSgate": {}, "Rgate": {}, "Interferometer": {}}

    blackbird_template = textwrap.dedent(
        """\
        name {name}
        version 1.0
        target gbsdevice_01 (shots={shots})

        # for n spatial degrees, first n signal modes, then n idler modes, phase zero
        S2gate({squeezing0}, 0.0) | [0, 2]
        S2gate({squeezing1}, 0.0) | [1, 3]

        # standard 2x2 interferometer for the signal modes (the lower ones in frequency)
        BSgate(0.0, {external_phase}) | [0, 1]
        BSgate({internal_phase}, 0.0) | [0, 1]

        #duplicate the interferometer for the idler modes (the higher ones in frequency)
        BSgate(0.0, {external_phase}) | [2, 3]
        BSgate({internal_phase}, 0.0) | [2, 3]

        # Measurement in Fock basis
        Measure | [0]
        Measure | [1]
        Measure | [2]
        Measure | [3]
        """
    )

    # returned DAG has all parameters set to 0
    topology = to_DiGraph(
        blackbird.loads(
            self.blackbird_template.format(
                name="topology",
                shots=1,
                squeezing0=0,
                squeezing1=0,
                external_phase=0,
                internal_phase=0,
            )
        )
    )

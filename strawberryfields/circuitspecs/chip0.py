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
"""Circuit class specification for the chip0 class of circuits."""
import textwrap

from .circuit_specs import CircuitSpecs


class Chip0Specs(CircuitSpecs):
    """Circuit specifications for the chip0 class of circuits."""

    short_name = 'chip0'
    modes = 4
    remote = True
    local = True
    interactive = True

    primitives = {"S2gate", "Interferometer", "MeasureFock", "Rgate", "BSgate"}
    decompositions = {"Interferometer": {}}

    circuit = textwrap.dedent(
        """\
        name template_2x2_chip0
        version 1.0
        target chip0 (shots=10)

        # for n spatial degrees, first n signal modes, then n idler modes, phase zero
        S2gate({squeezing_amplitude_0}, 0.0) | [0, 2]
        S2gate({squeezing_amplitude_1}, 0.0) | [1, 3]

        # standard 2x2 interferometer for the signal modes (the lower ones in frequency)
        Rgate({external_phase_0}) | [0]
        BSgate(pi/4, pi/2) | [0, 1]
        Rgate({internal_phase_0}) | [0]
        BSgate(pi/4, pi/2) | [0, 1]

        #duplicate the interferometer for the idler modes (the higher ones in frequency)
        Rgate({external_phase_0}) | [2]
        BSgate(pi/4, pi/2) | [2, 3]
        Rgate({internal_phase_0}) | [2]
        BSgate(pi/4, pi/2) | [2, 3]

        # Measurement in Fock basis
        MeasureFock() | [0, 1, 2, 3]
        """
    )

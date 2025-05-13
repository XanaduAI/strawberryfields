# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect

import networkx as nx
import numpy as np

from strawberryfields.device import Device
from strawberryfields.parameters import par_evaluate
from strawberryfields.program_utils import list_to_DAG

pi = np.pi

X8_layout = inspect.cleandoc(
    """
    name template_4x2_X8
    version 1.0
    target X8_01 (shots=1)

    # for n spatial degrees, first n signal modes, then n idler modes, all phases zero
    S2gate({squeezing_amplitude_0}, 0.0) | [0, 4]
    S2gate({squeezing_amplitude_1}, 0.0) | [1, 5]
    S2gate({squeezing_amplitude_2}, 0.0) | [2, 6]
    S2gate({squeezing_amplitude_3}, 0.0) | [3, 7]

    # standard 4x4 interferometer for the signal modes (the lower ones in frequency)
    # even phase indices correspond to internal Mach-Zehnder interferometer phases
    # odd phase indices correspond to external Mach-Zehnder interferometer phases
    MZgate({phase_0}, {phase_1}) | [0, 1]
    MZgate({phase_2}, {phase_3}) | [2, 3]
    MZgate({phase_4}, {phase_5}) | [1, 2]
    MZgate({phase_6}, {phase_7}) | [0, 1]
    MZgate({phase_8}, {phase_9}) | [2, 3]
    MZgate({phase_10}, {phase_11}) | [1, 2]

    # duplicate the interferometer for the idler modes (the higher ones in frequency)
    MZgate({phase_0}, {phase_1}) | [4, 5]
    MZgate({phase_2}, {phase_3}) | [6, 7]
    MZgate({phase_4}, {phase_5}) | [5, 6]
    MZgate({phase_6}, {phase_7}) | [4, 5]
    MZgate({phase_8}, {phase_9}) | [6, 7]
    MZgate({phase_10}, {phase_11}) | [5, 6]

    # add final dummy phases to allow mapping any unitary to this template (these do not
    # affect the photon number measurement)
    Rgate({final_phase_0}) | [0]
    Rgate({final_phase_1}) | [1]
    Rgate({final_phase_2}) | [2]
    Rgate({final_phase_3}) | [3]
    Rgate({final_phase_4}) | [4]
    Rgate({final_phase_5}) | [5]
    Rgate({final_phase_6}) | [6]
    Rgate({final_phase_7}) | [7]

    # measurement in Fock basis
    MeasureFock() | [0, 1, 2, 3, 4, 5, 6, 7]
    """
)

X8_spec = {
    "target": "X8_01",
    "layout": X8_layout,
    "modes": 8,
    "compiler": [],
    "gate_parameters": {
        "squeezing_amplitude_0": [0, 1],
        "squeezing_amplitude_1": [0, 1],
        "squeezing_amplitude_2": [0, 1],
        "squeezing_amplitude_3": [0, 1],
        "phase_0": [0, [0, 6.283185307179586]],
        "phase_1": [0, [0, 6.283185307179586]],
        "phase_2": [0, [0, 6.283185307179586]],
        "phase_3": [0, [0, 6.283185307179586]],
        "phase_4": [0, [0, 6.283185307179586]],
        "phase_5": [0, [0, 6.283185307179586]],
        "phase_6": [0, [0, 6.283185307179586]],
        "phase_7": [0, [0, 6.283185307179586]],
        "phase_8": [0, [0, 6.283185307179586]],
        "phase_9": [0, [0, 6.283185307179586]],
        "phase_10": [0, [0, 6.283185307179586]],
        "phase_11": [0, [0, 6.283185307179586]],
        "final_phase_0": [0, [0, 6.283185307179586]],
        "final_phase_1": [0, [0, 6.283185307179586]],
        "final_phase_2": [0, [0, 6.283185307179586]],
        "final_phase_3": [0, [0, 6.283185307179586]],
        "final_phase_4": [0, [0, 6.283185307179586]],
        "final_phase_5": [0, [0, 6.283185307179586]],
        "final_phase_6": [0, [0, 6.283185307179586]],
        "final_phase_7": [0, [0, 6.283185307179586]],
    },
}


X8_device = Device(spec=X8_spec)

borealis_layout = inspect.cleandoc(
    """
    name template_borealis
    version 1.0
    target borealis (shots=1)
    type tdm (temporal_modes=259, copies=1)

    float array p0[1, 259] =
        {s}
    float array p1[1, 259] =
        {r0}
    float array p2[1, 259] =
        {bs0}
    float array p3[1, 259] =
        {loop1_phase}
    float array p4[1, 259] =
        {r1}
    float array p5[1, 259] =
        {bs1}
    float array p6[1, 259] =
        {loop2_phase}
    float array p7[1, 259] =
        {r2}
    float array p8[1, 259] =
        {bs2}
    float array p9[1, 259] =
        {loop3_phase}


    Sgate({s}, 0.0) | 43
    Rgate({r0}) | 43
    BSgate({bs0}, 1.5707963267948966) | [42, 43]
    Rgate({loop0_phase}) | 43
    Rgate({r1}) | 42
    BSgate({bs1}, 1.5707963267948966) | [36, 42]
    Rgate({loop1_phase}) | 42
    Rgate({r2}) | 36
    BSgate({bs2}, 1.5707963267948966) | [0, 36]
    Rgate({loop2_phase}) | 36
    MeasureFock() | 0
    """
)
borealis_spec = {
    "target": "borealis",
    "layout": borealis_layout,
    "modes": {"temporal_max": 331, "concurrent": 44, "spatial": 1},
    "compiler": ["borealis"],
    "compiler_default": "borealis",
    "gate_parameters": {
        "s": [[0, 2]],
        "r0": [[-pi / 2, pi / 2]],
        "bs0": [[0, pi / 2]],
        "loop0_phase": [[-pi, pi]],
        "r1": [[-pi / 2, pi / 2]],
        "bs1": [[0, pi / 2]],
        "loop1_phase": [[-pi, pi]],
        "r2": [[-pi / 2, pi / 2]],
        "bs2": [[0, pi / 2]],
        "loop2_phase": [[-pi, pi]],
    },
}
borealis_cert = {
    "target": "borealis",
    "finished_at": "2022-02-03T15:00:59.641616+00:00",
    "loop_phases": [0.1, -0.1, 3],
    "schmidt_number": 1.333,
    "common_efficiency": 0.55,
    "loop_efficiencies": [0.9, 0.8, 0.7],
    "squeezing_parameters_mean": {"low": [0.1], "high": [0.5], "medium": [0.3]},
    "relative_channel_efficiencies": [],
}
borealis_device = Device(spec=borealis_spec, cert=borealis_cert)


def generate_X8_params(r, p):
    return {
        "squeezing_amplitude_0": r,
        "squeezing_amplitude_1": r,
        "squeezing_amplitude_2": r,
        "squeezing_amplitude_3": r,
        "phase_0": p,
        "phase_1": p,
        "phase_2": p,
        "phase_3": p,
        "phase_4": p,
        "phase_5": p,
        "phase_6": p,
        "phase_7": p,
        "phase_8": p,
        "phase_9": p,
        "phase_10": p,
        "phase_11": p,
        "final_phase_0": 1.24,
        "final_phase_1": 0.54,
        "final_phase_2": 4.12,
        "final_phase_3": 0,
        "final_phase_4": 1.24,
        "final_phase_5": 0.54,
        "final_phase_6": 4.12,
        "final_phase_7": 0,
    }

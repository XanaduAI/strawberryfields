# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Circuit specifications for general Gaussian simulator backends."""
from .compiler import Compiler


class Bosonic(Compiler):
    """Compiler for general Bosonic backends."""

    short_name = "bosonic"
    interactive = True

    primitives = {
        # meta operations
        "All",
        "_New_modes",
        "_Delete",
        # state preparations
        "Vacuum",
        "Coherent",
        "Squeezed",
        "DisplacedSqueezed",
        "Thermal",
        "Gaussian",
        "Fock",
        "Ket",
        "DensityMatrix",
        "Bosonic",
        "GKP",
        "Catstate",
        "Comb",
        # measurements
        "MeasureHomodyne",
        "MeasureHeterodyne",
        # TODO: "MeasureFock",
        "MeasureThreshold",
        # channels
        "LossChannel",
        "ThermalLossChannel",
        # single mode gates
        "Dgate",
        "Sgate",
        "Rgate",
        "BSgate",
        "MSgate",
    }

    decompositions = {
        "Pgate": {},
        "S2gate": {},
        "CXgate": {},
        "CZgate": {},
        "MZgate": {},
        "Xgate": {},
        "Zgate": {},
        "Fouriergate": {},
    }

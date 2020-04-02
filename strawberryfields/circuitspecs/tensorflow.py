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
"""Circuit specifications for the TensorFlow simulator backend."""
from .circuit_specs import CircuitSpecs


class TFSpecs(CircuitSpecs):
    """Circuit specifications for the TensorFlow backend."""

    short_name = "tf"
    modes = None
    local = True
    remote = False
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
        "Fock",
        "Catstate",
        "Ket",
        "DensityMatrix",
        # measurements
        "MeasureFock",
        "MeasureHomodyne",
        # channels
        "LossChannel",
        # single mode gates
        "Dgate",
        "Sgate",
        "Rgate",
        "Vgate",
        "Kgate",
        "Fouriergate",
        "BSgate",
        "CKgate",
    }

    decompositions = {
        "Interferometer": {},
        "GraphEmbed": {},
        "BipartiteGraphEmbed": {},
        "GaussianTransform": {},
        "Gaussian": {},
        "Pgate": {},
        "S2gate": {},
        "CXgate": {},
        "CZgate": {},
        "MZgate": {},
        "Xgate": {},
        "Zgate": {},
        "Fouriergate": {},
    }

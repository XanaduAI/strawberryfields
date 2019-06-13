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
"""Gaussian backend validation data"""
from .device_specs import DeviceSpecs


class GaussianSpecs(DeviceSpecs):
    """Validation data for the Gaussian backend"""

    modes = None
    local = True
    remote = True
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
        # measurements
        "MeasureHomodyne",
        "MeasureHeterodyne",
        # channels
        "LossChannel",
        "ThermalLossChannel",
        # single mode gates
        "Dgate",
        "Xgate",
        "Zgate",
        "Sgate",
        "Rgate",
        "Fouriergate",
        "BSgate",
    }

    decompositions = {
        "Interferometer": {},
        "GraphEmbed": {},
        "GaussianTransform": {},
        "Gaussian": {},
        "Pgate": {},
        "S2gate": {},
        "CXgate": {},
        "CZgate": {},
    }

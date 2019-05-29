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
"""Fock backend validation data"""
from .device_specs import DeviceSpecs


class BaseSpecs(DeviceSpecs):
    """Validation data for the Base backend. The information
    below matches the methods supported by the abstract base
    class :class:`~.BaseBackend`."""

    modes = None
    remote = False
    local = True
    interactive = True

    primitives = {
        # meta operations
        "All",
        "New_modes",
        "Delete",
        # state preparations
        "Vacuum",
        "Coherent",
        "Squeezed",
        "DisplacedSqueezed",
        "Thermal",
        # measurements
        "MeasureHomodyne",
        # channels
        "LossChannel",
        # single mode gates
        "Dgate",
        "Xgate",
        "Zgate",
        "Sgate",
        "Rgate",
        "Vgate",
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

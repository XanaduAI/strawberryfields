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
"""Device validation submodule"""
from .chip0 import Chip0Specs
from .fock import FockSpecs
from .gaussian import GaussianSpecs
from .tensorflow import TFSpecs


backend_databases = {
    "base": FockSpecs,
    "chip0": Chip0Specs,
    "fock": FockSpecs,
    "gaussian": GaussianSpecs,
    "tf": TFSpecs,
}
"""dict[str, DeviceSpecs]: dictionary mapping backend shortname
to the validation dataclass."""

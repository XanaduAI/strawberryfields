# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This package contains the modules for the low-level Strawberry Fields program
execution API. Specifically, the :class:`~strawberryfields.api.Result` and
:class:`~strawberryfields.api.DeviceSpec` classes are provided to represent
hardware results and device specifications, respectively.
"""

from .devicespec import DeviceSpec
from .result import Result

__all__ = ["DeviceSpec", "Result"]


class FailedJobError(Exception):
    """Raised when a job had a failure on the server side."""

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
execution API. The :class:`~strawberryfields.api.Connection` class mediates
the network connection to, and exposes operations provided by, a remote program
execution backend. The :class:`~strawberryfields.api.Job` and
:class:`~strawberryfields.api.Result` classes provide interfaces for managing
program execution jobs and job results respectively.
"""

from .connection import Connection, RequestFailedError
from .devicespec import DeviceSpec
from .job import InvalidJobOperationError, Job, JobStatus
from .result import Result

__all__ = ["Connection", "DeviceSpec", "Job", "Result"]

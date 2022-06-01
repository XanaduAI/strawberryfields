# Copyright 2019-2022 Xanadu Quantum Technologies Inc.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This package contains classes and functions for creating time-domain
algorithms in Strawberry Fields."""

from .program import TDMProgram, reshape_samples, is_ptype, shift_by
from .utils import (
    borealis_gbs,
    full_compile,
    get_mode_indices,
    loop_phase_from_device,
    make_phases_compatible,
    make_squeezing_compatible,
    move_vac_modes,
    vacuum_padding,
    random_r,
    random_bs,
    to_args_list,
    to_args_dict,
)

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

"""Temporary module for stuff that hasn't found its place yet."""

import strawberryfields.devicespecs as specs
import strawberryfields.ops as ops
from strawberryfields.program import (CircuitError, Command)



def check_GBS(prog):
    """Try to arrange the circuit into a form suitable for Gaussian boson sampling.

    This method checks whether the circuit can be implemented as a Gaussian boson sampling
    problem, i.e., if it is equivalent to a circuit A+B, where the sequence A only contains
    Gaussian operations, and B only contains Fock measurements.

    If the answer is yes, the circuit is arranged into the A+B order, and all the Fock measurements
    are combined into a single :class:`MeasureFock` operation.

    Args:
        prog (Program): quantum circuit to modify

    Raises:
        CircuitError: the circuit does not correspond to GBS
    """
    A, B, C = prog._group_operations(prog.circuit, lambda x: isinstance(x, ops.MeasureFock))

    # C should be empty
    if C:
        raise CircuitError('Operations following the Fock measurements.')

    # A should only contain Gaussian operations
    db = specs.backend_specs['gaussian']()
    gaussian_ops = db.primitives | set(db.decompositions.keys())
    for cmd in A:
        temp = cmd.op.__class__.__name__
        if temp not in gaussian_ops:
            raise CircuitError('Non-gaussian Operation: {}.'.format(temp))

    # without Fock measurements GBS is pointless
    if not B:
        raise CircuitError('No Fock measurements.')

    # there should be only Fock measurements in B
    measured = set()
    for cmd in B:
        if not isinstance(cmd.op, ops.MeasureFock):
            raise CircuitError('The Fock measurements are not consecutive.')
        else:
            # combine the Fock measurements
            temp = set(cmd.reg)
            if measured & temp:
                raise CircuitError('Measuring the same mode more than once.')
            measured |= temp

    # replace B with a single Fock measurement
    B = [Command(ops.MeasureFock(), list(measured))]

    prog.circuit = A+B

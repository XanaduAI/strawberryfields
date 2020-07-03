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
"""Compiler for the general Gaussian Boson Sampling class of circuits."""

from strawberryfields.program_utils import CircuitError, Command, group_operations
import strawberryfields.ops as ops

from .gaussian import Gaussian


class GBS(Gaussian):
    """Compiler for the general GBS class of circuits."""

    short_name = "gbs"
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
        "MeasureFock",
        "MeasureThreshold",
        # channels
        "LossChannel",
        "ThermalLossChannel",
        # single mode gates
        "Dgate",
        "Sgate",
        "Rgate",
        "Fouriergate",
        "BSgate",
    }
    decompositions = {"Xgate": {}, "Zgate": {}, "Fouriergate": {}, "S2gate": {}}

    def compile(self, seq, registers):
        """Try to arrange a quantum circuit into a form suitable for Gaussian boson sampling.

        This method checks whether the circuit can be implemented as a Gaussian boson sampling
        problem, i.e., if it is equivalent to a circuit A+B, where the sequence A only contains
        Gaussian operations, and B only contains Fock measurements.

        If the answer is yes, the circuit is arranged into the A+B order, and all the Fock
        measurements are combined into a single :class:`MeasureFock` operation.

        Args:
            seq (Sequence[Command]): quantum circuit to modify
            registers (Sequence[RegRefs]): quantum registers
        Returns:
            List[Command]: modified circuit
        Raises:
            CircuitError: the circuit does not correspond to GBS
        """
        A, B, C = group_operations(seq, lambda x: isinstance(x, ops.MeasureFock))

        # C should be empty
        if C:
            raise CircuitError("Operations following the Fock measurements.")

        # A should only contain Gaussian operations
        # (but this is already guaranteed by group_operations() and our primitive set)

        # without Fock measurements GBS is pointless
        if not B:
            raise CircuitError("GBS circuits must contain Fock measurements.")

        # there should be only Fock measurements in B
        measured = set()
        for cmd in B:
            if not isinstance(cmd.op, ops.MeasureFock):
                raise CircuitError("The Fock measurements are not consecutive.")

            # combine the Fock measurements
            temp = set(cmd.reg)
            if measured & temp:
                raise CircuitError("Measuring the same mode more than once.")
            measured |= temp

        # replace B with a single Fock measurement
        B = [Command(ops.MeasureFock(), sorted(list(measured), key=lambda x: x.ind))]
        return super().compile(A + B, registers)

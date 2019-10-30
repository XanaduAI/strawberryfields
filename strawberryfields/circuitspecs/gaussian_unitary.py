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
"""Circuit specifications for the Gaussian simulator backend."""
from .circuit_specs import CircuitSpecs
from strawberryfields.program_utils import Command
import numpy as np
from .matrices import *
from strawberryfields import ops
class GaussianUnitary(CircuitSpecs):
    """Circuit specifications for the Gaussian backend."""

    short_name = 'gaussian_unitary'
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
        #"Vacuum",
        #"Coherent",
        #"Squeezed",
        #"DisplacedSqueezed",
        #"Thermal",
        #"Gaussian",
        # measurements
        #"MeasureHomodyne",
        #"MeasureHeterodyne",
        #"MeasureFock",
        #"MeasureThreshold",
        # single mode gates
        "Dgate",
        "Sgate",
        "Rgate",
        # multi mode gates
        "BSgate",
        "S2gate",
        "Interferometer", #Note that interferometer is accepted as a primitive
    }

    decompositions = {
        #"Interferometer": {},
        "GraphEmbed": {},
        "BipartiteGraphEmbed": {},
        "GaussianTransform": {},
        "Gaussian": {},
        "Pgate": {},
        "CXgate": {},
        "CZgate": {},
        "MZgate": {},
        "Xgate": {},
        "Zgate": {},
        "Fouriergate": {},
    }


    def compile(self, seq, registers):
        """Try to arrange a quantum circuit into a the canonical Symplectic form.

        This method checks whether the circuit can be implemented as a sequence of Gaussian operations.
        If the answer is yes it arranges them in the canonical order with displacement at the end.


        Args:
            seq (Sequence[Command]): quantum circuit to modify
            registers (Sequence[RegRefs]): quantum registers
        Returns:
            List[Command]: modified circuit
        Raises:
            CircuitError: the circuit does not correspond to GBS
        """
        #[print(ops.op) for ops in seq]
        indices = [i.ind for i in registers]
        indices.sort()
        dict_indices = {indices[i]:i for i in range(len(indices))}

        nmodes = len(indices)
        Snet = np.identity(2 * nmodes)
        rnet = np.zeros(2 * nmodes)

        for operations in seq:
            name = operations.op.__class__.__name__
            params = [i.x for i in operations.op.p]
            modes = [thing.ind for thing in operations.reg]

            if name == 'Dgate':
                rnet = rnet+expand_vector(params[0]*(np.exp(1j*params[1])), dict_indices[modes[0]], nmodes)
            else:
                if name == 'Rgate':
                    S = expand(rotation(params[0]), dict_indices[modes[0]], nmodes)
                elif name == 'Sgate':
                    S = expand(squeezing(params[0], params[1]), dict_indices[modes[0]], nmodes)
                elif name == 'S2gate':
                    S = expand(two_mode_squeezing(params[0], params[1]), [dict_indices[modes[0]], dict_indices[modes[1]]], nmodes)
                elif name == 'Interferometer':
                    S = expand(interferometer(params[0]), [dict_indices[mode] for mode in modes], nmodes)
                Snet = S @ Snet
                rnet = S @ rnet
        #A = [Command(ops.GaussianTransform(Snet), indices)]
        A = [Command(ops.GaussianTransform(Snet), sorted(list(registers), key=lambda x: x.ind))]
        return A
        """
        A, B, C = group_operations(seq, lambda x: isinstance(x, ops.MeasureFock))

        # C should be empty
        if C:
            raise CircuitError('Operations following the Fock measurements.')

        # A should only contain Gaussian operations
        # (but this is already guaranteed by group_operations() and our primitive set)

        # without Fock measurements GBS is pointless
        if not B:
            raise CircuitError('GBS circuits must contain Fock measurements.')

        # there should be only Fock measurements in B
        measured = set()
        for cmd in B:
            if not isinstance(cmd.op, ops.MeasureFock):
                raise CircuitError('The Fock measurements are not consecutive.')

            # combine the Fock measurements
            temp = set(cmd.reg)
            if measured & temp:
                raise CircuitError('Measuring the same mode more than once.')
            measured |= temp

        # replace B with a single Fock measurement
        B = [Command(ops.MeasureFock(), sorted(list(measured), key=lambda x: x.ind))]
        return super().compile(A + B, registers)
        """
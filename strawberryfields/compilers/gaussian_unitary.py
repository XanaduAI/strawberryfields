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
"""This module contains a compiler to arrange a Gaussian quantum circuit into the canonical Symplectic form."""

import numpy as np
from strawberryfields.program_utils import Command
from strawberryfields import ops
from strawberryfields.parameters import par_evaluate
from thewalrus.symplectic import (
    expand_vector,
    expand,
    rotation,
    squeezing,
    two_mode_squeezing,
    interferometer,
    beam_splitter,
)

from .compiler import Compiler


class GaussianUnitary(Compiler):
    """Compiler to arrange a Gaussian quantum circuit into the canonical Symplectic form.

    This compiler checks whether the circuit can be implemented as a sequence of
    Gaussian operations. If so, it arranges them in the canonical order with displacement at the end.
    After compilation, the circuit will consist of at most two operations, a :class:`~.GaussianTransform`
    and a :class:`~.Dgate`.

    This compiler can be accessed by calling :meth:`.Program.compile` with `'gaussian_unitary'` specified.

    **Example:**

    Consider the following Strawberry Fields program, compiled using the `'gaussian_unitary'` compiler:

    .. code-block:: python3

        from strawberryfields.ops import Xgate, Zgate, Sgate, Dgate, Rgate
        import strawberryfields as sf

        circuit = sf.Program(1)
        with circuit.context as q:
            Xgate(0.4) | q[0]
            Zgate(0.5) | q[0]
            Sgate(0.6) | q[0]
            Dgate(1.0+2.0j) | q[0]
            Rgate(0.3) | q[0]
            Sgate(0.6, 1.0) | q[0]

        compiled_circuit = circuit.compile(compiler="gaussian_unitary")

    We can now print the compiled circuit, consisting of one
    :class:`~.GaussianTransform` and one :class:`~.Dgate`:

    >>> compiled_circuit.print()
    GaussianTransform([[ 0.3543 -1.3857]
                       [-0.0328  2.9508]]) | (q[0])
    Dgate(-1.151+3.91j, 0) | (q[0])
    """

    short_name = "gaussian_unitary"
    interactive = True

    primitives = {
        # meta operations
        "All",
        "_New_modes",
        "_Delete",
        # single mode gates
        "Dgate",
        "Sgate",
        "Rgate",
        # multi mode gates
        "MZgate",
        "BSgate",
        "S2gate",
        "Interferometer",  # Note that interferometer is accepted as a primitive
        "GaussianTransform",  # Note that GaussianTransform is accepted as a primitive
    }

    decompositions = {
        "GraphEmbed": {},
        "BipartiteGraphEmbed": {},
        "Gaussian": {},
        "Pgate": {},
        "CXgate": {},
        "CZgate": {},
        "Xgate": {},
        "Zgate": {},
        "Fouriergate": {},
    }
    # pylint: disable=too-many-branches
    def compile(self, seq, registers):
        """Try to arrange a quantum circuit into the canonical Symplectic form.

        This method checks whether the circuit can be implemented as a sequence of Gaussian operations.
        If the answer is yes it arranges them in the canonical order with displacement at the end.

        Args:
            seq (Sequence[Command]): quantum circuit to modify
            registers (Sequence[RegRefs]): quantum registers
        Returns:
            List[Command]: modified circuit
        Raises:
            CircuitError: the circuit does not correspond to a Gaussian unitary
        """

        # Check which modes are actually being used
        used_modes = []
        for operations in seq:
            modes = [modes_label.ind for modes_label in operations.reg]
            used_modes.append(modes)
        # pylint: disable=consider-using-set-comprehension
        used_modes = list(set([item for sublist in used_modes for item in sublist]))

        # dictionary mapping the used modes to consecutive non-negative integers
        dict_indices = {used_modes[i]: i for i in range(len(used_modes))}
        nmodes = len(used_modes)

        # This is the identity transformation in phase-space, multiply by the identity and add zero
        Snet = np.identity(2 * nmodes)
        rnet = np.zeros(2 * nmodes)

        # Now we will go through each operation in the sequence `seq` and apply it in quadrature space
        # We will keep track of the net transforation in the Symplectic matrix `Snet` and the quadrature
        # vector `rnet`.
        for operations in seq:
            name = operations.op.__class__.__name__
            params = par_evaluate(operations.op.p)
            modes = [modes_label.ind for modes_label in operations.reg]
            if name == "Dgate":
                rnet = rnet + expand_vector(
                    params[0] * (np.exp(1j * params[1])), dict_indices[modes[0]], nmodes
                )
            else:
                if name == "Rgate":
                    S = expand(rotation(params[0]), dict_indices[modes[0]], nmodes)
                elif name == "Sgate":
                    S = expand(squeezing(params[0], params[1]), dict_indices[modes[0]], nmodes)
                elif name == "S2gate":
                    S = expand(
                        two_mode_squeezing(params[0], params[1]),
                        [dict_indices[modes[0]], dict_indices[modes[1]]],
                        nmodes,
                    )
                elif name == "Interferometer":
                    S = expand(
                        interferometer(params[0]), [dict_indices[mode] for mode in modes], nmodes
                    )
                elif name == "GaussianTransform":
                    S = expand(params[0], [dict_indices[mode] for mode in modes], nmodes)
                elif name == "BSgate":
                    S = expand(
                        beam_splitter(params[0], params[1]),
                        [dict_indices[modes[0]], dict_indices[modes[1]]],
                        nmodes,
                    )
                elif name == "MZgate":
                    v = np.exp(1j * params[0])
                    u = np.exp(1j * params[1])
                    U = 0.5 * np.array([[u * (v - 1), 1j * (1 + v)], [1j * u * (1 + v), 1 - v]])
                    S = expand(
                        interferometer(U),
                        [dict_indices[modes[0]], dict_indices[modes[1]]],
                        nmodes,
                    )
                Snet = S @ Snet
                rnet = S @ rnet

        # Having obtained the net displacement we simply convert it into complex notation
        alphas = 0.5 * (rnet[0:nmodes] + 1j * rnet[nmodes : 2 * nmodes])
        # And now we just pass the net transformation as a big Symplectic operation plus displacements
        ord_reg = [r for r in list(registers) if r.ind in used_modes]
        ord_reg = sorted(list(ord_reg), key=lambda x: x.ind)
        if np.allclose(Snet, np.identity(2 * nmodes)):
            A = []
        else:
            A = [Command(ops.GaussianTransform(Snet), ord_reg)]
        B = [
            Command(ops.Dgate(np.abs(alphas[i]), np.angle(alphas[i])), ord_reg[i])
            for i in range(len(ord_reg))
            if not np.allclose(alphas[i], 0.0)
        ]
        return A + B

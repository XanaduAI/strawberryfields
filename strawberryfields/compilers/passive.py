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
"""This module contains a compiler to reduce a sequence of passive gates into a single multimode linear passive operation"""

import numpy as np
from numba import jit
from strawberryfields.program_utils import Command
from strawberryfields import ops
from strawberryfields.parameters import par_evaluate
from thewalrus.symplectic import (
    expand_passive,
    interferometer,
    beam_splitter,
)

from .compiler import Compiler


@jit(nopython=True)
def _apply_one_mode_gate(G, T, i):
    """In-place applies a one mode gate G into the process matrix T in mode i

    Args:
        G (complex/float): one mode gate
        T (array): passive transformation
        i (int): index of one mode gate
    """

    T[i] *= G


@jit(nopython=True)
def _apply_two_mode_gate(G, T, i, j):
    """In-place applies a two mode gate G into the process matrix T in modes i and j

    Args:
        G (array): 2x2 matrix for two mode gate
        T (array): passive transformation
        i (int): index of first mode of gate
        j (int): index of second mode of gate
    """
    (T[i], T[j]) = (G[0, 0] * T[i] + G[0, 1] * T[j], G[1, 0] * T[i] + G[1, 1] * T[j])


@jit(nopython=True)
def _beam_splitter_passive(theta, phi):
    """Beam-splitter.

    Args:
        theta (float): transmissivity parameter
        phi (float): phase parameter
        dtype (numpy.dtype): datatype to represent the Symplectic matrix

    Returns:
        array: symplectic-orthogonal transformation matrix of an interferometer with angles theta and phi
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    eip = np.cos(phi) + 1j * np.sin(phi)
    U = np.array(
        [
            [ct, -np.conj(eip) * st],
            [eip * st, ct],
        ]
    )
    return U


class Passive(Compiler):
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

    short_name = "passive"
    interactive = True

    primitives = {
        # meta operations
        "All",
        "_New_modes",
        "_Delete",
        # single mode gates
        "Rgate",
        "LossChannel",
        # multi mode gates
        "MZgate",
        "sMZgate",
        "BSgate",
        "Interferometer",  # Note that interferometer is accepted as a primitive
        "PassiveChannel",
    }

    decompositions = {}
    # pylint: disable=too-many-branches
    def compile(self, seq, registers):
        """Try to arrange a passive circuit into a single multimode passive operation

        This method checks whether the circuit can be implemented as a sequence of passive gates.
        If the answer is yes it arranges them into a single operation.

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
        T = np.identity(nmodes, dtype=np.complex128)

        # Now we will go through each operation in the sequence `seq` and apply it in quadrature space
        # We will keep track of the net transforation in the Symplectic matrix `Snet` and the quadrature
        # vector `rnet`.
        for operations in seq:
            name = operations.op.__class__.__name__
            params = par_evaluate(operations.op.p)
            modes = [modes_label.ind for modes_label in operations.reg]
            if name == "Rgate":
                G = np.exp(1j * params[0])
                _apply_one_mode_gate(G, T, dict_indices[modes[0]])
            elif name == "LossChannel":
                G = np.sqrt(params[0])
                _apply_one_mode_gate(G, T, dict_indices[modes[0]])
            elif name == "Interferometer":
                U = params[0]
                if U.shape == (1, 1):
                    _apply_one_mode_gate(U, T, dict_indices[modes[0]])
                elif U.shape == (2, 2):
                    _apply_two_mode_gate(U, T, dict_indices[modes[0]], dict_indices[modes[1]])
                else:
                    T = expand_passive(U, [dict_indices[mode] for mode in modes], nmodes) @ T
            elif name == "PassiveChannel":
                T0 = params[0]
                if T0.shape == (1, 1):
                    _apply_one_mode_gate(T0, T, dict_indices[modes[0]])
                elif T0.shape == (2, 2):
                    _apply_two_mode_gate(T0, T, dict_indices[modes[0]], dict_indices[modes[1]])
                else:
                    T = expand_passive(T0, [dict_indices[mode] for mode in modes], nmodes) @ T
            elif name == "BSgate":
                G = _beam_splitter_passive(params[0], params[1])
                _apply_two_mode_gate(G, T, dict_indices[modes[0]], dict_indices[modes[1]])
            elif name == "MZgate":
                v = np.exp(1j * params[0])
                u = np.exp(1j * params[1])
                U = 0.5 * np.array([[u * (v - 1), 1j * (1 + v)], [1j * u * (1 + v), 1 - v]])
                _apply_two_mode_gate(U, T, dict_indices[modes[0]], dict_indices[modes[1]])
            elif name == "sMZgate":
                exp_sigma = np.exp(1j * (params[0] + params[1]) / 2)
                delta = (params[0] - params[1]) / 2
                U = exp_sigma * np.array(
                    [[np.sin(delta), np.cos(delta)], [np.cos(delta), -np.sin(delta)]]
                )
                _apply_two_mode_gate(U, T, dict_indices[modes[0]], dict_indices[modes[1]])

        ord_reg = [r for r in list(registers) if r.ind in used_modes]
        ord_reg = sorted(list(ord_reg), key=lambda x: x.ind)

        return [Command(ops.PassiveChannel(T), ord_reg)]

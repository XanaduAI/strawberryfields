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
from strawberryfields.program_utils import Command
from strawberryfields import ops
from strawberryfields.parameters import par_evaluate

from .compiler import Compiler


def _apply_one_mode_gate(G, T, i):
    """In-place applies a one mode gate G into the process matrix T in mode i

    Args:
        G (complex or float): one mode gate
        T (array): passive transformation
        i (int): index of one mode gate

    Returns:
        T (array): updated passive transformation
    """

    T[i] *= G
    return T


def _apply_two_mode_gate(G, T, i, j):
    """In-place applies a two mode gate G into the process matrix T in modes i and j

    Args:
        G (array): 2x2 matrix for two mode gate
        T (array): passive transformation
        i (int): index of first mode of gate
        j (int): index of second mode of gate

    Returns:
        T (array): updated passive transformation
    """
    (T[i], T[j]) = (G[0, 0] * T[i] + G[0, 1] * T[j], G[1, 0] * T[i] + G[1, 1] * T[j])
    return T


def _beam_splitter_passive(theta, phi):
    """Beam-splitter.

    Args:
        theta (float): transmissivity parameter
        phi (float): phase parameter

    Returns:
        array: unitary 2x2 transformation matrix of an interferometer with angles theta and phi
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
    """Compiler to write a sequence of passive operations as a single passive operation

    This compiler checks whether the circuit can be implemented as a sequence of
    passive operations. If so, it arranges them in a single matrix, T. It then returns an PassiveChannel
    operation which can act this transformation.

    This compiler can be accessed by calling :meth:`.Program.compile` with `'passive'` specified.

    **Example:**

    Consider the following Strawberry Fields program, compiled using the `'passive'` compiler:

    .. code-block:: python3

        from strawberryfields.ops import BSgate, LossChannel, Rgate
        import strawberryfields as sf

        circuit = sf.Program(2)
        with circuit.context as q:
            Rgate(np.pi) | q[0]
            BSgate(0.25 * np.pi, 0) | (q[0], q[1])
            LossChannel(0.9) | q[1]

        compiled_circuit = circuit.compile(compiler="passive")

    We can now print the compiled circuit, consisting a single
    :class:`~.PassiveChannel`:

    >>> compiled_circuit.print()
    PassiveChannel([[-0.7071+8.6596e-17j -0.7071+0.0000e+00j]
     [-0.6708+8.2152e-17j  0.6708+0.0000e+00j]]) | (q[0], q[1])
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
        "PassiveChannel",  # and PassiveChannels!
    }

    decompositions = {}
    # pylint: disable=too-many-branches, too-many-statements
    def compile(self, seq, registers):
        """Try to arrange a passive circuit into a single multimode passive operation

        This method checks whether the circuit can be implemented as a sequence of passive gates.
        If the answer is yes it arranges them into a single operation.

        Args:
            seq (Sequence[Command]): passive quantum circuit to modify
            registers (Sequence[RegRefs]): quantum registers
        Returns:
            List[Command]: compiled circuit
        Raises:
            CircuitError: the circuit does not correspond to a passive unitary
        """

        # Check which modes are actually being used
        used_modes = []
        for operations in seq:
            modes = [modes_label.ind for modes_label in operations.reg]
            used_modes.append(modes)

        used_modes = list(set(item for sublist in used_modes for item in sublist))

        # dictionary mapping the used modes to consecutive non-negative integers
        dict_indices = {used_modes[i]: i for i in range(len(used_modes))}
        nmodes = len(used_modes)

        # We start with an identity then sequentially update with the gate transformations
        T = np.identity(nmodes, dtype=np.complex128)

        # Now we will go through each operation in the sequence `seq` and apply it to T
        for operations in seq:
            name = operations.op.__class__.__name__
            params = par_evaluate(operations.op.p)
            modes = [modes_label.ind for modes_label in operations.reg]
            if name == "Rgate":
                G = np.exp(1j * params[0])
                T = _apply_one_mode_gate(G, T, dict_indices[modes[0]])
            elif name == "LossChannel":
                G = np.sqrt(params[0])
                T = _apply_one_mode_gate(G, T, dict_indices[modes[0]])
            elif name == "Interferometer":
                U = params[0]
                if U.shape == (1, 1):
                    T = _apply_one_mode_gate(U[0, 0], T, dict_indices[modes[0]])
                elif U.shape == (2, 2):
                    T = _apply_two_mode_gate(U, T, dict_indices[modes[0]], dict_indices[modes[1]])
                else:
                    modes = [dict_indices[mode] for mode in modes]
                    U_expand = np.eye(nmodes, dtype=np.complex128)
                    U_expand[np.ix_(modes, modes)] = U
                    T = U_expand @ T
            elif name == "PassiveChannel":
                T0 = params[0]
                if T0.shape == (1, 1):
                    T = _apply_one_mode_gate(T0[0, 0], T, dict_indices[modes[0]])
                elif T0.shape == (2, 2):
                    T = _apply_two_mode_gate(T0, T, dict_indices[modes[0]], dict_indices[modes[1]])
                else:
                    modes = [dict_indices[mode] for mode in modes]
                    T0_expand = np.eye(nmodes, dtype=np.complex128)
                    T0_expand[np.ix_(modes, modes)] = T0
                    T = T0_expand @ T
            elif name == "BSgate":
                G = _beam_splitter_passive(params[0], params[1])
                T = _apply_two_mode_gate(G, T, dict_indices[modes[0]], dict_indices[modes[1]])
            elif name == "MZgate":
                v = np.exp(1j * params[0])
                u = np.exp(1j * params[1])
                U = 0.5 * np.array([[u * (v - 1), 1j * (1 + v)], [1j * u * (1 + v), 1 - v]])
                T = _apply_two_mode_gate(U, T, dict_indices[modes[0]], dict_indices[modes[1]])
            elif name == "sMZgate":
                exp_sigma = np.exp(1j * (params[0] + params[1]) / 2)
                delta = (params[0] - params[1]) / 2
                U = exp_sigma * np.array(
                    [[np.sin(delta), np.cos(delta)], [np.cos(delta), -np.sin(delta)]]
                )
                T = _apply_two_mode_gate(U, T, dict_indices[modes[0]], dict_indices[modes[1]])

        ord_reg = [r for r in list(registers) if r.ind in used_modes]
        ord_reg = sorted(list(ord_reg), key=lambda x: x.ind)

        return [Command(ops.PassiveChannel(T), ord_reg)]

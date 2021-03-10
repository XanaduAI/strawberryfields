# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import textwrap

import pytest
import numpy as np
import networkx as nx

from strawberryfields.api import DeviceSpec
from strawberryfields.parameters import par_evaluate
from strawberryfields.program_utils import list_to_DAG


test_spec = {
    "layout": textwrap.dedent(
        """\
        name template_4x2_X8
        version 1.0
        target X8_01 (shots=1)

        # for n spatial degrees, first n signal modes, then n idler modes, all phases zero
        S2gate({squeezing_amplitude_0}, 0.0) | [0, 4]
        S2gate({squeezing_amplitude_1}, 0.0) | [1, 5]
        S2gate({squeezing_amplitude_2}, 0.0) | [2, 6]
        S2gate({squeezing_amplitude_3}, 0.0) | [3, 7]

        # standard 4x4 interferometer for the signal modes (the lower ones in frequency)
        # even phase indices correspond to internal Mach-Zehnder interferometer phases
        # odd phase indices correspond to external Mach-Zehnder interferometer phases
        MZgate({phase_0}, {phase_1}) | [0, 1]
        MZgate({phase_2}, {phase_3}) | [2, 3]
        MZgate({phase_4}, {phase_5}) | [1, 2]
        MZgate({phase_6}, {phase_7}) | [0, 1]
        MZgate({phase_8}, {phase_9}) | [2, 3]
        MZgate({phase_10}, {phase_11}) | [1, 2]

        # duplicate the interferometer for the idler modes (the higher ones in frequency)
        MZgate({phase_0}, {phase_1}) | [4, 5]
        MZgate({phase_2}, {phase_3}) | [6, 7]
        MZgate({phase_4}, {phase_5}) | [5, 6]
        MZgate({phase_6}, {phase_7}) | [4, 5]
        MZgate({phase_8}, {phase_9}) | [6, 7]
        MZgate({phase_10}, {phase_11}) | [5, 6]

        # add final dummy phases to allow mapping any unitary to this template (these do not
        # affect the photon number measurement)
        Rgate({final_phase_0}) | [0]
        Rgate({final_phase_1}) | [1]
        Rgate({final_phase_2}) | [2]
        Rgate({final_phase_3}) | [3]
        Rgate({final_phase_4}) | [4]
        Rgate({final_phase_5}) | [5]
        Rgate({final_phase_6}) | [6]
        Rgate({final_phase_7}) | [7]

        # measurement in Fock basis
        MeasureFock() | [0, 1, 2, 3, 4, 5, 6, 7]
    """
    ),
    "modes": 8,
    "compiler": [],
    "gate_parameters": {
        "squeezing_amplitude_0": [0, 1],
        "squeezing_amplitude_1": [0, 1],
        "squeezing_amplitude_2": [0, 1],
        "squeezing_amplitude_3": [0, 1],
        "phase_0": [0, [0, 6.283185307179586]],
        "phase_1": [0, [0, 6.283185307179586]],
        "phase_2": [0, [0, 6.283185307179586]],
        "phase_3": [0, [0, 6.283185307179586]],
        "phase_4": [0, [0, 6.283185307179586]],
        "phase_5": [0, [0, 6.283185307179586]],
        "phase_6": [0, [0, 6.283185307179586]],
        "phase_7": [0, [0, 6.283185307179586]],
        "phase_8": [0, [0, 6.283185307179586]],
        "phase_9": [0, [0, 6.283185307179586]],
        "phase_10": [0, [0, 6.283185307179586]],
        "phase_11": [0, [0, 6.283185307179586]],
        "final_phase_0": [0, [0, 6.283185307179586]],
        "final_phase_1": [0, [0, 6.283185307179586]],
        "final_phase_2": [0, [0, 6.283185307179586]],
        "final_phase_3": [0, [0, 6.283185307179586]],
        "final_phase_4": [0, [0, 6.283185307179586]],
        "final_phase_5": [0, [0, 6.283185307179586]],
        "final_phase_6": [0, [0, 6.283185307179586]],
        "final_phase_7": [0, [0, 6.283185307179586]],
    },
}


X8_spec = DeviceSpec(target="X8_01", connection=None, spec=test_spec)


def generate_X8_params(r, p):
    return {
        "squeezing_amplitude_0": r,
        "squeezing_amplitude_1": r,
        "squeezing_amplitude_2": r,
        "squeezing_amplitude_3": r,
        "phase_0": p,
        "phase_1": p,
        "phase_2": p,
        "phase_3": p,
        "phase_4": p,
        "phase_5": p,
        "phase_6": p,
        "phase_7": p,
        "phase_8": p,
        "phase_9": p,
        "phase_10": p,
        "phase_11": p,
        "final_phase_0": 1.24,
        "final_phase_1": 0.54,
        "final_phase_2": 4.12,
        "final_phase_3": 0,
        "final_phase_4": 1.24,
        "final_phase_5": 0.54,
        "final_phase_6": 4.12,
        "final_phase_7": 0,
    }


def program_equivalence(prog1, prog2, compare_params=True, atol=1e-6, rtol=0):
    r"""Checks if two programs are equivalent.

    This function converts the program lists into directed acyclic graphs,
    and runs the NetworkX `is_isomorphic` graph function in order
    to determine if the two programs are equivalent.

    Note: when checking for parameter equality between two parameters
    :math:`a` and :math:`b`, we use the following formula:

    .. math:: |a - b| \leq (\texttt{atol} + \texttt{rtol}\times|b|)

    Args:
        prog1 (strawberryfields.program.Program): quantum program
        prog2 (strawberryfields.program.Program): quantum program
        compare_params (bool): Set to ``False`` to turn of comparing
            program parameters; equivalency will only take into
            account the operation order.
        atol (float): the absolute tolerance parameter for checking
            quantum operation parameter equality
        rtol (float): the relative tolerance parameter for checking
            quantum operation parameter equality

    Returns:
        bool: returns ``True`` if two quantum programs are equivalent
    """
    DAG1 = list_to_DAG(prog1.circuit)
    DAG2 = list_to_DAG(prog2.circuit)

    circuit = []
    for G in [DAG1, DAG2]:
        # relabel the DAG nodes to integers
        circuit.append(nx.convert_node_labels_to_integers(G))

        # add node attributes to store the operation name and parameters
        name_mapping = {i: n.op.__class__.__name__ for i, n in enumerate(G.nodes())}
        parameter_mapping = {i: par_evaluate(n.op.p) for i, n in enumerate(G.nodes())}

        # CXgate and BSgate are not symmetric wrt permuting the order of the two
        # modes it acts on; i.e., the order of the wires matter
        wire_mapping = {}
        for i, n in enumerate(G.nodes()):
            if n.op.__class__.__name__ == "CXgate":
                if np.allclose(n.op.p[0], 0):
                    # if the CXgate parameter is 0, wire order doesn't matter
                    wire_mapping[i] = 0
                else:
                    # if the CXgate parameter is not 0, order matters
                    wire_mapping[i] = [j.ind for j in n.reg]

            elif n.op.__class__.__name__ == "BSgate":
                if np.allclose([j % np.pi for j in par_evaluate(n.op.p)], [np.pi / 4, np.pi / 2]):
                    # if the beamsplitter is *symmetric*, then the order of the
                    # wires does not matter.
                    wire_mapping[i] = 0
                else:
                    # beamsplitter is not symmetric, order matters
                    wire_mapping[i] = [j.ind for j in n.reg]

            else:
                # not a CXgate or a BSgate, order of wires doesn't matter
                wire_mapping[i] = 0

        # TODO: at the moment, we do not check for whether an empty
        # wire will match an operation with trivial parameters.
        # Maybe we can do this in future, but this is a subgraph
        # isomorphism problem and much harder.

        nx.set_node_attributes(circuit[-1], name_mapping, name="name")
        nx.set_node_attributes(circuit[-1], parameter_mapping, name="p")
        nx.set_node_attributes(circuit[-1], wire_mapping, name="w")

    def node_match(n1, n2):
        """Returns True if both nodes have the same name and
        same parameters, within a certain tolerance"""
        name_match = n1["name"] == n2["name"]
        p_match = np.allclose(n1["p"], n2["p"], atol=atol, rtol=rtol)
        wire_match = n1["w"] == n2["w"]

        if compare_params:
            return name_match and p_match and wire_match

        return name_match and wire_match

    # check if circuits are equivalent
    return nx.is_isomorphic(circuit[0], circuit[1], node_match)

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

# The module docstring is in strawberryfields/circuitspecs/__init__.py
"""
**Module name:** :mod:`strawberryfields.circuitspecs.circuit_specs`
"""

from typing import List, Set, Dict, Union
import abc

import networkx as nx
import blackbird
from blackbird.utils import to_DiGraph

import strawberryfields.program_utils as pu


class CircuitSpecs(abc.ABC):
    """Abstract base class for describing circuit classes.

    This class stores information about :term:`classes of quantum circuits <circuit class>`.
    For some circuit classes (e.g, ones corresponding to physical hardware chips), the
    specifications can be quite rigid. For other classes, e.g., circuits supported by a particular
    simulator backend, the specifications can be more flexible and general.

    Key ingredients in a specification include: the primitive gates supported by the circuit class,
    the gates that can be decomposed to sequences of primitive gates, and the possible
    topology/connectivity restrictions.

    This information is used e.g., in :meth:`.Program.compile` for validation and compilation.
    """

    short_name = ''
    """str: short name of the circuit class"""

    @property
    @abc.abstractmethod
    def modes(self) -> Union[int, None]:
        """The number of modes supported by the circuit class.

        If the circuit class supports arbitrary number of modes, set this to 0.

        Returns:
            int: number of supported modes
        """

    @property
    @abc.abstractmethod
    def local(self) -> bool:
        """Whether the circuit class can be executed locally (i.e., within a simulator).

        Returns:
            bool: ``True`` if the circuit class supports local execution
        """

    @property
    @abc.abstractmethod
    def remote(self) -> bool:
        """Whether the circuit class supports remote execution.

        Returns:
            bool: ``True`` if the circuit class supports remote execution
        """

    @property
    @abc.abstractmethod
    def interactive(self) -> bool:
        """Whether the circuits in the class can be executed interactively, that is,
        the registers in the circuit are not reset between engine executions.

        Returns:
            bool: ``True`` if the circuit supports interactive use
        """

    @property
    @abc.abstractmethod
    def primitives(self) -> Set[str]:
        """The primitive set of quantum operations directly supported
        by the circuit class.

        Returns:
            set[str]: the names of the quantum primitives the circuit class supports
        """

    @property
    @abc.abstractmethod
    def decompositions(self) -> Dict[str, Dict]:
        """Quantum operations that are not quantum primitives for the
        circuit class, but are supported via specified decompositions.

        This should be of the form

        .. code-block:: python

            {'operation_name': {'option1': val, 'option2': val,...}}

        For each operation specified in the dictionary, the
        :meth:`.Operation.decompose` method will be called during
        :class:`.Program` compilation, with keyword arguments
        given by the dictionary value.

        Returns:
            dict[str, dict]: the quantum operations that are supported
            by the circuit class via decomposition
        """

    @property
    def parameter_ranges(self) -> Dict[str, List[List[float]]]:
        """Allowed parameter ranges for supported quantum operations.

        This property is optional.

        Returns:
            dict[str, list]: a dictionary mapping an allowed quantum operation
            to a nested list of the form ``[[p0_min, p0_max], [p1_min, p0_max], ...]``.
            where ``pi`` corresponds to the ``i`` th gate parameter
        """
        return dict()

    @property
    def graph(self):
        """The allowed circuit topologies or connectivity of the class, modelled as a directed
        acyclic graph.

        This property is optional; if arbitrary topologies are allowed in the circuit class,
        this will simply return ``None``.

        Returns:
            networkx.DiGraph: a directed acyclic graph
        """
        if self.circuit is None:
            return None

        # returned DAG has all parameters set to 0
        bb = blackbird.loads(self.circuit)

        if bb.is_template():
            params = bb.parameters
            kwargs = {p: 0 for p in params}

            # initialize the topology with all template
            # parameters set to zero
            topology = to_DiGraph(bb(**kwargs))

        else:
            topology = to_DiGraph(bb)

        return topology

    @property
    def circuit(self):
        """A rigid circuit template that defines this circuit specification.

        This property is optional. If arbitrary topologies are allowed in the circuit class,
        **do not define this property**. In such a case, it will simply return ``None``.

        If a backend device expects a specific template for the recieved Blackbird
        script, this method will return the serialized Blackbird circuit in string
        form.

        Returns:
            Union[str, None]: Blackbird program or template representing the circuit
        """
        return None

    def compile(self, seq):
        """Class-specific circuit compilation method.

        If additional compilation logic is required, child classes can redefine this method.

        Args:
            seq (Sequence[Command]): quantum circuit to modify
        Returns:
            List[Command]: modified circuit
        Raises:
            CircuitError: the given circuit cannot be validated to belong to this circuit class
        """
        if self.graph is not None:
            # check topology
            DAG = pu.list_to_DAG(seq)

            # relabel the DAG nodes to integers, with attributes
            # specifying the operation name. This allows them to be
            # compared, rather than using Command objects.
            mapping = {i: n.op.__class__.__name__ for i, n in enumerate(DAG.nodes())}
            circuit = nx.convert_node_labels_to_integers(DAG)
            nx.set_node_attributes(circuit, mapping, name='name')

            def node_match(n1, n2):
                """Returns True if both nodes have the same name"""
                return n1['name'] == n2['name']

            # check if topology matches
            if not nx.is_isomorphic(circuit, self.graph, node_match):
                # TODO: try and compile the program to match the topology
                # TODO: add support for parameter range matching/compilation
                raise pu.CircuitError(
                    "Program cannot be used with the CircuitSpec '{}' "
                    "due to incompatible topology.".format(self.short_name)
                )

        return seq

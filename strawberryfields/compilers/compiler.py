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

# The module docstring is in strawberryfields/compiler/__init__.py
"""
**Module name:** :mod:`strawberryfields.compilers.compiler`
"""

import abc
import sympy as sym
from typing import Sequence, Set, Dict, Optional

import networkx as nx
import blackbird
from blackbird.utils import to_DiGraph

import strawberryfields.program_utils as pu
from strawberryfields.program_utils import CircuitError, Command, RegRef


class Compiler(abc.ABC):
    """Abstract base class for describing circuit compilation.

    This class stores information about :term:`compilation of photonic quantum circuits <circuit
    class>`.

    Key ingredients in a specification include: the primitive gates supported by the circuit class,
    the gates that can be decomposed to sequences of primitive gates, and the possible
    topology/connectivity restrictions.

    This information is used e.g., in :meth:`.Program.compile` for validation and compilation.
    """

    short_name = ""
    """str: short name of the circuit class"""

    _layout = None
    """str: layout of the circuit"""

    _graph = None
    """DiGraph: circuit graph"""

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
    def graph(self) -> Optional[nx.DiGraph]:
        """The allowed circuit topologies or connectivity of the class, modelled as a directed
        acyclic graph.

        This property is optional; if arbitrary topologies are allowed in the circuit class,
        this will simply return ``None``.

        Returns:
            networkx.DiGraph: a directed acyclic graph
        """
        if self.circuit is None:
            return None
        if self._graph is not None:
            return self._graph

        # returned DAG has all parameters set to 0
        bb = blackbird.loads(self.circuit)
        topology = to_DiGraph(bb)

        self._set_graph(topology)
        return topology

    @classmethod
    def _set_graph(cls, topology: nx.DiGraph) -> None:
        """Sets the graph attribute in the class."""
        cls._graph = topology

    @property
    def circuit(self) -> Optional[str]:
        """A rigid circuit template that defines this circuit specification.

        If arbitrary topologies are allowed in the circuit class, this function
        will simply return ``None``.

        If a backend device expects a specific template for the received Blackbird
        script, this method will return the serialized Blackbird circuit in string
        form.

        Returns:
            Union[str, None]: Blackbird circuit or template representing the circuit
        """
        return self._layout

    @classmethod
    def init_circuit(cls, layout: str) -> None:
        """Sets the circuit in the compiler class.

        Args:
            layout (str): the circuit layout for the target device
        """
        if cls._layout:
            # if the exact same circuit is set (apart from newlines) then return
            if cls._layout.replace("\n", "") != layout.replace("\n", ""):
                raise CircuitError(
                    f"Circuit already set in compiler {cls.short_name}. Device layout incompatible "
                    "with compiler layout. Call the compiler's 'reset_circuit' method, or use a "
                    "different device layout."
                )
            return

        if not isinstance(layout, str):
            raise TypeError("Layout must be a string representing the Blackbird circuit.")

        cls._layout = layout

    @classmethod
    def reset_circuit(cls) -> None:
        """Resets the ``circuit`` and ``graph`` class attributes."""
        cls._layout = None
        cls._graph = None

    def compile(self, seq: Sequence[Command], registers: Sequence[RegRef]) -> Sequence[Command]:
        """Class-specific circuit compilation method.

        If additional compilation logic is required, child classes can redefine this method.

        Args:
            seq (Sequence[Command]): quantum circuit to modify
            registers (Sequence[RegRef]): quantum registers

        Returns:
            Sequence[Command]: modified circuit

        Raises:
            CircuitError: the given circuit cannot be validated to belong to this circuit class
        """
        # registers is not used here, but may be used if the method is overwritten pylint: disable=unused-argument
        if self.graph is not None:
            # check topology
            DAG = pu.list_to_DAG(seq)

            # relabel the DAG nodes to integers, with attributes
            # specifying the operation name. This allows them to be
            # compared, rather than using Command objects.
            mapping_name, mapping_args, mapping_modes = {}, {}, {}
            for i, n in enumerate(DAG.nodes()):
                mapping_name[i] = n.op.__class__.__name__
                mapping_args[i] = n.op.p
                mapping_modes[i] = tuple(m.ind for m in n.reg)

            circuit = nx.convert_node_labels_to_integers(DAG)
            nx.set_node_attributes(circuit, mapping_name, name="name")
            nx.set_node_attributes(circuit, mapping_args, name="args")
            nx.set_node_attributes(circuit, mapping_modes, name="modes")

            def node_match(n1, n2):
                """Returns True if both nodes have the same name and modes"""
                return n1["name"] == n2["name"] and n1["modes"] == n2["modes"]

            GM = nx.algorithms.isomorphism.DiGraphMatcher(self.graph, circuit, node_match)

            # check if topology matches
            if not GM.is_isomorphic():
                raise pu.CircuitError(
                    "Program cannot be used with the compiler '{}' "
                    "due to incompatible topology.".format(self.short_name)
                )

            # check if hard-coded parameters match
            G1nodes = self.graph.nodes().data()
            G2nodes = circuit.nodes().data()

            for n1, n2 in GM.mapping.items():
                for x, y in zip(G1nodes[n1]["args"], G2nodes[n2]["args"]):
                    if x != y and not (isinstance(x, sym.Symbol) or isinstance(y, sym.Expr)):
                        raise CircuitError(
                            "Program cannot be used with the compiler '{}' "
                            "due to incompatible parameter values.".format(self.short_name)
                        )

        return seq

    def decompose(self, seq: Sequence[Command]) -> Sequence[Command]:
        """Recursively decompose all gates in a given sequence, as allowed
        by the circuit specification.

        This method follows the directives defined in the
        :attr:`~.Compiler.primitives` and :attr:`~.Compiler.decompositions`
        class attributes to determine whether a command should be decomposed.

        The order of precedence to determine whether decomposition
        should be applied is as follows.

        1. First, we check if the operation is in :attr:`~.Compiler.decompositions`.
           If not, decomposition is skipped, and the operation is applied
           as a primitive (if supported by the ``Compiler``).

        2. Next, we check if (a) the operation supports decomposition, and (b) if the user
           has explicitly requested no decomposition.

           - If both (a) and (b) are true, the operation is applied
             as a primitive (if supported by the ``Compiler``).

           - Otherwise, we attempt to decompose the operation by calling
             :meth:`~.Operation.decompose` recursively.

        Args:
            list[strawberryfields.program_utils.Command]: list of commands to
                be decomposed

        Returns:
            list[strawberryfields.program_utils.Command]: list of compiled commands
            for the circuit specification
        """
        compiled = []
        for cmd in seq:
            op_name = cmd.op.__class__.__name__
            if op_name in self.decompositions:
                # target can implement this op decomposed
                if hasattr(cmd.op, "decomp") and not cmd.op.decomp:
                    # user has requested application of the op as a primitive
                    if op_name in self.primitives:
                        compiled.append(cmd)
                        continue

                    raise pu.CircuitError(
                        "The operation {} is not a primitive for the compiler '{}'".format(
                            cmd.op.__class__.__name__, self.short_name
                        )
                    )
                try:
                    kwargs = self.decompositions[op_name]
                    temp = cmd.op.decompose(cmd.reg, **kwargs)
                    # now compile the decomposition
                    temp = self.decompose(temp)
                    compiled.extend(temp)
                except NotImplementedError as err:
                    # Operation does not have _decompose() method defined!
                    # simplify the error message by suppressing the previous exception
                    raise err from None

            elif op_name in self.primitives:
                # target can handle the op natively
                compiled.append(cmd)

            else:
                raise pu.CircuitError(
                    "The operation {} cannot be used with the compiler '{}'.".format(
                        cmd.op.__class__.__name__, self.short_name
                    )
                )

        return compiled

    def update_params(self, program, device) -> None:
        """Updates and checks parameters in the program circuit.

        Child classes can override this method with compiler specific logic. If no parameters need
        to be updated, and are separately checked, this method should not be overridden and be left
        empty.

        Args:
            program (.Program): Program containing the circuit and gate parameters
            device (.DeviceSpec): device specification containing the valid parameter values
        """
        # unless overridden by subclass, no parameters need to be updated
        return None

    def add_loss(self, program, device):
        """Adds realistic loss to circuit.

        Child classes which are hardware compilers should override this method with device specific
        loss added to the circuit."""
        raise NotImplementedError


class Range:
    """Lightweight class for representing a range of floats.

    **Example**

    >>> x = Range(0.2, 0.5)
    >>> print(x)
    0.2≤x≤0.5
    >>> 0.34 in x
    True
    >>> -0.1 in x
    False

    Note that the upper bound is inclusive:
    >>> 0.5 in x
    True

    Leaving off the lower bound corresponds to a range of a single value:
    >>> x = Range(0.3)
    >>> 0.3 in x
    True
    >>> 0.30001 in x
    False


    Args:
        x (float): lower bound of the range

    Keyword Args:
        y (float): Upper bound of the range (inclusive). If not provided,
            the range will represent the single value ``x``.
        variable_name (str): the variable name to use when printing
            the range
        atol (float): positive float representing the absolute tolerance
            used when checking if items are within the range
    """

    def __init__(self, x, y=None, variable_name="x", atol=1e-5):
        self.x = x
        self.y = y if y is not None else x
        self.name = variable_name
        self.atol = atol

        if self.y < self.x:
            raise ValueError(
                "Upper bound of the range must be strictly larger than the lower bound."
            )

    def __contains__(self, item):
        return self.x - self.atol <= item <= self.y + self.atol

    def __repr__(self):
        if self.x == self.y:
            return "{}={}".format(self.name, self.x)

        return "{}≤{}≤{}".format(self.x, self.name, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Ranges:
    """Lightweight class for representing a set of ranges of floats.

    **Example**

    >>> x = Ranges([0], [0.2, 0.55], [1.0])
    >>> print(x)
    x=0, 0.2≤x≤0.55, x=1.0
    >>> test_data = [0, 0.34, 0.1, 1.0]
    >>> [i in x for i in test_data]
    [True, True, False, True]

    Args:
        r1, r2, r3,... (list[float]): Allowed ranges. Lists of size ``(1,)``
            correspond to a single allowed value, whereas lists of size ``(2,)``
            correspond to a lower and upper bound (inclusive).

    Keyword Args
        variable_name (str): the variable name to use when printing
            the range
    """

    def __init__(self, *args, variable_name="x"):
        self.ranges = [Range(*a, variable_name=variable_name) for a in args]

    def __contains__(self, item):
        for range_ in self.ranges:
            if item in range_:
                return True

        return False

    def __repr__(self):
        return ", ".join([str(i) for i in self.ranges])

    def __eq__(self, other):
        if len(self.ranges) != len(other.ranges):
            return False

        return all(i == j for i, j in zip(self.ranges, other.ranges))

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
"""
This module contains various utility classes and functions used
within the :class:`~.Program` class.
"""

from collections.abc import Sequence

import networkx as nx
import numpy as np

from .parameters import MeasuredParameter, par_evaluate


__all__ = [
    "Program_current_context",
    "RegRefError",
    "CircuitError",
    "MergeFailure",
    "Command",
    "RegRef",
    "list_to_grid",
    "grid_to_DAG",
    "DAG_to_list",
    "list_to_DAG",
    "group_operations",
    "optimize_circuit",
]


Program_current_context = None
"""Context for inputting a Program. Used to be a class attribute of :class:`.Program`, placed
here to avoid cyclic imports."""
# todo: Avoid issues with Program contexts and threading,
# cf. _pydecimal.py in the python standard distribution.


class RegRefError(IndexError):
    """Exception raised by :class:`.Program` when it encounters an invalid register reference.

    E.g., trying to apply a gate to a nonexistent or deleted subsystem.
    """


class CircuitError(RuntimeError):
    """Exception raised by :class:`.Program` when it encounters an illegal
    operation in the quantum circuit.

    E.g., trying to use an Operation type that is unsupported by the current compilation target.
    """


class MergeFailure(RuntimeError):
    """Exception raised by :meth:`strawberryfields.ops.Operation.merge` when an
    attempted merge fails.

    E.g., trying to merge two gates of different families.
    """


class Command:
    """Represents a quantum operation applied on specific subsystems of the register.

    A Command instance is immutable once created, and can be shared between
    several :class:`.Program` instances.

    Args:
        op (~strawberryfields.ops.Operation): quantum operation to apply
        reg (Sequence[RegRef]): Subsystems to which the operation is applied.
            Note that the order matters here.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, op, reg):
        # accept a single RegRef in addition to a Sequence
        if not isinstance(reg, Sequence):
            reg = [reg]

        #: Operation: quantum operation to apply
        self.op = op
        #: Sequence[RegRef]: subsystems to which the operation is applied
        self.reg = reg

    def __str__(self):
        """
        Return a string containing the command in Blackbird syntax.
        """

        operation = str(self.op)
        if self.op.ns == 0:
            # op takes no subsystems as parameters, do not print anything more
            code = operation
        else:
            subsystems = ", ".join([str(r) for r in self.reg])
            code = "{} | ({})".format(operation, subsystems)
        return code

    def __lt__(self, other):
        # Needed as a tiebreaker for NetworkX lexicographical_topological_sort()
        # due to a buggy implementation! Any order will do. Remove when NetworkX is fixed.
        return True

    def get_dependencies(self):
        """Subsystems the command depends on.

        Combination of ``self.reg`` and ``self.op.measurement_deps``.

        .. note:: ``measurement_deps`` are used to ensure that the measurement
            happens before the result is used, but this is a bit too strict:
            two gates depending on the same measurement result but otherwise
            acting on different subsystems should commute.

        Returns:
            set[RegRef]: set of subsystems the command depends on
        """
        deps = self.op.measurement_deps | set(self.reg)
        return deps


class RegRef:
    """Quantum register reference.

    The objects of this class refer to a specific subsystem (mode) of
    a quantum register.

    Within the scope of each :class:`.Program` instance, only one RegRef instance
    should exist per subsystem. Program keeps the authoritative mapping
    of subsystem indices to RegRef instances.
    Subsystem measurement results are stored in the "official" RegRef object.
    If other RegRef objects referring to the same subsystem exist, they will
    not be updated. Once a RegRef is assigned a subsystem index it will never
    change, not even if the subsystem is deleted.

    The RegRefs are constructed in :meth:`.Program._add_subsystems`.

    Args:
        ind (int): index of the register subsystem referred to
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, ind):
        self.ind = ind  #: int: subsystem index
        self.val = None  #: float, complex: Measurement result. None if the subsystem has not been measured yet.
        self.active = True  #: bool: True at construction, False after the subsystem is deleted

    def __str__(self):
        return "q[{}]".format(self.ind)

    def __hash__(self):
        """Hashing method.

        NOTE: Has to match :meth:`__eq__` such that if two RegRefs compare equal they must have equal hashes.
        """
        return hash((self.ind, self.active))

    def __eq__(self, other):
        """Equality comparison.

        Compares the index and the activity state of the two RegRefs, the val field does not matter.
        NOTE: Affects the hashability of RegRefs, see also :meth:`__hash__`.
        """
        if other.__class__ != self.__class__:
            print("---------------          regref.__eq__: compared reqref to ", other.__class__)
            return False
        return self.ind == other.ind and self.active == other.active

    @property
    def par(self):
        """Convert the RegRef into a measured parameter.

        Returns:
            MeasuredParameter: measured parameter linked to this RegRef
        """
        return MeasuredParameter(self)


# =================
# Utility functions
# =================


def list_to_grid(ls):
    """Transforms a list of Commands to a grid representation.

    The grid is a mapping from subsystem indices to lists of :class:`Command` instances touching
    that subsystem, in temporal order. The same Command instance will appear in each list that
    corresponds to one of its subsystems.

    Args:
        ls (Iterable[Command]): quantum circuit
    Returns:
        dict[int, list[Command]]: same circuit in grid form
    """
    grid = {}
    # enter every operation in the list to its proper position in the grid
    for cmd in ls:
        for r in cmd.get_dependencies():
            # Add cmd to the grid to the end of the line r.ind.
            grid.setdefault(r.ind, []).append(cmd)
    return grid


def grid_to_DAG(grid):
    """Transforms a grid of Commands to a DAG representation.

    In the DAG (directed acyclic graph) each node is a :class:`Command` instance,
    and edges point from Commands to their immediate dependents/followers.

    Args:
        grid (dict[int, list[Command]]): quantum circuit
    Returns:
        networkx.DiGraph[Command]: same circuit in DAG form
    """
    DAG = nx.DiGraph()
    for _, q in grid.items():
        if q:
            # add the first operation on the wire that does not depend on anything
            DAG.add_node(q[0])
        for i in range(1, len(q)):
            # add the edge between the operations, and the operation nodes themselves
            DAG.add_edge(q[i - 1], q[i])
    return DAG


def list_to_DAG(ls):
    """Transforms a list of Commands to a DAG representation.

    In the DAG (directed acyclic graph) each node is a :class:`Command` instance,
    and edges point from Commands to their immediate dependents/followers.

    Args:
        ls (Iterable[Command]): quantum circuit
    Returns:
        networkx.DiGraph[Command]: same circuit in DAG form
    """
    return grid_to_DAG(list_to_grid(ls))


def DAG_to_list(dag):
    """Transforms a Command DAG to a list representation.

    The list contains the :class:`Command` instances in (one possible) topological order,
    i.e., dependants following the operations they depend on.

    Args:
        dag (networkx.DiGraph[Command]): quantum circuit
    Returns:
        list[Command]: same circuit in list form
    """
    # sort the operation graph into topological order
    temp = nx.algorithms.dag.topological_sort(dag)
    return list(temp)


def group_operations(seq, predicate):
    """Group a set of Operations in a circuit together (if possible).

    For the purposes of this method, we call a :class:`Operation` instance *marked* iff
    ``predicate`` returns True on it.

    This method converts the quantum circuit in ``seq`` into an equivalent circuit ``A+B+C``,
    where the :class:`Command` instances in sequences ``A`` and ``C`` do not contain any
    marked Operations.
    The sequence ``B`` contains all marked Operations in the circuit, and possibly
    additional unmarked instances that could not be moved into ``A`` or ``C`` using the
    available commutation rules.
    Any of the three returned sequences can be empty (but if ``B`` is empty then so is ``C``).

    Args:
        seq (Sequence[Command]): quantum circuit
        predicate (Callable[[Operation], bool]): Grouping predicate. Returns True for the
            Operations to be grouped together, False for the others.
    Returns:
        Tuple[Sequence[Command]]: A, B, C such that A+B+C is equivalent to seq,
            and A and C do not contain any marked Operation instances.
    """

    def find_first_index(seq):
        """Index of the first element in the sequence for which the predicate function returns True.
        If no such element exists, returns the length of the sequence.
        """
        return next((i for i, e in enumerate(seq) if predicate(e.op)), len(seq))

    def marked_last(node):
        """Mapping from nodes to sorting keys to resolve ambiguities in the topological sort.
        Larger key values come later in the lexicographical-topological ordering.
        """
        if predicate(node.op):
            return 1
        return 0

    def lex_topo(seq, key):
        """Sorts a Command sequence lexicographical-topologically using the given lexicographic key function."""
        DAG = list_to_DAG(seq)
        return list(nx.algorithms.dag.lexicographical_topological_sort(DAG, key=key))

    C = lex_topo(seq, key=marked_last)
    ind = find_first_index(C)
    A = C[:ind]  # initial unmarked instances
    B = C[ind:]  # marked and possibly unmarked

    # re-sort B, marked instances first
    C = lex_topo(B, key=lambda x: -marked_last(x))
    # find last marked
    ind = len(C) - find_first_index(list(reversed(C)))
    B = C[:ind]  # marked and still possibly unmarked
    C = C[ind:]  # final unmarked instances
    return A, B, C


def optimize_circuit(seq):
    """Try to simplify and optimize a quantum circuit.

    The purpose of the optimizer is to simplify the circuit
    to make it cheaper and faster to execute. Different backends may require
    different types of optimization, but in general the fewer operations a circuit has,
    the faster it should run. The optimizer thus should convert the circuit into a
    simpler :term:`equivalent circuit`.

    The optimizations are based on the abstract algebraic properties of the Operations
    constituting the circuit, e.g., combining two consecutive gates of the same gate family,
    and at no point should require a matrix representation of any kind.
    The optimization must also not change the state of the RegRefs in any way.

    Currently the optimization is very simple. It

    * merges neighboring :class:`state preparations <.Preparation>` and :class:`gates <.Gate>`
      belonging to the same family and acting on the same sequence of subsystems
    * cancels neighboring pairs of a gate and its inverse

    Args:
        seq (Sequence[Command]): quantum circuit to optimize

    Returns:
        List[Command]: optimized circuit
    """

    def _print_list(i, q, print_fn=print):
        "For debugging."
        # pylint: disable=unreachable
        return
        print_fn("i: {},  len: {}   ".format(i, len(q)), end="")
        for x in q:
            print_fn(x.op, ", ", end="")
        print_fn()

    grid = list_to_grid(seq)

    # try merging neighboring operations on each wire
    # TODO the merging could also be done using the circuit DAG, which
    # might be smarter (ns>1 would be easy)
    for k in grid:
        q = grid[k]
        i = 0  # index along the wire
        _print_list(i, q)
        while i + 1 < len(q):
            # at least two operations left to merge on this wire
            try:
                a = q[i]
                b = q[i + 1]
                # the ops must have equal size and act on the same wires
                if a.op.ns == b.op.ns and a.reg == b.reg:
                    if a.op.ns != 1:
                        # ns > 1 is tougher. on no wire must there be anything
                        # between them, also deleting is more complicated
                        # todo treat it as a failed merge for now
                        i += 1
                        continue
                    op = a.op.merge(b.op)
                    # merge was successful, delete the old ops
                    del q[i : i + 2]
                    # insert the merged op (unless it's identity)
                    if op is not None:
                        q.insert(i, Command(op, a.reg))
                    # move one spot backwards to try another merge
                    if i > 0:
                        i -= 1
                    _print_list(i, q)
                    continue
            except MergeFailure:
                pass
            i += 1  # failed at merging the ops, move forward

    # convert the circuit back into a list (via a DAG)
    DAG = grid_to_DAG(grid)
    return DAG_to_list(DAG)


def program_equivalence(prog1, prog2, compare_params=True, atol=1e-6, rtol=0):
    r"""Checks if two programs are equivalent.

    This function converts the program lists into directed acyclic graphs,
    and runs the NetworkX `is_isomorphic` graph function in order
    to determine if the two programs are equivalent.

    .. note::

        When checking for parameter equality between two parameters
        :math:`a` and :math:`b`, we use the following formula:

        .. math:: |a - b| \leq (\texttt{atol} + \texttt{rtol}\times|b|)

    Args:
        prog1 (strawberryfields.program.Program): quantum program
        prog2 (strawberryfields.program.Program): quantum program
        compare_params (bool): Set to ``False`` to turn of comparing program parameters;
            equivalency will only take into account the operation order.
        atol (float): the absolute tolerance parameter for checking quantum operation
            parameter equality
        rtol (float): the relative tolerance parameter for checking quantum operation
            parameter equality

    Returns:
        bool: returns ``True`` if two quantum programs are equivalent
    """
    # TODO: at the moment, we do not check for whether an empty
    # wire will match an operation with trivial parameters.
    # Maybe we can do this in future, but this is a subgraph
    # isomorphism problem and much harder.

    # if the same program is passed twice, return ``True``
    if prog1 is prog2:
        return True

    DAG1 = list_to_DAG(prog1.circuit)
    DAG2 = list_to_DAG(prog2.circuit)

    circuit = []
    for G in (DAG1, DAG2):
        # relabel the DAG nodes to integers
        circuit.append(nx.convert_node_labels_to_integers(G))

        # ``CXgate`` and ``BSgate`` are not symmetric with respect to permuting the order of the two
        # modes it acts on; i.e., the order of the wires matter
        wire_mapping = {}
        for i, n in enumerate(G.nodes()):
            # not a ``CXgate`` or a ``BSgate``, order of wires doesn't matter
            wire_mapping[i] = 0

            if n.op.__class__.__name__ == "CXgate":
                # if the ``CXgate`` parameter is not 0, order matters
                if not np.allclose(n.op.p[0], 0):
                    wire_mapping[i] = [j.ind for j in n.reg]

            elif n.op.__class__.__name__ == "BSgate":
                # if the beamsplitter is not symmetric, order matters
                bs_params = [j % np.pi for j in par_evaluate(n.op.p)]
                if not np.allclose(bs_params, [np.pi / 4, np.pi / 2]):
                    wire_mapping[i] = [j.ind for j in n.reg]

        # add node attributes to store the operation wires
        nx.set_node_attributes(circuit[-1], wire_mapping, name="w")

        # add node attributes to store the operation parameters
        if compare_params:
            parameter_mapping = {i: par_evaluate(n.op.p) for i, n in enumerate(G.nodes())}
            nx.set_node_attributes(circuit[-1], parameter_mapping, name="p")

        # add node attributes to store the operation name
        name_mapping = {i: n.op.__class__.__name__ for i, n in enumerate(G.nodes())}
        nx.set_node_attributes(circuit[-1], name_mapping, name="name")

    def node_match(n1, n2):
        """Returns True if both nodes have the same name and
        same parameters, within a certain tolerance"""
        name_match = n1["name"] == n2["name"]
        wire_match = n1["w"] == n2["w"]

        if compare_params:
            p_match = np.allclose(n1["p"], n2["p"], atol=atol, rtol=rtol)
            return name_match and p_match and wire_match

        return name_match and wire_match

    # check if circuits are equivalent
    return nx.is_isomorphic(circuit[0], circuit[1], node_match)


def remove_loss(circuit):
    """Removes any ``LossChannel`` operations from a circuit sequence.
    Args:
        circuit (list[Command]: circuit with ``LossChannels`` to be removed
    Returns:
        list[Command]: circuit where the ``LossChannels`` have been removed
    """
    # pylint: disable=import-outside-toplevel
    from strawberryfields.ops import LossChannel

    lossless_circuit = circuit.copy()
    for cmd in circuit:
        if isinstance(cmd.op, LossChannel):
            lossless_circuit.remove(cmd)

    return lossless_circuit

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
Quantum program utilities. Module docs in program.py.
"""

from collections.abc import Sequence
import functools

import networkx as nx


__all__ = ['Program_current_context', '_convert', 'RegRefError', 'CircuitError', 'MergeFailure',
           'Command', 'RegRef', 'RegRefTransform',
           'list_to_grid', 'grid_to_DAG', 'DAG_to_list', 'list_to_DAG', 'group_operations']


Program_current_context = None
"""Context for inputting a Program. Used to be a class attribute of :class:`Program`, moved
here to avoid cyclic imports."""
# todo: Avoid issues with Program contexts and threading,
# cf. _pydecimal.py in the python standard distribution.


def _convert(func):
    r"""Decorator for converting user defined functions to a :class:`RegRefTransform`.

    This allows classical processing of measured qumodes values.

    Example usage:

    .. code-block:: python

        @convert
        def F(x):
            # some classical processing of x
            return f(x)

        with prog.context as q:
            MeasureX       | q[0]
            Dgate(F(q[0])) | q[1]

    Args:
        func (function): function to be converted to a :class:`RegRefTransform`.
    """
    @functools.wraps(func)
    def wrapper(*args):
        "Unused docstring."
        return RegRefTransform(args, func)
    return wrapper


class RegRefError(IndexError):
    """Exception raised by :class:`Program` when it encounters an invalid register reference.

    E.g., trying to apply a gate to a nonexistent or deleted subsystem.
    """


class CircuitError(RuntimeError):
    """Exception raised by :class:`Program` when it encounters an illegal
    operation in the quantum circuit.

    E.g., trying to use a measurement result before it is available.
    """


class MergeFailure(RuntimeError):
    """Exception raised by :meth:`strawberryfields.ops.Operation.merge` when an
    attempted merge fails.

    E.g., trying to merge two gates of different families.
    """


class Command:
    """Represents a quantum operation applied on specific subsystems of the register.

    A Command instance is immutable once created, and can be shared between
    several :class:`Program` instances.

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
        Return a string containing the Blackbird syntax.
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

        Combination of ``self.reg`` and ``self.op.extra_deps``.

        .. note:: ``extra_deps`` are used to ensure that the measurement
            happens before the result is used, but this is a bit too strict:
            two gates depending on the same measurement result but otherwise
            acting on different subsystems should commute.

        Returns:
            set[RegRef]: set of subsystems the command depends on
        """
        deps = self.op.extra_deps | set(self.reg)
        return deps


class RegRef:
    """Quantum register reference.

    The objects of this class refer to a specific subsystem (mode) of
    a quantum register.

    Within the scope of each :class:`Program` instance, only one RegRef instance
    should exist per subsystem. :class:`Program` keeps the authoritative mapping
    of subsystem indices to RegRef instances.
    Subsystem measurement results are stored in the "official" RegRef object.
    If other RegRef objects referring to the same subsystem exist, they will
    not be updated. Once a RegRef is assigned a subsystem index it will never
    change, not even if the subsystem is deleted.

    The RegRefs are constructed in :meth:`Program._add_subsystems`.

    Args:
        ind (int): index of the register subsystem referred to
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, ind):
        self.ind = ind   #: int: subsystem index
        self.val = None  #: float, complex: Measurement result. None if the subsystem has not been measured yet.
        self.active = True  #: bool: True at construction, False after the subsystem is deleted

    def __str__(self):
        return 'q[{}]'.format(self.ind)

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
        return self.ind == other.ind and self.active == other.active


class RegRefTransform:
    """Represents a scalar function of one or more register references.

    A RegRefTransform instance, given as a parameter to a
    :class:`~strawberryfields.ops.Operation` constructor, represents
    a dependence of the Operation on classical information obtained by
    measuring one or more subsystems.

    Used for deferred measurements, i.e., using a measurement's value
    symbolically in defining a gate before the numeric value of that
    measurement is available.

    Args:
        r (Sequence[RegRef]): register references that act as parameters for the function
        func (None, function): Scalar function that takes the values of the
            register references in r as parameters. None is equivalent to the identity
            transformation lambda x: x.
        func_str (str): an optional argument containing the string representation of the function.
            This is useful if a lambda function is passed to the RegRefTransform, which would otherwise
            show in the program queue as ``RegRefTransform(q[0], <lambda>)``.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, refs, func=None, func_str=None):
        # into a list of regrefs
        if isinstance(refs, RegRef):
            refs = [refs]

        if any([not r.active for r in refs]):
            # todo allow this if the regref already has a measurement result in it.
            # Maybe we want to delete a mode after measurement to save comp effort.
            raise ValueError('Trying to use inactive RegRefs.')

        #: list[RegRef]: register references that act as parameters for the function
        self.regrefs = refs
        self.func = func   #: None, function: the transformation itself, returns a scalar
        self.func_str = func_str

        if func is None and len(refs) > 1:
            raise ValueError('Identity transformation only takes one parameter.')

    def __str__(self):
        """Print the RegRefTransform using Blackbird syntax."""
        temp = [str(r) for r in self.regrefs]
        rr = ', '.join(temp)

        if len(temp) > 1:
            rr = '[' + rr + ']'

        if self.func is None:
            return 'RR({})'.format(rr)

        if self.func_str is None:
            return 'RR({}, {})'.format(rr, self.func.__name__)

        return 'RR({}, {})'.format(rr, self.func_str)

    def __format__(self, format_spec):
        return self.__str__()  # pragma: no cover

    def __eq__(self, other):
        "Not equal to anything."
        return False

    def evaluate(self):
        """Evaluates the numeric value of the function if all the measurement values are available.

        Returns:
            Number: function value
        """
        temp = [r.val for r in self.regrefs]
        if any(v is None for v in temp):
            # NOTE: "if None in temp" causes an error if temp contains arrays,
            # since it uses the == comparison in addition to "is"
            raise CircuitError("Trying to use a nonexistent measurement result (e.g., before it has been measured).")
        if self.func is None:
            return temp[0]
        return self.func(*temp)


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

    In the DAG each node is a :class:`Command` instance, and edges point from Commands to their dependents/followers.

    Args:
        grid (dict[int, list[Command]]): quantum circuit
    Returns:
        DAG[Command]: same circuit in DAG form
    """
    DAG = nx.DiGraph()
    for _, q in grid.items():
        if q:
            # add the first operation on the wire that does not depend on anything
            DAG.add_node(q[0])
        for i in range(1, len(q)):
            # add the edge between the operations, and the operation nodes themselves
            DAG.add_edge(q[i-1], q[i])
    return DAG


def list_to_DAG(ls):
    """Transforms a list of Commands to a DAG representation.

    In the DAG each node is a :class:`Command` instance, and edges point from Commands to their dependents/followers.

    Args:
        ls (Iterable[Command]): quantum circuit
    Returns:
        DAG[Command]: same circuit in DAG form
    """
    return grid_to_DAG(list_to_grid(ls))


def DAG_to_list(dag):
    """Transforms a Command DAG to a list representation.

    The list contains the :class:`Command` instances in (one possible) topological order,
    i.e., dependants following the operations they depend on.

    Args:
        dag (DAG[Command]): quantum circuit
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

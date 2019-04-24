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
Quantum programs
================

**Module name:** :mod:`strawberryfields.program`

.. currentmodule:: strawberryfields.program

This module implements the :class:`Program` class which acts as a representation for quantum circuits.
The Program object also acts as a context for defining the quantum circuit using the Python-embedded Blackbird syntax.

A typical use looks like
::

  prog = sf.Program(num_subsystems)
  with prog.context as q:
      Coherent(0.5)  | q[0]
      Vac            | q[1]
      Sgate(2)       | q[1]
      Dgate(0.5)     | q[0]
      BSgate(1)      | q
      Dgate(0.5).H   | q[0]
      Measure        | q
  eng = sf.Engine('fock')
  eng.run(prog, cutoff_dim=5)
  v1 = prog.register[1].val

The Program objects also keep track of the state of the quantum register they act on, using a dictionary of :class:`RegRef` objects.
The currently active register references can be accessed using the :meth:`~Program.register` method.


Program methods
---------------

.. currentmodule:: strawberryfields.program.Program

.. autosummary::
   context
   register
   num_subsystems
   __len__
   can_follow
   append
   compile
   optimize
   print
   draw_circuit
   lock

The following are internal Program methods. In most cases the user should not
call these directly.

.. autosummary::
       __enter__
       __exit__
       _clear_regrefs
       _add_subsystems
       _delete_subsystems
       _index_to_regref
       _test_regrefs
       _list_to_grid
       _grid_to_DAG
       _DAG_to_list


Helper classes
--------------

.. currentmodule:: strawberryfields.program

.. autosummary::
   Command
   RegRef
   RegRefTransform


Exceptions
----------

.. autosummary::
   MergeFailure
   CircuitError
   RegRefError
   ~strawberryfields.backends.base.NotApplicableError


Quantum circuit representation
------------------------------

The :class:`Command` instances in the circuit form a
`strict partially ordered set <http://en.wikipedia.org/wiki/Partially_ordered_set#Strict_and_non-strict_partial_orders>`_
in the sense that the order in which the operations have to be executed is usually not completely fixed.
For example, operations acting on different subsystems always commute with each other.
We denote :math:`a < b` if :math:`a` has to be executed before :math:`b`.
Each strict partial order corresponds to a
`directed acyclic graph <http://en.wikipedia.org/wiki/Directed_acyclic_graph>`_ (DAG),
and the transitive closure of any DAG is a strict partial order.
During the optimization three different (but equivalent) representations of the circuit are used.

* Initially, the circuit is represented as a Command queue (list), listing the Commands in
  the temporal order they are applied.
* The second representation, grid, essentially mimics a quantum circuit diagram.
  It is a mapping from subsystem indices to lists of Commands touching that subsystem,
  where each list is temporally ordered.
* Finally, the quantum circuit can be represented using a DAG by making each Command a node,
  and drawing an edge from each Command to all its immediate followers along each wire it touches.
  It can be converted back into a command queue by popping a maximal element until the graph
  is empty, that is, consuming it in a topological order.
  Note that a topological order is not always unique, there may be several equivalent topological orders.

.. currentmodule:: strawberryfields.program.Program

The three representations can be converted to each other
using the methods :func:`_list_to_grid`, :func:`_grid_to_DAG` and :func:`_DAG_to_list`.


Optimizer
---------

The purpose of the optimizer part of the compiler is to simplify the circuit
to make it cheaper and faster to execute. Different backends might require
different types of optimization, but in general the fewer operations a circuit has,
the faster it should run. The optimizer thus should convert the circuit into a
simpler circuit while preserving the probability distributions of the measurement results.
The optimization utilizes the abstract algebraic properties of the gates,
and in no point should require a matrix representation.

Currently the optimization is very simple. It

* merges neighboring gates belonging to the same gate family and sharing the same set of subsystems
* cancels neighboring pairs of a gate and its inverse

.. currentmodule:: strawberryfields.program


Code details
~~~~~~~~~~~~

"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init

# todo: Avoid issues with Program contexts and threading,
# cf. _pydecimal.py in the python standard distribution.

from collections.abc import Sequence
import copy
import functools
import numbers

import networkx as nx

import strawberryfields.circuitdrawer as sfcd



def _print_list(i, q, print_fn=print):
    "For debugging."
    # pylint: disable=unreachable
    return
    print_fn('i: {},  len: {}   '.format(i, len(q)), end='')
    for x in q:
        print_fn(x.op, ', ', end='')
    print_fn()

def _convert(func):
    r"""Decorator for converting user defined functions to a :class:`RegRefTransform`.

    This allows classical processing of measured qumodes values.

    Example usage:

    .. code-block:: python

        @convert
        def F(x):
            # some classical processing of x
            return f(x)

        with eng:
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


# TODO replace this mock backend capability database
backend_database = {
    'fock': {
        'New_modes': True,
        'Delete': True,
        'Vacuum': True,
        'Coherent': True,
        'Squeezed': True,
        'DisplacedSqueezed': True,
        'Ket': True,
        'DensityMatrix': True,
        'Fock': True,
        'Catstate': True,
        'Thermal': True,
        'LossChannel': True,
        'ThermalLossChannel': True,
        'Rgate': True,
        'Fouriergate': True,
        'Dgate': True,
        'Xgate': True,
        'Zgate': True,
        'Sgate': True,
        'Pgate': False,
        'Vgate': True,
        'Kgate': True,
        'BSgate': True,
        'CXgate': False,
        'CZgate': False,
        'CKgate': True,
        'S2gate': False,  # use a decomposition
        'Interferometer': False,
        'GraphEmbed': False,
        'Gaussian': False,
        'GaussianTransform': False,
        'MeasureHomodyne': True,
        'MeasureFock': True,
    },
    'gaussian': {
        'New_modes': True,
        'Delete': True,
        'Vacuum': True,
        'Coherent': True,
        'Squeezed': True,
        'DisplacedSqueezed': True,
        'Thermal': True,
        'LossChannel': True,
        'ThermalLossChannel': True,
        'Rgate': True,
        'Fouriergate': True,
        'Dgate': True,
        'Xgate': True,
        'Zgate': True,
        'Sgate': True,
        'Pgate': False,
        'BSgate': True,
        'CXgate': False,
        'CZgate': False,
        'S2gate': False,  # use a decomposition
        'Interferometer': False,
        'GraphEmbed': False,
        'Gaussian': False,
        'GaussianTransform': False,
        'MeasureHomodyne': True,
        'MeasureHeterodyne': True,
    },
}
backend_database['tf'] = backend_database['fock']  # tf can do the same things as fock
backend_database['base'] = backend_database['fock']


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

    def __init__(self, op, reg, decomp=False):
        # accept a single RegRef in addition to a Sequence
        if not isinstance(reg, Sequence):
            reg = [reg]

        #: Operation: quantum operation to apply
        self.op = op
        #: Sequence[RegRef]: subsystems to which the operation is applied
        self.reg = reg
        #: bool: is this Command a part of a decomposition?
        self.decomp = decomp

    def __str__(self):
        """Print the command using Blackbird syntax."""
        temp = str(self.op)
        if self.op.ns == 0:
            # op takes no subsystems as parameters, do not print anything more
            return temp
        return '{} | ({})'.format(temp, ", ".join([str(rr) for rr in self.reg]))

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
        self.val = None  #: Real: measured eigenvalue. None if the subsystem has not been measured yet
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
            raise CircuitError("Trying to use a nonexistent measurement result (e.g., before it can be measured).")
        if self.func is None:
            return temp[0]
        return self.func(*temp)


class Program:
    """Represents a quantum circuit.

    A quantum circuit is a set of quantum operations applied in a specific order
    to a set of subsystems (represented by wires in the circuit diagram) of the quantum register.
    The Program class represents a quantum circuit in general as a directed acyclic graph (DAG)
    whose nodes are :class:`Command` instances, and each (directed) edge in the graph
    corresponds to a specific wire along which the two associated Commands are connected.

    Program instances also act as context managers (and the context itself) for inputting
    quantum circuits using a :code:`with` block and :class:`Operation` instances.
    The contexts may not be nested.

    The quantum circuit is inputted by using the :meth:`~strawberryfields.ops.Operation.__or__`
    methods of the quantum operations, which call the :meth:`append` method of the Program.
    :meth:`append` checks that the register references are valid and then
    adds a new :class:`.Command` instance to the Program.

    The ``New`` and ``Del`` operations modify the quantum register itself by adding
    and deleting subsystems. The Program keeps track of these changes (using
    :class:`RegRef` instances that represent the register)
    as they are appended to it in order to be able to report register
    reference errors as soon as they happen.

    Program `p2` can be run after Program `p1` if the RegRef state at the end of `p1` matches the
    RegRef state at the start of `p2`. This can be enforced by constructing `p2` as an explicit
    successor of `p1`, in which case the regrefs are copied over.
    When a Program is run or it obtains a successor, it is locked and no more Commands can be appended to it.

    Args:
        num_subsystems (int, Program): Initial number of subsystems in the quantum register.
            Alternatively, another Program instance from which to inherit the register state.
        name (str): program name (optional)
    """
    _current_context = None  # context for inputting a program

    def __init__(self, num_subsystems, name=None):
        #: str: program name
        self.name = name
        #: list[Command]: Commands constituting the quantum circuit in temporal order
        self.circuit = []
        #: bool: True if no more Commands can be appended to the Program
        self.locked = False
        #: str: backend for which the circuit has been compiled
        self.backend = None
        #: Program: for compiled programs, this is the original
        self.source = None

        # create subsystem references
        if isinstance(num_subsystems, numbers.Integral):
            #: int: initial number of subsystems
            self.init_num_subsystems = num_subsystems
            #: dict[int, RegRef]: mapping from subsystem indices to corresponding RegRef objects
            self.reg_refs = {}
            #: set[int]: created subsystem indices that have not been used (operated on) yet
            self.unused_indices = set()
            self._add_subsystems(num_subsystems)
        elif isinstance(num_subsystems, Program):
            # it's the parent program
            parent = num_subsystems
            # copy the RegRef state from the parent program
            parent.lock()  # make sure the parent isn't accidentally updated by the user
            self.init_num_subsystems = parent.num_subsystems
            self.reg_refs = copy.deepcopy(parent.reg_refs)  # independent copy of the RegRefs
            self.unused_indices = copy.copy(parent.unused_indices)
        else:
            raise TypeError('First argument must be either the number of subsystems or the parent Program.')

        # save the initial regref state
        #: dict[int, RegRef]: like reg_refs
        self.init_reg_refs = copy.deepcopy(self.reg_refs)
        #: set[int]: like unused_indices
        self.init_unused_indices = copy.copy(self.unused_indices)

    def __str__(self):
        """String representation."""
        return self.__class__.__name__ + '({}, {}->{} subsystems, compiled for {})'.format(self.name, self.init_num_subsystems, self.num_subsystems, self.backend)

    def __len__(self):
        """Program length.

        Returns:
            int: number of Commands in the program
        """
        return len(self.circuit)

    def print(self, print_fn=print):
        """Print the program contents using Blackbird syntax.

        Args:
            print_fn (function): optional custom function to use for string printing
        """
        for k in self.circuit:
            print_fn(k)

    @property
    def context(self):
        """Syntactic sugar for defining a Program using the with statement.

        The Program object itself acts as the context manager.
        """
        return self

    def __enter__(self):
        """Enter the context for this program.

        Returns:
            tuple[RegRef]: subsystem references
        """
        if Program._current_context is None:
            Program._current_context = self
        else:
            raise RuntimeError('Only one Program context can be active at a time.')
        return self.register

    def __exit__(self, ex_type, ex_value, ex_tb):
        """Exit the quantum circuit context."""
        Program._current_context = None

    # =================================================
    #  RegRef accounting
    # =================================================
    @property
    def register(self):
        """Return symbolic references to all the currently valid register subsystems.

        Returns:
            tuple[RegRef]: valid subsystem references
        """
        return tuple(r for r in self.reg_refs.values() if r.active)

    @property
    def num_subsystems(self):
        """Return the current number of valid register subsystems.

        Returns:
            int: number of currently valid register subsystems
        """
        return len(self.register)

    def _clear_regrefs(self):
        """Clears any measurement values stored in the RegRefs.

        Called by :class:`~.engine.Engine` when resetting the backend.
        """
        for r in self.reg_refs.values():
            r.val = None

    def _add_subsystems(self, n):
        """Create new subsystem references, add them to the reg_ref dictionary.

        To avoid discrepancies with the backend this method must not be called directly,
        but rather indirectly by using :class:`~strawberryfields.ops.New_modes`
        instances in the Program context.

        .. note:: This is the only place where :class:`RegRef` instances are constructed.

        Args:
            n (int): number of subsystems to add
        Returns:
            tuple[RegRef]: tuple of the newly added subsystem references
        """
        if not isinstance(n, numbers.Integral) or n < 1:
            raise ValueError('Number of added subsystems {} is not a positive integer.'.format(n))
        first_unassigned_index = len(self.reg_refs)
        # create a list of RegRefs
        inds = [first_unassigned_index+i for i in range(n)]
        refs = tuple(RegRef(i) for i in inds)
        # add them to the index map
        for r in refs:
            self.reg_refs[r.ind] = r
        # all the newly reserved indices are unused for now
        self.unused_indices.update(inds)
        return refs

    def _delete_subsystems(self, refs):
        """Delete existing subsystem references.

        To avoid discrepancies with the backend this method must not be called directly,
        but rather indirectly by using :class:`~strawberryfields.ops.Delete` instances
        in the Program context.

        Args:
          refs (Sequence[RegRef]): subsystems to delete
        """
        # NOTE: refs have already been through _test_regrefs() in append() and thus should be valid
        for r in refs:
            # mark the RegRef as deleted
            r.active = False
            #self.reg_refs[r.ind].active = False
        # NOTE: deleted indices are *not* removed from self.unused_indices

    def lock(self):
        """Finalize the program.

        When a Program is locked, no more Commands can be appended to it.
        The locking happens when the program is run, compiled, or a successor Program is constructed,
        in order to ensure that the RegRef state of the Program does not change anymore.
        """
        self.locked = True

    def can_follow(self, prev):
        """Checks if this program can follow the given program.

        This requires that the final RegRef state of the first program matches
        the initial RegRef state of the second program, i.e., they have the same number
        number of RegRefs, all with identical indices and activity states.

        Args:
            prev (Program): preceding program fragment
        Returns:
            bool: True if the Program can follow prev
        """
        # TODO NOTE unused_indices is not compared here, in order to allow program fragment repetition
        return self.init_reg_refs == prev.reg_refs

    def _index_to_regref(self, ind):
        """Try to find a RegRef corresponding to a given subsystem index.

        Args:
            ind (int): subsystem index
        Returns:
            RegRef: corresponding register reference
        Raises:
            .RegRefError: if the subsystem cannot be found, or is invalid
        """
        # index must be found in the dict
        if ind not in self.reg_refs:
            raise RegRefError('Subsystem {} does not exist.'.format(ind))
        return self.reg_refs[ind]

    def _test_regrefs(self, reg):
        """Make sure reg is a valid selection of subsystems, convert them to RegRefs.

        A register reference is valid if it is properly recorded in self.reg_refs
        and has not been deleted. The selection is valid if it contains only
        valid RegRefs and no subsystem is repeated.

        Args:
            reg (Iterable[int, RegRef]): subsystem references
        Returns:
            list[RegRef]: converted subsystem references
        Raises:
            .RegRefError: if an invalid subsystem reference is found
        """
        temp = []
        for rr in reg:
            # must be either an integer or a RegRef
            if isinstance(rr, RegRef):
                # regref must be found in the dict values (the RegRefs are compared using __eq__, which, since we do not define it, defaults to "is")
                if rr not in self.reg_refs.values():
                    raise RegRefError('Unknown RegRef.')
                if self.reg_refs[rr.ind] is not rr:
                    raise RegRefError('RegRef state has become inconsistent.')
            elif isinstance(rr, numbers.Integral):
                rr = self._index_to_regref(rr)
            else:
                raise RegRefError('Subsystems can only be indexed using integers and RegRefs.')

            if not rr.active:
                raise RegRefError('Subsystem {} has already been deleted.'.format(rr.ind))
            if rr in temp:
                raise RegRefError('Trying to act on the same subsystem more than once.')
            temp.append(rr)
        return temp

    def append(self, op, reg):
        """Append a command to the program.

        Args:
            op (Operation): quantum operation
            reg (list[int, RegRef]): register subsystem(s) to apply it to
        Returns:
            list[RegRef]: subsystem list as RegRefs
        """
        if self.locked:
            raise RuntimeError('The Program is locked, no more Commands can be appended to it.')

        # test that the target subsystem references are ok
        reg = self._test_regrefs(reg)
        # also test possible RegRef dependencies
        self._test_regrefs(op.extra_deps)
        for rr in reg:
            # it's used now
            self.unused_indices.discard(rr.ind)
        self.circuit.append(Command(op, reg))
        return reg

    def compile(self, backend='fock', optimize=True):
        """Compile the program for the given backend.

        The compilation step validates the program, making sure all the Operations used are accepted by the backend.
        Additionally it may decompose certain gates into sequences of simpler gates.

        The compiled program shares its RegRefs with the original, which makes it easier to access the measurement
        results, but also necessitates the locking of both the compiled program and the original to make sure the
        RegRef state remains consistent.

        Args:
            backend (str): target backend
            optimize (bool): try to optimize the program by merging and canceling gates
        Returns:
            Program: compiled program
        """
        db = backend_database[backend]

        def compile_sequence(seq):
            """Compiles the given Command sequence."""
            compiled = []
            for cmd in seq:
                # None represents an identity gate
                if cmd.op is None:
                    continue
                elif cmd.op.__class__.__name__ in db:
                    # backend can handle the op
                    if db[cmd.op.__class__.__name__]:
                        compiled.append(cmd)
                    else:
                        # op not directly supported by the backend, try a decomposition instead
                        try:
                            temp = cmd.op.decompose(cmd.reg)
                            # now compile the decomposition
                            temp = compile_sequence(temp)
                            compiled.extend(temp)
                        except NotImplementedError as err:
                            # simplify the error message by suppressing the previous exception
                            raise err from None
                else:
                    raise CircuitError('The operation {} cannot be used with the {} backend.'.format(cmd.op.__class__.__name__, backend))
            return compiled

        self.lock()
        seq = compile_sequence(self.circuit)
        compiled = copy.copy(self)  # shares RegRefs with the source
        compiled.backend = backend
        compiled.circuit = seq
        # link to the original source Program
        if self.source is None:
            compiled.source = self
        else:
            compiled.source = self.source
        if optimize:
            compiled.optimize()
        return compiled

    @staticmethod
    def _list_to_grid(ls):
        """Transforms a list of commands to a grid representation.

        The grid is a mapping from subsystem indices to lists of :class:`Command` instances touching that subsystem.
        The same Command instance will appear in each list that corresponds to one of its subsystems.

        Args:
            ls (list[Command]): circuit to be transformed
        Returns:
            Grid[Command]: transformed circuit
        """
        grid = {}
        # enter every operation in the list to its proper position in the grid
        for cmd in ls:
            for r in cmd.get_dependencies():
                # Add cmd to the grid to the end of the line r.ind.
                if r.ind not in grid:
                    grid[r.ind] = []  # add a new line to the circuit
                grid[r.ind].append(cmd)
        return grid

    @staticmethod
    def _grid_to_DAG(grid):
        """Transforms a command grid to a DAG representation.

        In the DAG each node is a :class:`Command` instance, and edges point from Commands to their dependents/followers.

        Args:
            grid (Grid[Command]): circuit to be transformed
        Returns:
            DAG[Command]: transformed circuit
        """
        DAG = nx.DiGraph()
        for key in grid:
            q = grid[key]
            _print_list(0, q)
            if q:
                # add the first operation on the wire that does not depend on anything
                DAG.add_node(q[0])
            for i in range(1, len(q)):
                # add the edge between the operations, and the operation nodes themselves
                DAG.add_edge(q[i-1], q[i])
        return DAG

    @staticmethod
    def _DAG_to_list(dag):
        """Transforms a command DAG to a list representation.

        The list contains the :class:`Command` instances in (one possible) topological order.

        Args:
            dag (DAG[Command]): circuit to be transformed
        Returns:
            list[Command]: transformed circuit
        """
        # sort the operation graph into topological order
        # (dependants following the operations they depend on)
        temp = nx.algorithms.dag.topological_sort(dag)
        return list(temp)

    def optimize(self):
        """Try to simplify and optimize the quantum circuit.

        The simplifications are based on the algebraic properties of the gates,
        e.g., combining two consecutive gates of the same gate family.

        The optimization must not change the state of the RegRefs in any way.
        """
        # print('\n\nOptimizing...\nUnused inds: ', self.unused_indices)

        grid = self._list_to_grid(self.circuit)
        # for k in grid:
        #    print('mode {}, len {}'.format(k, len(grid[k])))

        # try merging neighboring operations on each wire
        # TODO the merging could also be done using the circuit DAG, which
        # might be smarter (ns>1 would be easy)
        for k in grid:
            q = grid[k]
            #print('\nqumode {}:\n'.format(k))
            i = 0  # index along the wire
            _print_list(i, q)
            while i+1 < len(q):
                # at least two operations left to merge on this wire
                try:
                    a = q[i]
                    b = q[i+1]
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
                        del q[i:i+2]
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
        DAG = self._grid_to_DAG(grid)
        self.circuit = self._DAG_to_list(DAG)


    def draw_circuit(self, tex_dir='./circuit_tex', write_to_file=True):
        r"""Draw the circuit using the Qcircuit :math:`\LaTeX` package.

        This will generate the LaTeX code required to draw the quantum circuit
        diagram corresponding to the Program.

        Args:
            tex_dir (str): relative directory for latex document output
            write_to_file (bool): if False, no output file is created

        Returns:
            list[str]: filename of the written tex document and the written tex content
        """
        drawer = sfcd.Circuit(wires=self.init_num_subsystems)
        self.print(drawer.parse_op)
        tex = drawer.dump_to_document()

        document = None
        if write_to_file:
            document = drawer.compile_document(tex_dir=tex_dir)

        return [document, tex]

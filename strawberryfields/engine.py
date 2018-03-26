# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Quantum compiler engine
========================================================

**Module name:** :mod:`strawberryfields.engine`

.. currentmodule:: strawberryfields.engine

This module implements the quantum compiler engine.
The :class:`Engine` class provides the toolchain that
starts with the user inputting a quantum program and ends with a backend that
could be e.g. a simulator, a hardware quantum processor, or a circuit drawer.

Syntactically, the compiler engine acts as the context for the quantum program.
A typical use looks like
::

  eng, q = sf.Engine(num_subsystems)
  with eng:
    Coherent(0.5)  | q[0]
    Vac            | q[1]
    Sgate(2)       | q[1]
    Dgate(0.5)     | q[0]
    BSgate(1)      | q
    Dgate(0.5).H   | q[0]
    Measure        | q
  eng.run(backend[,cutoff_dim])
  v1 = q[1].val


Engine methods
--------------------------------

.. currentmodule:: strawberryfields.engine.Engine

.. autosummary::
   __enter__
   __exit__
   register
   add_subsystems
   delete_subsystems
   reset
   reset_queue
   reset_backend
   append
   print_queue
   print_applied
   run_command_list
   run
   optimize

Helper classes
--------------

.. currentmodule:: strawberryfields.engine

.. autosummary::
   Command
   RegRef
   RegRefTransform


Optimizer
---------

The purpose of the optimizer part of the compiler engine is to simplify the program
to make it cheaper and faster to execute. Different backends might require different types of optimization,
but in general the fewer operations a program has, the faster it should run.
The optimizer thus should convert the program into a simpler but otherwise equivalent version of itself,
preserving the probability distributions of the measurement results.
The optimization utilizes the abstract algebraic properties of the gates, and in no point should require a
matrix representation.
Currently the optimization is somewhat simple, being able to merge neighboring gates belonging to the same
gate family and sharing the same set of subsystems, and canceling pairs of a gate and its inverse.

.. currentmodule:: strawberryfields.engine

Code details
~~~~~~~~~~~~

"""

# todo: Avoid issues with contexts and threading, cf. _pydecimal.py in the python standard distribution.
# todo: In the future we might wish to be able to commute gates past each other using a set of commutation rules/patterns.
#pylint: disable=too-many-instance-attributes

from collections.abc import Sequence
from functools import wraps

import networkx as nx

from .backends import load_backend
from .backends.base import NotApplicableError


def _print_list(i, q):
    "For debugging."
    # pylint: disable=unreachable
    return
    print('i: {},  len: {}   '.format(i, len(q)), end='')
    for x in q:
        print(x.op, ', ', end='')
    print()


def _convert(custom_function):
    r"""Decorator for converting user defined functions
    to a :class:`~.RegRefTransform`.

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
        custom_function (function): the user defined function
            to be converted to a :class:`~.RegRefTransform`.
    """
    @wraps(custom_function)
    def wrapper(*args, **kwargs):
        # pylint: disable=missing-docstring,unused-argument
        register = args[0]
        rr = RegRefTransform(
            register,
            func=custom_function,
            func_str=custom_function.__name__
        )
        return rr
    return wrapper


class SFProgramError(RuntimeError):
    """Exception raised by :class:`Engine` when it encounters an illegal operation in the quantum program.

    E.g. trying to use a measurement result before it is available.
    """
    pass


class Command:
    """Represents a quantum operation applied on specific subsystems of the register.

    Args:
      op (Operation): quantum operation to apply
      reg (Sequence[RegRef]): subsystems to which the operation is applied
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, op, reg, decomp=False):
        # accept a single RegRef in addition to a Sequence
        if not isinstance(reg, Sequence):
            reg = [reg]
        self.op = op   #: Operation: quantum operation to apply
        self.reg = reg  #: Sequence[RegRef]: subsystems to which the operation is applied
        self.decomp = decomp

    def __str__(self):
        return '{}, \t({})'.format(self.op, ", ".join([str(rr) for rr in self.reg]))

    def get_dependencies(self):
        """Subsystems the command depends on.

        Combination of ``self.reg`` and ``self.op.extra_deps``.

        .. note:: ``extra_deps`` are used to ensure that the measurement happens before the result is used, but this is a bit too strict: two gates depending on the same measurement result but otherwise acting on different subsystems commute.

        Returns:
          list[RegRef]: list of subsystems the command depends on
        """
        deps = list(self.reg)
        deps.extend(self.op.extra_deps)
        return deps


class RegRef:
    """Quantum register reference.

    The objects of this class refer to a specific subsystem (mode) of a quantum register.
    Only one RegRef instance should exist per subsystem: :class:`Engine` keeps the authoritative mapping of subsystem numbers to RegRef instances.
    Subsystem measurement results are stored in the "official" RegRef object. If other RegRefs objects referring to the same subsystem exist, they will not be updated.

    The RegRefs are constructed in :func:`Engine.add_subsystems`.

    Args:
      ind (int): subsystem index referred to
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, ind):
        self.ind = ind   #: int: subsystem index
        self.val = None  #: Real: measured eigenvalue, or None if the subsystem has not been measured yet

    def __str__(self):
        return 'reg[{}]'.format(self.ind)


class RegRefTransform:
    """Represents a scalar function of one or more register references.

    A RegRefTransform instance, given as a parameter to a :class:`~strawberryfields.ops.Operation` constructor, represents
    a dependence of the Operation on classical information obtained by measuring one or more subsystems.
    Used for deferred measurements, i.e., using a measurement's value symbolically in defining a gate before the numeric value of that measurement is available.

    Args:
      r (Sequence[RegRef]): register references that act as parameters for the function
      func (function): scalar function that takes the values of the register references in r as parameters
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, r, func=lambda x: x, func_str=None):
        # into a list of regrefs
        if isinstance(r, RegRef):
            r = [r]
        self.regrefs = r   #: list[RegRef]: register references that act as parameters for the function
        self.func = func   #: function: the transformation itself, returns a scalar
        self.func_str = func_str

    def __str__(self):
        # pylint: disable=no-else-return
        temp = [str(r) for r in self.regrefs]
        if self.func_str is None:
            if len(temp) == 1:
                return 'RR({})'.format(', '.join(temp))
            else:
                return 'RR([{}])'.format(', '.join(temp))
        else:
            if len(temp) == 1:
                return 'RR({}, {})'.format(', '.join(temp), self.func_str)
            else:
                return 'RR([{}], {})'.format(', '.join(temp), self.func_str)

    def __format__(self, format_spec):
        return self.__str__() # pragma: no cover

    def __eq__(self, other):
        "Not equal to anything."
        return False

    def __pos__(self):
        """Unary plus, used to evaluate the numeric value of the function once all the measurement values are available.

        .. note:: Hack: we use the unary plus for RegRefTransform evaluation because it also happens to be a NOP
              for plain numeric variables, thus producing the expected behavior in both cases.
        """
        temp = [r.val for r in self.regrefs]
        any_nones = sum([v is None for v in temp])
        if any_nones:
            raise SFProgramError('Trying to use a nonexistent measurement result (e.g. before it can be measured).')
        return self.func(*temp)


class Engine:
    r"""Quantum compiler engine.

    Acts as a context manager (and the context itself) for quantum programs.
    The contexts may not be nested.

    Args:
        num_subsystems (int): Number of subsystems in the quantum register.
        hbar (float): The value of :math:`\hbar` to initialise the engine with, depending on the
            conventions followed. By default, :math:`\hbar=2`. See
            :ref:`conventions` for more details.
    """
    _current_context = None

    def __init__(self, num_subsystems, hbar):
        self.num_subsystems = 0     #: int: number of subsystems in the quantum register
        self.cmd_queue = []       #: list[Command]: command queue
        self.cmd_applied = []         #: list[Command]: commands that have been run
        self.reg_refs = {}        #: dict[int->RegRef]: mapping from subsystem indices to corresponding RegRef objects
        self.unused_indices = set()  #: set[int]: created subsystem indices that have not been used (operated on) yet
        # create mode references
        self.add_subsystems(num_subsystems)
        self.backend = None
        self.init_modes = num_subsystems
        self.hbar = hbar

    def __str__(self):
        """String representation."""
        return self.__class__.__name__ + '({} subsystems, {})'.format(self.num_subsystems, self.backend.__class__.__name__)

    def __enter__(self):
        """Enter the quantum program context for this engine."""
        if Engine._current_context is None:
            Engine._current_context = self
        else:
            raise RuntimeError('Only one context can be active at a time.')
        return self

    def __exit__(self, ex_type, ex_value, ex_tb):
        """Exit the quantum program context."""
        Engine._current_context = None

    @property
    def register(self):
        """Return symbolic references to all the register subsystems.

        Returns:
          tuple[RegRef]: tuple of subsystem references
        """
        return tuple(self.reg_refs.values())

    def add_subsystems(self, n):
        """Create new subsystem references, add them to the reg_ref dictionary.

        Does *not* ask the backend to create the new modes.
        This is the only place where RegRef instances are constructed.

        Args:
          n (int): number of subsystems to add

        Returns:
          tuple[RegRef]: tuple of the newly added subsystem references
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError('{} is not a nonnegative integer.'.format(n))
        first_unassigned_index = len(self.reg_refs)
        # create a list of RegRefs
        inds = [first_unassigned_index+i for i in range(n)]
        refs = tuple(RegRef(i) for i in inds)
        # add them to the index map
        for rr in refs:
            self.reg_refs[rr.ind] = rr
        # optimization stuff
        self.unused_indices.update(inds)  # all the newly reserved indices are unused for now
        self.num_subsystems += len(inds)
        return refs

    def delete_subsystems(self, refs):
        """Delete existing subsystem references.

        Does *not* ask the backend to delete the modes right away. This only happens later
        when the corresponding Command is applied/executed.

        Args:
          refs (Sequence[RegRef]): subsystems to delete
        """
        for rr in refs:
            if rr.ind not in self.reg_refs:
                raise IndexError('Subsystem {} does not exist.'.format(rr.ind))
            temp = self.reg_refs[rr.ind]
            if temp is None:
                raise IndexError('Subsystem {} has already been deleted.'.format(rr.ind))
            if temp is not rr:
                raise RuntimeError('There should be only one RegRef instance for each index.')
            # mark the index as deleted/invalid
            self.reg_refs[rr.ind] = None
        self.num_subsystems -= len(refs)
        # NOTE: deleted indices are *not* removed from self.unused_indices

    def reset(self):
        """Clear the command queue, re-initialize the backend to vacuum.

        This command clears all queued gates, and if the circuit has been run, also resets the backend,
        setting the state of all modes back to the vacuum state.
        """
        self.reset_queue()
        self.reset_backend()

    def reset_queue(self):
        """Clear the command queue.

        This command clears all queued gates, but does not reset the current state of the circuit
        if it has been run at any point."""
        self.cmd_queue.clear()


    def reset_backend(self):
        """Clear the backend.

        Resets the backend, and the simulation status of the circuit.
        All modes are returned to the vacuum state. Note that this
        *does not* clear the program queue."""
        if self.backend:
            self.backend.reset()
            self.cmd_applied = []


    def _test_regrefs(self, reg):
        "Make sure reg is a valid selection of subsystems, convert them to RegRefs."
        temp = []
        for rr in reg:
            if isinstance(rr, int):
                rr = self.reg_refs[rr]
            elif rr not in self.reg_refs.values():
                raise IndexError('Trying to act on a nonexistent subsystem.')
            if rr in temp:
                raise IndexError('Trying to act on the same subsystem more than once.')
            temp.append(rr)
        return temp

    def append(self, op, reg):
        """Append a quantum program command to the engine command queue.

        Args:
          op (Operation): quantum operation
          reg (list[int, RegRef]): register subsystem(s) to apply it to
        """
        # test that the target subsystem references are ok
        reg = self._test_regrefs(reg)
        # also test possible RegRef dependencies
        self._test_regrefs(op.extra_deps)
        for rr in reg:
            # it's used now
            self.unused_indices.discard(rr.ind)
        self.cmd_queue.append(Command(op, reg))

    def print_queue(self):
        """Print the command queue.

        This contains the gates that will be applied on the next call to :meth:`~.Engine.run`."""
        for k in self.cmd_queue:
            print(k)

    def print_applied(self):
        """Print all commands applied to the qumodes since the backend was first initialized.

        This will be blank until the first call to :meth:`~.Engine.run`. The output may
        differ compared to :meth:`~.Engine.print_queue`, due to command decompositions
        and optimizations supported by the backend."""
        for k in self.cmd_applied:
            print(k)

    def return_state(self, modes=None, **kwargs):
        """This method returns the backend state object.

        Args:
            modes (Sequence[int]): integers containing the modes to be returned.
                If none, all modes are returned.

        Returns:
            BaseState: object containing details and methods for manipulation
                of the returned circuit state
        """
        return self.backend.state(modes=modes, **kwargs)

    def run_command_list(self, clist, **kwargs):
        """Execute the commands in the list.

        Args:
          clist (list[Command]): command list to run
        """
        for cmd in clist:
            if cmd.op is None:
                # None represents an identity gate
                continue
            else:
                try:
                    # try to apply it to the backend
                    cmd.op.apply(cmd.reg, self.backend, hbar=self.hbar, **kwargs)
                    self.cmd_applied.append(cmd)
                except NotApplicableError:
                    # command is not applicable to the current backend type
                    raise TypeError('The operation {} cannot be used with {}.'.format(cmd.op, self.backend)) from None
                except NotImplementedError:
                    # command not directly supported by backend API, try a decomposition instead
                    try:
                        temp = cmd.op.decompose(cmd.reg)
                        self.run_command_list(temp)
                    except NotImplementedError as err:
                        raise err from None

    def run(self, backend=None, reset_backend=True, return_state=True, modes=None, *args, **kwargs):
        """Execute the program in the command queue by sending it to the backend, does not empty the queue.

        Args:
            backend (str, BaseBackend, None): Backend for executing the commands.
                Either a backend name ("gaussian", "fock", or "tf"), in which case it is loaded and initialized,
                or a BaseBackend instance, or None if a backend is not required.
            reset_backend (bool): if set to True (default), the backend is reset before the engine is run.
                To avoid this behaviour, for instance if you would like to loop over
                engine to run to successively apply the same gates in the command queue, simply
                set ``reset=False``.
            return_state (bool): determines if the state is returned by the engine run (by default set to True)
            modes (Sequence[int]): a sequence of integers denoting the modes to be returned
                in the state object. If set to ``None``, all modes will be returned by default.
        """
        if reset_backend:
            self.reset_backend()

        if backend:
            # if backend is specified via a string and the engine already has that type of backend
            # loaded, then we should just use the existing backend
            if isinstance(backend, str) and self.backend is not None and self.backend._short_name == backend:
                pass
            else:
                # initialize a backend
                if isinstance(backend, str):
                    backend = load_backend(backend)
                    backend.begin_circuit(num_subsystems=self.init_modes, hbar=self.hbar, *args, **kwargs)

                self.backend = backend    #: BaseBackend: backend instance for executing the commands

        self.run_command_list(self.cmd_queue, **kwargs)

        if return_state:
            return self.return_state(modes=modes, **kwargs)


# The :class:`Command` instances in the program form a
# `strict partially ordered set <http://en.wikipedia.org/wiki/Partially_ordered_set#Strict_and_non-strict_partial_orders>`_
# in the sense that the order in which they have to be executed is usually not completely fixed.
# Specifically, operations acting on different subsystems always commute with each other.
# We denote :math:`a < b` if :math:`a` has to be executed before :math:`b`.
# Each strict partial order corresponds to a
# `directed acyclic graph <http://en.wikipedia.org/wiki/Directed_acyclic_graph>`_ (DAG),
# and the transitive closure of any DAG is a strict partial order.
# During the optimization three different (but equivalent) representations of the program are used.

# * Initially, the program is represented as a Command queue (list), listing the Commands in the temporal order they are applied.
# * The second representation, grid, essentially mimics a quantum circuit diagram.
#   It is a mapping from subsystem indices to lists of Commands touching that subsystem, where each list is temporally ordered.
# * Finally, the quantum circuit can be represented using a DAG by making each Command a node,
#   and drawing an edge from each Command to all its immediate followers along each wire it touches.
#   It can be converted back into a command queue by popping a maximal element until the graph is empty, that is, consuming it in a topological order.
#   Note that a topological order is not always unique, there may be several equivalent topological orders.

# .. currentmodule:: strawberryfields.engine.Engine

# The three representations can be converted to each other
# using the methods :func:`_list_to_grid`, :func:`_grid_to_DAG` and :func:`_DAG_to_list`.

    @staticmethod
    def _list_to_grid(ls):
        """Transforms a list of commands to a grid representation.

        The grid is a mapping from subsystem indices to lists of Commands touching that subsystem.
        The same Command will appear in each list that corresponds to one of its subsystems.

        Args:
          ls (list[Command]): program to be transformed
        Returns:
          Grid[Command]: transformed program
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

        In the DAG each node is a Command, and edges point from Commands to their dependents/followers.

        Args:
          grid (Grid[Command]): program to be transformed
        Returns:
          DAG[Command]: transformed program
        """
        DAG = nx.DiGraph()
        for key in grid:
            q = grid[key]
            _print_list(0, q)
            if len(q) > 0:
                DAG.add_node(q[0])  # add the first operation on the wire that does not depend on anything
            for i in range(1, len(q)):
                DAG.add_edge(q[i-1], q[i]) # add the edge between the operations, and the operation nodes themselves
        return DAG

    @staticmethod
    def _DAG_to_list(dag):
        """Transforms a command DAG to a list representation.

        The list contains the Commands in (one possible) topological (executable) order.

        Args:
          dag (DAG[Command]): program to be transformed
        Returns:
          list[Command]: transformed program
        """
        # sort the operation graph into topological order (dependants following the operations they depend on)
        temp = nx.algorithms.dag.topological_sort(dag)
        return list(temp)

    def optimize(self):
        """Try to simplify and optimize the program in the command queue.

        The simplifications are based on the algebraic properties of the gates,
        e.g., combining two consecutive gates of the same gate family.
        """
        # print('\n\nOptimizing...\nUnused inds: ', self.unused_indices)

        grid = self._list_to_grid(self.cmd_queue)
        #for k in grid:
        #    print('mode {}, len {}'.format(k, len(grid[k])))

        # try merging neighboring operations on each wire
        # todo the merging could also be done using the circuit DAG, which might be smarter (ns>1 would be easy)
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
                    if a.op.ns == b.op.ns and a.reg == b.reg:  # the ops must have equal size and act on the same wires
                        if a.op.ns != 1:
                            # ns > 1 is tougher. on no wire must there be anything between them, also deleting is more complicated
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
                except (TypeError, ValueError):
                    pass
                i += 1  # failed at merging the ops, move forward

        # convert the program back into a list (via a DAG)
        DAG = self._grid_to_DAG(grid)
        del grid
        self.cmd_queue = self._DAG_to_list(DAG)

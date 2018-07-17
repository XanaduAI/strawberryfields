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
starts with the user inputting a quantum circuit and ends with a backend that
could be e.g. a simulator, a hardware quantum processor, or a circuit drawer.

Syntactically, the compiler engine acts as the context for the quantum circuit.
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
--------------

.. currentmodule:: strawberryfields.engine.Engine

.. autosummary::
   register
   reset
   reset_queue
   append
   print_queue
   print_applied
   run
   return_state
   optimize

..
    The following are internal Engine methods. In most cases the user should not
    call these directly.
    .. autosummary::
       __enter__
       __exit__
       _add_subsystems
       _delete_subsystems
       _index_to_regref
       _test_regrefs
       _run_command_list
       _cmd_applied_all
       _retain_queue
       _list_to_grid
       _grid_to_DAG
       _DAG_to_list

Helper classes
--------------

.. currentmodule:: strawberryfields.engine

.. autosummary::
   Command
   RegRef
   RegRefTransform


Optimizer
---------

The purpose of the optimizer part of the compiler engine is to simplify the circuit
to make it cheaper and faster to execute. Different backends might require
different types of optimization, but in general the fewer operations a circuit has,
the faster it should run. The optimizer thus should convert the circuit into a
simpler but otherwise equivalent version of itself, preserving the probability
distributions of the measurement results. The optimization utilizes the abstract
algebraic properties of the gates, and in no point should require a
matrix representation.

Currently the optimization is somewhat simple, being able to merge neighboring
gates belonging to the same gate family and sharing the same set of subsystems,
and canceling pairs of a gate and its inverse.

.. currentmodule:: strawberryfields.engine



Exceptions
----------

.. autosummary::
   MergeFailure
   CircuitError
   RegRefError
   ~strawberryfields.backends.base.NotApplicableError


Code details
~~~~~~~~~~~~

"""
#pylint: disable=too-many-instance-attributes,attribute-defined-outside-init

# todo: Avoid issues with Engine contexts and threading,
# cf. _pydecimal.py in the python standard distribution.

from collections.abc import Sequence
import numbers
from functools import wraps
from itertools import chain

import networkx as nx

from .backends import load_backend
from .backends.base import NotApplicableError, BaseBackend


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
    @wraps(func)
    def wrapper(*args):
        "Unused docstring."
        return RegRefTransform(args, func)
    return wrapper


class CircuitError(RuntimeError):
    """Exception raised by :class:`Engine` when it encounters an illegal
    operation in the quantum circuit.

    E.g. trying to use a measurement result before it is available.
    """
    pass

class RegRefError(IndexError):
    """Exception raised by :class:`Engine` when it encounters an invalid register reference.

    E.g. trying to apply a gate to a nonexistent or deleted subsystem.
    """
    pass

class MergeFailure(RuntimeError):
    """Exception raised by :func:`~strawberryfields.ops.Operation.merge` when an
    attempted merge fails.

    E.g. trying to merge two gates of different families.
    """
    pass


class Command:
    """Represents a quantum operation applied on specific subsystems of the register.

    Args:
        op (Operation): quantum operation to apply
        reg (Sequence[RegRef]): Subsystems to which the operation is applied.
            Note that the order matters here.
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, op, reg, decomp=False):
        # accept a single RegRef in addition to a Sequence
        if not isinstance(reg, Sequence):
            reg = [reg]

        self.op = op   #: Operation: quantum operation to apply
        self.reg = reg  #: Sequence[RegRef]: subsystems to which the operation is applied
        self.decomp = decomp  #: bool: is this Command a part of a decomposition?

    def __str__(self):
        """Prints the command using proper Blackbird syntax."""
        temp = str(self.op)
        if temp[-1] == ' ':
            # HACK, trailing space means do not print anything more.
            return temp
        return '{} | ({})'.format(temp, ", ".join([str(rr) for rr in self.reg]))

    def get_dependencies(self):
        """Subsystems the command depends on.

        Combination of ``self.reg`` and ``self.op.extra_deps``.

        .. note:: ``extra_deps`` are used to ensure that the measurement
            happens before the result is used, but this is a bit too strict:
            two gates depending on the same measurement result but otherwise
            acting on different subsystems commute.

        Returns:
            set[RegRef]: set of subsystems the command depends on
        """
        deps = self.op.extra_deps | set(self.reg)
        return deps


class RegRef:
    """Quantum register reference.

    The objects of this class refer to a specific subsystem (mode) of
    a quantum register.

    Only one RegRef instance should exist per subsystem: :class:`Engine`
    keeps the authoritative mapping of subsystem indices to RegRef instances.
    Subsystem measurement results are stored in the "official" RegRef object.
    If other RegRefs objects referring to the same subsystem exist, they will
    not be updated. Once a RegRef is assigned a subsystem index it will never
    change, not even if the subsystem is deleted.

    The RegRefs are constructed in :func:`Engine._add_subsystems`.

    Args:
        ind (int): index of the register subsystem referred to
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, ind):
        self.ind = ind   #: int: subsystem index
        self.val = None  #: Real: measured eigenvalue. None if the subsystem has not been measured yet
        self.active = True  #: bool: True at construction, False when the subsystem is deleted

    def __str__(self):
        return 'q[{}]'.format(self.ind)


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
            it show in the engine queue as ``RegRefTransform(q[0], <lambda>)``.
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

        self.regrefs = refs   #: list[RegRef]: register references that act as parameters for the function
        self.func = func   #: None, function: the transformation itself, returns a scalar
        self.func_str = func_str

        if func is None and len(refs) > 1:
            raise ValueError('Identity transformation only takes one parameter.')

    def __str__(self):
        """Prints the RegRefTransform using Blackbird syntax."""
        temp = [str(r) for r in self.regrefs]
        rr = ', '.join(temp)

        if len(temp) > 1:
            rr = '[' +rr +']'

        if self.func is None:
            return 'RR({})'.format(rr)

        if self.func_str is None:
            return 'RR({}, {})'.format(rr, self.func.__name__)

        return 'RR({}, {})'.format(rr, self.func_str)

    def __format__(self, format_spec):
        return self.__str__() # pragma: no cover

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
            raise CircuitError("Trying to use a nonexistent measurement result "
                               "(e.g. before it can be measured).")
        if self.func is None:
            return temp[0]
        return self.func(*temp)


class Engine:
    r"""Quantum compiler engine.

    Acts as a context manager (and the context itself) for quantum circuits.
    The contexts may not be nested.

    .. currentmodule:: strawberryfields.engine.Engine

    The quantum circuit is inputted by using the :meth:`~strawberryfields.ops.Operation.__or__`
    methods of the quantum operations, which call the :meth:`append` method of the Engine.
    :meth:`append` checks that the register references are valid and then
    adds a new :class:`.Command` instance to the Engine command queue.

    :meth:`run` executes the command queue on the chosen backend, and makes
    measurement results available via the :class:`.RegRef` instances.

    The ``New`` and ``Del`` operations modify the quantum register itself by adding
    and deleting subsystems. The Engine keeps track of these changes as
    they enter the command queue in order to be able to report register
    reference errors as soon as they happen.

    The backend, however, only becomes aware of subsystem changes when
    the circuit is run. See :meth:`reset_queue` and :meth:`reset`.

    Args:
        num_subsystems (int): Number of subsystems in the quantum register.
    Keyword Args:
        hbar (float): The value of :math:`\hbar` to initialise the engine with, depending on the
            conventions followed. By default, :math:`\hbar=2`. See
            :ref:`conventions` for more details.
    """
    _current_context = None

    def __init__(self, num_subsystems, hbar=2):
        self.init_num_subsystems = num_subsystems  #: int: initial number of subsystems
        self.cmd_queue = []       #: list[Command]: command queue
        self.cmd_applied = []     #: list[list[Command]]: list of lists of commands that have been run (one list per run)
        self.backend = None       #: BaseBackend: backend for executing the quantum circuit
        self.hbar = hbar          #: float: Numerical value of hbar in the (implicit) units of position and momentum.
        # create mode references
        self.reg_refs = {}        #: dict[int->RegRef]: mapping from subsystem indices to corresponding RegRef objects
        self.unused_indices = set()  #: set[int]: created subsystem indices that have not been used (operated on) yet
        self._add_subsystems(num_subsystems)
        self._set_checkpoint()

    def __str__(self):
        """String representation."""
        return self.__class__.__name__ + '({} subsystems, {})'.format(self.num_subsystems, self.backend.__class__.__name__)

    def __enter__(self):
        """Enter the quantum circuit context for this engine."""
        if Engine._current_context is None:
            Engine._current_context = self
        else:
            raise RuntimeError('Only one context can be active at a time.')
        return self

    def __exit__(self, ex_type, ex_value, ex_tb):
        """Exit the quantum circuit context."""
        Engine._current_context = None

    #=================================================
    #  RegRef accounting
    #=================================================
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

    def _cmd_applied_all(self):
        """Return all applied commands in a single list.

        Returns:
            list[Command]: concatenation of all applied command lists
        """
        return list(chain.from_iterable(self.cmd_applied))

    def _set_checkpoint(self):
        """Make a RegRef checkpoint.

        Stores the activity state of the RegRefs in self.reg_refs, as well
        as self.unused_indices so that they can be restored in reset_queue().

        A RegRef checkpoint is made after (1) :meth:`__init__`, (2) :meth:`run`, (3) :meth:`reset`.
        """
        self._reg_refs_checkpoint = {k: r.active for k, r in self.reg_refs.items()}  #: dict[int->bool]: activity state of the current RegRefs
        self._unused_indices_checkpoint = self.unused_indices.copy()  #: set[int]: like unused_indices

    def _restore_checkpoint(self):
        """Restore a RegRef checkpoint.

        Called after command queue reset.
        Any RegRefs created after the checkpoint are made inactive and
        deleted from self.reg_refs.
        """
        for k, r in list(self.reg_refs.items()):  # convert to a list since we modify the dictionary during iteration
            if k in self._reg_refs_checkpoint:
                # existed in the last checkpoint
                r.active = self._reg_refs_checkpoint[k]
            else:
                # did not exist
                r.active = False
                del self.reg_refs[k]
        self.unused_indices = self._unused_indices_checkpoint.copy()

    def _add_subsystems(self, n):
        """Create new subsystem references, add them to the reg_ref dictionary.

        Does *not* ask the backend to create the new modes.
        To avoid discrepancies with the backend this method must not be called directly,
        but rather indirectly by using :class:`~strawberryfields.ops.New_modes`
        instances in the Engine context.

        .. note:: This is the only place where RegRef instances are constructed.

        Args:
            n (int): number of subsystems to add

        Returns:
            tuple[RegRef]: tuple of the newly added subsystem references
        """
        if not isinstance(n, numbers.Integral) or n < 1:
            raise ValueError('{} is not a nonnegative integer.'.format(n))
        first_unassigned_index = len(self.reg_refs)
        # create a list of RegRefs
        inds = [first_unassigned_index+i for i in range(n)]
        refs = tuple(RegRef(i) for i in inds)
        # add them to the index map
        for r in refs:
            self.reg_refs[r.ind] = r
        self.unused_indices.update(inds)  # all the newly reserved indices are unused for now
        return refs

    def _delete_subsystems(self, refs):
        """Delete existing subsystem references.

        Does *not* ask the backend to delete the modes right away. This only happens later
        when the corresponding Command is applied/executed.

        To avoid discrepancies with the backend this method must not be called directly,
        but rather indirectly by using :class:`~strawberryfields.ops.Delete` instances
        in the Engine context.

        Args:
          refs (Sequence[RegRef]): subsystems to delete
        """
        # NOTE: refs have already been through _test_regrefs() in append() and thus should be valid
        for r in refs:
            # mark the RegRef as deleted
            r.active = False
            #self.reg_refs[r.ind].active = False
        # NOTE: deleted indices are *not* removed from self.unused_indices


    def reset(self, keep_history=False, **kwargs):
        r"""Re-initialize the backend state to vacuum.

        Resets the state of the quantum circuit represented by the backend.

        * The original number of modes is restored.
        * All modes are reset to the vacuum state.
        * All known RegRefs are cleared of measured values.
        * A checkpoint is made of the initial register state.
        * If ``keep_history`` is False:

          * The command queue and the list of commands that have been run are cleared.
          * Any RegRefs for subsystems that were created after the init are rendered
            inactive and deleted.

        * If ``keep_history`` is True:

          * The command queue is prepended by the list of commands that have been run.
            The latter is then cleared. The purpose of this is to keep the circuit valid
            in the cases where previously run program segments have created or deleted
            subsystems, or made measurement on which the program in the command queue depends.
          * RegRef activity state is unchanged, active RegRefs remain valid.

        The keyword args are passed on to :meth:`strawberryfields.backends.base.BaseBackend.reset`.

        Args:
            keep_history (bool): allows the backend to be reset to the vacuum state while
                retaining the circuit history to be applied on the next call to :meth:`~.Engine.run`.
        """

        if self.backend is not None:
            self.backend.reset(**kwargs)

        # reset any measurement values stored in the RegRefs
        for r in self.reg_refs.values():
            r.val = None

        # make a checkpoint at the initial register state
        temp = range(self.init_num_subsystems)
        self._reg_refs_checkpoint = {k: True for k in temp}
        self._unused_indices_checkpoint = set(temp)

        # command queues and register state
        if keep_history:
            # insert all previously applied Commands in the front of the current
            # circuit to make it valid to run on the current state
            self._retain_queue()
        else:
            self.reset_queue()
        self.cmd_applied.clear()


    def _retain_queue(self):
        """Prepends the queue with previously applied commands"""
        self.cmd_queue[:0] = self._cmd_applied_all()


    def reset_queue(self):
        """Clear the command queue.

        Resets the currently queued circuit.

        * Clears all queued Commands, but does not reset the current state of the circuit.
        * self.reg_refs and self.unused_indices are restored to how they were at the last checkpoint.
        * Any extra RegRefs for subsystems that were created after last checkpoint are made inactive.
        """
        self.cmd_queue.clear()
        self._restore_checkpoint()


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
        rr = self.reg_refs[ind]
        if not rr.active:
            raise RegRefError('Subsystem {} has already been deleted.'.format(ind))
        return rr

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
            # either an integer or a RegRef
            if isinstance(rr, RegRef):
                # regref must be found in the dict values (the RegRefs are
                # compared using __eq__, which, since we do not define it, defaults to "is")
                if rr not in self.reg_refs.values():
                    raise RegRefError('Unknown RegRef.')
                if not rr.active:
                    raise RegRefError('Subsystem {} has already been deleted.'.format(rr.ind))
                if self.reg_refs[rr.ind] is not rr:
                    raise RegRefError('Should never happen!')
            elif isinstance(rr, numbers.Integral):
                rr = self._index_to_regref(rr)
            else:
                raise RegRefError('Subsystems can only be indexed using integers and RegRefs.')

            if rr in temp:
                raise RegRefError('Trying to act on the same subsystem more than once.')
            temp.append(rr)
        return temp

    def append(self, op, reg):
        """Append a quantum circuit command to the engine command queue.

        Args:
            op (Operation): quantum operation
            reg (list[int, RegRef]): register subsystem(s) to apply it to
        Returns:
            list[RegRef]: subsystem list as RegRefs
        """
        # test that the target subsystem references are ok
        reg = self._test_regrefs(reg)
        # also test possible RegRef dependencies
        self._test_regrefs(op.extra_deps)
        for rr in reg:
            # it's used now
            self.unused_indices.discard(rr.ind)
        self.cmd_queue.append(Command(op, reg))
        return reg

    def print_queue(self, print_fn=print):
        """Print the command queue.

        This contains the gates that will be applied on the next call to :meth:`run`.

        Args:
            print_fn (function): optional custom function to use for string printing.
        """
        for k in self.cmd_queue:
            print_fn(k)

    def print_applied(self, print_fn=print):
        """Print all commands applied to the qumodes since the backend was first initialized.

        This will be blank until the first call to :meth:`run`. The output may
        differ compared to :meth:`print_queue`, due to command decompositions
        and optimizations supported by the backend.

        Args:
            print_fn (function): optional custom function to use for string printing.
        """
        for k, r in enumerate(self.cmd_applied):
            print_fn('Run {}:'.format(k))
            for c in r:
                print_fn(c)

    def return_state(self, modes=None, **kwargs):
        """Return the backend state object.

        Args:
            modes (Sequence[int]): integers containing the modes to be returned.
                If none, all modes are returned.
        Returns:
            BaseState: object containing details and methods for manipulation of the returned circuit state
        """
        return self.backend.state(modes=modes, **kwargs)

    def _run_command_list(self, clist, **kwargs):
        """Execute the commands in the list.

        To avoid discrepancies with the backend this method must not be called directly,
        since it bypasses the circuit queue and the associated RegRef bookkeeping.

        Args:
            clist (list[Command]): command list to run

        Returns:
            list[Command]: commands that were applied to the backend
        """
        applied = []
        for cmd in clist:
            if cmd.op is None:
                # None represents an identity gate
                continue
            else:
                try:
                    # try to apply it to the backend
                    cmd.op.apply(cmd.reg, self.backend, hbar=self.hbar, **kwargs)
                    applied.append(cmd)
                except NotApplicableError:
                    # command is not applicable to the current backend type
                    raise NotApplicableError('The operation {} cannot be used with {}.'.format(
                        cmd.op, self.backend)) from None
                except NotImplementedError:
                    # command not directly supported by backend API, try a decomposition instead
                    try:
                        temp = cmd.op.decompose(cmd.reg)
                        # run the decomposition
                        applied_cmds = self._run_command_list(temp)
                        # todo should we store the decomposition or the original Command?
                        # if we change backends and re-run the circuit, the decomposition
                        # may not be valid or useful.
                        applied.extend(applied_cmds)
                    except NotImplementedError as err:
                        # simplify the error message by suppressing the previous exception
                        raise err from None
        return applied


    def run(self, backend=None, return_state=True, modes=None, apply_history=False, **kwargs):
        """Execute the circuit in the command queue by sending it to the backend.

        The backend state is updated, and a new RegRef checkpoint is created.
        The circuit queue is emptied, and its contents (possibly decomposed) are
        appended to self.cmd_applied.

        Args:
            backend (str, BaseBackend, None): Backend for executing the commands.
                Either a backend name ("gaussian", "fock", or "tf"), in which case it
                is loaded and initialized, or a BaseBackend instance, or None to keep
                the current backend.
            return_state (bool): If True, returns the state of the circuit after the
                circuit has been run like :meth:`return_state` was called.
            modes (Sequence[int]): Modes to be returned in the state object. If None, returns all modes.
            apply_history (bool): If True, all applied operations from the previous calls to
                eng.run are reapplied to the backend state, before applying recently
                queued quantum operations.
        """
        if apply_history:
            self._retain_queue()
            self.cmd_applied.clear()

        if backend is None:
            # keep the current backend
            if self.backend is None:
                # if backend does not exist, raise error
                raise ValueError("Please provide a simulation backend.") from None
        elif isinstance(backend, str):
            # if backend is specified via a string and the engine already has that type of backend
            # loaded, then we should just use the existing backend
            # pylint: disable=protected-access
            if self.backend is not None and self.backend._short_name == backend:
                pass
            else:
                # initialize a backend
                self.cmd_applied.clear()
                self.backend = load_backend(backend)
                self.backend.begin_circuit(num_subsystems=self.init_num_subsystems, hbar=self.hbar, **kwargs)
        else:
            if isinstance(backend, BaseBackend):
                self.cmd_applied.clear()
                self.backend = backend
            else:
                raise ValueError("Please provide a valid Strawberry Fields backend.") from None

        # todo unsuccessful run due to exceptions should ideally result in keeping
        # the command queue as is but bringing the backend back to the last checkpoint
        try:
            temp = self._run_command_list(self.cmd_queue, **kwargs)
        except Exception as e:
            # todo: reset the backend to the last checkpoint here
            raise e
        else:
            # command execution was successful, reset the queue
            self.cmd_applied.append(temp)
            self.cmd_queue.clear()
            self._set_checkpoint()

        if return_state:
            return self.return_state(modes=modes, **kwargs)

        return None


# The :class:`Command` instances in the circuit form a
# `strict partially ordered set
# <http://en.wikipedia.org/wiki/Partially_ordered_set#Strict_and_non-strict_partial_orders>`_
# in the sense that the order in which they have to be executed is usually not completely fixed.
# Specifically, operations acting on different subsystems always commute with each other.
# We denote :math:`a < b` if :math:`a` has to be executed before :math:`b`.
# Each strict partial order corresponds to a
# `directed acyclic graph <http://en.wikipedia.org/wiki/Directed_acyclic_graph>`_ (DAG),
# and the transitive closure of any DAG is a strict partial order.
# During the optimization three different (but equivalent) representations of the circuit are used.

# * Initially, the circuit is represented as a Command queue (list), listing the Commands in
#   the temporal order they are applied.
# * The second representation, grid, essentially mimics a quantum circuit diagram.
#   It is a mapping from subsystem indices to lists of Commands touching that subsystem,
#   where each list is temporally ordered.
# * Finally, the quantum circuit can be represented using a DAG by making each Command a node,
#   and drawing an edge from each Command to all its immediate followers along each wire it touches.
#   It can be converted back into a command queue by popping a maximal element until the graph
#   is empty, that is, consuming it in a topological order.
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

        In the DAG each node is a Command, and edges point from Commands to their dependents/followers.

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

        The list contains the Commands in (one possible) topological (executable) order.

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
        """Try to simplify and optimize the circuit in the command queue.

        The simplifications are based on the algebraic properties of the gates,
        e.g., combining two consecutive gates of the same gate family.
        """
        # print('\n\nOptimizing...\nUnused inds: ', self.unused_indices)

        grid = self._list_to_grid(self.cmd_queue)
        #for k in grid:
        #    print('mode {}, len {}'.format(k, len(grid[k])))

        # try merging neighboring operations on each wire
        # todo the merging could also be done using the circuit DAG, which
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
        del grid
        self.cmd_queue = self._DAG_to_list(DAG)

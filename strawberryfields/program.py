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
This module implements the :class:`.Program` class which acts as a representation for quantum circuits.

Quantum circuit representation
------------------------------

The :class:`.Command` instances in the circuit form a
`strict partially ordered set <http://en.wikipedia.org/wiki/Partially_ordered_set#Strict_and_non-strict_partial_orders>`_
in the sense that the order in which the operations have to be executed is usually not completely fixed.
For example, operations acting on different subsystems always commute with each other.
We denote :math:`a < b` if :math:`a` has to be executed before :math:`b`.
Each strict partial order corresponds to a
`directed acyclic graph <http://en.wikipedia.org/wiki/Directed_acyclic_graph>`_ (DAG),
and the transitive closure of any DAG is a strict partial order.
Three different (but equivalent) representations of the circuit are used.

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

.. currentmodule:: strawberryfields.program_utils

The three representations can be converted to each other
using the functions :func:`list_to_grid`, :func:`grid_to_DAG` and :func:`DAG_to_list`.

.. currentmodule:: strawberryfields.program
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init

import copy
import numbers
import warnings

import networkx as nx

import strawberryfields.circuitdrawer as sfcd
import strawberryfields.circuitspecs as specs
import strawberryfields.program_utils as pu
from .program_utils import Command, RegRef, CircuitError, RegRefError
from .parameters import FreeParameter, ParameterError


# for automodapi, do not include the classes that should appear under the top-level strawberryfields namespace
__all__ = []


class Program:
    """Represents a photonic quantum circuit.

    The program class provides a context manager for:

    * accessing the quantum register associated with the program, and
    * appending :doc:`/introduction/ops` to the program.

    Within the context, operations are appended to the program using the
    Python-embedded Blackbird syntax

    .. code-block:: python3

        ops.GateName(arg1, arg2, ...) | (q[i], q[j], ...)

    where ``ops.GateName`` is a valid quantum operation, and ``q`` is a list
    of the programs quantum modes.
    All operations are appended to the program in the order they are
    listed within the context.

    In addition, some 'meta-operations' (such as :func:`~.New` and :attr:`~.Del`)
    are provided to modify the programs quantum register itself by adding
    and deleting subsystems.

    .. note::

        Two programs can be run successively on the same engine if and only if
        the number of registers at the end of the first program matches the
        number of modes at the beginning of the second program.

        This can be enforced by constructing the second program as an explicit
        successor of the first, in which case the registers are directly copied over.

        When a Program is run or it obtains a successor, it is locked and no more
        operations can be appended to it.

    **Example:**

    .. code-block:: python3

        import strawberryfields as sf
        from strawberryfields import ops

        # create a 3 mode quantum program
        prog = sf.Program(3)

        with prog.context as q:
            ops.Sgate(0.54) | q[0]
            ops.Sgate(0.54) | q[1]
            ops.Sgate(0.54) | q[2]
            ops.BSgate(0.43, 0.1) | (q[0], q[2])
            ops.BSgate(0.43, 0.1) | (q[1], q[2])
            ops.MeasureFock() | q

    The currently active register references can be accessed using the :meth:`~Program.register` method.

    Args:
        num_subsystems (int, Program): Initial number of modes (subsystems) in the quantum register.
            Alternatively, another Program instance from which to inherit the register state.
        name (str): program name (optional)
    """

    def __init__(self, num_subsystems, name=None):
        #: str: program name
        self.name = name
        #: list[Command]: Commands constituting the quantum circuit in temporal order
        self.circuit = []
        #: bool: if True, no more Commands can be appended to the Program
        self.locked = False
        #: str, None: for compiled Programs, the short name of the target CircuitSpecs template, otherwise None
        self._target = None
        #: Program, None: for compiled Programs, this is the original, otherwise None
        self.source = None
        #: dict[str, Parameter]: free circuit parameters owned by this Program
        self.free_params = {}
        self.run_options = {}
        """dict[str, Any]: dictionary of default run options, to be passed to the engine upon
        execution of the program. Note that if the ``run_options`` dictionary is passed
        directly to :meth:`~.Engine.run`, it takes precedence over the run options specified
        here.
        """

        # create subsystem references
        # Program keeps track of the state of the quantum register using a dictionary of :class:`RegRef` objects.
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
            raise TypeError(
                "First argument must be either the number of subsystems or the parent Program."
            )

        # save the initial regref state
        #: dict[int, RegRef]: like reg_refs
        self.init_reg_refs = copy.deepcopy(self.reg_refs)
        #: set[int]: like unused_indices
        self.init_unused_indices = copy.copy(self.unused_indices)

    def __str__(self):
        """String representation."""
        return self.__class__.__name__ + "({}, {}->{} subsystems, compiled for '{}')".format(
            self.name, self.init_num_subsystems, self.num_subsystems, self.target
        )

    def __len__(self):
        """Program length.

        Returns:
            int: number of Commands in the program
        """
        return len(self.circuit)

    def print(self, print_fn=print):
        """Print the program contents using Blackbird syntax.

        **Example:**

        .. code-block:: python

            # create a 3 mode quantum program
            prog = sf.Program(3)

            with prog.context as q:
                ops.Sgate(0.54) | q[0]
                ops.Sgate(0.54) | q[1]
                ops.Sgate(0.54) | q[2]
                ops.BSgate(0.43, 0.1) | (q[0], q[2])
                ops.BSgate(0.43, 0.1) | (q[1], q[2])
                ops.MeasureFock() | q

        >>> prog.print()
        Sgate(0.54, 0) | (q[0])
        Sgate(0.54, 0) | (q[1])
        Sgate(0.54, 0) | (q[2])
        BSgate(0.43, 0.1) | (q[0], q[2])
        BSgate(0.43, 0.1) | (q[1], q[2])
        MeasureFock | (q[0], q[1], q[2])

        Args:
            print_fn (function): optional custom function to use for string printing
        """
        for k in self.circuit:
            print_fn(k)

    @property
    def context(self):
        """Syntactic sugar for defining a Program using the :code:`with` statement.

        The Program object itself acts as the context manager.
        """
        return self

    def __enter__(self):
        """Enter the context for this program.

        Returns:
            tuple[RegRef]: subsystem references
        """
        if pu.Program_current_context is None:
            pu.Program_current_context = self
        else:
            raise RuntimeError("Only one Program context can be active at a time.")
        return self.register

    def __exit__(self, ex_type, ex_value, ex_tb):
        """Exit the quantum circuit context."""
        pu.Program_current_context = None

    # =================================================
    #  RegRef accounting
    # =================================================
    @property
    def register(self):
        """Return a tuple of all the currently valid quantum modes.

        Returns:
            tuple[RegRef]: valid subsystem references
        """
        return tuple(r for r in self.reg_refs.values() if r.active)

    @property
    def num_subsystems(self):
        """Return the current number of valid quantum modes.

        Returns:
            int: number of currently valid register subsystems
        """
        return len(self.register)

    def _clear_regrefs(self):
        """Clear any measurement values stored in the RegRefs.

        Called by :class:`~.engine.Engine` when resetting the backend.
        """
        for r in self.reg_refs.values():
            r.val = None

    def _add_subsystems(self, n):
        """Create new subsystem references, add them to the reg_ref dictionary.

        To avoid discrepancies with the backend this method must not be called directly,
        but rather indirectly by using :func:`~strawberryfields.ops.New`
        in a Program context.

        .. note:: This is the only place where :class:`RegRef` instances are constructed.

        Args:
            n (int): number of subsystems to add
        Returns:
            tuple[RegRef]: tuple of the newly added subsystem references
        """
        if self.locked:
            raise CircuitError("The Program is locked, no new subsystems can be created.")
        if not isinstance(n, numbers.Integral) or n < 1:
            raise ValueError("Number of added subsystems {} is not a positive integer.".format(n))

        first_unassigned_index = len(self.reg_refs)
        # create a list of RegRefs
        inds = [first_unassigned_index + i for i in range(n)]
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
        but rather indirectly by using :class:`~strawberryfields.ops._Delete` instances
        in the Program context.

        Args:
          refs (Sequence[RegRef]): subsystems to delete
        """
        # NOTE: refs have already been through _test_regrefs() in append() and thus should be valid
        for r in refs:
            # mark the RegRef as deleted
            r.active = False
            # self.reg_refs[r.ind].active = False
        # NOTE: deleted indices are *not* removed from self.unused_indices

    def lock(self):
        """Finalize the program.

        When a Program is locked, no more Commands can be appended to it.
        The locking happens when the program is run, compiled, or a successor Program is constructed,
        in order to ensure that the RegRef state of the Program does not change anymore.
        """
        self.locked = True

    def can_follow(self, prev):
        """Check whether this program can follow the given program.

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
            raise RegRefError("Subsystem {} does not exist.".format(ind))
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
                    raise RegRefError("Unknown RegRef.")
                if self.reg_refs[rr.ind] is not rr:
                    raise RegRefError("RegRef state has become inconsistent.")
            elif isinstance(rr, numbers.Integral):
                rr = self._index_to_regref(rr)
            else:
                raise RegRefError("Subsystems can only be indexed using integers and RegRefs.")

            if not rr.active:
                raise RegRefError("Subsystem {} has already been deleted.".format(rr.ind))
            if rr in temp:
                raise RegRefError("Trying to act on the same subsystem more than once.")
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
            raise CircuitError("The Program is locked, no more Commands can be appended to it.")

        # test that the target subsystem references are ok
        reg = self._test_regrefs(reg)
        # also test possible Parameter-related dependencies
        self._test_regrefs(op.measurement_deps)
        for rr in reg:
            # it's used now
            self.unused_indices.discard(rr.ind)
        self.circuit.append(Command(op, reg))
        return reg

    def _linked_copy(self):
        """Create a copy of the Program, linked to the original.

        Both the original and the copy are :meth:`locked <lock>`, since they share their RegRefs.
        FreeParameters are also shared.

        Returns:
            Program: a copy of the Program
        """
        self.lock()
        p = copy.copy(self)  # shares RegRefs with the source
        # link to the original source Program
        if self.source is None:
            p.source = self
        else:
            p.source = self.source
        return p

    def compile(self, target, **kwargs):
        """Compile the program targeting the given circuit specification.

        Validates the program against the given target, making sure all the
        :doc:`/introduction/ops` used are accepted by the target specification.

        Additionally, depending on the target, the compilation may modify the quantum circuit
        into an equivalent circuit, e.g., by decomposing certain gates into sequences
        of simpler gates, or optimizing the gate ordering using commutation rules.

        **Example:**

        The ``gbs`` compile target will
        compile a circuit consisting of Gaussian operations and Fock measurements
        into canonical Gaussian boson sampling form.

        >>> prog2 = prog.compile('gbs')

        Args:
            target (str, ~strawberryfields.circuitspecs.CircuitSpecs): short name of the target
                circuit specification, or the specification object itself

        Keyword Args:
            optimize (bool): If True, try to optimize the program by merging and canceling gates.
                The default is False.
            warn_connected (bool): If True, the user is warned if the quantum circuit is not weakly
                connected. The default is True.

        Returns:
            Program: compiled program
        """
        if isinstance(target, specs.CircuitSpecs):
            db = target
            target = db.short_name
        elif target in specs.circuit_db:
            db = specs.circuit_db[target]()
        else:
            raise ValueError(
                "Could not find target '{}' in the Strawberry Fields circuit database.".format(
                    target
                )
            )

        if db.modes is not None:
            # subsystems may be created and destroyed, this is total number that has ever existed
            modes_total = len(self.reg_refs)
            if modes_total > db.modes:
                raise CircuitError(
                    "This program requires {} modes, but the target '{}' "
                    "only supports a {}-mode program".format(modes_total, target, db.modes)
                )

        seq = db.decompose(self.circuit)

        if kwargs.get("warn_connected", True):
            DAG = pu.list_to_DAG(seq)
            temp = nx.algorithms.components.number_weakly_connected_components(DAG)
            if temp > 1:
                warnings.warn("The circuit consists of {} disconnected components.".format(temp))

        # run optimizations
        if kwargs.get("optimize", False):
            seq = pu.optimize_circuit(seq)

        # does the circuit spec  have its own compilation method?
        if db.compile is not None:
            seq = db.compile(seq, self.register)

        # create the compiled Program
        compiled = self._linked_copy()
        compiled.circuit = seq
        compiled._target = db.short_name

        # get run options of compiled program
        # for the moment, shots is the only supported run option.
        if "shots" in kwargs:
            compiled.run_options["shots"] = kwargs["shots"]

        compiled.backend_options = {}
        if "cutoff_dim" in kwargs:
            compiled.backend_options["cutoff_dim"] = kwargs["cutoff_dim"]

        return compiled

    def optimize(self):
        """Simplify and optimize the program.

        The simplifications are based on the algebraic properties of the gates,
        e.g., combining two consecutive gates of the same gate family.

        Returns a copy of the program, sharing RegRefs with the original.

        See :func:`~strawberryfields.program_utils.optimize_circuit`.

        Returns:
            Program: optimized copy of the program
        """
        opt = self._linked_copy()
        opt.circuit = pu.optimize_circuit(self.circuit)
        return opt

    def draw_circuit(self, tex_dir="./circuit_tex", write_to_file=True):
        r"""Draw the circuit using the Qcircuit :math:`\LaTeX` package.

        This will generate the LaTeX code required to draw the quantum circuit
        diagram corresponding to the Program.


        The drawing of the following Xanadu supported operations are currently supported:

        .. rst-class:: docstable

        +-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |     Gate type     |                                                                            Supported gates                                                                             |
        +===================+========================================================================================================================================================================+
        | Single mode gates | :class:`~.Dgate`, :class:`~.Xgate`, :class:`~.Zgate`, :class:`~.Sgate`, :class:`~.Rgate`, :class:`~.Pgate`, :class:`~.Vgate`, :class:`~.Kgate`, :class:`~.Fouriergate` |
        +-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        | Two mode gates    | :class:`~.BSgate`, :class:`~.S2gate`, :class:`~.CXgate`, :class:`~.CZgate`, :class:`~.CKgate`                                                                          |
        +-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

        .. note::

            Measurement operations :class:`~.MeasureHomodyne`, :class:`~.MeasureHeterodyne`,
            and :class:`~.MeasureFock` are not currently supported.

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

    @property
    def target(self):
        """The target specification the program has been compiled against.

        If the program has not been compiled, this will return ``None``.

        Returns:
            str or None: the short name of the target CircuitSpecs template if
            compiled, otherwise None
        """
        return self._target

    def params(self, *args):
        """Create and access free circuit parameters.

        Returns the named free parameters. If a parameter does not exist yet, it is created and returned.

        Args:
            *args (tuple[str]): name(s) of the free parameters to access

        Returns:
            FreeParameter, list[FreeParameter]: requested parameter(s)
        """
        ret = []
        for a in args:
            if not isinstance(a, str):
                raise TypeError("Parameter names must be strings.")

            if a not in self.free_params:
                if self.locked:
                    raise CircuitError(
                        "The Program is locked, no more free parameters can be created."
                    )
                p = FreeParameter(a)
                self.free_params[a] = p
            else:
                p = self.free_params[a]
            ret.append(p)

        if len(ret) == 1:
            return ret[0]
        return ret

    def bind_params(self, binding):
        """Binds the free parameters of the program to the given values.

        Args:
            binding (dict[Union[str, FreeParameter], Any]): mapping from parameter names (or the
                parameters themselves) to parameter values

        Raises:
            ParameterError: tried to bind an unknown parameter
        """
        for k, v in binding.items():
            temp = self.free_params.get(k)  # it's a name
            if temp:
                temp.val = v
            elif k in self.free_params.values():  # it's a parameter
                k.val = v
            else:
                raise ParameterError("Unknown free parameter '{}'".format(k))

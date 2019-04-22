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
Quantum compiler engine
========================================================

**Module name:** :mod:`strawberryfields.engine`

.. currentmodule:: strawberryfields.engine

The :class:`Engine` class is responsible for communicating
quantum programs represented by :class:`.Program` objects
to a backend that could be e.g. a simulator, a hardware quantum processor, or a circuit drawer.

A typical use looks like
::

  prog = sf.Program(num_subsystems)
  with prog.context as q:
      Coherent(0.5)  | q[0]
      Vac            | q[1]
      Sgate(2)       | q[1]
      Dgate(0.5)     | q[0]
      BSgate(1)      | q[0:2]
      Dgate(0.5).H   | q[0]
      Measure        | q
  eng = sf.Engine(backend='fock')
  eng.run(prog, cutoff_dim=5)
  v1 = prog.register[1].val


Engine methods
--------------

.. currentmodule:: strawberryfields.engine.Engine

.. autosummary::
   run
   reset
   print_applied
   return_state

..
    The following are internal Engine methods. In most cases the user should not
    call these directly.
    .. autosummary::
       _run_program


.. currentmodule:: strawberryfields.engine

Exceptions
----------

.. autosummary::
   ~strawberryfields.backends.base.NotApplicableError


Code details
~~~~~~~~~~~~

"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init

# todo: Avoid issues with Engine contexts and threading,
# cf. _pydecimal.py in the python standard distribution.

from collections.abc import Sequence

from .backends import load_backend
from .backends.base import (NotApplicableError, BaseBackend)


class Engine:
    r"""Quantum program executor engine.

    :meth:`run` executes :class:`.Program` instances on the chosen backend, and makes
    measurement results available via the :class:`.RegRef` instances.

    Args:
        backend (str): name of the backend
        host    (str): URL of the remote backend host, or None if a local backend is used
    Keyword Args:
        hbar (float): The value of :math:`\hbar` to initialise the engine with, depending on the
            conventions followed. By default, :math:`\hbar=2`. See
            :ref:`conventions` for more details.
    """
    _current_context = None

    def __init__(self, backend, host=None, *, hbar=2, **kwargs):
        #: list[Program]: list of Programs that have been run
        self.run_progs = []
        #: str: URL of the remote backend host
        self.host = host
        #: float: numerical value of hbar in the (implicit) units of position * momentum
        self.hbar = hbar

        if isinstance(backend, str):
            #: str: short name of the backend
            self.backend_name = backend
            #: BaseBackend: backend for executing the quantum circuit
            self.backend = load_backend(backend)
        elif isinstance(backend, BaseBackend):
            self.backend_name = backend._short_name
            self.backend = backend
        else:
            raise TypeError('backend must be a string or a BaseBackend instance.')
        # TODO kwargs for backend?

    def __str__(self):
        """String representation."""
        return self.__class__.__name__ + '({})'.format(self.backend_name)

    def reset(self, **kwargs):
        r"""Re-initialize the backend state to vacuum.

        Resets the state of the quantum circuit represented by the backend.

        * The original number of modes is restored.
        * All modes are reset to the vacuum state.
        * All RegRefs of previously run Programs are cleared of measured values.
        * List of previously run Progams is cleared.

        Note that the reset does nothing to any Program objects in existence, beyond erasing the measured values.

        Keyword Args:
            kwargs: The keyword args are passed on to :meth:`strawberryfields.backends.base.BaseBackend.reset`.
        """
        self.backend.reset(**kwargs)

        for p in self.run_progs:
            p._clear_regrefs()
        self.run_progs.clear()

    def print_applied(self, print_fn=print):
        """Print all commands applied to the qumodes since the backend was first initialized.

        This will be blank until the first call to :meth:`run`. The output may
        differ compared to :meth:`Program.print`, due to command decompositions
        and optimizations made by :meth:`run`.

        Args:
            print_fn (function): optional custom function to use for string printing.
        """
        for k, r in enumerate(self.run_progs):
            print_fn('Run {}:'.format(k))
            r.print(print_fn)

    def return_state(self, modes=None, **kwargs):
        """Return the backend state object.

        Args:
            modes (Sequence[int]): Modes to be returned. If None, all modes are returned.
        Returns:
            BaseState: object containing details and methods for manipulation of the returned circuit state
        """
        return self.backend.state(modes=modes, **kwargs)

    def _run_program(self, prog, **kwargs):
        """Execute a program on the local backend.

        This method should not be called directly.

        Args:
            prog (Program): program to run
        Returns:
            list[Command]: commands that were applied to the backend
        """
        applied = []
        for cmd in prog.circuit:
            try:
                # try to apply it to the backend
                cmd.op.apply(cmd.reg, self.backend, hbar=self.hbar, **kwargs)
                applied.append(cmd)
            except NotApplicableError:
                # command is not applicable to the current backend type
                raise NotApplicableError('The operation {} cannot be used with {}.'.format(cmd.op, self.backend)) from None
            except NotImplementedError:
                # command not directly supported by backend API, try a decomposition instead
                try:
                    temp = cmd.op.decompose(cmd.reg)
                    # run the decomposition
                    applied_cmds = self._run_command_list(temp)
                    applied.extend(applied_cmds)
                except NotImplementedError as err:
                    # simplify the error message by suppressing the previous exception
                    raise err from None
        return applied

    def run(self, program, return_state=True, modes=None, compile=True, **kwargs):
        """Execute the given program by sending it to the backend.

        * The backend state is updated.
        * The executed Program(s) are appended to self.run_progs.

        Args:
            program (Program, Sequence[Program]): quantum circuit(s) to run
            return_state (bool): If True, returns the state of the circuit after the
                circuit has been run like :meth:`return_state` was called.
            modes (Sequence[int]): Modes to be returned in the state object. If None, returns all modes.
            compile (bool): if True, compile the Program instances before sending them to the backend
        """
        if not isinstance(program, Sequence):
            program = [program]

        if not self.run_progs:
            # initialize the backend
            self.backend.begin_circuit(num_subsystems=program[0].init_num_subsystems, hbar=self.hbar, **kwargs)

        # TODO unsuccessful run due to exceptions should ideally have no effect on the backend state (state checkpoints?)
        try:
            prev = None  # previous program segment
            for p in program:
                if self.run_progs:
                    prev = self.run_progs[-1]

                if prev is not None and not p.can_follow(prev):
                    raise RuntimeError("Register mismatch after Program {}, '{}'.".format(len(self.run_progs)-1, prev.name))

                # if the program hasn't been compiled for this backend, do it now
                if compile and p.backend != self.backend._short_name:
                    p = p.compile(self.backend._short_name)
                p.lock()

                # TODO handle remote backends here, store measurement results in the RegRefs or maybe in the Engine object.
                temp = self._run_program(p, **kwargs)
                self.run_progs.append(p)
        except Exception as e:
            # TODO reset the backend to the last checkpoint here?
            raise e
        else:
            # program execution was successful
            pass

        if return_state:
            return self.return_state(modes=modes, **kwargs)

        return None

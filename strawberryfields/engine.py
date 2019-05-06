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
Quantum program executor engine
===============================

**Module name:** :mod:`strawberryfields.engine`

.. currentmodule:: strawberryfields.engine

The :class:`BaseEngine` subclasses are responsible for communicating
quantum programs represented by :class:`.Program` objects
to a backend that could be e.g., a simulator, a hardware quantum processor, or a circuit drawer,
and returning the result to the user.

A typical use looks like

.. include:: example_use.rst


Classes
-------

.. autosummary::
   BaseEngine
   LocalEngine
   Result


BaseEngine methods
------------------

.. currentmodule:: strawberryfields.engine.BaseEngine

.. autosummary::
   run
   print_applied
   reset

..
    The following are internal BaseEngine methods. In most cases the user should not
    call these directly.
    .. autosummary::
       _init_backend
       _run_program


.. currentmodule:: strawberryfields.engine

Exceptions
----------

.. autosummary::
   ~strawberryfields.backends.base.NotApplicableError


Code details
~~~~~~~~~~~~

"""

import abc
from collections.abc import Sequence

from .backends import load_backend
from .backends.base import (NotApplicableError, BaseBackend)


class Result:
    """Result of a quantum computation.

    Represents the results of the execution of a quantum program.
    Returned by :meth:`.BaseEngine.run`.
    """
    def __init__(self, samples):
        #: BaseState: quantum state object returned by a local backend, if any
        self.state = None
        #: List[List[Number]]: measurement samples, shape == (modes, shots)
        self.samples = samples

    def __str__(self):
        return self.__class__.__name__ + ': {} subsystems, state: {}'.format(len(self.samples), self.state) + '\n' + str(self.samples)


class BaseEngine(abc.ABC):
    r"""ABC for quantum program executor engines.

    Args:
        backend (str): backend short name
        backend_options (Dict[str, Any]): keyword arguments for the backend
    """
    def __init__(self, backend, backend_options={}):
        #: str: short name of the backend
        self.backend_name = backend
        #: Dict[str, Any]: keyword arguments for the backend
        self.backend_options = backend_options.copy()  # dict is mutable
        #: List[Program]: list of Programs that have been run
        self.run_progs = []
        #: List[List[Number]]: latest measurement results, shape == (modes, shots)
        self.samples = None

    @abc.abstractmethod
    def __str__(self):
        """String representation."""

    @abc.abstractmethod
    def reset(self,  backend_options):
        r"""Re-initialize the quantum computation.

        Resets the state of the engine and the quantum circuit represented by the backend.

        * The original number of modes is restored.
        * All modes are reset to the vacuum state.
        * All RegRefs of previously run Programs are cleared of measured values.
        * List of previously run Progams is cleared.

        Note that the reset does nothing to any Program objects in existence, beyond erasing the measured values.

        Args:
           backend_options (Dict[str, Any]): keyword arguments for the backend,
              overriding the old values
        """
        self.backend_options.update(backend_options)
        for p in self.run_progs:
            p._clear_regrefs()
        self.run_progs.clear()
        self.samples = None

    def print_applied(self, print_fn=print):
        """Print all the Programs run since the backend was initialized.

        This will be blank until the first call to :meth:`run`. The output may
        differ compared to :meth:`Program.print`, due to command decompositions
        and optimizations made by :meth:`run`.

        Args:
            print_fn (function): optional custom function to use for string printing.
        """
        for k, r in enumerate(self.run_progs):
            print_fn('Run {}:'.format(k))
            r.print(print_fn)

    @abc.abstractmethod
    def _init_backend(self, init_num_subsystems):
        """Initialize the backend.

        Args:
            init_num_subsystems (int): number of subsystems the backend is initialized to
        """

    @abc.abstractmethod
    def _run_program(self, prog, **kwargs):
        """Execute a program on the backend.

        This method should not be called directly.

        Args:
            prog (Program): program to run
        Returns:
            list[Command]: commands that were applied to the backend
        """

    def run(self, program, *, modes=None, compile=True, **kwargs):
        """Execute the given program by sending it to the backend.

        * Each executed Program is compiled and optimized
        * The compiled versions of the executed Programs are appended to self.run_progs.
        * The backend state is updated.
        * The result of the computation is returned.
        * Additionally, the latest measurement results of each subsystem are stored
          in the :class:`.RegRef` instances of the corresponding Program.

        Args:
            program (Program, Sequence[Program]): quantum programs to run
            modes (None, Sequence[int]): Modes to be returned in the state object.
                An empty sequence [] means no state object is returned. None returns all the modes.
            compile (bool): If True, compile the Program instances before sending them to the backend.

        The keyword args are passed to :meth:`_run_program`.

        Returns:
            Result: results of the computation
        """
        if not isinstance(program, Sequence):
            program = [program]

        try:
            prev = self.run_progs[-1] if self.run_progs else None  # previous program segment
            for p in program:
                if prev is None:
                    # initialize the backend
                    self._init_backend(p.init_num_subsystems)
                else:
                    if not p.can_follow(prev):
                        raise RuntimeError("Register mismatch after Program {}, '{}'.".format(len(self.run_progs)-1, prev.name))

                    # todo copy vals from self.samples in case same prog is used in two engines....
                    # copy the measured values in RegRefs over from prev
                    for k, v in prev.reg_refs.items():
                        p.reg_refs[k].val = v.val

                # if the program hasn't been compiled for this backend, do it now
                if compile and p.backend != self.backend_name:
                    p = p.compile(self.backend_name)
                p.lock()

                self._run_program(p, **kwargs)
                self.run_progs.append(p)
                prev = p
        except Exception as e:
            raise e
        else:
            # program execution was successful
            pass

        if prev:
            # store the latest measurement results
            self.samples = [prev.reg_refs[k].val for k in sorted(prev.reg_refs)]

        return Result(self.samples)


class LocalEngine(BaseEngine):
    """Local quantum program executor engine.

    Executes :class:`.Program` instances on the chosen local backend, and makes
    the results available via :class:`.Result`.

    Args:
        backend (str, BaseBackend): name of the backend, or a pre-constructed backend instance
        backend_options (Dict[str, Any]): keyword arguments to be passed to the backend
    """
    def __init__(self, backend, *, backend_options={}):
        super().__init__(backend, backend_options)

        if isinstance(backend, str):
            self.backend_name = backend
            #: BaseBackend: backend for executing the quantum circuit
            self.backend = load_backend(backend)
        elif isinstance(backend, BaseBackend):
            self.backend_name = backend._short_name
            self.backend = backend
        else:
            raise TypeError('backend must be a string or a BaseBackend instance.')

    def __str__(self):
        return self.__class__.__name__ + '({})'.format(self.backend_name)

    def reset(self, backend_options={}):
        r"""Re-initialize the backend state to vacuum.

        Resets the state of the quantum circuit represented by the backend.

        * The original number of modes is restored.
        * All modes are reset to the vacuum state.
        * All RegRefs of previously run Programs are cleared of measured values.
        * List of previously run Progams is cleared.

        Note that the reset does nothing to any Program objects in existence, beyond erasing the measured values.

        The keyword args are passed on to :meth:`strawberryfields.backends.base.BaseBackend.reset`.
        """
        super().reset(backend_options)
        self.backend_options.pop('batch_size', None)  # HACK to make tests work for now
        self.backend.reset(**self.backend_options)
        # TODO should backend.reset and backend.begin_circuit be combined?

    def _init_backend(self, init_num_subsystems):
        self.backend.begin_circuit(init_num_subsystems, **self.backend_options)

    def _run_program(self, prog, **kwargs):
        applied = []
        for cmd in prog.circuit:
            try:
                # try to apply it to the backend
                res = cmd.op.apply(cmd.reg, self.backend, **kwargs)  # TODO we could also handle storing the measurement vals here
                applied.append(cmd)
            except NotApplicableError:
                # command is not applicable to the current backend type
                raise NotApplicableError('The operation {} cannot be used with {}.'.format(cmd.op, self.backend)) from None
            except NotImplementedError:
                # command not directly supported by backend API, try a decomposition instead
                try:
                    temp = cmd.op.decompose(cmd.reg)
                    # run the decomposition
                    applied_cmds = self._run_program(temp, **kwargs)
                    applied.extend(applied_cmds)
                except NotImplementedError as err:
                    # simplify the error message by suppressing the previous exception
                    raise err from None
        return applied

    def run(self, program, *, modes=None, compile=True, **kwargs):
        result = super().run(program, modes=modes, compile=compile, **kwargs)
        if isinstance(modes, Sequence) and not modes:
            # empty sequence
            pass
        else:
            result.state = self.backend.state(modes=modes)  # TODO tfbackend.state can use kwargs
        return result


Engine = LocalEngine  # alias for backwards compatibility

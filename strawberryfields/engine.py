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
Execution engine
================

**Module name:** :mod:`strawberryfields.engine`

.. currentmodule:: strawberryfields.engine

This module implements :class:`BaseEngine` and its subclasses that are responsible for
communicating quantum programs represented by :class:`.Program` objects
to a backend that could be e.g., a simulator, a hardware quantum processor,
or a circuit drawer, and returning the result to the user.
One can think of each BaseEngine instance as a separate quantum computation.

A typical use looks like

.. include:: example_use.rst


Classes
-------

.. autosummary::
   BaseEngine
   LocalEngine
   Result


LocalEngine methods
-------------------

.. currentmodule:: strawberryfields.engine.LocalEngine

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
       _run


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
import numbers

import numpy as np

from strawberryfields.backends import load_backend
from strawberryfields.backends.base import (NotApplicableError, BaseBackend)


class Result:
    """Result of a quantum computation.

    Represents the results of the execution of a quantum program.
    Returned by :meth:`.BaseEngine.run`.

    Args:
        samples (Dict[int, array[Number]]): dictionary of measurement samples;
            keys are mode numbers, entries have shape ``([batch_size,] shots,)``
    """
    def __init__(self, samples):
        #: BaseState: quantum state object returned by a local backend, if any
        self.state = None

        #: Dict[int, array[Number]]
        self.samples = samples

        #: list[int]: list of measured modes
        self.measured_modes = list(samples.keys())

        # samples arrives as a dictionary, need to convert here to a multidimensional array
        if not samples:
            self.samples_array = np.zeros((0, 0))
        else:
            arrs = list(samples.values())
            #: array[Number]: measurement samples, with shape ``(modes, [batch_size,] shots)``
            self.samples_array = np.stack(arrs)  # gives an error if the arrs are not all the same shape


    def __str__(self):
        """String representation."""
        return 'Result: {} subsystems, state: {}\n samples: {}'.format(len(self.samples), self.state, self.samples)


class BaseEngine(abc.ABC):
    r"""ABC for quantum program executor engines.

    Args:
        backend (str): backend short name
        backend_options (Dict[str, Any]): keyword arguments for the backend
    """
    def __init__(self, backend, backend_options=None):
        if backend_options is None:
            backend_options = {}

        #: str: short name of the backend
        self.backend_name = backend
        #: Dict[str, Any]: keyword arguments for the backend
        self.backend_options = backend_options.copy()  # dict is mutable
        #: List[Program]: list of Programs that have been run
        self.run_progs = []
        #: Dict[int, array[Number]]: measurement results for measured modes, array shape == ([batch_size,] shots,)
        self.samples = None

    @abc.abstractmethod
    def __str__(self):
        """String representation."""

    @abc.abstractmethod
    def reset(self, backend_options):
        r"""Re-initialize the quantum computation.

        Resets the state of the engine and the quantum circuit represented by the backend.

        * The original number of modes is restored.
        * All modes are reset to the vacuum state.
        * All RegRefs of previously run Programs are cleared of measured values.
        * List of previously run Progams is cleared.

        Note that the reset does nothing to any Program objects in existence, beyond erasing the measured values.

        Args:
           backend_options (Dict[str, Any]): keyword arguments for the backend,
              updating (overriding) old values
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
        """Execute a single program on the backend.

        This method should not be called directly.

        Args:
            prog (Program): program to run
            shots (int): number of independent measurement evaluations for this program
        Returns:
            list[Command]: commands that were applied to the backend
        """

    def _run(self, program, *, shots=1, compile_options={}, **kwargs):
        """Execute the given programs by sending them to the backend.

        If multiple Programs are given they will be executed sequentially as
        parts of a single computation.
        For each :class:`Program` instance given as input, the following happens:

        * The Program instance is compiled and optimized for the target backend.
        * The compiled program is executed on the backend.
        * The measurement results of each subsystem (if any) are stored
          in the :class:`.RegRef` instances of the corresponding Program, as well as in :attr:`~.samples`.
        * The compiled program is appended to self.run_progs.

        Finally, the result of the computation is returned.

        Args:
            program (Program, Sequence[Program]): quantum programs to run
            shots (int): number of times the program measurement evaluation is repeated
            compile_options (Dict[str, Any]): keyword arguments for :meth:`.Program.compile`

        The ``kwargs`` keyword arguments are passed to :meth:`_run_program`.

        Returns:
            Result: results of the computation
        """

        if not isinstance(program, Sequence):
            program = [program]

        if not isinstance(shots, numbers.Integral) or shots < 1:
            raise ValueError("Number of shots must be an integer greater than zero.")
        kwargs["shots"] = shots
        # Note: by putting ``shots`` into keyword arguments, it allows for the
        # signatures of methods in Operations to remain cleaner, since only
        # Measurements need to know about shots

        prev = self.run_progs[-1] if self.run_progs else None  # previous program segment
        for p in program:
            if prev is None:
                # initialize the backend
                self._init_backend(p.init_num_subsystems)
            else:
                # there was a previous program segment
                if not p.can_follow(prev):
                    raise RuntimeError("Register mismatch: program {}, '{}'.".format(len(self.run_progs), p.name))

                # Copy the latest measured values in the RegRefs of p.
                # We cannot copy from prev directly because it could be used in more than one engine.
                for r in p.reg_refs.values():
                    r.val = self.samples.get(r.ind, None)

            # if the program hasn't been compiled for this backend, do it now
            if p.backend != self.backend_name:
                p = p.compile(self.backend_name, **compile_options) # TODO: shots might be relevant for compilation?
            p.lock()


            self._run_program(p, **kwargs)
            self.run_progs.append(p)
            # store the latest measurement results
            self.samples = {r.ind: r.val for r in p.reg_refs.values() if r.val is not None}
            prev = p

        return Result(self.samples.copy())


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

    def reset(self, backend_options=None):
        if backend_options is None:
            backend_options = {}

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
                cmd.op.apply(cmd.reg, self.backend, **kwargs)  # NOTE we could also handle storing measured vals here
                applied.append(cmd)
            except NotApplicableError:
                # NOTE: This exception has largely been supplanted by the compilation process, but it's good
                # to have it as a backup if the backend circuit template has a bug.
                # command is not applicable to the current backend type
                raise NotApplicableError('The operation {} cannot be used with {}.'.format(cmd.op, self.backend)) from None
            except NotImplementedError as ex:
                # command not supported by backend API
                # amend the error message
                raise NotImplementedError(
                    "The operation {} has not been implemented in the '{}' backend for the arguments {}:  ".format(cmd.op, self.backend_name, kwargs)
                    +ex.args[0]
                ) from None
        return applied

    def run(self, program, *, shots=1, compile_options={}, modes=None, state_options={}, **kwargs):
        """Execute the given programs by sending them to the backend.

        Extends :meth:`BaseEngine._run`.

        Args:
            program (Program, Sequence[Program]): quantum programs to run
            shots (int): number of times the program measurement evaluation is repeated
            compile_options (Dict[str, Any]): keyword arguments for :meth:`.Program.compile`
            modes (None, Sequence[int]): Modes to be returned in the ``Result.state`` :class:`.BaseState` object.
                An empty sequence [] means no state object is returned. None returns all the modes.
            state_options (Dict[str, Any]): keyword arguments for :meth:`.BaseBackend.state`

        The ``kwargs`` keyword arguments are passed to :meth:`_run_program`.

        Returns:
            Result: results of the computation
        """

        result = super()._run(program, shots=shots, compile_options=compile_options, **kwargs)
        if isinstance(modes, Sequence) and not modes:
            # empty sequence
            pass
        else:
            result.state = self.backend.state(modes, **state_options)  # tfbackend.state can use kwargs
        return result


Engine = LocalEngine  # alias for backwards compatibility

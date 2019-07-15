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
to a backend that could be e.g., a simulator or a hardware quantum processor.
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
from numpy import stack, shape

from .backends import load_backend
from .backends.base import (NotApplicableError, BaseBackend)


class Result:
    """Result of a quantum computation.

    Represents the results of the execution of a quantum program.
    Returned by :meth:`.BaseEngine.run`.
    """
    def __init__(self, samples):
        self._state = None

        # ``samples`` arrives as a list of arrays, need to convert here to a multidimensional array
        if len(shape(samples)) > 1:
            samples = stack(samples, 1)
        self._samples = samples

    @property
    def samples(self):
        """Measurement samples.

        Returned measurement samples will have shape ``(modes,)``. If multiple
        shots are requested during execution, the returned measurement samples
        will instead have shape ``(shots, modes)``.

        Returns:
            array[array[float, int]]: measurement samples returned from
            program execution
        """
        return self._samples

    @property
    def state(self):
        """The quantum state object.

        The quantum state object contains details and methods
        for manipulation of the final circuit state.

        .. note::

            Only local simulators will return a state object. Remote
            simulators and hardware backends will return
            :attr:`measurement samples <Result.samples>`,
            but the return value of ``state`` will be ``None``.

        Returns:
            BaseState: quantum state returned from program execution
        """
        return self._state

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
        #: List[List[Number]]: latest measurement results, shape == (modes, shots)
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
        Returns:
            list[Command]: commands that were applied to the backend
        """

    def _run(self, program, *, compile_options=None, **kwargs):
        """Execute the given programs by sending them to the backend.

        If multiple Programs are given they will be executed sequentially as
        parts of a single computation.
        For each :class:`.Program` instance given as input, the following happens:

        * The Program instance is compiled for the target backend.
        * The compiled program is executed on the backend.
        * The measurement results of each subsystem (if any) are stored in the :class:`.RegRef`
          instances of the corresponding Program, as well as in :attr:`~BaseEngine.samples`.
        * The compiled program is appended to :attr:`~BaseEngine.run_progs`.

        Finally, the result of the computation is returned.

        Args:
            program (Program, Sequence[Program]): quantum programs to run
            compile_options (None, Dict[str, Any]): keyword arguments for :meth:`.Program.compile`

        The ``kwargs`` keyword arguments are passed to the backend API calls via :meth:`Operation.apply`.

        Returns:
            Result: results of the computation
        """

        def _broadcast_nones(val, dim):
            """Helper function to ensure register values have same shape, even if not measured"""
            if val is None and dim > 1:
                return [None] * dim
            return val

        compile_options = compile_options or {}

        if not isinstance(program, Sequence):
            program = [program]

        kwargs.setdefault("shots", 1)
        # NOTE: by putting ``shots`` into keyword arguments, it allows for the
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
                for k, v in enumerate(self.samples):
                    p.reg_refs[k].val = v

            # if the program hasn't been compiled for this backend, do it now
            target = self.backend.circuit_spec
            if target is not None and p.target != target:
                p = p.compile(target, **compile_options)
            p.lock()

            self._run_program(p, **kwargs)
            self.run_progs.append(p)
            # store the latest measurement results
            self.samples = [_broadcast_nones(p.reg_refs[k].val, kwargs["shots"]) for k in sorted(p.reg_refs)]
            prev = p

        return Result(self.samples.copy())


class LocalEngine(BaseEngine):
    """Local quantum program executor engine.

    Executes :class:`.Program` instances on the chosen local backend, and makes
    the results available via :class:`.Result`.

    Args:
        backend (str, BaseBackend): short name of the backend, or a pre-constructed backend instance
        backend_options (None, Dict[str, Any]): keyword arguments to be passed to the backend
    """
    def __init__(self, backend, *, backend_options=None):
        backend_options = backend_options or {}
        super().__init__(backend, backend_options)

        if isinstance(backend, str):
            self.backend_name = backend
            #: BaseBackend: backend for executing the quantum circuit
            self.backend = load_backend(backend)
        elif isinstance(backend, BaseBackend):
            self.backend_name = backend.short_name
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
                # command is not applicable to the current backend type
                raise NotApplicableError('The operation {} cannot be used with {}.'.format(cmd.op, self.backend)) from None
            except NotImplementedError:
                # command not directly supported by backend API
                raise NotImplementedError('The operation {} has not been implemented in {} for the arguments {}.'.format(cmd.op, self.backend, kwargs)) from None
        return applied

    def run(self, program, *, compile_options=None, run_options=None):
        """Execute the given programs by sending them to the backend.

        Extends :meth:`BaseEngine._run`.

        Args:
            program (Program, Sequence[Program]): quantum programs to run
            compile_options (None, Dict[str, Any]): keyword arguments for :meth:`.Program.compile`
            run_options (None, Dict[str, Any]): keyword arguments that are needed by the backend during execution
                e.g., in :meth:`Operation.apply` or :meth:`.BaseBackend.state`

        Returns:
            Result: results of the computation

        ``run_options`` can contain the following:

        Keyword Args:
            shots (int): number of times the program measurement evaluation is repeated
            modes (None, Sequence[int]): Modes to be returned in the ``Result.state`` :class:`.BaseState` object.
                ``None`` returns all the modes (default). An empty sequence means no state object is returned.
            eval (bool): If False, the backend returns unevaluated tf.Tensors instead of
                numbers/arrays for returned measurement results and state (default: True). TF backend only.
            session (tf.Session): TensorFlow session, used when evaluating returned measurement results and state.
                TF backend only.
            feed_dict (dict[str, Any]): TensorFlow feed dictionary, used when evaluating returned measurement results
                and state. TF backend only.
        """
        compile_options = compile_options or {}
        temp_run_options = {}

        if isinstance(program, Sequence):
            # succesively update all run option defaults.
            # the run options of successive programs
            # overwrite the run options of previous programs
            # in the list
            [temp_run_options.update(p.run_options) for p in program]
        else:
            # single program to execute
            temp_run_options.update(program.run_options)

        temp_run_options.update(run_options or {})
        temp_run_options.setdefault("shots", 1)
        temp_run_options.setdefault('modes', None)

        # avoid unexpected keys being sent to Operations
        eng_run_keys = ["eval", "session", "feed_dict", "shots"]
        eng_run_options = {key: temp_run_options[key] for key in temp_run_options.keys() & eng_run_keys}

        result = super()._run(program, compile_options=compile_options, **eng_run_options)

        modes = temp_run_options["modes"]

        if modes is None or modes:
            # state object requested
            # session and feed_dict are needed by TF backend both during simulation (if program
            # contains measurements) and state object construction.
            result._state = self.backend.state(**temp_run_options)

        return result


Engine = LocalEngine  # alias for backwards compatibility

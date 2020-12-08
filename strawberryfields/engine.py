# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

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
This module implements :class:`BaseEngine` and its subclasses that are responsible for
communicating quantum programs represented by :class:`.Program` objects
to a backend that could be e.g., a simulator or a hardware quantum processor.
One can think of each BaseEngine instance as a separate quantum computation.
"""
import abc
import collections.abc
import time
from typing import Optional

import numpy as np

from strawberryfields.api import Connection, Job, Result
from strawberryfields.api.job import FailedJobError
from strawberryfields.logger import create_logger
from strawberryfields.program import Program

from .backends import load_backend
from .backends.base import BaseBackend, NotApplicableError

# for automodapi, do not include the classes that should appear under the top-level strawberryfields namespace
__all__ = ["BaseEngine", "LocalEngine"]


class BaseEngine(abc.ABC):
    r"""Abstract base class for quantum program executor engines.

    Args:
        backend (str): backend short name
        backend_options (Dict[str, Any]): keyword arguments for the backend
    """

    def __init__(self, backend, *, backend_options=None):
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
        self.all_samples = None

        if isinstance(backend, str):
            self.backend_name = backend
            self.backend = load_backend(backend)
        elif isinstance(backend, BaseBackend):
            self.backend_name = backend.short_name
            self.backend = backend
        else:
            raise TypeError("backend must be a string or a BaseBackend instance.")

    @abc.abstractmethod
    def __str__(self):
        """String representation."""

    @abc.abstractmethod
    def reset(self, backend_options):
        r"""Re-initialize the quantum computation.

        Resets the state of the engine and the quantum circuit represented by the backend.

        * The original number of modes is restored.
        * All modes are reset to the vacuum state.
        * All registers of previously run Programs are cleared of measured values.
        * List of previously run Progams is cleared.

        Note that the reset does nothing to any Program objects in existence, beyond
        erasing the measured values.

        **Example:**

        .. code-block:: python

            # create a program
            prog = sf.Program(3)

            with prog.context as q:
                ops.Sgate(0.543) | q[1]
                ops.BSgate(0.6, 0.1) | (q[2], q[0])

            # create an engine
            eng = sf.Engine("gaussian")

        Running the engine with the above program will
        modify the state of the statevector simulator:

        >>> eng.run(prog)
        >>> eng.backend.is_vacuum()
        False

        Resetting the engine will return the backend simulator
        to the vacuum state:

        >>> eng.reset()
        >>> eng.backend.is_vacuum()
        True

        .. note:: The ``reset()`` method only applies to statevector backends.

        Args:
            backend_options (Dict[str, Any]): keyword arguments for the backend,
                updating (overriding) old values
        """
        self.backend_options.update(backend_options)
        for p in self.run_progs:
            p._clear_regrefs()
        self.run_progs.clear()
        self.samples = None
        self.all_samples = None

    def print_applied(self, print_fn=print):
        """Print all the Programs run since the backend was initialized.

        This will be blank until the first call to :meth:`~.LocalEngine.run`. The output may
        differ compared to :meth:`.Program.print`, due to backend-specific
        device compilation performed by :meth:`~.LocalEngine.run`.

        **Example:**

        .. code-block:: python

            # create a program
            prog = sf.Program(2)

            with prog.context as q:
                ops.S2gate(0.543) | (q[0], q[1])

            # create an engine
            eng = sf.Engine("gaussian")

        Initially, the engine will have applied no operations:

        >>> eng.print_applied()
        None

        After running the engine, we can now see the quantum
        operations that were applied:

        >>> eng.run(prog)
        >>> eng.print_applied()
        Run 0:
        BSgate(0.7854, 0) | (q[0], q[1])
        Sgate(0.543, 0) | (q[0])
        Sgate(0.543, 0).H | (q[1])
        BSgate(0.7854, 0).H | (q[0], q[1])

        Note that the :class:`~.S2gate` has been decomposed into
        single-mode squeezers and beamsplitters, which are supported
        by the ``'gaussian'`` backend.

        Subsequent program runs can also be viewed:

        .. code-block:: python

            # a second program
            prog2 = sf.Program(2)
            with prog2.context as q:
                ops.S2gate(0.543) | (q[0], q[1])

        >>> eng.run(prog2)
        >>> eng.print_applied()
        Run 0:
        BSgate(0.7854, 0) | (q[0], q[1])
        Sgate(0.543, 0) | (q[0])
        Sgate(0.543, 0).H | (q[1])
        BSgate(0.7854, 0).H | (q[0], q[1])
        Run 1:
        Dgate(0.06, 0) | (q[0])

        Args:
            print_fn (function): optional custom function to use for string printing.
        """
        for k, r in enumerate(self.run_progs):
            print_fn("Run {}:".format(k))
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
            list[array, tensor]: samples returned from the backend
        """

    def _run(self, program, *, args, compile_options, **kwargs):
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
            args (Dict[str, Any]): values for the free parameters in the program(s) (if any)
            compile_options (Dict[str, Any]): keyword arguments for :meth:`.Program.compile`

        The ``kwargs`` keyword arguments are passed to the backend API calls via :meth:`Operation.apply`.

        Returns:
            Result: results of the computation
        """

        if not isinstance(program, collections.abc.Sequence):
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
                    raise RuntimeError(
                        "Register mismatch: program {}, '{}'.".format(len(self.run_progs), p.name)
                    )

                # Copy the latest measured values in the RegRefs of p.
                # We cannot copy from prev directly because it could be used in more than one
                # engine.
                for k, v in enumerate(self.samples):
                    p.reg_refs[k].val = v

            # bind free parameters to their values
            p.bind_params(args)

            # compile the program for the correct backend
            target = self.backend.compiler
            if target is not None:
                p = p.compile(compiler=target, **compile_options)
            p.lock()

            _, self.samples, self.all_samples = self._run_program(p, **kwargs)
            self.run_progs.append(p)

            prev = p

        return Result(self.samples, all_samples=self.all_samples)


class LocalEngine(BaseEngine):
    """Local quantum program executor engine.

    The Strawberry Fields engine is used to execute :class:`.Program` instances
    on the chosen local backend, and makes the results available via :class:`.Result`.

    **Example:**

    The following example creates a Strawberry Fields
    quantum :class:`~.Program` and runs it using an engine.

    .. code-block:: python

        # create a program
        prog = sf.Program(2)

        with prog.context as q:
            ops.S2gate(0.543) | (q[0], q[1])

    We initialize the engine with the name of the local backend,
    and can pass optional backend options.

    >>> eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})

    The :meth:`~.LocalEngine.run` method is used to execute quantum
    programs on the attached backend, and returns a :class:`.Result`
    object containing the results of the execution.

    >>> results = eng.run(prog)

    Args:
        backend (str, BaseBackend): short name of the backend, or a pre-constructed backend instance
        backend_options (None, Dict[str, Any]): keyword arguments to be passed to the backend
    """

    def __str__(self):
        return self.__class__.__name__ + "({})".format(self.backend_name)

    def reset(self, backend_options=None):
        backend_options = backend_options or {}
        super().reset(backend_options)
        self.backend.reset(**self.backend_options)
        # TODO should backend.reset and backend.begin_circuit be combined?

    def _init_backend(self, init_num_subsystems):
        self.backend.begin_circuit(init_num_subsystems, **self.backend_options)

    def _run_program(self, prog, **kwargs):
        applied = []
        samples_dict = {}
        all_samples = {}
        batches = self.backend_options.get("batch_size", 0)

        for cmd in prog.circuit:
            try:
                # try to apply it to the backend and, if op is a measurement, store it in values
                val = cmd.op.apply(cmd.reg, self.backend, **kwargs)
                if val is not None:
                    for i, r in enumerate(cmd.reg):
                        if batches:
                            samples_dict[r.ind] = val[:, :, i]

                            # Internally also store all the measurement outcomes
                            if r.ind not in all_samples:
                                all_samples[r.ind] = list()
                            all_samples[r.ind].append(val[:, :, i])
                        else:
                            samples_dict[r.ind] = val[:, i]

                            # Internally also store all the measurement outcomes
                            if r.ind not in all_samples:
                                all_samples[r.ind] = list()
                            all_samples[r.ind].append(val[:, i])

                applied.append(cmd)

            except NotApplicableError:
                # command is not applicable to the current backend type
                raise NotApplicableError(
                    "The operation {} cannot be used with {}.".format(cmd.op, self.backend)
                ) from None

            except NotImplementedError:
                # command not directly supported by backend API
                raise NotImplementedError(
                    "The operation {} has not been implemented in {} for the arguments {}.".format(
                        cmd.op, self.backend, kwargs
                    )
                ) from None

        samples = self._combine_and_sort_samples(samples_dict)

        return applied, samples, all_samples

    def _combine_and_sort_samples(self, samples_dict):
        """Helper function to combine the values in the samples dictionary sorted by its keys."""
        batches = self.backend_options.get("batch_size", 0)

        if not samples_dict:
            return np.empty((0, 0))

        samples = np.transpose([i for _, i in sorted(samples_dict.items())])

        # pylint: disable=import-outside-toplevel
        if self.backend_name == "tf":
            from tensorflow import convert_to_tensor

            if batches:
                samples = [
                    np.transpose([i[b] for _, i in sorted(samples_dict.items())])
                    for b in range(batches)
                ]

            return convert_to_tensor(samples)

        return samples

    # pylint:disable=too-many-branches
    def run(self, program, *, args=None, compile_options=None, **kwargs):
        """Execute quantum programs by sending them to the backend.

        Args:
            program (Program, Sequence[Program]): quantum programs to run
            args (dict[str, Any]): values for the free parameters in the program(s) (if any)
            compile_options (None, Dict[str, Any]): keyword arguments for :meth:`.Program.compile`

        Keyword Args:
            shots (int): number of times the program measurement evaluation is repeated
            modes (None, Sequence[int]): Modes to be returned in the ``Result.state`` :class:`.BaseState` object.
                ``None`` returns all the modes (default). An empty sequence means no state object is returned.

        Returns:
            Result: results of the computation
        """
        # pylint: disable=import-outside-toplevel
        from strawberryfields.tdm.tdmprogram import reshape_samples

        # At this time we do not support lists of tdm programs
        valid_tdm_program = (
            not isinstance(program, collections.abc.Sequence) and program.type == "tdm"
        )
        if valid_tdm_program:
            # priority order for the shots value should be kwargs > run_options > 1
            shots = kwargs.get("shots", program.run_options.get("shots", 1))

            program.unroll(shots=shots)
            # Shots >1 for a TDM program simply corresponds to creating
            # multiple copies of the program, and appending them to run sequentially.
            # As a result, we set the backend shots to 1 for the Gaussian backend.
            kwargs["shots"] = 1

        args = args or {}
        compile_options = compile_options or {}
        temp_run_options = {}

        if isinstance(program, collections.abc.Sequence):
            # successively update all run option defaults.
            # the run options of successive programs
            # overwrite the run options of previous programs
            # in the list
            program_lst = program
            for p in program:
                if p.type == "tdm":
                    raise NotImplementedError("Lists of TDM programs are not currently supported")

                temp_run_options.update(p.run_options)
        else:
            # single program to execute
            program_lst = [program]
            temp_run_options.update(program.run_options)

        temp_run_options.update(kwargs or {})
        temp_run_options.setdefault("shots", 1)
        temp_run_options.setdefault("modes", None)

        # avoid unexpected keys being sent to Operations
        eng_run_keys = ["eval", "session", "feed_dict", "shots"]
        eng_run_options = {
            key: temp_run_options[key] for key in temp_run_options.keys() & eng_run_keys
        }

        # check that batching is not used together with shots > 1
        if self.backend_options.get("batch_size", 0) and eng_run_options["shots"] > 1:
            raise NotImplementedError("Batching cannot be used together with multiple shots.")

        # check that post-selection and feed-forwarding is not used together with shots > 1
        for p in program_lst:
            for c in p.circuit:
                try:
                    if c.op.select and eng_run_options["shots"] > 1:
                        raise NotImplementedError(
                            "Post-selection cannot be used together with multiple shots."
                        )
                except AttributeError:
                    pass

                if c.op.measurement_deps and eng_run_options["shots"] > 1:
                    raise NotImplementedError(
                        "Feed-forwarding of measurements cannot be used together with multiple shots."
                    )

        result = super()._run(
            program, args=args, compile_options=compile_options, **eng_run_options
        )

        if valid_tdm_program:
            result._all_samples = reshape_samples(
                result.all_samples, program.measured_modes, program.N, program.timebins
            )
            # transpose the samples so that they have shape `(shots, spatial modes, timebins)`
            result._samples = np.array(list(result.all_samples.values())).transpose(1, 0, 2)
            program.roll()
        modes = temp_run_options["modes"]

        if modes is None or modes:
            # state object requested
            # session and feed_dict are needed by TF backend both during simulation (if program
            # contains measurements) and state object construction.
            result._state = self.backend.state(**temp_run_options)

        return result


class RemoteEngine:
    """A quantum program executor engine that provides a simple interface for
    running remote jobs in a blocking or non-blocking manner.

    **Example:**

    The following examples instantiate an engine with the default configuration, and
    run both blocking and non-blocking jobs.

    Run a blocking job:

    >>> engine = RemoteEngine("X8_01")
    >>> result = engine.run(program, shots=1) # blocking call
    >>> result
    [[0 1 0 2 1 0 0 0]]

    Run a non-blocking job:

    >>> job = engine.run_async(program, shots=1)
    >>> job.status
    "queued"
    >>> job.result
    InvalidJobOperationError
    >>> job.refresh()
    >>> job.status
    "complete"
    >>> job.result
    [[0 1 0 2 1 0 0 0]]

    Args:
        target (str): the target device
        connection (strawberryfields.api.Connection): a connection to the remote job
            execution platform
        backend_options (Dict[str, Any]): keyword arguments for the backend
    """

    POLLING_INTERVAL_SECONDS = 1
    DEFAULT_TARGETS = {"X8": "X8_01", "X12": "X12_01"}

    def __init__(self, target: str, connection: Connection = None, backend_options: dict = None):
        self._target = self.DEFAULT_TARGETS.get(target, target)
        self._spec = None
        self._connection = connection or Connection()
        self._backend_options = backend_options or {}
        self.log = create_logger(__name__)

    @property
    def target(self) -> str:
        """The target device used by the engine.

        Returns:
            str: the name of the target
        """
        return self._target

    @property
    def connection(self) -> Connection:
        """The connection object used by the engine.

        Returns:
            strawberryfields.api.Connection
        """
        return self._connection

    @property
    def device_spec(self):
        """The device specifications for target device"""
        if self._spec is None:
            self._spec = self._connection.get_device_spec(self.target)
        return self._spec

    def run(
        self, program: Program, *, compile_options=None, recompile=False, **kwargs
    ) -> Optional[Result]:
        """Runs a blocking job.

        In the blocking mode, the engine blocks until the job is completed, failed, or
        cancelled. A job in progress can be cancelled with a keyboard interrupt (`ctrl+c`).

        If the job completes successfully, the result is returned; if the job
        fails or is cancelled, ``None`` is returned.

        Args:
            program (strawberryfields.Program): the quantum circuit
            compile_options (None, Dict[str, Any]): keyword arguments for :meth:`.Program.compile`
            recompile (bool): Specifies if ``program`` should be recompiled
                using ``compile_options``, or if not provided, the default compilation options.

        Keyword Args:
            shots (Optional[int]): The number of shots for which to run the job. If this
                argument is not provided, the shots are derived from the given ``program``.

        Returns:
            strawberryfields.api.Result, None: the job result if successful, and ``None`` otherwise
        """
        job = self.run_async(
            program, compile_options=compile_options, recompile=recompile, **kwargs
        )
        try:
            while True:
                job.refresh()
                if job.status == "complete":
                    self.log.info("The remote job %s has been completed.", job.id)
                    return job.result

                if job.status == "failed":
                    message = (
                        "The remote job {} failed due to an internal "
                        "server error. Please try again. {}".format(job.id, job.meta)
                    )
                    self.log.error(message)

                    raise FailedJobError(message)

                time.sleep(self.POLLING_INTERVAL_SECONDS)
        except KeyboardInterrupt as e:
            self._connection.cancel_job(job.id)
            raise KeyboardInterrupt("The job has been cancelled.") from e

    def run_async(
        self, program: Program, *, compile_options=None, recompile=False, **kwargs
    ) -> Job:
        """Runs a non-blocking remote job.

        In the non-blocking mode, a ``Job`` object is returned immediately, and the user can
        manually refresh the status and check for updated results of the job.

        Args:
            program (strawberryfields.Program): the quantum circuit
            compile_options (None, Dict[str, Any]): keyword arguments for :meth:`.Program.compile`
            recompile (bool): Specifies if ``program`` should be recompiled
                using ``compile_options``, or if not provided, the default compilation options.

        Keyword Args:
            shots (Optional[int]): The number of shots for which to run the job. If this
                argument is not provided, the shots are derived from the given ``program``.

        Returns:
            strawberryfields.api.Job: the created remote job
        """
        # get the specific chip to submit the program to
        compile_options = compile_options or {}
        kwargs.update(self._backend_options)

        device = self.device_spec
        if program.type == "tdm" and not device.layout_is_formatted():
            device.fill_template(program)

        compiler_name = compile_options.get("compiler", device.default_compiler)

        program_is_compiled = program.compile_info is not None

        if program_is_compiled and not recompile:
            # error handling for program compilation:
            # program was compiled but recompilation was not allowed by the
            # user

            if (
                program.compile_info[0].target != device.target
                or program.compile_info[0]._spec != device._spec
            ):
                # program was compiled for a different device
                raise ValueError(
                    "Cannot use program compiled with "
                    f"{program._compile_info[0].target} for target {self.target}. "
                    'Pass the "recompile=True" keyword argument '
                    f"to compile with {compiler_name}."
                )

        if not program_is_compiled:
            # program is not compiled
            msg = f"Compiling program for device {device.target} using compiler {compiler_name}."
            self.log.info(msg)
            program = program.compile(device=device, **compile_options)

        elif recompile:
            # recompiling program
            if compile_options:
                msg = f"Recompiling program for device {device.target} using the specified compiler options: {compile_options}."
            else:
                msg = f"Recompiling program for device {device.target} using compiler {compiler_name}."

            self.log.info(msg)
            program = program.compile(device=device, **compile_options)

        else:
            # validating program
            msg = (
                f"Program previously compiled for {device.target} using {program.compile_info[1]}. "
                f"Validating program against the Xstrict compiler."
            )
            self.log.info(msg)
            program = program.compile(device=device, compiler="Xstrict")

        # update the run options if provided
        run_options = {}
        run_options.update(program.run_options)
        run_options.update(kwargs or {})

        if "shots" not in run_options:
            raise ValueError("Number of shots must be specified.")

        return self._connection.create_job(self.target, program, run_options)

    def __repr__(self):
        return "<{}: target={}, connection={}>".format(
            self.__class__.__name__, self.target, self.connection
        )

    def __str__(self):
        return self.__repr__()


class Engine(LocalEngine):
    """dummy"""

    # alias for backwards compatibility
    __doc__ = LocalEngine.__doc__

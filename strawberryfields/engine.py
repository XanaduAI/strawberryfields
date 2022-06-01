# Copyright 2019-2022 Xanadu Quantum Technologies Inc.

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
import warnings
from typing import Any, Dict, Optional

import numpy as np
import xcc

from strawberryfields.device import Device
from strawberryfields.result import Result
from strawberryfields.io import to_blackbird
from strawberryfields.logger import create_logger
from strawberryfields.program import Program
from strawberryfields.tdm import TDMProgram, reshape_samples

from .backends import load_backend, BosonicBackend
from .backends.base import BaseBackend, NotApplicableError
from ._version import __version__

# for automodapi, do not include the classes that should appear under the top-level strawberryfields namespace
__all__ = ["BaseEngine", "LocalEngine", "BosonicEngine"]


class FailedJobError(Exception):
    """Raised when a job had a failure on the server side."""


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
        #: Dict[Any, List]: the measurement results as a dictionary with measured modes as keys
        self.samples_dict = None

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
            print_fn(f"Run {k}:")
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
            tuple(list[Command], list[array, tensor], dict[int, list]): tuple containing the
            commands that were applied to the backend, the samples returned from the backend, and
            the samples as a dictionary with the measured modes as keys
        """
        # `_run_program` is called in `BaseEngine._run` but should never be called
        # from a `BaseEngine` object, in which case an error should be raised.
        raise NotImplementedError("'run_program' method must be implemented on child class")

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
        # pop modes so that it's not passed on to the backend API calls via 'op.apply'
        modes = kwargs.pop("modes")

        if not isinstance(program, collections.abc.Sequence):
            program = [program]

        kwargs.setdefault("shots", 1)
        # NOTE: by putting ``shots`` into keyword arguments, it allows for the
        # signatures of methods in Operations to remain cleaner, since only
        # Measurements need to know about shots

        prev = self.run_progs[-1] if self.run_progs else None  # previous program segment
        for p in program:

            if self.backend.compiler:
                default_compiler = getattr(
                    compile_options.get("device"), "default_compiler", self.backend.compiler
                )
                compile_options.setdefault("compiler", default_compiler)

            # compile the program for the correct backend if a compiler or a device exists
            if "compiler" in compile_options or "device" in compile_options:
                p = p.compile(**compile_options)

            received_rolled = False  # whether a TDMProgram had a rolled circuit
            if isinstance(p, TDMProgram):
                tdm_options = self.get_tdm_options(p, **kwargs)
                # pop modes so that it's not passed on to the backend API calls via 'op.apply'
                modes = tdm_options.pop("modes")
                received_rolled = tdm_options.pop("received_rolled")
                kwargs.update(tdm_options)

            if prev is None:
                # initialize the backend
                self._init_backend(p.init_num_subsystems)
            else:
                # there was a previous program segment
                if not p.can_follow(prev):
                    raise RuntimeError(
                        f"Register mismatch: program {len(self.run_progs)}, '{p.name}'."
                    )

                # Copy the latest measured values in the RegRefs of p.
                # We cannot copy from prev directly because it could be used in more than one
                # engine.
                for k, v in enumerate(self.samples):
                    p.reg_refs[k].val = v

            # bind free parameters to their values
            p.bind_params(args)
            p.lock()

            _, self.samples, self.samples_dict = self._run_program(p, **kwargs)
            self.run_progs.append(p)

            if isinstance(p, TDMProgram) and received_rolled:
                p.roll()

            prev = p

        ancillae_samples = None
        if isinstance(self.backend, BosonicBackend):
            ancillae_samples = self.backend.ancillae_samples_dict.copy()

        samples = {"output": [self.samples]}
        result = Result(samples, samples_dict=self.samples_dict, ancillae_samples=ancillae_samples)

        # if `modes`` is empty (i.e. `modes==[]`) return no state, else return state with
        # selected modes (all if `modes==None`)
        if modes is None or modes:
            # state object requested
            # session and feed_dict are needed by TF backend both during simulation (if program
            # contains measurements) and state object construction.
            result.state = self.backend.state(modes=modes, **kwargs)

        return result

    @staticmethod
    def get_tdm_options(program, **kwargs):
        """Retrieves the shots and modes for the TDM program and (space)unrolls it if necessary."""
        # priority order for the shots value should be kwargs > run_options > 1
        shots = kwargs.get("shots", program.run_options.get("shots", 1))

        # setting shots >1 for a TDM program corresponds to further unrolling the program,
        # meaning that we still only need to execute it once
        tdm_options = {"modes": None, "shots": 1 if shots else None, "received_rolled": False}
        if program.is_unrolled:
            tdm_options["received_rolled"] = True

        # if a tdm program is input in a rolled state, then unroll it
        if kwargs.get("space_unroll", False):
            if program.space_unrolled_circuit is None:
                program.space_unroll(shots=shots or 1)
        else:
            # if `space_unroll != True`, only unroll it iff it isn't already unrolled
            if not program.is_unrolled:
                program.unroll(shots=shots or 1)

        if program.space_unrolled_circuit is not None:
            if kwargs.get("crop", False):
                tdm_options["modes"] = range(program.get_crop_value(), program.timebins)
            else:
                tdm_options["modes"] = range(0, program.timebins)
        elif kwargs.get("crop", False):
            warnings.warn(
                "Cropping the state is only possible for space unrolled circuits, which "
                "can be done by executing `Program.space_unroll()` prior to running the "
                "circuit. Note, this might cause a significant performance hit if sampling!"
            )

        return tdm_options


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

    def __new__(cls, backend, *, backend_options=None):
        if backend == "bosonic":
            bos_eng = super().__new__(BosonicEngine)
            bos_eng.__init__(backend, backend_options=backend_options)
            return bos_eng

        return super().__new__(cls)

    def __str__(self):
        return self.__class__.__name__ + f"({self.backend_name})"

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
        batches = self.backend_options.get("batch_size", 0)

        for cmd in prog.circuit:
            try:
                # try to apply it to the backend and, if op is a measurement, store it in values
                val = cmd.op.apply(cmd.reg, self.backend, **kwargs)
                if val is not None:
                    for i, r in enumerate(cmd.reg):
                        if batches:
                            # Internally also store all the measurement outcomes
                            if r.ind not in samples_dict:
                                samples_dict[r.ind] = []
                            samples_dict[r.ind].append(val[:, :, i])
                        else:
                            # Internally also store all the measurement outcomes
                            if r.ind not in samples_dict:
                                samples_dict[r.ind] = []
                            samples_dict[r.ind].append(val[:, i])

                applied.append(cmd)

            except NotApplicableError:
                # command is not applicable to the current backend type
                raise NotApplicableError(
                    f"The operation {cmd.op} cannot be used with {self.backend}."
                ) from None

            except NotImplementedError:
                # command not directly supported by backend API
                raise NotImplementedError(
                    f"The operation {cmd.op} has not been implemented in {self.backend} for the "
                    f"arguments {kwargs}."
                ) from None

        # combine the samples sorted by mode, and cast correctly to array or tensor
        samples, samples_dict = self._combine_and_sort_samples(samples_dict)

        if isinstance(prog, TDMProgram) and samples_dict:
            samples_dict = reshape_samples(samples_dict, prog.measured_modes, prog.N, prog.timebins)
            # crop vacuum modes arriving at the detector before the first computational mode
            if kwargs.get("crop", False):
                samples_dict[0] = samples_dict[0][:, prog.get_crop_value() :]

            # transpose the samples so that they have shape `(shots, spatial modes, timebins)`
            samples = np.array(list(samples_dict.values())).transpose(1, 0, 2)

        return applied, samples, samples_dict

    def _combine_and_sort_samples(self, samples_dict):
        """Helper function to combine the values in the samples dictionary sorted by its keys."""
        batches = self.backend_options.get("batch_size", 0)

        if not samples_dict:
            return np.empty((0, 0)), samples_dict

        single_sample_dict = {key: val[-1] for key, val in samples_dict.items()}
        samples = np.transpose([i for _, i in sorted(single_sample_dict.items())])

        # pylint: disable=import-outside-toplevel
        if self.backend_name == "tf":
            from tensorflow import convert_to_tensor

            if batches:
                samples = [
                    np.transpose([i[b] for _, i in sorted(single_sample_dict.items())])
                    for b in range(batches)
                ]

            samples_dict = {k: [convert_to_tensor(i) for i in v] for k, v in samples_dict.items()}
            return convert_to_tensor(samples), samples_dict

        samples_dict = {k: [np.array(i) for i in v] for k, v in samples_dict.items()}
        return samples, samples_dict

    # pylint:disable=too-many-branches
    def run(self, program, *, args=None, compile_options=None, **kwargs):
        """Execute quantum programs by sending them to the backend.

        Args:
            program (Program, Sequence[Program]): quantum programs to run
            args (dict[str, Any]): values for the free parameters in the program(s) (if any)
            compile_options (None, Dict[str, Any]): keyword arguments for :meth:`.Program.compile`

        Keyword Args:
            shots (int): number of times the program measurement evaluation is repeated
            crop (bool): Crop vacuum modes present before the first computational modes.
                This option crops samples from ``Result.samples`` and ``Result.samples_dict``.
                If the program is space unrolled the state in `Result.state` is also cropped.
            modes (None, Sequence[int]): Modes to be returned in the ``Result.state`` :class:`.BaseState` object.
                ``None`` returns all the modes (default). An empty sequence means no state object is returned.

        Returns:
            Result: results of the computation
        """
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
                if isinstance(p, TDMProgram):
                    raise NotImplementedError("Lists of TDM programs are not currently supported")

                temp_run_options.update(p.run_options)
        else:
            # single program to execute
            program_lst = [program]
            temp_run_options.update(program.run_options)

        temp_run_options.update(kwargs or {})
        temp_run_options.setdefault("shots", 1)
        temp_run_options.setdefault("modes", None)

        # avoid unexpected keys being sent to Operations (note, "modes" shouldn't be passed, but is
        # needed in `BaseEngine.run()`; it's popped before being passed on to Operations)
        eng_run_keys = ["eval", "session", "feed_dict", "shots", "crop", "space_unroll", "modes"]
        eng_run_options = {
            key: temp_run_options[key] for key in temp_run_options.keys() & eng_run_keys
        }

        # check that batching is not used together with shots > 1
        if self.backend_options.get("batch_size", 0) and eng_run_options["shots"] > 1:
            raise NotImplementedError("Batching cannot be used together with multiple shots.")

        # check that post-selection and feed-forwarding is not used together with shots > 1
        for p in program_lst:
            for c in p.circuit or []:
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

        return super()._run(
            program_lst, args=args, compile_options=compile_options, **eng_run_options
        )


class RemoteEngine:
    """A quantum program executor engine that provides a simple interface for
    running remote jobs in a blocking or non-blocking manner.

    **Example:**

    The following examples instantiate an engine with the default settings, and
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
    >>> job.wait()
    >>> job.status
    "complete"
    >>> result = sf.Result(job.result)
    >>> result.samples
    array([[0 1 0 2 1 0 0 0]])

    Args:
        target (str): the target device
        connection (xcc.Connection, optional): a connection to the Xanadu Cloud
        backend_options (Dict[str, Any], optional): keyword arguments for the backend
    """

    POLLING_INTERVAL_SECONDS = 1
    DEFAULT_TARGETS = {"X8": "X8_01", "X12": "X12_02"}

    def __init__(
        self,
        target: str,
        connection: Optional[xcc.Connection] = None,
        backend_options: Optional[Dict[str, Any]] = None,
    ):
        self._target = self.DEFAULT_TARGETS.get(target, target)
        self._device = None
        self._connection = connection
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
    def connection(self) -> xcc.Connection:
        """The connection used by the engine. A new :class:`xcc.Connection` will
        be created and returned each time this property is accessed if the
        connection supplied to :meth:`~.RemoteEngine.__init__` was ``None``.

        Returns:
            xcc.Connection
        """
        if self._connection is not None:
            return self._connection

        settings = xcc.Settings()

        return xcc.Connection(
            refresh_token=settings.REFRESH_TOKEN,
            access_token=settings.ACCESS_TOKEN,
            host=settings.HOST,
            port=settings.PORT,
            tls=settings.TLS,
            headers={"User-Agent": f"StrawberryFields/{__version__}"},
        )

    @property
    def device(self) -> Device:
        """The representation of the target device.

        Returns:
            .strawberryfields.Device: the target device representation

        Raises:
            requests.exceptions.RequestException: if there was an issue fetching
                the device specifications or the device certificate from the Xanadu Cloud
        """
        if self._device is None:
            device = xcc.Device(target=self.target, connection=self.connection)
            self._device = Device(spec=device.specification, cert=device.certificate)
        return self._device

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
            integer_overflow_protection (Optional[bool]): Whether to enable the
                conversion of integral job results into ``np.int64`` objects.
                By default, integer overflow protection is enabled. For more
                information, see `xcc.Job.get_result
                <https://xanadu-cloud-client.readthedocs.io/en/stable/api/xcc.Job.html#xcc.Job.get_result>`_.

        Returns:
            strawberryfields.Result, None: the job result if successful, and ``None`` otherwise

        Raises:
            requests.exceptions.RequestException: if there was an issue fetching
                the device specifications from the Xanadu Cloud
            FailedJobError: if the remote job fails on the server side ("cancelled" or "failed")
        """
        job = self.run_async(
            program, compile_options=compile_options, recompile=recompile, **kwargs
        )
        try:
            while True:
                # TODO: needed to refresh connection; remove once xcc.Connection
                # is able to refresh config info dynamically
                job._connection = self.connection
                job.clear()

                if job.finished:
                    break
                time.sleep(self.POLLING_INTERVAL_SECONDS)

        except KeyboardInterrupt as e:
            xcc.Job(id_=job.id, connection=self.connection).cancel()
            raise KeyboardInterrupt("The job has been cancelled.") from e

        if job.status == "failed":
            message = (
                f"The remote job {job.id} failed due to an internal "
                f"server error: {job.metadata}. Please try again."
            )
            self.log.error(message)
            raise FailedJobError(message)

        if job.status == "complete":
            self.log.info(f"The remote job {job.id} has been completed.")

            integer_overflow_protection = kwargs.get("integer_overflow_protection", True)
            result = job.get_result(integer_overflow_protection=integer_overflow_protection)
            output = result.get("output")

            # crop vacuum modes arriving at the detector before the first computational mode
            if output and isinstance(program, TDMProgram) and kwargs.get("crop", False):
                output[0] = output[0][:, :, program.get_crop_value() :]

            return Result(result)

        message = f"The remote job {job.id} has failed with status {job.status}: {job.metadata}."
        self.log.info(message)

        raise FailedJobError(message)

    def run_async(
        self, program: Program, *, compile_options=None, recompile=False, **kwargs
    ) -> xcc.Job:
        """Runs a non-blocking remote job.

        In the non-blocking mode, a ``xcc.Job`` object is returned immediately, and the user can
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
            xcc.Job: the created remote job
        """
        # get the specific chip to submit the program to
        compile_options = compile_options or {}
        kwargs.update(self._backend_options)

        device = self.device
        compiler_name = compile_options.get("compiler", device.default_compiler)

        program_is_compiled = program.compile_info is not None

        if program_is_compiled and not recompile:
            # error handling for program compilation:
            # program was compiled but recompilation was not allowed by the
            # user

            if program.compile_info and (
                program.compile_info[0].target != device.target
                or program.compile_info[0]._spec != device._spec
            ):
                compile_target = program.compile_info[0].target
                # program was compiled for a different device
                raise ValueError(
                    "Cannot use program compiled with "
                    f"{compile_target} for target {self.target}. "
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

        elif program.compile_info[1][0] == "X":
            # validating program
            msg = (
                f"Program previously compiled for {device.target} using {program.compile_info[1]}. "
                f"Validating program against the Xstrict compiler."
            )
            self.log.info(msg)
            program = program.compile(device=device, compiler="Xstrict")

        # Update and filter the run options if provided.
        # Forwarding the boolean `crop` run option to the hardware will result in undesired
        # behaviour as og-tdm also uses that keyword arg, hence it is removed.
        temp_run_options = {**program.run_options, **kwargs}
        skip_run_keys = ["crop"]
        run_options = {
            key: temp_run_options[key] for key in temp_run_options.keys() - skip_run_keys
        }

        if "shots" not in run_options:
            raise ValueError("Number of shots must be specified.")

        # Serialize a Blackbird circuit for network transmission
        bb = to_blackbird(program)
        bb._target["options"] = run_options
        circuit = bb.serialize()

        connection = self.connection

        job = xcc.Job.submit(
            connection=connection,
            name=program.name,
            target=self.target,
            circuit=circuit,
            language=f"blackbird:{bb.version}",
        )

        return job

    def __repr__(self):
        return f"<{self.__class__.__name__}: target={self.target}, connection={self._connection}>"

    def __str__(self):
        return self.__repr__()


class Engine(LocalEngine):
    """dummy"""

    # alias for backwards compatibility
    __doc__ = LocalEngine.__doc__


class BosonicEngine(LocalEngine):
    """Local quantum program executor engine for programs executed on the bosonic backend.

    The BosonicEngine is used to execute :class:`.Program` instances on the bosonic backend,
    and makes the results available via :class:`.Result`.
    """

    def _run_program(self, prog, **kwargs):
        # Custom Bosonic run code
        applied, samples_dict = self.backend.run_prog(prog, **kwargs)
        samples, samples_dict = self._combine_and_sort_samples(samples_dict)
        return applied, samples, samples_dict

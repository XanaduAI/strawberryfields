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
This module implements :class:`BaseEngine` and its subclasses that are responsible for
communicating quantum programs represented by :class:`.Program` objects
to a backend that could be e.g., a simulator or a hardware quantum processor.
One can think of each BaseEngine instance as a separate quantum computation.
"""
import abc
import collections.abc
import time

import numpy as np

from .backends import load_backend
from .backends.base import (NotApplicableError, BaseBackend)

from strawberryfields.api_client import APIClient, Job, JobNotQueuedError, JobExecutionError
from strawberryfields.io import to_blackbird
from strawberryfields.configuration import DEFAULT_CONFIG


class OneJobAtATimeError(Exception):
    """Raised when a user attempts to execute more than one job on the same engine instance."""


# for automodapi, do not include the classes that should appear under the top-level strawberryfields namespace
__all__ = ["Result", "BaseEngine", "LocalEngine"]



class Result:
    """Result of a quantum computation.

    Represents the results of the execution of a quantum program
    returned by the :meth:`.LocalEngine.run` method.

    The returned :class:`~Result` object provides several useful properties
    for accessing the results of your program execution:

    * ``results.state``: The quantum state object contains details and methods
      for manipulation of the final circuit state. Not available for remote
      backends. See :doc:`/introduction/states` for more information regarding available
      state methods.

    * ``results.samples``: Measurement samples from any measurements performed.

    **Example:**

    The following examples run an existing Strawberry Fields
    quantum :class:`~.Program` on the Gaussian engine to get
    a results object.

    Using this results object, the measurement samples
    can be returned, as well as quantum state information.

    >>> eng = sf.Engine("gaussian")
    >>> results = eng.run(prog)
    >>> print(results)
    Result: 3 subsystems
        state: <GaussianState: num_modes=3, pure=True, hbar=2>
        samples: [[0, 0, 0]]
    >>> results.samples
    np.array([[0, 0, 0]])
    >>> results.state.is_pure()
    True

    .. note::

        Only local simulators will return a state object. Remote
        simulators and hardware backends will return
        measurement samples  (:attr:`Result.samples`),
        but the return value of ``Result.state`` will be ``None``.
    """

    def __init__(self, samples):
        self._state = None

        # samples arrives as either a list of arrays (for shots > 1) or a list (for shots = 1)
        # need to be converted to a multidimensional array with shape (shots, modes)
        if np.ndim(samples) == 1:
            samples = np.array([samples])
        else:
            samples = np.stack(samples, 1)
        self._samples = samples

    @property
    def samples(self):
        """Measurement samples.

        Returned measurement samples will have shape ``(shots, modes)``. If a
        single shot is requested during execution, the returned measurement
        sample will have shape ``(1, modes)``.

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

        See :doc:`/introduction/states` for more information regarding available
        state methods.

        .. note::

            Only local simulators will return a state object. Remote
            simulators and hardware backends will return
            measurement samples (:attr:`Result.samples`),
            but the return value of ``Result.state`` will be ``None``.

        Returns:
            BaseState: quantum state returned from program execution
        """
        return self._state

    def __str__(self):
        """String representation."""
        return "Result: {} subsystems, state: {}\n samples: {}".format(
            len(self.samples), self.state, self.samples
        )


class BaseEngine(abc.ABC):
    r"""Abstract base class for quantum program executor engines.

    Args:
        backend (str): backend short name
        backend_options (Dict[str, Any]): keyword arguments for the backend
    """

    REMOTE = False

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

        def _broadcast_nones(val, dim):
            """Helper function to ensure register values have same shape, even if not measured"""
            if val is None and dim > 1:
                return [None] * dim
            return val

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

            # if the program hasn't been compiled for this backend, do it now
            target = self.backend.circuit_spec
            if target is not None and p.target != target:
                p = p.compile(target, **compile_options)
            p.lock()

            if self.REMOTE:
                self.samples = self._run_program(p, **kwargs)
            else:
                self._run_program(p, **kwargs)
                shots = kwargs.get("shots", 1)
                self.samples = [
                    _broadcast_nones(p.reg_refs[k].val, shots) for k in sorted(p.reg_refs)
                ]
                self.run_progs.append(p)

            prev = p

        if self.samples is not None:
            return Result(self.samples.copy())


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

    def __init__(self, backend, *, backend_options=None):
        backend_options = backend_options or {}
        super().__init__(backend, backend_options)

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
        for cmd in prog.circuit:
            try:
                # try to apply it to the backend
                # NOTE we could also handle storing measured vals here
                cmd.op.apply(cmd.reg, self.backend, **kwargs)
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
        return applied

    def run(self, program, *, args=None, compile_options=None, run_options=None):
        """Execute quantum programs by sending them to the backend.

        Args:
            program (Program, Sequence[Program]): quantum programs to run
            args (dict[str, Any]): values for the free parameters in the program(s) (if any)
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
        args = args or {}
        compile_options = compile_options or {}
        temp_run_options = {}

        if isinstance(program, collections.abc.Sequence):
            # succesively update all run option defaults.
            # the run options of successive programs
            # overwrite the run options of previous programs
            # in the list
            program_lst = program
            for p in program:
                temp_run_options.update(p.run_options)
        else:
            # single program to execute
            program_lst = [program]
            temp_run_options.update(program.run_options)

        temp_run_options.update(run_options or {})
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
                        raise NotImplementedError("Post-selection cannot be used together with multiple shots.")
                except AttributeError:
                    pass

                if c.op.measurement_deps and eng_run_options["shots"] > 1:
                    raise NotImplementedError("Feed-forwarding of measurements cannot be used together with multiple shots.")

        result = super()._run(program, args=args, compile_options=compile_options, **eng_run_options)

        modes = temp_run_options["modes"]

        if modes is None or modes:
            # state object requested
            # session and feed_dict are needed by TF backend both during simulation (if program
            # contains measurements) and state object construction.
            result._state = self.backend.state(**temp_run_options)

        return result


class StarshipEngine(BaseEngine):
    """
    Starship quantum program executor engine.

    Executes :class:`.Program` instances on the chosen remote backend, and makes
    the results available via :class:`.Result`.

    Args:
        backend (str, BaseBackend): name of the backend, or a pre-constructed backend instance
        polling_delay_seconds (int): the number of seconds to wait between queries when polling for
            job results
    """

    # This engine will execute jobs remotely.
    REMOTE = True

    def __init__(self, backend, polling_delay_seconds=1, **kwargs):
        super().__init__(backend)

        api_client_params = {k: v for k, v in kwargs.items() if k in DEFAULT_CONFIG["api"].keys()}
        self.client = APIClient(**api_client_params)
        self.polling_delay_seconds = polling_delay_seconds
        self.jobs = []

    def __str__(self):
        return self.__class__.__name__ + "({})".format(self.backend_name)

    def _init_backend(self, *args):
        """
        TODO: This does not do anything right now.
        """
        # Do nothing for now...
        pass

    def reset(self):
        """
        Reset must be called in order to submit a new job. This clears the job queue as well as
        any ran Programs.
        """
        super().reset(backend_options={})
        self.jobs.clear()

    def _get_blackbird(self, shots, program):
        """
        Returns a Blackbird object to be sent later to the server when creating a job.
        Assumes the current backend as the target.

        Args:
            shots (int): the number of shots
            program (Program): program to be converted to Blackbird code

        Returns:
            blackbird.BlackbirdProgram
        """
        bb = to_blackbird(program, version="1.0")

        # TODO: This is potentially not needed here
        bb._target["name"] = self.backend_name
        bb._target["options"] = {"shots": shots, **program.backend_options}
        return bb

    def _queue_job(self, job_content):
        """
        Create a Job instance based on job_content, and send the job to the API. Append to list
        of jobs.

        Args:
            job_content (str): the Blackbird code to execute

        Returns:
            (strawberryfields.api_client.Job): a Job instance referencing the queued job
        """
        job = Job(client=self.client)
        job.manager.create(circuit=job_content)
        self.jobs.append(job)
        print("Job {} is sent to server.".format(job.id.value))
        return job

    def _run_program(self, program, **kwargs):
        """
        Given a compiled program, gets the blackbird circuit code and creates (or resumes) a job
        via the API. If the job is completed, returns the job result.

        A queued job can be interrupted by a ``KeyboardInterrupt`` event, at which point if the
        job ID was retrieved from the server, the job will be accessible via
        :meth:`~.Starship.jobs`.

        Args:
            program (strawberryfields.program.Program): program to be executed remotely

        Returns:
            (list): a list representing the result samples

        Raises:
            Exception: In case a job could not be submitted or completed.
            TypeError: In case a job is already queued and a user is trying to submit a new job.
        """
        if self.jobs:
            raise OneJobAtATimeError(
                "A job is already queued. Please reset the engine and try again."
            )

        kwargs.update(program.run_options)
        job_content = self._get_blackbird(program=program, **kwargs).serialize()
        job = self._queue_job(job_content)

        try:
            while not job.is_failed and not job.is_complete:
                job.reload()
                time.sleep(self.polling_delay_seconds)
        except KeyboardInterrupt:
            if job.id:
                print("Job {} is queued in the background.".format(job.id.value))
            else:
                self.reset()
                raise JobNotQueuedError("Job was not sent to server. Please try again.")

        # Job either failed or is complete - in either case, clear the job queue so that the engine is
        # ready for future jobs.
        self.reset()

        if job.is_failed:
            message = str(job.manager.http_response_data["meta"])
            raise JobExecutionError(message)
        elif job.is_complete:
            job.result.manager.get()
            return job.result.result.value

    def run(self, program, shots=1, **kwargs):
        """Compile the given program and execute it by queuing a job in the Starship.

        For the :class:`Program` instance given as input, the following happens:

        * The Program instance is compiled for the target backend.
        * The compiled program is sent as a job to the Starship
        * The measurement results of each subsystem (if any) are stored in the :attr:`~.BaseEngine.samples`.
        * The compiled program is appended to self.run_progs.
        * The queued or completed jobs are appended to self.jobs.

        Finally, the result of the computation is returned.

        Args:
            program (Program, Sequence[Program]): quantum programs to run
            shots (int): number of times the program measurement evaluation is repeated

        The ``kwargs`` keyword arguments are passed to :meth:`_run_program`.

        Returns:
            Result: results of the computation
        """

        return super()._run(program, args={}, compile_options={}, shots=shots, **kwargs)


class Engine(LocalEngine):
    """dummy"""
    # alias for backwards compatibility
    __doc__ = LocalEngine.__doc__

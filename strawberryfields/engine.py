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
import enum
import io
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin

import numpy as np
import requests

from strawberryfields.configuration import DEFAULT_CONFIG
from strawberryfields.io import to_blackbird
from strawberryfields.program import Program

from .backends import load_backend
from .backends.base import BaseBackend, NotApplicableError

# for automodapi, do not include the classes that should appear under the top-level strawberryfields namespace
__all__ = ["Result", "BaseEngine", "LocalEngine", "Connection"]


class Result:
    """Result of a quantum computation.

    Represents the results of the execution of a quantum program on a local or
    remote backend.

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
        samples: [0, 0, 0]
    >>> results.samples
    [0, 0, 0]
    >>> results.state.is_pure()
    True

    .. note::

        Only local simulators will return a state object. Remote
        simulators and hardware backends will return
        measurement samples  (:attr:`Result.samples`),
        but the return value of ``Result.state`` will be ``None``.
    """

    def __init__(self, samples, is_stateful=True):
        self._state = None
        self._is_stateful = is_stateful

        # ``samples`` arrives as a list of arrays, need to convert here to a multidimensional array
        # if len(np.shape(samples)) > 1:
        # samples = np.stack(samples, 1)
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
        if not self._is_stateful:
            raise AttributeError("The state is undefined for a stateless computation.")
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

        prev = (
            self.run_progs[-1] if self.run_progs else None
        )  # previous program segment
        for p in program:
            if prev is None:
                # initialize the backend
                self._init_backend(p.init_num_subsystems)
            else:
                # there was a previous program segment
                if not p.can_follow(prev):
                    raise RuntimeError(
                        "Register mismatch: program {}, '{}'.".format(
                            len(self.run_progs), p.name
                        )
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
                    _broadcast_nones(p.reg_refs[k].val, shots)
                    for k in sorted(p.reg_refs)
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
                    "The operation {} cannot be used with {}.".format(
                        cmd.op, self.backend
                    )
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
            for p in program:
                temp_run_options.update(p.run_options)
        else:
            # single program to execute
            temp_run_options.update(program.run_options)

        temp_run_options.update(run_options or {})
        temp_run_options.setdefault("shots", 1)
        temp_run_options.setdefault("modes", None)

        # avoid unexpected keys being sent to Operations
        eng_run_keys = ["eval", "session", "feed_dict", "shots"]
        eng_run_options = {
            key: temp_run_options[key] for key in temp_run_options.keys() & eng_run_keys
        }

        result = super()._run(
            program, args=args, compile_options=compile_options, **eng_run_options
        )

        modes = temp_run_options["modes"]

        if modes is None or modes:
            # state object requested
            # session and feed_dict are needed by TF backend both during simulation (if program
            # contains measurements) and state object construction.
            result._state = self.backend.state(**temp_run_options)

        return result


class RequestFailedError(Exception):
    """Raised when a request to the remote platform returns an error response."""


class InvalidJobOperationError(Exception):
    """Raised when an invalid operation is performed on a job."""


class JobFailedError(Exception):
    """Raised when a remote job enters a 'failed' status."""


class JobStatus(enum.Enum):
    """Represents the status of a remote job.

    This class maps a set of job statuses to the string representations returned by the
    remote platform.
    """

    OPEN = "open"
    QUEUED = "queued"
    CANCELLED = "cancelled"
    COMPLETE = "complete"
    FAILED = "failed"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        """Checks if this status represents a final and immutable state.

        This method is primarily used to determine if an operation is valid for a given
        status.

        Returns:
            bool: ``True`` if the job is terminal, and ``False`` otherwise
        """
        return self in (JobStatus.CANCELLED, JobStatus.COMPLETE, JobStatus.FAILED)


class Job:
    """Represents a remote job that can be queried for its status or result.

    This object should not be instantiated directly, but returned by an `Engine` or
    `Connection` when a job is run.

    Args:
        id_ (str): the job ID
        status (strawberryfields.engine.JobStatus): the job status
        connection (strawberryfields.engine.Connection): the connection over which the
            job is managed
    """

    def __init__(self, id_: str, status: JobStatus, connection: "Connection"):
        self._id = id_
        self._status = status
        self._connection = connection
        self._result = None

    @property
    def id(self) -> str:
        """The job ID.

        Returns:
            str: the job ID
        """
        return self._id

    @property
    def status(self) -> JobStatus:
        """The job status.

        Returns:
            strawberryfields.engine.JobStatus: the job status
        """
        return self._status

    @property
    def result(self) -> Result:
        """The job result.

        This is only defined for complete jobs, and raises an exception for any other
        status.

        Returns:
            strawberryfields.engine.Result: the result
        """
        if self.status != JobStatus.COMPLETE:
            raise AttributeError(
                "The result is undefined for jobs that are not complete "
                "(current status: {})".format(self.status.value)
            )
        return self._result

    def refresh(self):
        """Refreshes the status of the job, along with the job result if the job is
        newly completed.

        Only a non-terminal (open or queued job) can be refreshed; an exception is
        raised otherwise.
        """
        if self.status.is_terminal:
            raise InvalidJobOperationError(
                "A {} job cannot be refreshed".format(self.status.value)
            )
        self._status = self._connection.get_job_status(self.id)
        if self._status == JobStatus.COMPLETE:
            self._result = self._connection.get_job_result(self.id)

    def cancel(self):
        """Cancels the job.

        Only a non-terminal (open or queued job) can be cancelled; an exception is
        raised otherwise.
        """
        if self.status.is_terminal:
            raise InvalidJobOperationError(
                "A {} job cannot be cancelled".format(self.status.value)
            )
        self._connection.cancel_job(self.id)


class RequestMethod(enum.Enum):
    """Defines the valid request methods for messages sent to the remote job platform."""

    GET = "get"
    POST = "post"
    PATCH = "patch"


class Connection:
    """Manages remote connections to the remote job execution platform and exposes
    advanced job operations.

    For basic usage, it is not necessary to manually instantiate this object; the user
    is encouraged to use the higher-level interface provided by :class:`~StarshipEngine`.

    Args:
        token (str): the API authentication token
        host (str): the hostname of the remote platform
        port (int): the port to connect to on the remote host
        use_ssl (bool): whether to use SSL for the connection
    """

    MAX_JOBS_REQUESTED = 100
    JOB_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

    # pylint: disable=bad-continuation
    # See: https://github.com/PyCQA/pylint/issues/289
    def __init__(
        self, token, host="platform.strawberryfields.ai", port=443, use_ssl=True
    ):
        self._token = token
        self._host = host
        self._port = port
        self._use_ssl = use_ssl

    @property
    def token(self) -> str:
        """The API authentication token.

        Returns:
            str: the authentication token
        """
        return self._token

    @property
    def host(self) -> str:
        """The host for the remote platform.

        Returns:
            str: the hostname
        """
        return self._host

    @property
    def port(self) -> int:
        """The port to connect to on the remote host.

        Returns:
            int: the port number
        """
        return self._port

    @property
    def use_ssl(self) -> bool:
        """Whether to use SSL for the connection.

        Returns:
            bool: ``True`` if SSL should be used, and ``False`` otherwise
        """
        return self._use_ssl

    @property
    def base_url(self) -> str:
        """The base URL used for the connection.

        Returns:
            str: the base URL
        """
        return "http{}://{}:{}".format(
            "s" if self.use_ssl else "", self.host, self.port
        )

    def create_job(self, circuit: str) -> Job:
        """Creates a job with the given circuit.

        Args:
            circuit (str): the serialized Blackbird program

        Returns:
            strawberryfields.engine.Job: the created job
        """
        response = self._post("/jobs", data=json.dumps({"circuit": circuit}))
        if response.status_code == 201:
            return Job(
                id_=response.json()["id"],
                status=JobStatus(response.json()["status"]),
                connection=self,
            )
        raise RequestFailedError(
            "Job creation failed: {}".format(self._request_error_message(response))
        )

    def get_all_jobs(self, after: datetime = datetime(1970, 1, 1)) -> List[Job]:
        """Gets all jobs created by the user, optionally filtered by datetime.

        Args:
            after (datetime.datetime): if provided, only jobs more recently created
                                       then ``after`` are returned

        Returns:
            List[strawberryfields.engine.Job]: the jobs
        """
        response = self._get("/jobs?page[size]={}".format(self.MAX_JOBS_REQUESTED))
        if response.status_code == 200:
            return [
                Job(id_=info["id"], status=info["status"], connection=self)
                for info in response.json()["data"]
                if datetime.strptime(info["created_at"], self.JOB_TIMESTAMP_FORMAT)
                > after
            ]
        raise RequestFailedError(self._request_error_message(response))

    def get_job(self, job_id: str) -> Job:
        """Gets a job.

        Args:
            job_id (str): the job ID

        Returns:
            strawberryfields.engine.Job: the job
        """
        response = self._get("/jobs/{}".format(job_id))
        if response.status_code == 200:
            return Job(
                id_=response.json()["id"],
                status=JobStatus(response.json()["status"]),
                connection=self,
            )
        raise RequestFailedError(self._request_error_message(response))

    def get_job_status(self, job_id: str) -> JobStatus:
        """Returns the status of a job.

        Args:
            job_id (str): the job ID

        Returns:
            strawberryfields.engine.JobStatus: the job status
        """
        return JobStatus(self.get_job(job_id).status)

    def get_job_result(self, job_id: str) -> Result:
        """Returns the result of a job.

        Args:
            job_id (str): the job ID

        Returns:
            strawberryfields.engine.Result: the job result
        """
        response = self._get(
            "/jobs/{}/result".format(job_id), {"Accept": "application/x-numpy"},
        )
        if response.status_code == 200:
            # Read the numpy binary data in the payload into memory
            with io.BytesIO() as buf:
                buf.write(response.content)
                buf.seek(0)
                samples = np.load(buf)
            return Result(samples, is_stateful=False)
        raise RequestFailedError(self._request_error_message(response))

    def cancel_job(self, job_id: str):
        """Cancels a job.

        Args:
            job_id (str): the job ID
        """
        response = self._patch(
            "/jobs/{}".format(job_id), data={"status", JobStatus.CANCELLED.value}
        )
        if response.status_code == 204:
            return
        raise RequestFailedError(self._request_error_message(response))

    def ping(self) -> bool:
        """Tests the connection to the remote backend.

        Returns:
            bool: ``True`` if the connection is successful, and ``False`` otherwise
        """
        response = self._get("/healthz")
        return response.status_code == 200

    def _get(self, path: str, headers: Dict = None, **kwargs) -> requests.Response:
        return self._request(RequestMethod.GET, path, headers, **kwargs)

    def _post(self, path: str, headers: Dict = None, **kwargs) -> requests.Response:
        return self._request(RequestMethod.POST, path, headers, **kwargs)

    def _patch(self, path: str, headers: Dict = None, **kwargs) -> requests.Response:
        return self._request(RequestMethod.PATCH, path, headers, **kwargs)

    def _request(
        self, method: RequestMethod, path: str, headers: Dict = None, **kwargs
    ) -> requests.Response:
        headers = {} if headers is None else headers
        return getattr(requests, method.value)(
            urljoin(self.base_url, path),
            headers={"Authorization": self.token, **headers},
            **kwargs
        )

    @staticmethod
    def _request_error_message(response: requests.Response) -> str:
        body = response.json()
        return "{} ({}): {}".format(
            body.get("status_code", ""), body.get("code", ""), body.get("detail", "")
        )


class StarshipEngine:
    """A quantum program executor engine that that provides a simple interface for
    running remote jobs in a synchronous or asynchronous manner.

    **Example:**

    The following example instantiates an engine with the default configuration, and
    runs jobs both synchronously and asynchronously.

    .. code-block:: python

        engine = StarshipEngine("chip2")

        # Run a job synchronously
        result = engine.run(program, shots=1)
        # (Engine blocks until job is complete)
        result  # [[0, 1, 0, 2, 1, 0, 0, 0]]

        # Run a job synchronously, but cancel it before it is completed
        result = engine.run(program, shots=1)
        ^C # KeyboardInterrupt cancels the job

        # Run a job asynchronously
        job = engine.run_async(program, shots=1)
        job.status  # "queued"
        job.result  # InvalidJobOperationError
        # (After some time...)
        job.refresh()
        job.status  # "complete"
        job.result  # [[0, 1, 0, 2, 1, 0, 0, 0]]

    Args:
        target (str): the target backend
        connection (strawberryfields.engine.Connection): a connection to the remote job
            execution platform
    """

    POLLING_INTERVAL_SECONDS = 1
    VALID_TARGETS = ("chip2",)

    def __init__(self, target: str, connection: Connection = None):
        if target not in self.VALID_TARGETS:
            raise ValueError(
                "Invalid engine target: {} (valid targets: {})".format(
                    target, self.VALID_TARGETS
                )
            )
        if connection is None:
            # TODO use `load_config` when implemented
            config = DEFAULT_CONFIG["api"]
            connection = Connection(
                token=config["authentication_token"],
                host=config["hostname"],
                port=config["port"],
                use_ssl=config["use_ssl"],
            )

        self._target = target
        self._connection = connection

    @property
    def target(self) -> str:
        """The target backend used by the engine.

        Returns:
            str: the target backend used by the engine
        """
        return self._target

    @property
    def connection(self) -> Connection:
        """The connection object used by the engine.

        Returns:
            strawberryfields.engine.Connection: the connection object used by the engine
        """
        return self._connection

    def run(self, program: Program, shots: int = 1) -> Optional[Result]:
        """Runs a remote job synchronously.

        In the synchronous mode, the engine blocks until the job is completed, failed, or
        cancelled. If the job completes successfully, the result is returned; if the job
        fails or is cancelled, ``None`` is returned.

        Args:
            program (strawberryfields.Program): the quantum circuit
            shots (int): the number of shots for which to run the job

        Returns:
            [strawberryfields.engine.Result, None]: the job result if successful, and
                ``None`` otherwise
        """
        job = self.run_async(program, shots)
        try:
            while True:
                job.refresh()
                if job.status == JobStatus.COMPLETE:
                    return job.result
                if job.status == JobStatus.FAILED:
                    raise JobFailedError("The computation failed; please try again.")
                time.sleep(self.POLLING_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            self._connection.cancel_job(job.id)

    def run_async(self, program: Program, shots: int = 1) -> Job:
        """Runs a remote job asynchronously.

        In the asynchronous mode, a `Job` is returned immediately, and the user can
        manually refresh the status and result of the job.

        Args:
            program (strawberryfields.Program): the quantum circuit
            shots (int): the number of shots for which to run the job

        Returns:
            strawberryfields.engine.Job: the created remote job
        """
        bb = to_blackbird(program)
        # pylint: disable=protected-access
        bb._target["name"] = self.target
        # pylint: disable=protected-access
        bb._target["options"] = {"shots": shots}
        return self._connection.create_job(bb.serialize())


class Engine(LocalEngine):
    """dummy"""

    # alias for backwards compatibility
    __doc__ = LocalEngine.__doc__

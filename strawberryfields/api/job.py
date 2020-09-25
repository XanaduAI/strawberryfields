# Copyright 2020 Xanadu Quantum Technologies Inc.

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
This module provides classes for interfacing with program execution jobs on a remote backend.
"""
import enum

from strawberryfields.logger import create_logger

from .result import Result


class InvalidJobOperationError(Exception):
    """Raised when an invalid operation is performed on a job."""


class FailedJobError(Exception):
    """Raised when a job had a failure on the server side."""


class JobStatus(enum.Enum):
    """Represents the status of a remote job.

    This class maps a set of job statuses to the string representations returned by the
    remote platform.
    """

    OPEN = "open"
    QUEUED = "queued"
    CANCELLED = "cancelled"
    CANCEL_PENDING = "cancel_pending"
    COMPLETED = "complete"
    FAILED = "failed"

    @property
    def is_final(self) -> bool:
        """Checks if this status represents a final and immutable state.

        This method is primarily used to determine if an operation is valid for a given
        status.

        Returns:
            bool
        """
        return self in (JobStatus.CANCELLED, JobStatus.COMPLETED, JobStatus.FAILED)

    def __repr__(self) -> str:
        return "<JobStatus: {}>".format(self.value)

    def __str__(self) -> str:
        return self.__repr__()


class Job:
    """Represents a remote job that can be queried for its status or result.

    This object should typically not be instantiated directly, but returned by an
    :class:`strawberryfields.RemoteEngine` or :class:`strawberryfields.api.Connection`
    when a job is run.

    Args:
        id_ (str): the job ID
        status (strawberryfields.api.JobStatus): the job status
        connection (strawberryfields.api.Connection): the connection over which the
            job is managed
        meta (dict[str, str]): metadata related to job execution
    """

    def __init__(self, id_: str, status: JobStatus, connection, meta: dict = None):
        self._id = id_
        self._status = status
        self._connection = connection
        self._result = None
        self._meta = meta or {}

        self.log = create_logger(__name__)

    @property
    def id(self) -> str:
        """The job ID.

        Returns:
            str
        """
        return self._id

    @property
    def status(self) -> str:
        """Returns the current job status, with possible values including ``"open"``,
        ``"queued"``, ``"cancelled"``, ``"complete"``, and ``"failed"``.

        Returns:
            str
        """
        return self._status.value

    @property
    def result(self) -> Result:
        """Returns the :class:`~.Result` of a completed job.

        This is only defined for completed jobs, and raises an exception for any other
        status.

        Returns:
            strawberryfields.api.Result
        """
        if self._status != JobStatus.COMPLETED:
            raise AttributeError(
                "The result is undefined for jobs that are not completed "
                "(current status: {})".format(self._status.value)
            )
        return self._result

    @property
    def meta(self) -> dict:
        """Returns a dictionary of metadata on job execution, such as error
        details.

        Returns:
            dict[str, str]
        """
        return self._meta

    def refresh(self):
        """Refreshes the status and metadata of an open or queued job,
        along with the job result if the job is newly completed.
        """
        if self._status.is_final:
            self.log.warning("A %s job cannot be refreshed", self._status.value)
            return
        job_info = self._connection.get_job(self.id)
        self._status = JobStatus(job_info.status)
        self._meta = job_info.meta
        self.log.debug("Job %s metadata: %s", self.id, job_info.meta)
        if self._status == JobStatus.COMPLETED:
            self._result = self._connection.get_job_result(self.id)

    def cancel(self):
        """Cancels an open or queued job.

        Only an open or queued job can be cancelled; an exception is raised otherwise.
        """
        if self._status.is_final:
            raise InvalidJobOperationError(
                "A {} job cannot be cancelled".format(self._status.value)
            )
        self._connection.cancel_job(self.id)

    def __repr__(self):
        return "<{}: id={}, status={}>".format(self.__class__.__name__, self.id, self._status.value)

    def __str__(self):
        return self.__repr__()

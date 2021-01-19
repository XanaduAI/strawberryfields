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
This module provides an interface to a remote program execution backend.
"""
import io
from datetime import datetime
from typing import List, Dict

import numpy as np
import requests

from strawberryfields.configuration import load_config
from strawberryfields.io import to_blackbird
from strawberryfields.logger import create_logger
from strawberryfields.program import Program

from .job import Job, JobStatus
from .result import Result
from .devicespec import DeviceSpec

# pylint: disable=bad-continuation,protected-access


class RequestFailedError(Exception):
    """Raised when a request to the remote platform returns an error response."""


class Connection:
    """Manages remote connections to the remote job execution platform and exposes
    various job operations.

    For basic usage, it is not necessary to manually instantiate this object; the user
    is encouraged to use the higher-level interface provided by
    :class:`~strawberryfields.RemoteEngine`.

    **Example:**

    The following example instantiates a :class:`~strawberryfields.api.Connection` for a
    given API authentication token, tests the connection, submits a new job, and
    retrieves an existing job.

    >>> connection = Connection(token="abc")
    >>> success = connection.ping()  # `True` if successful, `False` if the connection fails
    >>> job = connection.create_job("chip_name", program, run_options={"shots":123})
    >>> job
    <Job: id=d177cbf5-1816-4779-802f-cef2c117dc1a, ...>
    >>> job.status
    "queued"
    >>> job.result
    AttributeError                            Traceback (most recent call last)
    <ipython-input-12-d54011f90b17> in <module>
    ...
    AttributeError: The result is undefined for jobs that are not completed (current status: queued)
    >>> job = connection.get_job(known_job_id)
    >>> job
    <Job: id=59a1c0b1-c6a7-4f9b-ae37-0ac5eec9c413, ...>
    >>> job.status
    "complete"
    >>> job.result
    [[0 1 0 2 1 0 0 0]]

    Args:
        token (str): the API authentication token
        host (str): the hostname of the remote platform
        port (int): the port to connect to on the remote host
        use_ssl (bool): whether to use SSL for the connection
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        token: str = None,
        host: str = None,
        port: int = None,
        use_ssl: bool = None,
        verbose: bool = True,
    ):
        default_config = load_config()

        self._token = token or default_config["api"]["authentication_token"]
        self._host = host or default_config["api"]["hostname"]
        self._port = port or default_config["api"]["port"]
        self._use_ssl = use_ssl or default_config["api"]["use_ssl"]
        self._verbose = verbose

        self._base_url = "http{}://{}:{}".format("s" if self.use_ssl else "", self.host, self.port)

        self._headers = {"Accept-Version": self.api_version}

        self.log = create_logger(__name__)

    @property
    def api_version(self) -> str:
        """str: The platform API version to request."""
        return "1.0.0"

    @property
    def token(self) -> str:
        """The API authentication token.

        Returns:
            str
        """
        return self._token

    @property
    def host(self) -> str:
        """The host for the remote platform.

        Returns:
            str
        """
        return self._host

    @property
    def port(self) -> int:
        """The port to connect to on the remote host.

        Returns:
            int
        """
        return self._port

    @property
    def use_ssl(self) -> bool:
        """Whether to use SSL for the connection.

        Returns:
            bool
        """
        return self._use_ssl

    def get_device_spec(self, target: str) -> DeviceSpec:
        """Gets the device specifications for target.

        Args:
            target (str): the target device

        Returns:
            strawberryfields.api.DeviceSpec: the created device specification
        """
        device_dict = self._get_device_dict(target)
        return DeviceSpec(target=target, spec=device_dict, connection=self)

    def _get_device_dict(self, target: str) -> dict:
        """Returns the device specifications as a dictionary"""
        path = f"/devices/{target}/specifications"
        response = self._request("GET", self._url(path), headers=self._headers)

        if response.status_code == 200:
            self.log.info("The device spec %s has been successfully retrieved.", target)
            return response.json()
        raise RequestFailedError(
            "Failed to get device specifications: {}".format(self._format_error_message(response))
        )

    def create_job(self, target: str, program: Program, run_options: dict = None) -> Job:
        """Creates a job with the given circuit.

        Args:
            target (str): the target device
            program (strawberryfields.Program): the quantum circuit
            run_options (Dict[str, Any]): Runtime arguments for the program execution.
                If provided, overrides the value stored in the given ``program``.

        Returns:
            strawberryfields.api.Job: the created job
        """
        # Serialize a blackbird circuit for network transmission
        bb = to_blackbird(program)
        bb._target["name"] = target
        bb._target["options"] = run_options

        circuit = bb.serialize()

        path = "/jobs"
        response = self._request(
            "POST", self._url(path), headers=self._headers, json={"circuit": circuit}
        )
        if response.status_code == 201:
            job_id = response.json()["id"]
            if self._verbose:
                self.log.info("Job %s was successfully submitted.", job_id)
            return Job(
                id_=job_id,
                status=JobStatus(response.json()["status"]),
                connection=self,
            )
        raise RequestFailedError(
            "Failed to create job: {}".format(self._format_error_message(response))
        )

    def get_all_jobs(self, after: datetime = datetime(1970, 1, 1)) -> List[Job]:
        """Gets a list of jobs created by the user, optionally filtered by datetime.

        Args:
            after (datetime.datetime): if provided, only jobs more recently created
                then ``after`` are returned

        Returns:
            List[strawberryfields.api.Job]: the jobs
        """
        raise NotImplementedError("This feature is not yet implemented")

    def get_job(self, job_id: str) -> Job:
        """Gets a job.

        Args:
            job_id (str): the job ID

        Returns:
            strawberryfields.api.Job: the job
        """
        path = "/jobs/{}".format(job_id)
        response = self._request("GET", self._url(path), headers=self._headers)
        if response.status_code == 200:
            return Job(
                id_=response.json()["id"],
                status=JobStatus(response.json()["status"]),
                connection=self,
                meta=response.json()["meta"],
            )
        raise RequestFailedError(
            "Failed to get job: {}".format(self._format_error_message(response))
        )

    def get_job_status(self, job_id: str) -> str:
        """Returns the status of a job.

        Args:
            job_id (str): the job ID

        Returns:
            str: the job status
        """
        return self.get_job(job_id).status

    def get_job_result(self, job_id: str) -> Result:
        """Returns the result of a job.

        Args:
            job_id (str): the job ID

        Returns:
            strawberryfields.api.Result: the job result
        """
        path = "/jobs/{}/result".format(job_id)
        response = self._request(
            "GET", self._url(path), headers={"Accept": "application/x-numpy", **self._headers}
        )
        if response.status_code == 200:
            # Read the numpy binary data in the payload into memory
            with io.BytesIO() as buf:
                buf.write(response.content)
                buf.seek(0)

                samples = np.load(buf, allow_pickle=False)

                if np.issubdtype(samples.dtype, np.integer):
                    # Samples represent photon numbers.
                    # Convert to int64, to avoid unexpected behaviour
                    # when users postprocess these samples.
                    samples = samples.astype(np.int64)

            return Result(samples, is_stateful=False)
        raise RequestFailedError(
            "Failed to get job result: {}".format(self._format_error_message(response))
        )

    def cancel_job(self, job_id: str):
        """Cancels a job.

        Args:
            job_id (str): the job ID
        """
        path = "/jobs/{}".format(job_id)
        response = self._request(
            "PATCH",
            self._url(path),
            headers=self._headers,
            json={"status": JobStatus.CANCELLED.value},
        )
        if response.status_code == 204:
            if self._verbose:
                self.log.info("The job %s was successfully cancelled.", job_id)
            return
        raise RequestFailedError(
            "Failed to cancel job: {}".format(self._format_error_message(response))
        )

    def ping(self) -> bool:
        """Tests the connection to the remote backend.

        Returns:
            bool: ``True`` if the connection is successful, and ``False`` otherwise
        """
        path = "/healthz"
        response = self._request("GET", self._url(path), headers=self._headers)
        return response.status_code == 200

    def _url(self, path: str) -> str:
        return self._base_url + path

    def _refresh_access_token(self):
        """Use the offline token to request a new access token."""
        self._headers.pop("Authorization", None)
        path = "/auth/realms/platform/protocol/openid-connect/token"
        headers = {**self._headers}
        response = requests.post(
            self._url(path),
            headers=headers,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._token,
                "client_id": "public",
            },
        )
        if response.status_code == 200:
            access_token = response.json().get("access_token")
            self._headers["Authorization"] = f"Bearer {access_token}"
        else:
            raise RequestFailedError(
                "Could not retrieve access token. Please check that your API key is correct."
            )

    def _request(self, method: str, path: str, headers: Dict = None, **kwargs):
        """Wrap all API requests with an authentication token refresh if a 401 status
        is received from the initial request.

        Args:
            method (str): the HTTP request method to use
            path (str): path of the endpoint to use
            headers (dict): dictionary containing the headers of the request

        Returns:
            requests.Response: the response received for the sent request
        """
        headers = headers or {}
        request_headers = {**headers, **self._headers}
        response = requests.request(method, path, headers=request_headers, **kwargs)
        if response.status_code == 401:
            # Refresh the access_token and retry the request
            self._refresh_access_token()
            request_headers = {**headers, **self._headers}
            response = requests.request(method, path, headers=request_headers, **kwargs)
        return response

    @staticmethod
    def _format_error_message(response: requests.Response) -> str:
        body = response.json()
        return "{} ({}): {} ({})".format(
            body.get("status_code", ""),
            body.get("code", ""),
            body.get("detail", ""),
            body.get("meta", ""),
        )

    def __repr__(self):
        return "<{}: token={}, host={}>".format(self.__class__.__name__, self.token, self.host)

    def __str__(self):
        return self.__repr__()

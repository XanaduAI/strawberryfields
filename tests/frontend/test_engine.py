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
r"""Unit tests for engine.py"""
from datetime import datetime
import io
from unittest.mock import MagicMock

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.engine import (
    Connection,
    InvalidJobOperationError,
    Job,
    JobStatus,
    RequestFailedError,
    Result,
    StarshipEngine,
)

pytestmark = pytest.mark.frontend

# pylint: disable=redefined-outer-name,no-self-use


@pytest.fixture
def prog():
    """A simple program for testing purposes."""
    program = sf.Program(2)
    with program.context as q:
        # pylint: disable=expression-not-assigned
        ops.Dgate(0.5) | q[0]
    return program


@pytest.fixture
def connection():
    """A mock connection object."""
    return Connection(token="token", host="host", port=123, use_ssl=True)


def mock_return(return_value):
    """A helper function for defining a mock function that returns the given value for
    any arguments.
    """
    return lambda *args, **kwargs: return_value


def mock_response(status_code, json_body=None, binary_body=None):
    """A helper function for creating a mock response with a JSON or binary body."""
    response = MagicMock()
    response.status_code = status_code
    if json_body:
        response.json.return_value = json_body
    if binary_body:
        response.content = binary_body
    return response


class MockServer:
    """A mock platform server that fakes a processing delay by counting requests."""

    REQUESTS_BEFORE_COMPLETE = 3

    def __init__(self):
        self.request_count = 0

    def get_job_status(self, _id):
        """Returns a 'queued' job status until the number of requests exceeds a defined
        threshold, beyond which a 'complete' job status is returned.
        """
        self.request_count += 1
        return (
            JobStatus.COMPLETE
            if self.request_count >= self.REQUESTS_BEFORE_COMPLETE
            else JobStatus.QUEUED
        )


class TestResult:
    """Tests for the ``Result`` class."""

    def stateless_result_raises_on_state_access(self):
        """Tests that `result.state` raises an error for a stateless result.
        """
        result = Result([[1, 2], [3, 4]], is_stateful=False)

        with pytest.raises(AttributeError):
            _ = result.state


class TestJob:
    """Tests for the ``Job`` class."""

    def incomplete_job_raises_on_result_access(self):
        """Tests that `job.result` raises an error for an incomplete job."""
        job = Job("abc", status=JobStatus.QUEUED, connection=Connection)

        with pytest.raises(AttributeError):
            _ = job.result

    def terminal_job_raises_on_refresh(self):
        """Tests that `job.refresh()` raises an error for a complete, failed, or
        cancelled job."""
        job = Job("abc", status=JobStatus.COMPLETE, connection=Connection)

        with pytest.raises(InvalidJobOperationError):
            job.refresh()

    def terminal_job_raises_on_cancel(self):
        """Tests that `job.cancel()` raises an error for a complete, failed, or
        aleady cancelled job."""
        job = Job("abc", status=JobStatus.COMPLETE, connection=Connection)

        with pytest.raises(InvalidJobOperationError):
            job.cancel()


class TestConnection:
    """Tests for the ``Connection`` class."""

    def test_init(self):
        """Tests that a ``Connection`` is initialized correctly."""
        token, host, port, use_ssl = "token", "host", 123, True
        connection = Connection(token, host, port, use_ssl)

        assert connection.token == token
        assert connection.host == host
        assert connection.port == port
        assert connection.use_ssl == use_ssl

        assert connection.base_url == "https://host:123"

    def test_create_job(self, connection, monkeypatch):
        """Tests a successful job creation flow."""
        id_, status = "123", JobStatus.QUEUED

        monkeypatch.setattr(
            Connection,
            "_post",
            mock_return(mock_response(201, {"id": id_, "status": status})),
        )

        job = connection.create_job("circuit")

        assert job.id == id_
        assert job.status == status

    def test_create_job_error(self, connection, monkeypatch):
        """Tests a failed job creation flow."""
        monkeypatch.setattr(Connection, "_post", mock_return(mock_response(400, {})))

        with pytest.raises(RequestFailedError):
            connection.create_job("circuit")

    def test_get_all_jobs(self, connection, monkeypatch):
        """Tests a successful job list request."""
        jobs = [
            {
                "id": str(i),
                "status": JobStatus.COMPLETE,
                "created_at": "2020-01-{:02d}T12:34:56.123456Z".format(i),
            }
            for i in range(1, 10)
        ]
        monkeypatch.setattr(
            Connection, "_get", mock_return(mock_response(200, {"data": jobs})),
        )

        jobs = connection.get_all_jobs(after=datetime(2020, 1, 5))

        assert [job.id for job in jobs] == [str(i) for i in range(5, 10)]

    def test_get_all_jobs_error(self, connection, monkeypatch):
        """Tests a failed job list request."""
        monkeypatch.setattr(Connection, "_get", mock_return(mock_response(404, {})))

        with pytest.raises(RequestFailedError):
            connection.get_all_jobs()

    def test_get_job(self, connection, monkeypatch):
        """Tests a successful job request."""
        id_, status = "123", JobStatus.COMPLETE

        monkeypatch.setattr(
            Connection,
            "_get",
            mock_return(mock_response(200, {"id": id_, "status": status.value})),
        )

        job = connection.get_job(id_)

        assert job.id == id_
        assert job.status == status

    def test_get_job_error(self, connection, monkeypatch):
        """Tests a failed job request."""
        monkeypatch.setattr(Connection, "_get", mock_return(mock_response(404, {})))

        with pytest.raises(RequestFailedError):
            connection.get_job("123")

    def test_get_job_status(self, connection, monkeypatch):
        """Tests a successful job status request."""
        id_, status = "123", JobStatus.COMPLETE

        monkeypatch.setattr(
            Connection,
            "_get",
            mock_return(mock_response(200, {"id": id_, "status": status.value})),
        )

        assert connection.get_job_status(id_) == status

    def test_get_job_status_error(self, connection, monkeypatch):
        """Tests a failed job status request."""
        monkeypatch.setattr(Connection, "_get", mock_return(mock_response(404, {})))

        with pytest.raises(RequestFailedError):
            connection.get_job_status("123")

    def test_get_job_result(self, connection, monkeypatch):
        """Tests a successful job result request."""
        result_samples = np.array([[1, 2], [3, 4]], dtype=np.int8)

        with io.BytesIO() as buf:
            np.save(buf, result_samples)
            buf.seek(0)
            monkeypatch.setattr(
                Connection,
                "_get",
                mock_return(mock_response(200, binary_body=buf.getvalue())),
            )

        result = connection.get_job_result("123")

        assert np.array_equal(result.samples, result_samples)

    def test_get_job_result_error(self, connection, monkeypatch):
        """Tests a failed job result request."""
        monkeypatch.setattr(Connection, "_get", mock_return(mock_response(404, {})))

        with pytest.raises(RequestFailedError):
            connection.get_job_result("123")

    def test_cancel_job(self, connection, monkeypatch):
        """Tests a successful job cancellation request."""
        monkeypatch.setattr(Connection, "_patch", mock_return(mock_response(204, {})))

        # A successful cancellation does not raise an exception
        connection.cancel_job("123")

    def test_cancel_job_error(self, connection, monkeypatch):
        """Tests a successful job cancellation request."""
        monkeypatch.setattr(Connection, "_patch", mock_return(mock_response(404, {})))

        with pytest.raises(RequestFailedError):
            connection.cancel_job("123")

    def test_ping_success(self, connection, monkeypatch):
        """Tests a successful ping to the remote host."""
        monkeypatch.setattr(Connection, "_get", mock_return(mock_response(200, {})))

        assert connection.ping()

    def test_ping_failure(self, connection, monkeypatch):
        """Tests a failed ping to the remote host."""
        monkeypatch.setattr(Connection, "_get", mock_return(mock_response(500, {})))

        assert not connection.ping()


class TestStarshipEngine:
    """Tests for the ``StarshipEngine`` class."""

    def test_run_complete(self, connection, prog, monkeypatch):
        """Tests a successful synchronous job execution."""
        id_, result_expected = "123", np.array([[1, 2], [3, 4]], dtype=np.int8)

        server = MockServer()
        monkeypatch.setattr(
            Connection,
            "create_job",
            mock_return(Job(id_=id_, status=JobStatus.OPEN, connection=connection)),
        )
        monkeypatch.setattr(Connection, "get_job_status", server.get_job_status)
        monkeypatch.setattr(
            Connection,
            "get_job_result",
            mock_return(Result(result_expected, is_stateful=False)),
        )

        engine = StarshipEngine("chip2", connection=connection)
        result = engine.run(prog)

        assert np.array_equal(result.samples.T, result_expected)

        with pytest.raises(AttributeError):
            _ = result.state

    def test_run_cancelled(self):
        """Tests a manual cancellation of synchronous job execution."""
        # TODO

    def test_run_async(self, connection, prog, monkeypatch):
        """Tests a successful asynchronous job execution."""
        id_, result_expected = "123", np.array([[1, 2], [3, 4]], dtype=np.int8)

        server = MockServer()
        monkeypatch.setattr(
            Connection,
            "create_job",
            mock_return(Job(id_=id_, status=JobStatus.OPEN, connection=connection)),
        )
        monkeypatch.setattr(Connection, "get_job_status", server.get_job_status)
        monkeypatch.setattr(
            Connection,
            "get_job_result",
            mock_return(Result(result_expected, is_stateful=False)),
        )

        engine = StarshipEngine("chip2", connection=connection)
        job = engine.run_async(prog)
        assert job.status == JobStatus.OPEN

        for _ in range(server.REQUESTS_BEFORE_COMPLETE):
            job.refresh()

        assert job.status == JobStatus.COMPLETE
        assert np.array_equal(job.result.samples.T, result_expected)

        with pytest.raises(AttributeError):
            _ = job.result.state

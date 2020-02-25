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
r"""Unit tests for engine.py"""
from datetime import datetime
import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.backends.base import BaseBackend
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

# pylint: disable=redefined-outer-name,no-self-use,bad-continuation,expression-not-assigned,pointless-statement


@pytest.fixture
def eng(backend):
    """Engine fixture."""
    return sf.LocalEngine(backend)


@pytest.fixture
def prog():
    """Program fixture."""
    prog = sf.Program(2)
    with prog.context as q:
        ops.Dgate(0.5) | q[0]
    return prog


class TestEngine:
    """Test basic engine functionality"""

    def test_load_backend(self):
        """Backend can be correctly loaded via strings"""
        eng = sf.LocalEngine("base")
        assert isinstance(eng.backend, BaseBackend)

    def test_bad_backend(self):
        """Backend must be a string or a BaseBackend instance."""
        with pytest.raises(
            TypeError, match="backend must be a string or a BaseBackend instance"
        ):
            _ = sf.LocalEngine(0)


class TestEngineProgramInteraction:
    """Test the Engine class and its interaction with Program instances."""

    def test_history(self, eng, prog):
        """Engine history."""
        # no programs have been run
        assert not eng.run_progs
        eng.run(prog)
        # one program has been run
        assert len(eng.run_progs) == 1
        assert eng.run_progs[-1] == prog  # no compilation required with BaseBackend

    def test_reset(self, eng, prog):
        """Running independent programs with an engine reset in between."""
        assert not eng.run_progs
        eng.run(prog)
        assert len(eng.run_progs) == 1

        eng.reset()
        assert not eng.run_progs
        p2 = sf.Program(3)
        with p2.context as q:
            ops.Rgate(1.0) | q[2]
        eng.run(p2)
        assert len(eng.run_progs) == 1

    def test_regref_mismatch(self, eng):
        """Running incompatible programs sequentially gives an error."""
        p1 = sf.Program(3)
        p2 = sf.Program(p1)
        p1.locked = False
        with p1.context as q:
            ops.Del | q[0]

        with pytest.raises(RuntimeError, match="Register mismatch"):
            eng.run([p1, p2])

    def test_sequential_programs(self, eng):
        """Running several program segments sequentially."""
        D = ops.Dgate(0.2)
        p1 = sf.Program(3)
        with p1.context as q:
            D | q[1]
            ops.Del | q[0]
        assert not eng.run_progs
        eng.run(p1)
        assert len(eng.run_progs) == 1

        # p2 succeeds p1
        p2 = sf.Program(p1)
        with p2.context as q:
            D | q[1]
        eng.run(p2)
        assert len(eng.run_progs) == 2

        # p2 does not alter the register so it can be repeated
        eng.run([p2] * 3)
        assert len(eng.run_progs) == 5

        eng.reset()
        assert not eng.run_progs

    def test_print_applied(self, eng):
        """Tests the printing of executed programs."""
        a = 0.23
        r = 0.1

        def inspect():
            res = []
            print_fn = lambda x: res.append(x.__str__())
            eng.print_applied(print_fn)
            return res

        p1 = sf.Program(2)
        with p1.context as q:
            ops.Dgate(a) | q[1]
            ops.Sgate(r) | q[1]

        eng.run(p1)
        expected1 = [
            "Run 0:",
            "Dgate({}, 0) | (q[1])".format(a),
            "Sgate({}, 0) | (q[1])".format(r),
        ]
        assert inspect() == expected1

        # run the program again
        eng.reset()
        eng.run(p1)
        assert inspect() == expected1

        # apply more commands to the same backend
        p2 = sf.Program(2)
        with p2.context as q:
            ops.Rgate(r) | q[1]

        eng.run(p2)
        expected2 = expected1 + ["Run 1:", "Rgate({}) | (q[1])".format(r)]
        assert inspect() == expected2

        # reapply history
        eng.reset()
        eng.run([p1, p2])
        assert inspect() == expected2


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

    REQUESTS_BEFORE_COMPLETED = 3

    def __init__(self):
        self.request_count = 0

    def get_job_status(self, _id):
        """Returns a 'queued' job status until the number of requests exceeds a defined
        threshold, beyond which a 'complete' job status is returned.
        """
        self.request_count += 1
        return (
            JobStatus.COMPLETED
            if self.request_count >= self.REQUESTS_BEFORE_COMPLETED
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

    def final_job_raises_on_refresh(self):
        """Tests that `job.refresh()` raises an error for a complete, failed, or
        cancelled job."""
        job = Job("abc", status=JobStatus.COMPLETED, connection=Connection)

        with pytest.raises(InvalidJobOperationError):
            job.refresh()

    def final_job_raises_on_cancel(self):
        """Tests that `job.cancel()` raises an error for a complete, failed, or
        aleady cancelled job."""
        job = Job("abc", status=JobStatus.COMPLETED, connection=Connection)

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

        assert connection._url("/abc") == "https://host:123/abc"

    def test_create_job(self, prog, connection, monkeypatch):
        """Tests a successful job creation flow."""
        id_, status = "123", JobStatus.QUEUED

        monkeypatch.setattr(
            requests,
            "post",
            mock_return(mock_response(201, {"id": id_, "status": status})),
        )

        job = connection.create_job("chip2", prog, 1)

        assert job.id == id_
        assert job.status == status

    def test_create_job_error(self, prog, connection, monkeypatch):
        """Tests a failed job creation flow."""
        monkeypatch.setattr(requests, "post", mock_return(mock_response(400, {})))

        with pytest.raises(RequestFailedError):
            connection.create_job("chip2", prog, 1)

    @pytest.mark.skip(reason="method not yet implemented")
    def test_get_all_jobs(self, connection, monkeypatch):
        """Tests a successful job list request."""
        jobs = [
            {
                "id": str(i),
                "status": JobStatus.COMPLETED,
                "created_at": "2020-01-{:02d}T12:34:56.123456Z".format(i),
            }
            for i in range(1, 10)
        ]
        monkeypatch.setattr(
            requests, "get", mock_return(mock_response(200, {"data": jobs})),
        )

        jobs = connection.get_all_jobs(after=datetime(2020, 1, 5))

        assert [job.id for job in jobs] == [str(i) for i in range(5, 10)]

    @pytest.mark.skip(reason="method not yet implemented")
    def test_get_all_jobs_error(self, connection, monkeypatch):
        """Tests a failed job list request."""
        monkeypatch.setattr(requests, "get", mock_return(mock_response(404, {})))

        with pytest.raises(RequestFailedError):
            connection.get_all_jobs()

    def test_get_job(self, connection, monkeypatch):
        """Tests a successful job request."""
        id_, status = "123", JobStatus.COMPLETED

        monkeypatch.setattr(
            requests,
            "get",
            mock_return(mock_response(200, {"id": id_, "status": status.value})),
        )

        job = connection.get_job(id_)

        assert job.id == id_
        assert job.status == status

    def test_get_job_error(self, connection, monkeypatch):
        """Tests a failed job request."""
        monkeypatch.setattr(requests, "get", mock_return(mock_response(404, {})))

        with pytest.raises(RequestFailedError):
            connection.get_job("123")

    def test_get_job_status(self, connection, monkeypatch):
        """Tests a successful job status request."""
        id_, status = "123", JobStatus.COMPLETED

        monkeypatch.setattr(
            requests,
            "get",
            mock_return(mock_response(200, {"id": id_, "status": status.value})),
        )

        assert connection.get_job_status(id_) == status

    def test_get_job_status_error(self, connection, monkeypatch):
        """Tests a failed job status request."""
        monkeypatch.setattr(requests, "get", mock_return(mock_response(404, {})))

        with pytest.raises(RequestFailedError):
            connection.get_job_status("123")

    def test_get_job_result(self, connection, monkeypatch):
        """Tests a successful job result request."""
        result_samples = np.array([[1, 2], [3, 4]], dtype=np.int8)

        with io.BytesIO() as buf:
            np.save(buf, result_samples)
            buf.seek(0)
            monkeypatch.setattr(
                requests,
                "get",
                mock_return(mock_response(200, binary_body=buf.getvalue())),
            )

        result = connection.get_job_result("123")

        assert np.array_equal(result.samples.T, result_samples)

    def test_get_job_result_error(self, connection, monkeypatch):
        """Tests a failed job result request."""
        monkeypatch.setattr(requests, "get", mock_return(mock_response(404, {})))

        with pytest.raises(RequestFailedError):
            connection.get_job_result("123")

    def test_cancel_job(self, connection, monkeypatch):
        """Tests a successful job cancellation request."""
        monkeypatch.setattr(requests, "patch", mock_return(mock_response(204, {})))

        # A successful cancellation does not raise an exception
        connection.cancel_job("123")

    def test_cancel_job_error(self, connection, monkeypatch):
        """Tests a successful job cancellation request."""
        monkeypatch.setattr(requests, "patch", mock_return(mock_response(404, {})))

        with pytest.raises(RequestFailedError):
            connection.cancel_job("123")

    def test_ping_success(self, connection, monkeypatch):
        """Tests a successful ping to the remote host."""
        monkeypatch.setattr(requests, "get", mock_return(mock_response(200, {})))

        assert connection.ping()

    def test_ping_failure(self, connection, monkeypatch):
        """Tests a failed ping to the remote host."""
        monkeypatch.setattr(requests, "get", mock_return(mock_response(500, {})))

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

        for _ in range(server.REQUESTS_BEFORE_COMPLETED):
            job.refresh()

        assert job.status == JobStatus.COMPLETED
        assert np.array_equal(job.result.samples.T, result_expected)

        with pytest.raises(AttributeError):
            _ = job.result.state

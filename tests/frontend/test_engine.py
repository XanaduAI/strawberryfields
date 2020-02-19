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

import pytest
from unittest.mock import MagicMock, call

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.backends.base import BaseBackend

from strawberryfields.configuration import Configuration
from strawberryfields.engine import (
    StarshipEngine,
    Connection,
    Job,
    JobStatus,
    Result,
    InvalidEngineTargetError,
    IncompleteJobError,
    CreateJobRequestError,
    GetAllJobsRequestError,
    GetJobRequestError,
    GetJobResultRequestError,
    GetJobCircuitRequestError,
    CancelJobRequestError,
    RefreshTerminalJobError,
    CancelTerminalJobError,
)

pytestmark = pytest.mark.frontend


@pytest.fixture
def prog(backend):
    """Program fixture."""
    prog = sf.Program(2)
    with prog.context as q:
        ops.Dgate(0.5) | q[0]
    return prog


def mock_return(return_value):
    return lambda *args, **kwargs: return_value


def mock_response(status_code, json_return_value):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_return_value
    return response


@pytest.fixture
def config():
    # TODO anything to do here?
    return Configuration()


# TODO should mock an actual http server here (e.g. with `http.server`)
class MockServer:
    # Fake a job processing delay
    REQUESTS_BEFORE_COMPLETE = 3

    def __init__(self):
        self.request_count = 0

    def get_job_status(self, _id):
        self.request_count += 1
        return (
            JobStatus.COMPLETE
            if self.request_count >= self.REQUESTS_BEFORE_COMPLETE
            else JobStatus.QUEUED
        )


class TestStarshipEngine:
    def test_run_complete(self, config, prog, monkeypatch):
        id_, result = "123", [[1, 2], [3, 4]]

        server = MockServer()
        monkeypatch.setattr(
            Connection,
            "create_job",
            mock_return(
                Job(id_=id_, status=JobStatus.OPEN, connection=Connection(config))
            ),
        )
        monkeypatch.setattr(Connection, "get_job_status", server.get_job_status)
        monkeypatch.setattr(Connection, "get_job_result", mock_return(Result(result)))

        engine = StarshipEngine("chip2", connection=Connection(config))
        job_result = engine.run(prog)

        assert job_result.samples.T.tolist() == result

    def test_run_cancelled(self, config, prog, monkeypatch):
        server = MockServer()
        # TODO

    def test_run_async(self):
        server = MockServer()
        # TODO


class TestConnection:
    @pytest.fixture
    def connection(self, config):
        return Connection(
            token=config.api["authentication_token"],
            host=config.api["hostname"],
            port=config.api["port"],
            use_ssl=config.api["use_ssl"],
        )

    def test_create_job(self, connection, monkeypatch):
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
        monkeypatch.setattr(Connection, "_post", mock_return(mock_response(400, {})))

        with pytest.raises(CreateJobRequestError):
            connection.create_job("circuit")

    def test_get_all_jobs(self, connection, monkeypatch):
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
        monkeypatch.setattr(Connection, "_get", mock_return(mock_response(404, {})))

        with pytest.raises(GetAllJobsRequestError):
            connection.get_all_jobs()

    def test_get_job(self, connection, monkeypatch):
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
        monkeypatch.setattr(Connection, "_get", mock_return(mock_response(404, {})))

        with pytest.raises(GetJobRequestError):
            connection.get_job("123")

    def test_get_job_status(self, connection, monkeypatch):
        id_, status = "123", JobStatus.COMPLETE

        monkeypatch.setattr(
            Connection,
            "_get",
            mock_return(mock_response(200, {"id": id_, "status": status.value})),
        )

        assert connection.get_job_status(id_) == status

    def test_get_job_status_error(self, connection, monkeypatch):
        monkeypatch.setattr(Connection, "_get", mock_return(mock_response(404, {})))

        with pytest.raises(GetJobRequestError):
            connection.get_job_status("123")

    def test_get_job_result(self, connection, monkeypatch):
        result_samples = [[1, 2], [3, 4]]

        monkeypatch.setattr(
            Connection,
            "_get",
            mock_return(mock_response(200, {"result": result_samples})),
        )

        result = connection.get_job_result("123")

        assert result.samples.T.tolist() == result_samples

    def test_get_job_result_error(self, connection, monkeypatch):
        monkeypatch.setattr(Connection, "_get", mock_return(mock_response(404, {})))

        with pytest.raises(GetJobResultRequestError):
            connection.get_job_result("123")

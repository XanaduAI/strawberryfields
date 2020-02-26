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
"""TODO"""
import numpy as np
import pytest

from strawberryfields.api import Connection, Job, JobStatus, Result
from strawberryfields.engine import StarshipEngine


def mock_return(return_value):
    """A helper function for defining a mock function that returns the given value for
    any arguments.
    """
    return lambda *args, **kwargs: return_value


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

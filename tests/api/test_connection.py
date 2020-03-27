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
Unit tests for strawberryfields.api.connection
"""
from datetime import datetime
import io

import numpy as np
import pytest
import requests

from strawberryfields.api import Connection, JobStatus, RequestFailedError
from .conftest import mock_return

# pylint: disable=no-self-use,unused-argument

pytestmark = pytest.mark.api


class MockResponse:
    """A mock response with a JSON or binary body."""

    def __init__(self, status_code, json_body=None, binary_body=None):
        self.status_code = status_code
        self.json_body = json_body
        self.binary_body = binary_body

    def json(self):
        """Mocks the ``requests.Response.json()`` method."""
        return self.json_body

    @property
    def content(self):
        """Mocks the ``requests.Response.content`` property."""
        return self.binary_body


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

        # pylint: disable=protected-access
        assert connection._url("/abc") == "https://host:123/abc"

    def test_create_job(self, prog, connection, monkeypatch):
        """Tests a successful job creation flow."""
        id_, status = "123", JobStatus.QUEUED

        monkeypatch.setattr(
            requests, "post", mock_return(MockResponse(201, {"id": id_, "status": status})),
        )

        job = connection.create_job("X8_01", prog, 1)

        assert job.id == id_
        assert job.status == status.value

    def test_create_job_error(self, prog, connection, monkeypatch):
        """Tests a failed job creation flow."""
        monkeypatch.setattr(requests, "post", mock_return(MockResponse(400, {})))

        with pytest.raises(RequestFailedError, match="Failed to create job"):
            connection.create_job("X8_01", prog, 1)

    @pytest.mark.xfail(reason="method not yet implemented")
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
            requests, "get", mock_return(MockResponse(200, {"data": jobs})),
        )

        jobs = connection.get_all_jobs(after=datetime(2020, 1, 5))

        assert [job.id for job in jobs] == [str(i) for i in range(5, 10)]

    @pytest.mark.xfail(reason="method not yet implemented")
    def test_get_all_jobs_error(self, connection, monkeypatch):
        """Tests a failed job list request."""
        monkeypatch.setattr(requests, "get", mock_return(MockResponse(404, {})))

        with pytest.raises(RequestFailedError, match="Failed to get all jobs"):
            connection.get_all_jobs()

    def test_get_job(self, connection, monkeypatch):
        """Tests a successful job request."""
        id_, status = "123", JobStatus.COMPLETED

        monkeypatch.setattr(
            requests, "get", mock_return(MockResponse(200, {"id": id_, "status": status.value})),
        )

        job = connection.get_job(id_)

        assert job.id == id_
        assert job.status == status.value

    def test_get_job_error(self, connection, monkeypatch):
        """Tests a failed job request."""
        monkeypatch.setattr(requests, "get", mock_return(MockResponse(404, {})))

        with pytest.raises(RequestFailedError, match="Failed to get job"):
            connection.get_job("123")

    def test_get_job_status(self, connection, monkeypatch):
        """Tests a successful job status request."""
        id_, status = "123", JobStatus.COMPLETED

        monkeypatch.setattr(
            requests, "get", mock_return(MockResponse(200, {"id": id_, "status": status.value})),
        )

        assert connection.get_job_status(id_) == status.value

    def test_get_job_status_error(self, connection, monkeypatch):
        """Tests a failed job status request."""
        monkeypatch.setattr(requests, "get", mock_return(MockResponse(404, {})))

        with pytest.raises(RequestFailedError, match="Failed to get job"):
            connection.get_job_status("123")

    @pytest.mark.parametrize(
        "result_dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float32,
            np.float64,
        ],
    )
    def test_get_job_result(self, connection, result_dtype, monkeypatch):
        """Tests a successful job result request."""
        result_samples = np.array([[1, 2], [3, 4]], dtype=result_dtype)

        with io.BytesIO() as buf:
            np.save(buf, result_samples)
            buf.seek(0)
            monkeypatch.setattr(
                requests, "get", mock_return(MockResponse(200, binary_body=buf.getvalue())),
            )

        result = connection.get_job_result("123")

        assert np.array_equal(result.samples, result_samples)

    def test_get_job_result_error(self, connection, monkeypatch):
        """Tests a failed job result request."""
        monkeypatch.setattr(requests, "get", mock_return(MockResponse(404, {})))

        with pytest.raises(RequestFailedError, match="Failed to get job result"):
            connection.get_job_result("123")

    def test_cancel_job(self, connection, monkeypatch):
        """Tests a successful job cancellation request."""
        # A custom `mock_return` that checks for expected arguments
        def _mock_return(return_value):
            def function(*args, **kwargs):
                assert kwargs.get("json") == {"status": "cancelled"}
                return return_value

            return function

        monkeypatch.setattr(requests, "patch", _mock_return(MockResponse(204, {})))

        # A successful cancellation does not raise an exception
        connection.cancel_job("123")

    def test_cancel_job_error(self, connection, monkeypatch):
        """Tests a failed job cancellation request."""
        monkeypatch.setattr(requests, "patch", mock_return(MockResponse(404, {})))

        with pytest.raises(RequestFailedError, match="Failed to cancel job"):
            connection.cancel_job("123")

    def test_ping_success(self, connection, monkeypatch):
        """Tests a successful ping to the remote host."""
        monkeypatch.setattr(requests, "get", mock_return(MockResponse(200, {})))

        assert connection.ping()

    def test_ping_failure(self, connection, monkeypatch):
        """Tests a failed ping to the remote host."""
        monkeypatch.setattr(requests, "get", mock_return(MockResponse(500, {})))

        assert not connection.ping()

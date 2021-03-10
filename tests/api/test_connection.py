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
import os

import numpy as np
import pytest
import requests

from strawberryfields.api import Connection, JobStatus, RequestFailedError
from strawberryfields import configuration as conf
from strawberryfields.compilers import Ranges

from .conftest import mock_return

# pylint: disable=no-self-use,unused-argument

pytestmark = pytest.mark.api

TEST_CONFIG_FILE_1 = """\
[api]
# Options for the Strawberry Fields Cloud API
authentication_token = "DummyToken"
hostname = "DummyHost"
use_ssl = false
port = 1234
"""

TEST_CONFIG_FILE_2 = """\
[api]
# Options for the Strawberry Fields Cloud API
authentication_token = "071cdcce-9241-4965-93af-4a4dbc739135"
hostname = "platform.strawberryfields.ai"
use_ssl = true
port = 443
"""

test_host = "SomeHost"
test_token = "SomeToken"

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

    def test_get_device_spec(self, prog, connection, monkeypatch):
        """Tests a successful device spec request."""
        target = "abc"
        layout = ""
        modes = 42
        compiler = []
        gate_parameters = {"param": Ranges([0, 1], variable_name="param")}

        monkeypatch.setattr(
            requests,
            "request",
            mock_return(MockResponse(
                200,
                {"layout": "", "modes": 42, "compiler": [], "gate_parameters": {"param": [[0, 1]]}}
            )),
        )

        device_spec = connection.get_device_spec(target)

        assert device_spec.target == target
        assert device_spec.layout == layout
        assert device_spec.modes == modes
        assert device_spec.compiler == compiler

        spec_params = device_spec.gate_parameters
        assert gate_parameters == spec_params

    def test_get_device_spec_error(self, connection, monkeypatch):
        """Tests a failed device spec request."""
        monkeypatch.setattr(requests, "request", mock_return(MockResponse(404, {})))

        with pytest.raises(RequestFailedError, match="Failed to get device specifications"):
            connection.get_device_spec("123")

    def test_create_job(self, prog, connection, monkeypatch):
        """Tests a successful job creation flow."""
        id_, status = "123", JobStatus.QUEUED

        monkeypatch.setattr(
            requests, "request", mock_return(MockResponse(201, {"id": id_, "status": status})),
        )

        job = connection.create_job("X8_01", prog, {"shots": 1})

        assert job.id == id_
        assert job.status == status.value

    def test_create_job_error(self, prog, connection, monkeypatch):
        """Tests a failed job creation flow."""
        monkeypatch.setattr(requests, "request", mock_return(MockResponse(400, {})))

        with pytest.raises(RequestFailedError, match="Failed to create job"):
            connection.create_job("X8_01", prog, {"shots": 1})

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
            requests, "request", mock_return(MockResponse(200, {"data": jobs})),
        )

        jobs = connection.get_all_jobs(after=datetime(2020, 1, 5))

        assert [job.id for job in jobs] == [str(i) for i in range(5, 10)]

    @pytest.mark.xfail(reason="method not yet implemented")
    def test_get_all_jobs_error(self, connection, monkeypatch):
        """Tests a failed job list request."""
        monkeypatch.setattr(requests, "request", mock_return(MockResponse(404, {})))

        with pytest.raises(RequestFailedError, match="Failed to get all jobs"):
            connection.get_all_jobs()

    def test_get_job(self, connection, monkeypatch):
        """Tests a successful job request."""
        id_, status, meta = "123", JobStatus.COMPLETED, {"abc": "def"}

        monkeypatch.setattr(
            requests,
            "request",
            mock_return(MockResponse(200, {"id": id_, "status": status.value, "meta": meta})),
        )

        job = connection.get_job(id_)

        assert job.id == id_
        assert job.status == status.value
        assert job.meta == meta

    def test_get_job_error(self, connection, monkeypatch):
        """Tests a failed job request."""
        monkeypatch.setattr(requests, "request", mock_return(MockResponse(404, {})))

        with pytest.raises(RequestFailedError, match="Failed to get job"):
            connection.get_job("123")

    def test_get_job_status(self, connection, monkeypatch):
        """Tests a successful job status request."""
        id_, status = "123", JobStatus.COMPLETED

        monkeypatch.setattr(
            requests,
            "request",
            mock_return(MockResponse(200, {"id": id_, "status": status.value, "meta": {}})),
        )

        assert connection.get_job_status(id_) == status.value

    def test_get_job_status_error(self, connection, monkeypatch):
        """Tests a failed job status request."""
        monkeypatch.setattr(requests, "request", mock_return(MockResponse(404, {})))

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
                requests, "request", mock_return(MockResponse(200, binary_body=buf.getvalue())),
            )

        result = connection.get_job_result("123")

        assert np.array_equal(result.samples, result_samples)

    def test_get_job_result_error(self, connection, monkeypatch):
        """Tests a failed job result request."""
        monkeypatch.setattr(requests, "request", mock_return(MockResponse(404, {})))

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

        monkeypatch.setattr(requests, "request", _mock_return(MockResponse(204, {})))

        # A successful cancellation does not raise an exception
        connection.cancel_job("123")

    def test_cancel_job_error(self, connection, monkeypatch):
        """Tests a failed job cancellation request."""
        monkeypatch.setattr(requests, "request", mock_return(MockResponse(404, {})))

        with pytest.raises(RequestFailedError, match="Failed to cancel job"):
            connection.cancel_job("123")

    def test_ping_success(self, connection, monkeypatch):
        """Tests a successful ping to the remote host."""
        monkeypatch.setattr(requests, "request", mock_return(MockResponse(200, {})))

        assert connection.ping()

    def test_ping_failure(self, connection, monkeypatch):
        """Tests a failed ping to the remote host."""
        monkeypatch.setattr(requests, "request", mock_return(MockResponse(500, {})))

        assert not connection.ping()

    def test_refresh_access_token(self, mocker, monkeypatch):
        """Test that the access token is created by passing the expected headers."""
        path = "/auth/realms/platform/protocol/openid-connect/token"

        data={
            "grant_type": "refresh_token",
            "refresh_token": test_token,
            "client_id": "public",
        }

        monkeypatch.setattr(requests, "post", mock_return(MockResponse(200, {})))
        spy = mocker.spy(requests, "post")

        conn = Connection(token=test_token, host=test_host)
        conn._refresh_access_token()
        expected_headers = {
            'Accept-Version': conn.api_version,
            'User-Agent': conn.user_agent,
        }
        expected_url = f"https://{test_host}:443{path}"
        spy.assert_called_once_with(expected_url, headers=expected_headers, data=data)

    def test_refresh_access_token_raises(self, monkeypatch):
        """Test that an error is raised when the access token could not be
        generated while creating the Connection object."""
        monkeypatch.setattr(requests, "post", mock_return(MockResponse(500, {})))
        conn = Connection(token=test_token, host=test_host)
        with pytest.raises(RequestFailedError, match="Could not retrieve access token"):
            conn._refresh_access_token()

    def test_wrapped_request_refreshes(self, mocker, monkeypatch):
        """Test that the _request method refreshes the access token when
        getting a 401 response."""
        # Mock post function used while refreshing
        monkeypatch.setattr(requests, "post", mock_return(MockResponse(200, {})))

        # Mock request function used for general requests
        monkeypatch.setattr(requests, "request", mock_return(MockResponse(401, {})))

        conn = Connection(token=test_token, host=test_host)

        spy = mocker.spy(conn, "_refresh_access_token")
        conn._request("SomeRequestMethod", "SomePath")
        spy.assert_called_once_with()


class TestConnectionIntegration:
    """Integration tests for using instances of the Connection."""

    def test_configuration_deleted_integration(self, monkeypatch, tmpdir):
        """Check that once two Connection instances indeed differ in their
        configuration if the configuration is being deleted in the meantime.

        The logic of the test goes as follows:
        1. Two temporary paths and files are being created
        2. The directories to be checked are mocked out to the temporary paths
        3. A connection object is created, using the configuration from the
        first config
        4. The first configuration is deleted, leaving the second config as
        default
        5. Another connection object is created, using the configuration from the
        second config
        6. Checks for the configuration for each Connection instances
        """
        test_file_name = "config.toml"

        # Creating the two temporary paths and files
        path1 = tmpdir.mkdir("sub1")
        path2 = tmpdir.mkdir("sub2")

        file1 = path1.join(test_file_name)
        file2 = path2.join(test_file_name)

        with open(file1, "w") as f:
            f.write(TEST_CONFIG_FILE_1)

        with open(file2, "w") as f:
            f.write(TEST_CONFIG_FILE_2)

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: path1)
            m.delenv("SF_CONF", raising=False)
            m.setattr(conf, "user_config_dir", lambda *args: path2)

            a = Connection()
            assert os.path.exists(file1)

            assert a.token == "DummyToken"
            assert a.host == "DummyHost"
            assert a.port == 1234
            assert a.use_ssl == False
            conf.delete_config(directory=path1)

            assert not os.path.exists(file1)

            b = Connection()
            assert b.token == "071cdcce-9241-4965-93af-4a4dbc739135"
            assert b.host == "platform.strawberryfields.ai"
            assert b.port == 443
            assert b.use_ssl == True

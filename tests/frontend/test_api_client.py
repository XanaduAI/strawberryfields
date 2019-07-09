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

r"""
Unit tests for API client
"""

import pytest
import json
from strawberryfields import api_client
from strawberryfields._dev import configuration
from strawberryfields.api_client import (
    requests,
    Job,
    ResourceManager,
    ObjectAlreadyCreatedException,
    MethodNotSupportedException,
)

from unittest.mock import MagicMock

pytestmark = pytest.mark.frontend

status_codes = requests.status_codes.codes


@pytest.fixture
def client():
    return api_client.APIClient()


SAMPLE_JOB_CREATE_RESPONSE = {
    "id": 29583,
    "status": "queued",
    "result_url": "https://platform.xanadu.ai/jobs/29583/result",
    "circuit_url": "https://platform.xanadu.ai/jobs/29583/circuit",
    "created_at": "2019-05-24T15:55:43.872531Z",
    "started_at": None,
    "finished_at": None,
    "running_time": None,
}

SAMPLE_JOB_RESPONSE = {
    "id": 19856,
    "status": "complete",
    "result_url": "https://platform.xanadu.ai/jobs/19856/result",
    "circuit_url": "https://platform.xanadu.ai/jobs/19856/circuit",
    "created_at": "2019-05-24T15:55:43.872531Z",
    "started_at": "2019-05-24T16:01:12.145636Z",
    "finished_at": "2019-05-24T16:01:12.145645Z",
    "running_time": "9µs",
}


class MockResponse:
    """
    A helper class to generate a mock response based on status code. Mocks
    the `json` and `text` attributes of a requests.Response class.
    """

    status_code = None

    def __init__(self, status_code):
        self.status_code = status_code

    def json(self):
        return self.possible_responses[self.status_code]

    @property
    def text(self):
        return json.dumps(self.json())

    def raise_for_status(self):
        raise requests.exceptions.HTTPError()


class MockPOSTResponse(MockResponse):
    possible_responses = {
        201: SAMPLE_JOB_CREATE_RESPONSE,
        400: {
            "code": "parse-error",
            "detail": (
                "The blackbird script could not be parsed. "
                "Please fix errors in the script and try again."
            ),
        },
        401: {"code": "unauthenticated", "detail": "Requires authentication"},
        409: {
            "code": "unsupported-circuit",
            "detail": ("This circuit is not compatible with the specified hardware."),
        },
        500: {
            "code": "server-error",
            "detail": ("Unexpected server error. Please try your request again " "later."),
        },
    }


class MockGETResponse(MockResponse):
    possible_responses = {
        200: SAMPLE_JOB_RESPONSE,
        401: {"code": "unauthenticated", "detail": "Requires authentication"},
        404: {
            "code": "not-found",
            "detail": "The requested resource could not be found or does not exist.",
        },
        500: {
            "code": "server-error",
            "detail": ("Unexpected server error. Please try your request again " "later."),
        },
    }

    status_code = None

    def __init__(self, status_code):
        self.status_code = status_code

    def json(self):
        return self.possible_responses[self.status_code]

    def raise_for_status(self):
        raise requests.exceptions.HTTPError()


@pytest.mark.api_client
class TestAPIClient:
    def test_init_default_client(self):
        """
        Test that initializing a default client generates an APIClient with the expected params.
        """
        client = api_client.APIClient()
        assert client.USE_SSL is True
        assert not client.AUTHENTICATION_TOKEN
        assert client.BASE_URL == "https://localhost"
        assert client.HEADERS["User-Agent"] == client.USER_AGENT

    def test_init_default_client_no_ssl(self):
        """
        Test setting use_ssl to False when initializing a client generates the correct base URL and
        sets the correct flag.
        """
        client = api_client.APIClient(use_ssl=False)
        assert client.USE_SSL is False
        assert not client.AUTHENTICATION_TOKEN
        assert client.BASE_URL == "http://localhost"
        assert client.HEADERS["User-Agent"] == client.USER_AGENT

    def test_init_custom_token_client(self):
        """
        Test that the token is correctly set when initializing a client.
        """
        test_token = "TEST"
        client = api_client.APIClient(authentication_token=test_token)
        assert client.AUTHENTICATION_TOKEN == test_token

    def test_init_custom_token_client_headers_set(self, monkeypatch):
        """
        Test that set_authentication_token is being called when setting a custom token.
        """
        test_token = "TEST"
        mock_set_authorization_header = MagicMock()
        monkeypatch.setattr(
            api_client.APIClient, "set_authorization_header", mock_set_authorization_header
        )
        api_client.APIClient(authentication_token=test_token)
        mock_set_authorization_header.assert_called_once_with(test_token)

    def test_set_authorization_header(self):
        """
        Test that the authentication token is added to the header correctly.
        """
        client = api_client.APIClient()

        authentication_token = MagicMock()
        client.set_authorization_header(authentication_token)
        assert client.HEADERS["Authorization"] == authentication_token

    def test_get_configuration_from_config(self, client, monkeypatch):
        """
        Test that the configuration is loaded from file correctly (not yet implemented).
        """
        mock_configuration = MagicMock()
        monkeypatch.setattr(configuration, "Configuration", mock_configuration.Configuration)
        assert client.get_configuration_from_config() == mock_configuration.Configuration().api

    def test_authenticate(self, client):
        """
        Test that the client can authenticate correctly (not yet implemented).
        """
        with pytest.raises(NotImplementedError):
            username = "TEST_USER"
            password = "TEST_PASSWORD"
            client.authenticate(username, password)

    def test_join_path(self, client):
        """
        Test that two paths can be joined and separated by a forward slash.
        """
        assert client.join_path("jobs") == "{client.BASE_URL}/jobs".format(client=client)


@pytest.mark.api_client
class TestResourceManager:
    def test_init(self):
        """
        Test that a resource manager instance can be initialized correctly with a resource and
        client instance. Assets that both manager.resource and manager.client are set.
        """
        resource = MagicMock()
        client = MagicMock()
        manager = ResourceManager(resource, client)

        assert manager.resource == resource
        assert manager.client == client

    def test_join_path(self):
        """
        Test that the resource path can be joined corectly with the base path.
        """
        mock_resource = MagicMock()
        mock_resource.PATH = "some-path"

        manager = ResourceManager(mock_resource, MagicMock())
        assert manager.join_path("test") == "some-path/test"

    def test_get_unsupported(self):
        """
        Test a GET request with a resource that does not support it. Asserts that
        MethodNotSupportedException is raised.
        """
        mock_resource = MagicMock()
        mock_resource.SUPPORTED_METHODS = ()
        manager = ResourceManager(mock_resource, MagicMock())
        with pytest.raises(MethodNotSupportedException):
            manager.get(1)

    def test_get(self, monkeypatch):
        """
        Test a successful GET request. Tests that manager.handle_response is being called with
        the correct Response object.
        """
        mock_resource = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.get = MagicMock(return_value=mock_response)

        mock_resource.SUPPORTED_METHODS = ("GET",)

        manager = ResourceManager(mock_resource, mock_client)
        monkeypatch.setattr(manager, "handle_response", MagicMock())

        manager.get(1)

        # TODO test that this is called with correct path
        mock_client.get.assert_called_once()
        manager.handle_response.assert_called_once_with(mock_response)

    def test_create_unsupported(self):
        """
        Test a POST (create) request with a resource that does not support that type or request.
        Asserts that MethodNotSupportedException is raised.
        """
        mock_resource = MagicMock()
        mock_resource.SUPPORTED_METHODS = ()
        manager = ResourceManager(mock_resource, MagicMock())
        with pytest.raises(MethodNotSupportedException):
            manager.create()

    def test_create_id_already_exists(self):
        """
        Tests that once an object is created, create method can not be called again. Asserts that
        ObjectAlreadyCreatedException is raised.
        """
        mock_resource = MagicMock()
        mock_resource.SUPPORTED_METHODS = ("POST",)
        mock_resource.id = MagicMock()
        manager = ResourceManager(mock_resource, MagicMock())
        with pytest.raises(ObjectAlreadyCreatedException):
            manager.create()

    def test_create(self, monkeypatch):
        """
        Tests a successful POST (create) method. Asserts that handle_response is called with the
        correct Response object.
        """
        mock_resource = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)

        mock_resource.SUPPORTED_METHODS = ("POST",)
        mock_resource.id = None

        manager = ResourceManager(mock_resource, mock_client)
        monkeypatch.setattr(manager, "handle_response", MagicMock())

        manager.create()

        # TODO test that this is called with correct path and params
        mock_client.post.assert_called_once()
        manager.handle_response.assert_called_once_with(mock_response)

    def test_handle_response(self, monkeypatch):
        """
        Tests that a successful response initiates a call to handle_success_response, and that an
        error response initiates a call to handle_error_response.
        """
        mock_resource = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_handle_success_response = MagicMock()
        mock_handle_error_response = MagicMock()

        manager = ResourceManager(mock_resource, mock_client)

        monkeypatch.setattr(manager, "handle_success_response", mock_handle_success_response)

        monkeypatch.setattr(manager, "handle_error_response", mock_handle_error_response)

        manager.handle_response(mock_response)
        assert manager.http_status_code == mock_response.status_code
        mock_handle_error_response.assert_called_once_with(mock_response)

        mock_response.status_code = 200
        manager.handle_response(mock_response)
        mock_handle_success_response.assert_called_once_with(mock_response)

    def test_handle_refresh_data(self):
        """
        Tests the ResourceManager.refresh_data method. Ensures that Field.set is called once with
        the correct data value.
        """
        mock_resource = MagicMock()
        mock_client = MagicMock()

        fields = [MagicMock() for i in range(5)]

        mock_resource.fields = {f: MagicMock() for f in fields}
        mock_data = {f.name: MagicMock() for f in fields}

        manager = ResourceManager(mock_resource, mock_client)

        manager.refresh_data(mock_data)

        for field in mock_resource.fields:
            field.set.assert_called_once_with(mock_data[field.name])


@pytest.mark.api_client
class TestJob:
    def test_create_created(self, monkeypatch):
        """
        Tests a successful Job creatioin with a mock POST response. Asserts that all fields on
        the Job instance have been set correctly and match the mock data.
        """
        monkeypatch.setattr(requests, "post", lambda url, headers, data: MockPOSTResponse(201))
        job = Job()
        job.manager.create(params={})

        keys_to_check = SAMPLE_JOB_CREATE_RESPONSE.keys()
        for key in keys_to_check:
            assert getattr(job, key).value == SAMPLE_JOB_CREATE_RESPONSE[key]

    def test_create_bad_request(self, monkeypatch):
        """
        Tests that the correct error code is returned when a bad request is sent to the server.
        """
        monkeypatch.setattr(requests, "post", lambda url, headers, data: MockPOSTResponse(400))
        job = Job()

        job.manager.create(params={})
        assert job.manager.http_status_code == 400

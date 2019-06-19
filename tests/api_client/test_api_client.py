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
from strawberryfields import api_client
from strawberryfields.api_client import (
    requests,
    ResourceManager,
    ObjectAlreadyCreatedException,
    MethodNotSupportedException,
)

from unittest.mock import MagicMock

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
    "running_time": "9µs"
}


class MockResponse:
    status_code = None

    def __init__(self, status_code):
        self.status_code = status_code

    def json(self):
        return self.possible_responses[self.status_code]

    def raise_for_status(self):
        raise requests.exceptions.HTTPError()


class MockPOSTResponse(MockResponse):
    possible_responses = {
        201: SAMPLE_JOB_CREATE_RESPONSE,
        400: {
            "code": "parse-error",
            "detail": (
                "The blackbird script could not be parsed. "
                "Please fix errors in the script and try again.")
        },
        401: {
            "code": "unauthenticated",
            "detail": "Requires authentication"
        },
        409: {
            "code": "unsupported-circuit",
            "detail": (
                "This circuit is not compatible with the specified hardware.")
        },
        500: {
            "code": "server-error",
            "detail": (
                "Unexpected server error. Please try your request again "
                "later.")
        },
    }


class MockGETResponse(MockResponse):
    possible_responses = {
        200: SAMPLE_JOB_RESPONSE,
        401: {
            "code": "unauthenticated",
            "detail": "Requires authentication"
        },
        404: {
            "code": "",
            "detail": "",
        },
        500: {
            "code": "server-error",
            "detail": (
                "Unexpected server error. Please try your request again "
                "later.")
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
        client = api_client.APIClient()
        assert client.USE_SSL is True
        assert client.AUTHENTICATION_TOKEN == ''
        assert client.BASE_URL == 'localhost'
        assert client.HEADERS == {}

    def test_init_custom_token_client(self):
        test_token = 'TEST'
        client = api_client.APIClient(authentication_token=test_token)
        assert client.AUTHENTICATION_TOKEN == test_token

    def test_load_configuration(self, client):
        with pytest.raises(NotImplementedError):
            client.load_configuration()

    def test_authenticate(self, client):
        with pytest.raises(NotImplementedError):
            username = 'TEST_USER'
            password = 'TEST_PASSWORD'
            client.authenticate(username, password)

    def test_set_authorization_header(self):
        assert True

    def test_join_path(self, client):
        assert client.join_path('jobs') == 'localhost/jobs'

    def test_get(self, client, monkeypatch):
        assert True

    def test_post(self, client):
        assert True


@pytest.mark.api_client
class TestResourceManager:
    def test_init(self):
        resource = MagicMock()
        client = MagicMock()
        manager = ResourceManager(resource, client)

        assert manager.resource == resource
        assert manager.client == client

    def test_join_path(self):
        mock_resource = MagicMock()
        mock_resource.PATH = 'some-path'

        manager = ResourceManager(mock_resource, MagicMock())
        assert manager.join_path('test') == "some-path/test"

    def test_get_unsupported(self):
        mock_resource = MagicMock()
        mock_resource.SUPPORTED_METHODS = ()
        manager = ResourceManager(mock_resource, MagicMock())
        with pytest.raises(MethodNotSupportedException):
            manager.get(1)

    def test_get(self, monkeypatch):
        mock_resource = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.get = MagicMock(return_value=mock_response)

        mock_resource.SUPPORTED_METHODS = ('GET',)

        manager = ResourceManager(mock_resource, mock_client)
        monkeypatch.setattr(manager, "handle_response", MagicMock())

        manager.get(1)

        # TODO test that this is called with correct path
        mock_client.get.assert_called_once()
        manager.handle_response.assert_called_once_with(mock_response)

    def test_create_unsupported(self):
        mock_resource = MagicMock()
        mock_resource.SUPPORTED_METHODS = ()
        manager = ResourceManager(mock_resource, MagicMock())
        with pytest.raises(MethodNotSupportedException):
            manager.create({})

    def test_create_id_already_exists(self):
        mock_resource = MagicMock()
        mock_resource.SUPPORTED_METHODS = ('POST',)
        mock_resource.id = MagicMock()
        manager = ResourceManager(mock_resource, MagicMock())
        with pytest.raises(ObjectAlreadyCreatedException):
            manager.create({})

    def test_create(self, monkeypatch):
        mock_resource = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)

        mock_resource.SUPPORTED_METHODS = ('POST',)
        mock_resource.id = None

        manager = ResourceManager(mock_resource, mock_client)
        monkeypatch.setattr(manager, "handle_response", MagicMock())

        manager.create({})

        # TODO test that this is called with correct path and params
        mock_client.post.assert_called_once()
        manager.handle_response.assert_called_once_with(mock_response)

    def test_handle_response(self, monkeypatch):
        mock_resource = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_handle_success_response = MagicMock()
        mock_handle_error_response = MagicMock()

        manager = ResourceManager(mock_resource, mock_client)

        monkeypatch.setattr(
            manager, "handle_success_response", mock_handle_success_response)

        monkeypatch.setattr(
            manager, "handle_error_response", mock_handle_error_response)

        manager.handle_response(mock_response)
        assert manager.http_status_code == mock_response.status_code
        mock_handle_error_response.assert_called_once_with(mock_response)

        mock_response.status_code = 200
        manager.handle_response(mock_response)
        mock_handle_success_response.assert_called_once_with(mock_response)

    def test_handle_refresh_data(self):
        mock_resource = MagicMock()
        mock_client = MagicMock()

        fields = (
            "id",
            "status",
            "result_url",
            "circuit_url",
            "created_at",
            "started_at",
            "finished_at",
            "running_time",
        )

        mock_resource.FIELDS = {f: MagicMock() for f in fields}
        mock_data = {f: MagicMock() for f in fields}

        manager = ResourceManager(mock_resource, mock_client)

        manager.refresh_data(mock_data)

        for key, value in mock_resource.FIELDS.items():
            value.assert_called_once()
            # TODO: test that the attributes on the resource were set correctly


@pytest.mark.api_client
class TestJob:
    def test_init(self):
        assert True

    def test_get(self):
        assert True

    def test_refresh_data(self):
        assert True

    def test_create_created(self, monkeypatch):
        monkeypatch.setattr(
            requests,
            "post",
            lambda url, headers, data: MockPOSTResponse(201))
        job = api_client.Job()
        job.manager.create(params={})

        keys_to_check = SAMPLE_JOB_CREATE_RESPONSE.keys()
        for key in keys_to_check:
            assert getattr(job, key) == SAMPLE_JOB_CREATE_RESPONSE[key]

    def test_create_bad_request(self, monkeypatch):
        monkeypatch.setattr(
            requests,
            "post",
            lambda url, headers, data: MockPOSTResponse(400))
        job = api_client.Job()

        job.manager.create(params={})
        assert job.manager.http_status_code == 400

    def test_join_path(self):
        assert True

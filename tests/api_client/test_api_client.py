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
from strawberryfields.api_client import requests

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


class MockCreatedResponse:
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

    status_code = None

    def __init__(self, status_code):
        self.status_code = status_code

    def json(self):
        return self.possible_responses[self.status_code]


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
class TestJob:
    def test_init(self):
        assert True

    def test_get(self):
        assert True

    def test_refresh_data(self):
        assert True

    def test_create(self, monkeypatch):
        monkeypatch.setattr(
            requests,
            "post",
            lambda url, headers, data: MockCreatedResponse(201))
        job = api_client.Job()
        job.manager.create(params={})
        assert job.id == SAMPLE_JOB_CREATE_RESPONSE['id']
        assert job.status == SAMPLE_JOB_CREATE_RESPONSE['status']
        assert job.result_url == SAMPLE_JOB_CREATE_RESPONSE['result_url']
        assert job.created_at == SAMPLE_JOB_CREATE_RESPONSE['created_at']
        assert job.started_at == SAMPLE_JOB_CREATE_RESPONSE['started_at']
        assert job.finished_at == SAMPLE_JOB_CREATE_RESPONSE['finished_at']
        assert job.running_time == SAMPLE_JOB_CREATE_RESPONSE['running_time']

    def test_join_path(self):
        assert True

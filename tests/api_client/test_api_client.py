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


@pytest.fixture
def client():
    return api_client.APIClient()


@pytest.mark.api_client
class TestAPIClient:
    def test_init_default_client(self):
        client = api_client.APIClient()
        assert client.USE_SSL is True
        assert client.AUTHENTICATION_TOKEN == ''
        assert client.BASE_URL == 'localhost/'
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

    def test_get(self, client):
        assert True

    def test_post(self, client):
        assert True


@pytest.mark.api_client
class TestJob:
    def test_init(self):
        assert True

    def test_get(self):
        assert True

    def test_update_job(self):
        assert True

    def test_create(self):
        assert True

    def test_join_path(self):
        assert True

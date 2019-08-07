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
import pytest
from unittest.mock import MagicMock

from strawberryfields.starship import Starship

pytestmark = pytest.mark.frontend


@pytest.fixture
def starship(monkeypatch):
    """
    Create a reusable Starship fixture without a real APIClient.
    """
    mock_api_client = MagicMock()
    monkeypatch.setattr("strawberryfields.starship.APIClient", mock_api_client)
    engine = Starship(polling_delay_seconds=0)
    return engine


class TestStarship:
    """
    Tests various methods on the remote engine Starship.
    """

    def test_init(self, monkeypatch):
        """
        Tests that a Starship instance is correctly initialized when additional APIClient
        parameters are passed.
        """
        mock_api_client = MagicMock()
        monkeypatch.setattr("strawberryfields.starship.APIClient", mock_api_client)
        engine = Starship()
        assert engine.client == mock_api_client()
        assert engine.jobs == []
        assert engine.threads == []
        assert engine.complete_jobs == []
        assert engine.failed_jobs == []

        assert engine.complete_jobs_queue.qsize() == 0
        assert engine.failed_jobs_queue.qsize() == 0

    def test__process_job_threading(self, starship, monkeypatch):
        mock__poll_for_job_results = MagicMock()
        mock_job = MagicMock()
        mock_job.is_processing = False
        mock_job.is_complete = True

        monkeypatch.setattr(
            "strawberryfields.starship.Starship._poll_for_job_results",
            mock__poll_for_job_results)

        test = starship._process_job(mock_job)
        assert starship.threads == [test]
        mock__poll_for_job_results.assert_called_once_with(mock_job)

    def test__process_job_synchronicity(self, starship, monkeypatch):
        mock_job = MagicMock()
        mock_thread = MagicMock()

        monkeypatch.setattr(
            "strawberryfields.starship.Thread", mock_thread)

        test = starship._process_job(mock_job, asynchronous=True)
        test.start.assert_called_once()
        test.join.assert_not_called()

        test.reset_mock()

        test = starship._process_job(mock_job, asynchronous=False)
        test.start.assert_called_once()
        test.join.assert_called_once()

    def test__send_program_as_job(self, starship, monkeypatch):
        methods = MagicMock()
        mock_program = MagicMock()

        monkeypatch.setattr("strawberryfields.starship.to_blackbird", methods.to_blackbird)
        monkeypatch.setattr(starship, "_create_job", methods._create_job)

        test = starship._send_program_as_job(mock_program)

        mock_job_content = methods.to_blackbird(mock_program, version="1.0").serialize()
        mock_job = methods._create_job(mock_job_content)

        assert starship.jobs == [test]
        assert test == mock_job

    def test__compile_program(self, starship, monkeypatch):
        mock_program = MagicMock()
        compile_options = {'some_parameter': MagicMock()}

        test = starship._compile_program(mock_program, compile_options=compile_options)

        mock_program.compile.assert_called_once_with(
            starship.backend.circuit_spec, **compile_options)

        assert test == mock_program.compile(starship.backend.circuit_spec, **compile_options)
        test.lock.assert_called_once()

    def test__create_job(self, starship, monkeypatch):
        mock_job = MagicMock()
        mock_job_content = MagicMock()
        monkeypatch.setattr("strawberryfields.starship.Job", mock_job)

        test = starship._create_job(mock_job_content)

        assert test == mock_job(client=starship.client)
        test.manager.create.assert_called_once_with(circuit=mock_job_content)

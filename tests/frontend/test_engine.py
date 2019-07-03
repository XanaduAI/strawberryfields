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

pytestmark = pytest.mark.frontend

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.backends.base import BaseBackend
from unittest.mock import MagicMock
from strawberryfields import StarshipEngine
from strawberryfields.api_client import APIClient
from strawberryfields.ops import S2gate, MeasureFock, Rgate, BSgate


@pytest.fixture
def eng(backend):
    """Engine fixture."""
    return sf.LocalEngine(backend)


@pytest.fixture
def prog(backend):
    """Program fixture."""
    prog = sf.Program(2)
    with prog.context as q:
        ops.Dgate(0.5) | q[0]
    return prog


@pytest.fixture
def starship_engine(monkeypatch):
    """
    Create a reusable StarshipEngine fixture without a real APIClient.
    """
    mock_api_client = MagicMock()
    monkeypatch.setattr("strawberryfields.engine.APIClient", mock_api_client)
    engine = StarshipEngine()
    monkeypatch.setattr(engine, "API_DEFAULT_REFRESH_SECONDS", 0)
    return engine


class TestEngine:
    """Test basic engine functionality"""

    def test_load_backend(self):
        """Backend can be correctly loaded via strings"""
        eng = sf.LocalEngine("base")
        assert isinstance(eng.backend, BaseBackend)

    def test_bad_backend(self):
        """Backend must be a string or a BaseBackend instance."""
        with pytest.raises(TypeError, match="backend must be a string or a BaseBackend instance"):
            eng = sf.LocalEngine(0)


class TestEngineProgramInteraction:
    """Test the Engine class and its interaction with Program instances."""

    def test_history(self, eng, prog):
        """Engine history."""
        # no programs have been run
        assert not eng.run_progs
        eng.run(prog)
        # one program has been run
        assert len(eng.run_progs) == 1
        assert eng.run_progs[-1].source == prog

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
        expected1 = ["Run 0:", "Dgate({}, 0) | (q[1])".format(a), "Sgate({}, 0) | (q[1])".format(r)]
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


class TestStarshipEngine:
    """
    Tests various methods on the remote engine StarshipEngine.
    """

    def test_init(self, monkeypatch):
        """
        Tests that a StarshipEngine instance is correctly initialized when additional APIClient
        parameters are passed.
        """
        mock_api_client = MagicMock()
        mock_api_client_params = {"param": MagicMock()}
        monkeypatch.setattr("strawberryfields.engine.APIClient", mock_api_client)
        engine = StarshipEngine(mock_api_client_params)
        mock_api_client.assert_called_once_with(param=mock_api_client_params["param"])
        assert engine.jobs == []

    def test_reset(self, starship_engine):
        """
        Tests that StarshipEngine.jobs is correctly cleared when callling StarshipEngine.reset.
        """
        starship_engine.jobs.append(MagicMock())
        assert len(starship_engine.jobs) == 1
        starship_engine.reset()
        assert len(starship_engine.jobs) == 0

    def test_generate_job_content(self, starship_engine):
        """
        Tests that StarshipEngine.generate_job_content returns the correct string given name,
        shots, and blackbird_code parameters.
        """
        name = MagicMock()
        shots = MagicMock()
        blackbird_code = MagicMock()

        output = starship_engine.generate_job_content(name, shots, blackbird_code)
        lines = output.split("\n")
        assert lines[0] == "name {}".format(name)
        assert lines[1] == "version 1.0"
        assert lines[2] == "target {} (shots={})".format(starship_engine.backend_name, shots)
        assert lines[3] == ""
        assert lines[4] == str(blackbird_code)

    def test_queue_job(self, starship_engine, monkeypatch):
        mock_job = MagicMock()
        monkeypatch.setattr("strawberryfields.engine.Job", mock_job)
        mock_job_content = MagicMock()

        result = starship_engine._queue_job(mock_job_content)
        mock_job.assert_called_once_with(client=starship_engine.client)
        result.manager.create.assert_called_once_with(circuit=mock_job_content)
        assert starship_engine.jobs == [result]

    def test__run_program(self, starship_engine, monkeypatch):
        """
        Tests StarshipEngine._run_program. Asserts that a program is converted to blackbird code,
        compiled into a job content string and that the job is queued. Also asserts that a
        completed job's result samples are returned.
        """
        mock_to_blackbird = MagicMock()
        mock_generate_job_content = MagicMock()
        program = MagicMock()
        mock_job = MagicMock()
        mock_job.is_complete = True
        mock_job.is_failed = False

        monkeypatch.setattr("strawberryfields.engine.to_blackbird", mock_to_blackbird)
        monkeypatch.setattr(starship_engine, "generate_job_content", mock_generate_job_content)
        monkeypatch.setattr(starship_engine, "_queue_job", lambda job_content: mock_job)

        some_params = {"param": MagicMock()}
        result = starship_engine._run_program(program, **some_params)

        mock_to_blackbird.assert_called_once_with(program)
        mock_generate_job_content.assert_called_once_with(
            blackbird_code=mock_to_blackbird(program), param=some_params["param"]
        )

        assert result == mock_job.result.result.value

    def test__run_program_fails(self, starship_engine, monkeypatch):
        """
        Tests that an Exception is raised when a job has failed.
        """
        mock_to_blackbird = MagicMock()
        mock_generate_job_content = MagicMock()
        program = MagicMock()
        mock_job = MagicMock()
        mock_job.is_complete = False
        mock_job.is_failed = True

        monkeypatch.setattr("strawberryfields.engine.to_blackbird", mock_to_blackbird)
        monkeypatch.setattr(starship_engine, "generate_job_content", mock_generate_job_content)
        monkeypatch.setattr(starship_engine, "_queue_job", lambda job_content: mock_job)

        some_params = {"param": MagicMock()}

        with pytest.raises(Exception):
            starship_engine._run_program(program, **some_params)

    def test__run(self, starship_engine, monkeypatch):
        """
        Tests StarshipEngine._run, with the assumption that the backend is a hardware backend
        that supports running only a single program. This test ensures that a program is compiled
        for the hardware backend, is locked, is added to self.run_progs, that it is run and that
        a Result object is returned populated with the result samples.
        """

        mock_hardware_backend_name = str(MagicMock())
        mock_run_program = MagicMock()
        mock_program = MagicMock()
        mock_result = MagicMock()
        mock_shots = MagicMock()

        monkeypatch.setattr(starship_engine, "backend_name", mock_hardware_backend_name)
        monkeypatch.setattr(starship_engine, "HARDWARE_BACKENDS", [mock_hardware_backend_name])
        monkeypatch.setattr(starship_engine, "_run_program", mock_run_program)
        monkeypatch.setattr("strawberryfields.engine.Result", mock_result)

        result = starship_engine._run(mock_program, shots=mock_shots)

        assert starship_engine.backend_name in starship_engine.HARDWARE_BACKENDS
        mock_program.compile.assert_called_once_with(starship_engine.backend_name)
        mock_compiled_program = mock_program.compile(starship_engine.backend_name)
        mock_compiled_program.lock.assert_called_once()
        mock_run_program.assert_called_once_with(mock_compiled_program, shots=mock_shots)
        mock_samples = mock_run_program(mock_compiled_program, shots=mock_shots)
        assert starship_engine.run_progs == [mock_compiled_program]
        assert result == mock_result(mock_samples)

    def test_run(self, starship_engine, monkeypatch):
        """
        Tests StarshipEngine.run. It is expected that StarshipEngine._run is called with the correct
        parameters.
        """
        mock_run = MagicMock()
        monkeypatch.setattr("strawberryfields.engine.BaseEngine._run", mock_run)

        name = MagicMock()
        program = MagicMock()
        shots = MagicMock()
        params = {"param": MagicMock()}

        starship_engine.run(program, shots, name, **params)
        mock_run.assert_called_once_with(program, shots=shots, name=name, param=params["param"])

    def test_engine_with_mocked_api_client_sample_job(self, monkeypatch):
        """
        This is an integration test that tests and actual program being submitted to a mock API, and
        how the engine handles a successful response from the server (first by queuing a job then by
        fetching the result.)
        """

        # NOTE: this is currently more of an integration test, currently a WIP / under development.

        api_client_params = {"hostname": "localhost"}
        engine = StarshipEngine(api_client_params)
        monkeypatch.setattr(engine, "API_DEFAULT_REFRESH_SECONDS", 0)

        # We don't want to actually send any requests, though we should make sure POST was called
        mock_api_client_post = MagicMock()
        mock_get = MagicMock()
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"status": "COMPLETE", "id": 1234}
        mock_get.return_value = mock_get_response

        mock_post_response = MagicMock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = {"status": "QUEUED", "id": 1234}
        mock_api_client_post.return_value = mock_post_response

        monkeypatch.setattr(APIClient, "post", mock_api_client_post)
        monkeypatch.setattr(APIClient, "get", mock_get)

        prog = sf.Program(4)
        with prog.context as q:
            S2gate(2) | [0, 2]
            S2gate(2) | [1, 3]
            Rgate(3) | 0
            BSgate() | [0, 1]
            Rgate(3) | 0
            Rgate(3) | 1
            Rgate(3) | 2
            BSgate() | [2, 3]
            Rgate(3) | 2
            Rgate(3) | 3
            MeasureFock() | [0]
            MeasureFock() | [1]
            MeasureFock() | [2]
            MeasureFock() | [3]

        engine.run(prog)

        mock_api_client_post.assert_called_once()

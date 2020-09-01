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
Unit tests for strawberryfields.engine.RemoteEngine
"""
import numpy as np
import pytest

from strawberryfields.api import Connection, Job, JobStatus, Result
from strawberryfields.engine import RemoteEngine

from .conftest import mock_return

# pylint: disable=bad-continuation,unused-argument,no-self-use,redefined-outer-name,pointless-statement

pytestmark = pytest.mark.api


class MockServer:
    """A mock platform server that fakes a processing delay by counting requests."""

    REQUESTS_BEFORE_COMPLETED = 3

    def __init__(self):
        self.request_count = 0

    def get_job(self, _id):
        """Returns a 'queued' job status until the number of requests exceeds a defined
        threshold, beyond which a 'complete' job status is returned.
        """
        self.request_count += 1
        status = (
            JobStatus.COMPLETED
            if self.request_count >= self.REQUESTS_BEFORE_COMPLETED
            else JobStatus.QUEUED
        )
        return Job(id_="123", status=status, connection=None, meta={"foo": "bar"})


@pytest.fixture
def job_to_complete(connection, monkeypatch):
    """Mocks a remote job that is completed after a certain number of requests."""
    monkeypatch.setattr(
        Connection,
        "create_job",
        mock_return(Job(id_="123", status=JobStatus.OPEN, connection=connection)),
    )
    server = MockServer()
    monkeypatch.setattr(Connection, "get_job", server.get_job)
    monkeypatch.setattr(
        Connection,
        "get_job_result",
        mock_return(Result(np.array([[1, 2], [3, 4]]), is_stateful=False)),
    )


class TestRemoteEngine:
    """Tests for the ``RemoteEngine`` class."""

    def test_run_complete(self, connection, prog, job_to_complete):
        """Tests a successful blocking job execution."""
        engine = RemoteEngine("X8_01", connection=connection)
        result = engine.run(prog, shots=10)

        assert np.array_equal(result.samples, np.array([[1, 2], [3, 4]]))

        with pytest.raises(
            AttributeError, match="The state is undefined for a stateless computation."
        ):
            result.state

    def test_run_async(self, connection, prog, job_to_complete):
        """Tests a successful non-blocking job execution."""

        engine = RemoteEngine("X8_01", connection=connection)
        job = engine.run_async(prog, shots=10)
        assert job.status == JobStatus.OPEN.value

        for _ in range(MockServer.REQUESTS_BEFORE_COMPLETED):
            job.refresh()

        assert job.status == JobStatus.COMPLETED.value
        assert job.meta == {"foo": "bar"}
        assert np.array_equal(job.result.samples, np.array([[1, 2], [3, 4]]))

        with pytest.raises(
            AttributeError, match="The state is undefined for a stateless computation."
        ):
            job.result.state

    def test_device_class_target(self):
        """Test that the remote engine correctly instantiates itself
        when provided with a non-specific target"""
        target = "X8"
        engine = RemoteEngine(target)
        assert engine.target == engine.DEFAULT_TARGETS[target]

    def test_run_options_from_kwargs(self, prog, monkeypatch):
        """Test that the remote engine run_async method correctly
        passes all keyword argument backend and runtime options to the create_job
        method."""
        monkeypatch.setattr(Connection, "create_job", lambda *args: args)
        engine = RemoteEngine("X8", backend_options={"cutoff_dim": 12})
        _, _, _, run_options = engine.run_async(prog, shots=1234)
        assert run_options == {"shots": 1234, "cutoff_dim": 12}

        # run options from keyword arguments overwrite
        # run options provided by the program object
        prog = prog.compile("X8", shots=15)
        _, _, _, run_options = engine.run_async(prog, shots=1234)
        assert run_options == {"shots": 1234, "cutoff_dim": 12}

    def test_run_options_from_program(self, prog, monkeypatch):
        """Test that the remote engine run_async method correctly
        parses runtime options compiled into the program"""
        monkeypatch.setattr(Connection, "create_job", lambda *args: args)
        engine = RemoteEngine("X8")

        prog = prog.compile("X8", shots=15)
        assert prog.run_options == {"shots": 15}

        _, _, _, run_options = engine.run_async(prog)
        assert run_options == {"shots": 15}

    def test_no_shots(self, prog, connection):
        """Test that if the number of shots is not provided, an
        exception is raised"""
        engine = RemoteEngine("X8", connection=connection)

        with pytest.raises(ValueError, match="Number of shots must be specified"):
            engine.run_async(prog)


class TestRemoteEngineIntegration:
    """Integration tests for the remote engine"""

    def test_compilation(self, prog, monkeypatch):
        """Test that the remote engine correctly compiles a program
        for the intended backend"""
        monkeypatch.setattr(Connection, "create_job", lambda *args: args)

        engine = RemoteEngine("X8")
        _, target, res_prog, _ = engine.run_async(prog, shots=10)

        assert target == RemoteEngine.DEFAULT_TARGETS["X8"]

        # check program is compiled to match the chip template
        expected = prog.compile("X8").circuit
        res = res_prog.circuit

        for cmd1, cmd2 in zip(res, expected):
            # loop through all commands in res and expected

            # check gates are the same
            assert type(cmd1.op) is type(cmd2.op)
            # check modes are the same
            assert all(i.ind == j.ind for i, j in zip(cmd1.reg, cmd2.reg))
            # check parameters are the same
            assert all(p1 == p2 for p1, p2 in zip(cmd1.op.p, cmd2.op.p))

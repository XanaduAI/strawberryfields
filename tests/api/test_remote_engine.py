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
Unit and integration tests for :mod:`strawberryfields.engine.RemoteEngine`.
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
import xcc

import strawberryfields as sf
from strawberryfields.devicespec import DeviceSpec
from strawberryfields.result import Result
from strawberryfields.engine import RemoteEngine

from .conftest import mock_return

# pylint: disable=bad-continuation,unused-argument,no-self-use,redefined-outer-name,pointless-statement

pytestmark = pytest.mark.api


REQUESTS_BEFORE_COMPLETED = 3


@pytest.fixture
def job(connection, monkeypatch):
    """Mocks a job which completes after a certain number of requests."""
    refresh_counter = 0

    @property
    def finished(self):
        nonlocal refresh_counter
        refresh_counter += 1
        if refresh_counter >= REQUESTS_BEFORE_COMPLETED:
            self._details = {"status": "complete"}
            return True
        return False

    _details = {"status": "open"}

    job = xcc.Job(id_="123", connection=connection)
    job._details = {"status": "open"}

    result = {"output": [np.array([[1, 2], [3, 4]])], "foo": [np.array([5, 6])]}

    monkeypatch.setattr(xcc.Job, "submit", mock_return(job))
    monkeypatch.setattr(xcc.Job, "result", result)
    monkeypatch.setattr(xcc.Job, "clear", mock_return(None))
    monkeypatch.setattr(xcc.Job, "finished", finished)
    return job


@pytest.fixture
def device(connection, monkeypatch):
    """Mocks an X8 device with the "fock" compiler."""
    device = xcc.Device(target="X8_01", connection=connection)
    device.specification = {
        "target": "X8_01",
        "layout": "",
        "modes": 8,
        "compiler": ["fock"],
        "gate_parameters": {},
    }

    monkeypatch.setattr(sf.engine.xcc, "Connection", MagicMock())
    monkeypatch.setattr(sf.engine.xcc, "Device", mock_return(device))
    return device


@pytest.fixture
def blackbird(monkeypatch):
    """Mocks a Blackbird program."""
    blackbird = MagicMock()
    blackbird.return_value = blackbird
    blackbird._target = {}
    monkeypatch.setattr(sf.engine, "to_blackbird", blackbird)
    return blackbird


@pytest.fixture
def infolog(caplog):
    """Sets the log capture level to ``logging.INFO``."""
    caplog.set_level(logging.INFO)
    return caplog


@pytest.mark.usefixtures("job", "device")
class TestRemoteEngine:
    """Tests for the ``RemoteEngine`` class."""

    def test_generic_target(self):
        """Test that :meth:`RemoteEngine.__init__` resolves the correct target
        when instantiated with a non-specific target.
        """
        engine = RemoteEngine("X8")
        assert engine.target == engine.DEFAULT_TARGETS["X8"]

    def test_run(self, prog):
        """Tests that a blocking job execution can succeed."""
        engine = RemoteEngine("X8_01")
        result = engine.run(prog, shots=10)

        assert result is not None
        assert np.array_equal(result.samples, [[1, 2], [3, 4]])

        result.state is None

    def test_run_async(self, prog):
        """Tests that a non-blocking job execution can succeed."""
        engine = RemoteEngine("X8_01")
        job = engine.run_async(prog, shots=10)

        # job.status calls job.finished, incrementing the request counter
        assert job.status == "open"

        for _ in range(REQUESTS_BEFORE_COMPLETED - 1):
            assert job.finished is False
        assert job.finished is True

        assert job.status == "complete"
        assert np.array_equal(job.result["foo"], [np.array([5, 6])])
        assert np.array_equal(job.result["output"], [np.array([[1, 2], [3, 4]])])

        result = Result(job.result)
        result.state is None

    def test_run_async_options_from_kwargs(self, prog, blackbird):
        """Tests that :meth:`RemoteEngine.run_async` passes all keyword
        argument backend and runtime options to the Blackbird program.
        """
        engine = RemoteEngine("X8", backend_options={"cutoff_dim": 12})
        engine.run_async(prog, shots=1234)
        assert blackbird._target["options"] == {"cutoff_dim": 12, "shots": 1234}

    def test_run_async_options_from_program(self, prog, blackbird):
        """Test that :meth:`RemoteEngine.run_async` correctly parses runtime
        options compiled into the program.
        """
        engine = RemoteEngine("X8")

        prog = prog.compile(device=engine.device_spec, shots=15)
        assert prog.run_options == {"shots": 15}

        engine.run_async(prog)
        assert blackbird._target["options"] == {"shots": 15}

    def test_run_async_without_shots(self, prog):
        """Tests that a ValueError is raised if no shots are specified when a
        remote engine is instantiated.
        """
        engine = RemoteEngine("X8")
        with pytest.raises(ValueError, match=r"Number of shots must be specified"):
            engine.run_async(prog)


@pytest.mark.usefixtures("job", "device")
class TestRemoteEngineIntegration:
    """Integration tests for the ``RemoteEngine`` class."""

    def test_compilation(self, prog, blackbird, infolog):
        """Test that :class:`RemoteEngine` can compile a program for the
        intended backend.
        """
        engine = RemoteEngine("X8")
        engine.run_async(prog, shots=10)

        program = blackbird.mock_calls[0].args[0]

        want_message = "Compiling program for device X8_01 using compiler fock."
        assert infolog.records[-1].message == want_message

        # Check that the program is compiled to match the chip template.
        expected = prog.compile(device=engine.device_spec).circuit
        res = program.circuit

        for cmd1, cmd2 in zip(res, expected):
            # Check that the gates are the same.
            assert type(cmd1.op) is type(cmd2.op)
            # Check that the modes are the same.
            assert all(i.ind == j.ind for i, j in zip(cmd1.reg, cmd2.reg))
            # Check that the parameters are the same.
            assert all(p1 == p2 for p1, p2 in zip(cmd1.op.p, cmd2.op.p))

    def test_default_compiler(self, prog, infolog, device):
        """Test that the Xunitary compiler is used by default if a device does
        not explicitly provide a default compiler.
        """
        device.specification["compiler"] = []

        engine = RemoteEngine("X8")
        engine.run_async(prog, shots=10)

        assert engine.device_spec.default_compiler == "Xunitary"

        want_message = "Compiling program for device X8_01 using compiler Xunitary."
        assert infolog.records[-1].message == want_message

    def test_compile_device_invalid_device_error(self, prog, device):
        """Tests that a ValueError is raised if a program is compiled for one
        device but run on a different device without recompilation.
        """
        device.specification["compiler"] = []

        # Setting compile_info with a dummy devicespec and compiler name
        dummy_spec = {
            "target": "DummyDevice",
            "modes": 2,
            "layout": None,
            "gate_parameters": None,
            "compiler": [None],
        }
        X8_spec = DeviceSpec(spec=dummy_spec)
        prog._compile_info = (X8_spec, "dummy_compiler")

        engine = sf.RemoteEngine("X8")
        with pytest.raises(ValueError, match="Cannot use program compiled"):
            engine.run_async(prog, shots=10)

    def test_compile(self, prog, infolog, device):
        """Tests that compilation happens by default if no compile_info is
        specified when :meth:`RemoteEngine.run_async` is invoked.
        """
        device.specification["compiler"] = []

        # Leaving compile_info as None.
        assert prog.compile_info == None

        engine = RemoteEngine("X8")
        engine.run_async(prog, shots=10)

        want_message = "Compiling program for device X8_01 using compiler Xunitary."
        assert infolog.records[-1].message == want_message

    def test_recompilation_precompiled(self, prog, infolog, device):
        """Test that recompilation happens when:

            1. the program was precompiled, but
            2. the recompile keyword argument was set to ``True``.

        The program is considered to be precompiled if ``program.compile_info``
        is set.
        """
        device.specification["compiler"] = []

        # Set compile_info to a fake device specification and compiler name.
        dummy_spec = {
            "target": "DummyDevice",
            "modes": 2,
            "layout": None,
            "gate_parameters": None,
            "compiler": [None],
        }
        X8_spec = DeviceSpec(spec=dummy_spec)
        prog._compile_info = (X8_spec, "fake_compiler")

        engine = sf.RemoteEngine("X8")
        engine.run_async(prog, shots=10, compile_options=None, recompile=True)

        want_message = "Recompiling program for device X8_01 using compiler Xunitary."
        assert infolog.records[-1].message == want_message

    def test_recompilation_run_async(self, prog, infolog, device):
        """Test that recompilation happens when the recompile keyword argument
        is set to ``True``.
        """
        device.specification["compiler"] = ["Xunitary"]

        engine = sf.RemoteEngine("X8")
        device = engine.device_spec

        prog._compile_info = (device, device.compiler)

        compile_options = {"compiler": "Xunitary"}
        engine.run_async(prog, shots=10, compile_options=compile_options, recompile=True)

        want_message = (
            f"Recompiling program for device {device.target} using the "
            f"specified compiler options: {compile_options}."
        )
        assert infolog.records[-1].message == want_message

    def test_recompilation_run(self, prog, infolog, device):
        """Test that recompilation happens when the recompile keyword argument
        was set to ``True`` and :meth:`RemoteEngin.run` is called."""
        device.specification["compiler"] = ["Xunitary"]

        engine = sf.RemoteEngine("X8")
        device = engine.device_spec

        prog._compile_info = (device, device.compiler)

        compile_options = {"compiler": "Xunitary"}
        engine.run(prog, shots=10, compile_options=compile_options, recompile=True)

        want_message_1 = "The remote job 123 has been completed."
        assert infolog.records[-1].message == want_message_1

        want_message_2 = (
            f"Recompiling program for device {device.target} using the "
            f"specified compiler options: {compile_options}."
        )
        assert infolog.records[-2].message == want_message_2

    def test_validation(self, prog, infolog, device):
        """Test that validation happens (i.e., no recompilation) when the target
        device and device specification match.
        """
        device.specification["compiler"] = ["Xunitary"]

        engine = sf.RemoteEngine("X8_01")
        device = engine.device_spec

        prog._compile_info = (device, device.compiler)

        engine.run_async(prog, shots=10)

        want_message = (
            f"Program previously compiled for {device.target} using {prog.compile_info[1]}. "
            f"Validating program against the Xstrict compiler."
        )
        assert infolog.records[-1].message == want_message

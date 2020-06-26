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
Unit tests for strawberryfields.api.devicespec
"""
import pytest
import textwrap

from strawberryfields.api import DeviceSpec
from strawberryfields.circuitspecs import Ranges
import strawberryfields as sf

import blackbird

# pylint: disable=bad-continuation,no-self-use,pointless-statement

pytestmark = pytest.mark.api

mock_layout = textwrap.dedent(
    """\
    name mock
    version 1.0

    S2gate({squeezing_amplitude_0}, 0.0) | [0, 1]
    MZgate({phase_0}, {phase_1}) | [0, 1]
    MeasureFock() | [0, 1]
    """
)

device_dict = {
    "target": "abc",
    "layout": mock_layout,
    "modes": 2,
    "compiler": [],
    "gate_parameters": {
        "squeezing_amplitude_0": [0, 1],
        "phase_0": [0, [0, 6.3]],
        "phase_1": [[0.5, 1.4]],
    },
    "connection": None,
}


class TestDeviceSpec:
    """Tests for the ``DeviceSpec`` class."""

    def test_gate_parameters(self):
        """Test that gate_parameters outputs the correctly parsed parameters"""
        true_params = {
            "squeezing_amplitude_0": Ranges([0], [1], variable_name="squeezing_amplitude_0"),
            "phase_0": Ranges([0], [0, 6.3], variable_name="phase_0"),
            "phase_1": Ranges([0.5, 1.4], variable_name="phase_1")
        }
        spec_params = DeviceSpec(**device_dict).gate_parameters

        assert [tp_key == sp_key
                for tp_key, sp_key in zip(true_params.keys(), spec_params.keys())]
        assert [str(tp_val) == str(sp_val)
                for tp_val, sp_val in zip(true_params.values(), spec_params.values())]

    def test_create_program(self, monkeypatch):
        """Test that the program creation works"""
        circuit = ["S2gate(0, 0) | (q[0], q[1])",
                   "MZgate(1.23, 0.5) | (q[0], q[1])",
                   "MeasureFock | (q[0], q[1])"]

        params = {"phase_0": 1.23}
        prog = DeviceSpec(**device_dict).create_program(**params)

        assert prog.target is None
        assert prog.name == "mock"
        for cmd in prog.circuit:
            assert str(cmd) == circuit.pop(0)

    @pytest.mark.parametrize("params",
        [{"phase_0": 7.5}, {"phase_1": 0.4}, {"squeezing_amplitude_0": 0.5}]
        )
    def test_wrong_parameter_value(self, params):
        """Test that error is raised when non-supported parameter value is supplied"""
        with pytest.raises(ValueError, match="has invalid value"):
            DeviceSpec(**device_dict).create_program(**params)

    @pytest.mark.parametrize("params", [
        {"invalid_type": 7.5}, {"phase_42": 0.4}, {"squeezing_amplitude_1": 0.5}]
        )
    def test_wrong_parameter_type(self, params):
        """Test that error is raised when non-supported parameter type is supplied"""
        with pytest.raises(ValueError, match="not a valid parameter for this device"):
            DeviceSpec(**device_dict).create_program(**params)

    def test_refresh(self, connection, monkeypatch):
        """Tests that the refresh method refreshes the device spec"""
        device_dict.update({"connection": connection})
        return_dict = {"abc": {
            "layout": mock_layout,
            "modes": 42,
            "compiler": [],
            "gate_parameters": {}
            }}
        monkeypatch.setattr(connection, "_get_device_dict", return_dict)

        spec = DeviceSpec(**device_dict)
        spec.refresh()

        assert spec.modes == 42
        assert spec._connection is not None

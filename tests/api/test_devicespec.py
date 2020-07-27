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
from strawberryfields.compilers import Ranges
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
    "layout": mock_layout,
    "modes": 2,
    "compiler": [],
    "gate_parameters": {
        "squeezing_amplitude_0": [0, 1],
        "phase_0": [0, [0, 6.3]],
        "phase_1": [[0.5, 1.4]],
    },
}


class TestDeviceSpec:
    """Tests for the ``DeviceSpec`` class."""

    def test_initialization(self, connection):
        """Test that the device spec class initializes correctly."""
        spec = DeviceSpec(connection=None, spec=device_dict, target="abc")

        assert spec.target == "abc"
        assert spec.layout == device_dict["layout"]
        assert spec.modes == device_dict["modes"]
        assert spec.compiler == device_dict["compiler"]

    def test_gate_parameters(self):
        """Test that gate_parameters outputs the correctly parsed parameters"""
        true_params = {
            "squeezing_amplitude_0": Ranges([0], [1], variable_name="squeezing_amplitude_0"),
            "phase_0": Ranges([0], [0, 6.3], variable_name="phase_0"),
            "phase_1": Ranges([0.5, 1.4], variable_name="phase_1"),
        }
        spec_params = DeviceSpec(connection=None, spec=device_dict, target="abc").gate_parameters
        assert true_params == spec_params

    def test_create_program(self, monkeypatch):
        """Test that the program creation works"""
        circuit = [
            "S2gate(0, 0) | (q[0], q[1])",
            "MZgate(1.23, 0.5) | (q[0], q[1])",
            "MeasureFock | (q[0], q[1])",
        ]

        params = {"phase_0": 1.23}
        prog = DeviceSpec(connection=None, spec=device_dict, target="abc").create_program(**params)

        assert prog.target is None
        assert prog.name == "mock"
        assert [str(cmd) for cmd in prog.circuit] == circuit

    @pytest.mark.parametrize(
        "params", [{"phase_0": 7.5}, {"phase_1": 0.4}, {"squeezing_amplitude_0": 0.5}]
    )
    def test_invalid_parameter_value(self, params):
        """Test that error is raised when an invalid parameter value is supplied"""
        with pytest.raises(ValueError, match="has invalid value"):
            DeviceSpec(connection=None, spec=device_dict, target="abc").create_program(**params)

    @pytest.mark.parametrize(
        "params", [{"invalid_type": 7.5}, {"phase_42": 0.4}, {"squeezing_amplitude_1": 0.5}]
    )
    def test_unknown_parameter(self, params):
        """Test that error is raised when an unknown parameter is supplied"""
        with pytest.raises(ValueError, match="not a valid parameter for this device"):
            DeviceSpec(connection=None, spec=device_dict, target="abc").create_program(**params)

    def test_refresh(self, connection, monkeypatch):
        """Tests that the refresh method refreshes the device spec"""
        spec = DeviceSpec(connection=connection, spec=device_dict, target="abc")
        assert spec.modes == device_dict["modes"]

        new_spec_dict = device_dict.copy()
        new_spec_dict["modes"] = 42
        monkeypatch.setattr(connection, "_get_device_dict", lambda target: new_spec_dict)

        spec.refresh()
        assert spec.modes != device_dict["modes"]
        assert spec.modes == 42

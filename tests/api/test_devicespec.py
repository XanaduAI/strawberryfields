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
Unit tests for strawberryfields.device
"""
import inspect

import pytest

from strawberryfields.device import Device
from strawberryfields.compilers import Ranges

pytestmark = pytest.mark.api

mock_layout = inspect.cleandoc(
    """
    name mock
    version 1.0

    S2gate({squeezing_amplitude_0}, 0.0) | [0, 1]
    MZgate({phase_0}, {phase_1}) | [0, 1]
    MeasureFock() | [0, 1]
    """
)

device_spec = {
    "target": "abc",
    "layout": mock_layout,
    "modes": 2,
    "compiler": ["Xcov"],
    "gate_parameters": {
        "squeezing_amplitude_0": [0, 1],
        "phase_0": [0, [0, 6.3]],
        "phase_1": [[0.5, 1.4]],
    },
}

device_certificate = {
    "target": "X8_01",
    "finished_at": "2022-02-07T15:29:48.080220+00:00",
    "identity_unitary": {},
    "shots_per_benchmark": 999999,
    "anti_identity_unitary": {},
    "laser_wavelength_meters": 42,
}

mock_layout_tdm = inspect.cleandoc(
    """
    name template_td2
    version 1.0
    target {target} (shots=1)
    type tdm (temporal_modes={tm})

    float array p0[1, {tm}] =
        {{rs_array}}
    float array p1[1, {tm}] =
        {{bs_array}}
    float array p2[1, {tm}] =
        {{r_array}}
    float array p3[1, {tm}] =
        {{m_array}}

    Sgate(p0) | 1
    BSgate(p1) | (1, 0)
    Rgate(p2) | 1
    MeasureHomodyne(p3) | 0
    """
)

device_spec_tdm = {
    "target": "abc",
    "layout": mock_layout_tdm,
    "modes": {"concurrent": 2, "spatial": 1, "temporal": {"max": 100}},
    "compiler": ["TD2"],
    "gate_parameters": {
        "p0": [0.56],
        "p1": [0, [0, 6.28]],
        "p2": [0, [0, 3.14], 3.14],
        "p3": [0, [0, 6.28]],
    },
}


class TestDevice:
    """Tests for the ``Device`` class."""

    def test_initialization(self):
        """Test that the device spec class initializes correctly."""
        device = Device(spec=device_spec, cert=device_certificate)

        assert device.target == "abc"
        assert device.layout == device_spec["layout"]
        assert device.modes == device_spec["modes"]
        assert device.compiler == device_spec["compiler"]
        assert device.certificate == device_certificate

    def test_gate_parameters(self):
        """Test that gate_parameters outputs the correctly parsed parameters"""
        true_params = {
            "squeezing_amplitude_0": Ranges([0], [1], variable_name="squeezing_amplitude_0"),
            "phase_0": Ranges([0], [0, 6.3], variable_name="phase_0"),
            "phase_1": Ranges([0.5, 1.4], variable_name="phase_1"),
        }
        spec_params = Device(spec=device_spec).gate_parameters
        assert true_params == spec_params

    def test_create_program(self):
        """Test that the program creation works"""
        circuit = [
            "S2gate(0, 0) | (q[0], q[1])",
            "MZgate(1.23, 0.5) | (q[0], q[1])",
            "MeasureFock | (q[0], q[1])",
        ]

        params = {"phase_0": 1.23}
        prog = Device(spec=device_spec).create_program(**params)

        assert prog.target is None
        assert prog.name == "mock"
        assert prog.circuit
        assert [str(cmd) for cmd in prog.circuit] == circuit

    def test_create_program_no_layout(self):
        """Test that the program creation raises an error if the device spec contains no layout"""

        params = {"phase_0": 1.23}
        device_dict_no_layout = {
            "target": "abc",
            "layout": None,
            "modes": 2,
            "compiler": ["Xcov"],
            "gate_parameters": {
                "squeezing_amplitude_0": [0, 1],
                "phase_0": [0, [0, 6.3]],
                "phase_1": [[0.5, 1.4]],
            },
        }
        with pytest.raises(ValueError, match="missing a circuit layout"):
            Device(spec=device_dict_no_layout).create_program(**params)

    def test_different_targets(self):
        """Test that different targets in layout and spec raises an error."""
        layout_different_target = inspect.cleandoc(
            """
            name mock
            version 1.0
            target banana

            S2gate({squeezing_amplitude_0}, 0.0) | [0, 1]
            MZgate({phase_0}, {phase_1}) | [0, 1]
            MeasureFock() | [0, 1]
            """
        )
        device_dict_different_target = {
            "target": "pawpaw",
            "layout": layout_different_target,
            "modes": 2,
            "compiler": ["Xcov"],
            "gate_parameters": {
                "squeezing_amplitude_0": [0, 1],
                "phase_0": [0, [0, 6.3]],
                "phase_1": [[0.5, 1.4]],
            },
        }
        with pytest.raises(
            ValueError,
            match="Target in specification 'pawpaw' differs from the target in layout 'banana'.",
        ):
            Device(spec=device_dict_different_target)

    @pytest.mark.parametrize(
        "params", [{"phase_0": 7.5}, {"phase_1": 0.4}, {"squeezing_amplitude_0": 0.5}]
    )
    def test_invalid_parameter_value(self, params):
        """Test that error is raised when an invalid parameter value is supplied"""
        with pytest.raises(ValueError, match="has invalid value"):
            Device(spec=device_spec).create_program(**params)

    @pytest.mark.parametrize(
        "params", [{"invalid_type": 7.5}, {"phase_42": 0.4}, {"squeezing_amplitude_1": 0.5}]
    )
    def test_unknown_parameter(self, params):
        """Test that error is raised when an unknown parameter is supplied"""
        with pytest.raises(ValueError, match="not a valid parameter for this device"):
            Device(spec=device_spec).create_program(**params)

    def test_invalid_spec(self):
        """Test that error is raised when a specification with missing entries is supplied"""
        invalid_spec = {
            "target": "abc",
            "modes": 2,
            "compiler": ["Xcov"],
        }
        with pytest.raises(
            ValueError, match=r"missing the following keys: \['gate_parameters', 'layout'\]"
        ):
            Device(spec=invalid_spec)

    @pytest.mark.parametrize(
        "params",
        [
            {"phase_0": 1.23},
            {"phase_0": 0},
            {"phase_0": 6.29999},
            {"phase_0": [6.29999, 0, 0.14]},
            {"phase_0": [[0.1, 0, 2], [[0.5]]]},
        ],
    )
    def test_valid_parameters(self, params):
        """Test that valid parameters pass the validate_parameters validation"""
        Device(spec=device_spec).validate_parameters(**params)

    def test_invalid_parameter(self):
        """Test that invalid parameter names raise an error in validate_parameters"""
        with pytest.raises(ValueError, match=r"not a valid parameter for this device"):
            Device(spec=device_spec).validate_parameters(phase_42=0)

    def test_invalid_parameters_value(self):
        """Test that invalid parameter values raise an error in validate_parameters"""
        with pytest.raises(ValueError, match=r"has invalid value"):
            Device(spec=device_spec).validate_parameters(phase_0=123)

    @pytest.mark.parametrize(
        "params",
        [
            {"phase_0": 1.23},
            {"phase_0": 0},
            {"phase_0": 6.29999},
            {"phase_0": 123},
            {"phase_42": 123},
        ],
    )
    def test_gate_parameters_none(self, params):
        """Test that any parameters a valid when gate_parameters is None"""
        device_dict = {
            "target": "abc",
            "layout": mock_layout,
            "modes": 2,
            "compiler": ["Xcov"],
            "gate_parameters": None,
        }
        Device(spec=device_dict).validate_parameters(**params)

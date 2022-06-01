# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Unit tests for the Xcov compiler"""
import inspect
import logging
import pytest
import copy

import numpy as np

import strawberryfields as sf
from strawberryfields.compilers import compiler_db
from strawberryfields import Device, TDMProgram, ops
from strawberryfields.program_utils import CircuitError
from strawberryfields.tdm.utils import borealis_gbs, get_mode_indices, random_bs, random_r

from conftest import borealis_device

pytestmark = pytest.mark.frontend


def singleloop_program(r, alpha, phi, theta):
    """Single delay loop with program.

    Args:
        r (float): squeezing parameter
        alpha (Sequence[float]): beamsplitter angles
        phi (Sequence[float]): rotation angles
        theta (Sequence[float]): homodyne measurement angles
    Returns:
        (array): homodyne samples from the single loop simulation
    """
    prog = TDMProgram(N=2)
    with prog.context(alpha, phi, theta) as (p, q):
        ops.Sgate(r, 0) | q[1]
        ops.BSgate(p[0]) | (q[1], q[0])
        ops.Rgate(p[1]) | q[1]
        ops.MeasureHomodyne(p[2]) | q[0]
    return prog


def reset_compiler(target):
    """Resets compiler circuit layout since different layouts are used"""
    compiler_db[target].reset_circuit()


class TestTDMCompiler:
    """Tests for the general TDM compiler."""

    target = "TDM"
    tm = 4
    layout_template = inspect.cleandoc(
        """
        name template_tdm
        version 1.0
        target {target} (shots=1)
        type tdm (temporal_modes={tm}, copies=1)

        float array p1[1, {tm}] =
            {{r}}
        float array p2[1, {tm}] =
            {{bs}}
        float array p3[1, {tm}] =
            {{m}}

        Sgate(0.5643, 0) | 1
        BSgate({{bs}}, 0) | (1, 0)
        Rgate({{r}}) | 1
        MeasureHomodyne({{m}}) | 0
        """
    )
    layout = layout_template.format(target=target, tm=tm)

    device_spec = {
        "target": target,
        "layout": layout,
        "modes": {"concurrent": 2, "spatial": 1, "temporal_max": 100},
        "compiler": ["TDM"],
        "gate_parameters": {
            "bs": [0, [0, 6.283185307179586]],
            "r": [0, [0, 3.141592653589793], 3.141592653589793],
            "m": [0, [0, 6.283185307179586]],
        },
    }
    device = Device(device_spec)

    def test_tdm_wrong_layout(self):
        """Test the correct error is raised when the tdm circuit gates don't match the device spec"""
        sq_r = 0.5643
        c = 2
        alpha = [np.pi / 4, 0] * c
        phi = [0, np.pi / 2] * c
        theta = [0.0, 0.0] + [np.pi / 2, np.pi / 2]
        prog = TDMProgram(N=2)
        with prog.context(alpha, phi, theta) as (p, q):
            ops.Dgate(sq_r) | q[1]  # Here one should have an Sgate
            ops.BSgate(p[0]) | (q[1], q[0])
            ops.Rgate(p[1]) | q[1]
            ops.MeasureHomodyne(p[2]) | q[0]

        reset_compiler(self.target)

        with pytest.raises(
            CircuitError,
            match=f"cannot be used with the compiler 'TDM",
        ):
            prog.compile(device=self.device, compiler="TDM")

    def test_tdm_wrong_modes(self):
        """Test the correct error is raised when the tdm circuit registers don't match the device spec"""
        sq_r = 0.5643
        c = 2
        alpha = [np.pi / 4, 0] * c
        phi = [0, np.pi / 2] * c
        theta = [0.0, 0.0] + [np.pi / 2, np.pi / 2]
        prog = TDMProgram(N=2)
        with prog.context(alpha, phi, theta) as (p, q):
            ops.Sgate(sq_r) | q[1]
            ops.BSgate(p[0]) | (q[0], q[1])  # The order should be (q[1], q[0])
            ops.Rgate(p[1]) | q[1]
            ops.MeasureHomodyne(p[2]) | q[0]

        reset_compiler(self.target)

        with pytest.raises(CircuitError, match=f"cannot be used with the compiler 'TDM"):
            prog.compile(device=self.device, compiler="TDM")

    def test_tdm_wrong_parameters_explicit(self):
        """Test the correct error is raised when the tdm circuit explicit parameters are not within the allowed ranges"""
        sq_r = 2  # This squeezing value is not the allowed value 0.5643 (hard-coded into the layout above)
        c = 2
        alpha = [np.pi / 4, 0] * c
        phi = [0, np.pi / 2] * c
        theta = [0.0, 0.0] + [np.pi / 2, np.pi / 2]
        prog = singleloop_program(sq_r, alpha, phi, theta)

        reset_compiler(self.target)

        with pytest.raises(CircuitError, match="due to incompatible parameter values"):
            prog.compile(device=self.device, compiler="TDM")

    def test_tdm_wrong_parameters_explicit_in_list(self):
        """Test the correct error is raised when the tdm circuit explicit parameters are not within the allowed ranges"""
        sq_r = 0.5643
        c = 2
        alpha = [
            np.pi / 4,
            27,
        ] * c  # This beamsplitter phase is not in the allowed range of squeezing parameters
        phi = [0, np.pi / 2] * c
        theta = [0.0, 0.0] + [np.pi / 2, np.pi / 2]
        prog = singleloop_program(sq_r, alpha, phi, theta)

        reset_compiler(self.target)

        with pytest.raises(ValueError, match="has invalid value 27"):
            prog.compile(device=self.device, compiler="TDM")

    def test_tdm_wrong_parameter_second_argument(self):
        """Test the correct error is raised when the tdm circuit explicit parameters are not within the allowed ranges"""
        sq_r = 0.5643
        c = 2
        alpha = [np.pi / 4, 0] * c
        phi = [0, np.pi / 2] * c
        theta = [0.0, 0.0] + [np.pi / 2, np.pi / 2]
        prog = TDMProgram(N=2)
        with prog.context(alpha, phi, theta) as (p, q):
            ops.Sgate(sq_r, 0.4) | q[
                1
            ]  # Note that the Sgate has a second parameter that is non-zero
            ops.BSgate(p[0]) | (q[1], q[0])
            ops.Rgate(p[1]) | q[1]
            ops.MeasureHomodyne(p[2]) | q[0]

        reset_compiler(self.target)

        with pytest.raises(CircuitError, match="due to incompatible parameter value"):
            prog.compile(device=self.device, compiler="TDM")

    def test_tdm_wrong_parameters_symbolic(self):
        """Test the correct error is raised when the tdm circuit symbolic parameters are not within the allowed ranges"""
        sq_r = 0.5643
        c = 2
        alpha = [137, 0] * c  # Note that alpha is outside the allowed range
        phi = [0, np.pi / 2] * c
        theta = [0.0, 0.0] + [np.pi / 2, np.pi / 2]
        prog = singleloop_program(sq_r, alpha, phi, theta)

        reset_compiler(self.target)

        with pytest.raises(ValueError, match="has invalid value 137"):
            prog.compile(device=self.device, compiler="TDM")

    def test_tdm_parameters_not_in_devicespec(self):
        """Test the correct error is raised when the tdm circuit symbolic parameters are not found
        in the device specification."""
        spec = copy.deepcopy(self.device_spec)
        # "bs" removed from device spec, but is still used in layout
        del spec["gate_parameters"]["bs"]

        c = 2
        prog = singleloop_program(
            0.5643, [np.pi / 4, 0] * c, [0, np.pi / 2] * c, [0, 0, np.pi / 2, np.pi / 2]
        )

        reset_compiler(self.target)

        # If a parameter isn't explicitly available in the device spec, but is still used in the
        # layout, then this error will be raised. Directed towards developers to make sure that
        # ``spec["gate_parameters"]`` contains all the necessary gates.
        with pytest.raises(
            ValueError, match="Parameter 'bs' not a valid parameter for this device"
        ):
            prog.compile(device=Device(spec), compiler="TDM")

    def test_tdm_inconsistent_temporal_modes(self):
        """Test the correct error is raised when the tdm circuit has too many temporal modes"""
        sq_r = 0.5643
        c = 100  # Note that we are requesting more temporal modes (2*c = 200) than what is allowed.
        alpha = [0.5, 0] * c
        phi = [0, np.pi / 2] * c
        theta = [0, 0] * c
        prog = singleloop_program(sq_r, alpha, phi, theta)

        reset_compiler(self.target)

        with pytest.raises(CircuitError, match="temporal modes, but the device"):
            prog.compile(device=self.device, compiler="TDM")

    def test_tdm_inconsistent_concurrent_modes(self):
        """Test the correct error is raised when the tdm circuit has too many concurrent modes"""
        device_spec1 = copy.deepcopy(self.device_spec)
        device_spec1["modes"][
            "concurrent"
        ] = 100  # Note that singleloop_program has only two concurrent modes
        device1 = Device(device_spec1)
        c = 1
        sq_r = 0.5643
        alpha = [0.5, 0] * c
        phi = [0, np.pi / 2] * c
        theta = [0, 0] * c
        prog = singleloop_program(sq_r, alpha, phi, theta)

        reset_compiler(self.target)

        with pytest.raises(CircuitError, match="concurrent modes, but the device"):
            prog.compile(device=device1, compiler="TDM")

    def test_tdm_inconsistent_spatial_modes(self):
        """Test the correct error is raised when the tdm circuit has too many spatial modes"""
        device_spec1 = copy.deepcopy(self.device_spec)
        device_spec1["modes"][
            "spatial"
        ] = 100  # Note that singleloop_program has only one spatial mode
        device1 = Device(device_spec1)
        c = 1
        sq_r = 0.5643
        alpha = [0.5, 0] * c
        phi = [0, np.pi / 2] * c
        theta = [0, 0] * c
        prog = singleloop_program(sq_r, alpha, phi, theta)

        reset_compiler(self.target)

        with pytest.raises(CircuitError, match="spatial modes, but the device"):
            prog.compile(device=device1, compiler="TDM")

    def test_error_without_devicespec(self):
        """Test the correct error is raised when the tdm circuit is compiled
        without a device specification"""
        c = 1
        sq_r = 0.5643
        alpha = [0.5, 0] * c
        phi = [0, np.pi / 2] * c
        theta = [0, 0] * c
        prog = singleloop_program(sq_r, alpha, phi, theta)

        reset_compiler(self.target)

        with pytest.raises(
            CircuitError, match="TDM programs cannot be compiled without a valid circuit layout."
        ):
            prog.compile(compiler="TDM")


class TestBorealisCompiler:
    """Tests for the general TDM compiler."""

    target = "borealis"

    @staticmethod
    def borealis_program(gate_args):
        delays = [1, 6, 36]

        n, N = get_mode_indices(delays)
        prog = sf.TDMProgram(N)

        with prog.context(*gate_args) as (p, q):
            ops.Sgate(p[0]) | q[n[0]]
            for i in range(len(delays)):
                ops.Rgate(p[2 * i + 1]) | q[n[i]]
                ops.BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i + 1]], q[n[i]])
            ops.MeasureFock() | q[0]

        return prog

    @staticmethod
    def borealis_program_with_offsets(gate_args):
        delays = [1, 6, 36]
        cert = borealis_device.certificate or {"loop_phases": []}
        loop_phases = cert["loop_phases"]

        n, N = get_mode_indices(delays)
        prog = sf.TDMProgram(N)

        with prog.context(*gate_args) as (p, q):
            ops.Sgate(p[0]) | q[n[0]]
            for i in range(len(delays)):
                ops.Rgate(p[2 * i + 1]) | q[n[i]]
                ops.BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i + 1]], q[n[i]])
                ops.Rgate(loop_phases[i]) | q[n[i]]
            ops.MeasureFock() | q[0]

        return prog

    def test_compile(self):
        """Test that a program can be compiled correctly using the Borealis compiler"""
        gate_args = borealis_gbs(borealis_device)
        prog = self.borealis_program(gate_args)
        reset_compiler(self.target)

        prog.compile(device=borealis_device)

    def test_compile_with_integer_phases(self):
        """Test that a program can be compiled correctly using the Borealis compiler and integer valued phases"""
        gate_args = borealis_gbs(borealis_device)
        gate_args[1] = [1] * len(
            gate_args[1]
        )  # exchange the phase-gate arguments with integer values
        prog = self.borealis_program(gate_args)
        reset_compiler(self.target)

        prog.compile(device=borealis_device)

    def test_loop_compensation_warning(self, caplog):
        """Test that a program logs the correct warning when loop offset compensation takes place."""
        gate_args = borealis_gbs(borealis_device)
        assert isinstance(gate_args, list)

        prog = self.borealis_program(gate_args)
        reset_compiler(self.target)

        caplog.clear()
        match = "have been shifted by pi in order to be compatible with the phase modulators"
        with caplog.at_level(logging.WARNING):
            prog.compile(device=borealis_device)
        assert match not in caplog.text

    def test_compile_wrong_params(self, caplog):
        """Test that a program logs the correct warning when used parameters are compensated
        values with the Borealis compiler"""
        gate_args = np.ones((7, 259))  # ~50% of the phase-shifts will need to be compensated

        prog = self.borealis_program(gate_args)
        reset_compiler(self.target)

        caplog.clear()
        match = "are beyond the range of the Borealis phase modulators and have been offset by pi."
        with caplog.at_level(logging.WARNING):
            prog.compile(device=borealis_device)
        assert match in caplog.text

    def test_compile_incompatible_topology(self):
        """Test that a program raises the correct error when using an invalid circuit
        with the Borealis compiler"""
        gate_args = borealis_gbs(borealis_device)

        prog = self.borealis_program(gate_args)
        assert isinstance(prog.circuit, list)
        assert isinstance(prog.circuit[0].op, ops.Sgate)

        # change Sgate to an Rgate (applied to the same modes)
        prog.circuit[0].op = ops.Rgate(0.12)
        reset_compiler(self.target)

        # should raise error inside of `Borealis.compile()`
        with pytest.raises(
            CircuitError, match="Compilation not possible due to incompatible topologies."
        ):
            prog.compile(device=borealis_device)

    def test_compile_with_offsets(self):
        """Test that a program can be compiled correctly using the Borealis compiler"""
        gate_args = [
            [0] * 216,
            random_r(216, low=-np.pi / 2, high=np.pi / 2),
            random_bs(216),
            random_r(216, low=-np.pi / 2, high=np.pi / 2),
            random_bs(216),
            random_r(216, low=-np.pi / 2, high=np.pi / 2),
            random_bs(216),
        ]
        prog = self.borealis_program_with_offsets(gate_args)
        reset_compiler(self.target)

        prog.compile(device=borealis_device)

    @pytest.mark.skip(reason="Should pass if `Device.create_program` works with TDM programs")
    def test_compile_with_create_program(self):
        """Test that a program can be compiled correctly using the Borealis compiler when using
        ``Device.create_program``"""
        params = borealis_gbs(borealis_device, return_list=True)
        assert isinstance(params, list)

        # add loop-phases
        params.extend([[4.0], [2.0], [1.0]])
        names = [
            "s",
            "r0",
            "bs0",
            "r1",
            "bs1",
            "r2",
            "bs2",
            "loop0_phase",
            "loop1_phase",
            "loop2_phase",
        ]
        params = dict(zip(names, params))
        prog = borealis_device.create_program(**params)

        reset_compiler(self.target)

        prog.compile(compiler="borealis")

    def test_compile_error_without_devicespec(self):
        """Test the correct error is raised when the tdm circuit is compiled
        without a device specification"""
        gate_args = borealis_gbs(borealis_device)
        prog = self.borealis_program(gate_args)
        reset_compiler(self.target)

        with pytest.raises(
            CircuitError, match="TDM programs cannot be compiled without a valid circuit layout"
        ):
            prog.compile(compiler=self.target)

    def test_compile_error_invalid_loop_offset_name(self):
        """Test the correct error is raised when the tdm circuit is compiled
        with a device specification having incorrectly named loop offsets"""
        invalid_spec = copy.deepcopy(borealis_device._spec)
        invalid_spec["layout"] = invalid_spec["layout"].replace("loop1_phase", "loop1_phase42")
        invalid_device = Device(spec=invalid_spec, cert=borealis_device.certificate)

        gate_args = borealis_gbs(invalid_device)
        prog = self.borealis_program(gate_args)
        reset_compiler(self.target)

        with pytest.raises(
            ValueError, match="Expected single integer as part of name, got 'loop1_phase42'"
        ):
            prog.compile(device=invalid_device)


class TestTDMValidation:
    """Test the validation of TDMProgram against the device specs"""

    target = "TDM"
    tm = 4
    layout = f"""
        name template_tdm
        version 1.0
        target {target} (shots=1)
        type tdm (temporal_modes=2)
        float array p0[1, {tm}] =
            {{rs}}
        float array p1[1, {tm}] =
            {{r}}
        float array p2[1, {tm}] =
            {{bs}}
        float array p3[1, {tm}] =
            {{m}}
        Sgate({{rs}}) | 1
        Rgate({{r}}) | 0
        BSgate({{bs}}, 0) | (0, 1)
        MeasureHomodyne({{m}}) | 0
    """
    device_spec = {
        "target": target,
        "layout": inspect.cleandoc(layout),
        "modes": {"concurrent": 2, "spatial": 1, "temporal_max": 100},
        "compiler": [target],
        "gate_parameters": {
            "rs": [-1],
            "r": [1],
            "bs": [2],
            "m": [3],
        },
    }
    device = Device(device_spec)

    @staticmethod
    def compile_test_program(dev, args=(-1, 1, 2, 3)):
        """Compiles a test program with the given gate arguments."""
        alpha = [args[1]]
        beta = [args[2]]
        gamma = [args[3]]
        prog = TDMProgram(N=2)
        with prog.context(alpha, beta, gamma) as (p, q):
            ops.Sgate(args[0]) | q[1]  # Note that the Sgate has a second parameter that is non-zero
            ops.Rgate(p[0]) | q[0]
            ops.BSgate(p[1]) | (q[0], q[1])
            ops.MeasureHomodyne(p[2]) | q[0]

        prog.compile(device=dev, compiler=dev.default_compiler)

    def test_validation_correct_args(self):
        """Test that no error is raised when the tdm circuit explicit parameters within the allowed ranges"""

        reset_compiler(self.target)

        self.compile_test_program(self.device, args=(-1, 1, 2, 3))

    @pytest.mark.parametrize("incorrect_index", list(range(4)))
    def test_validation_incorrect_args(self, incorrect_index):
        """Test the correct error is raised when the tdm circuit explicit parameters are not within the allowed ranges"""
        args = [-1, 1, 2, 3]
        args[incorrect_index] = -999

        reset_compiler(self.target)

        with pytest.raises(ValueError, match="has invalid value -999"):
            self.compile_test_program(self.device, args=args)

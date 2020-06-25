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
**Module name:** :mod:`strawberryfields.api.devicespec`
"""
import strawberryfields as sf
import blackbird


class DeviceSpec:
    """The specifications for a specific hardware device.

    Args:
        target (str): target chip
        layout (str): string containing the Blackbird circuit layout
        modes (int): number of modes supported by the target
        compiler (list): list of supported compilers
        gate_parameters (dict): parameters for the circuit gates
        connection (strawberryfields.api.Connection): connection over which the
            job is managed
    """

    def __init__(self, target, layout, modes, compiler, gate_parameters, connection):
        self._target = target
        self._layout = layout
        self._modes = modes
        self._compiler = compiler
        self._gate_parameters = gate_parameters

        self._connection = connection

    @property
    def target(self):
        return self._target

    @property
    def layout(self):
        return self._layout

    @property
    def modes(self):
        return self._modes

    @property
    def compiler(self):
        return self.compiler

    @property
    def gate_parameters(self):
        return self._gate_parameters

    def create_program(self, **kwargs):
        bb = blackbird.loads(self._layout)
        parameters = dict(self._gate_parameters, **kwargs)
        bb(**{f"{name}": par for name, par in parameters})

        prog = sf.io.to_program(bb)
        return prog

    def refresh(self):
        """Refreshes the device specifications"""
        device = self._connection._get_device_dict(self.target)

        self._target = device["target"]
        self._layout = device["layout"]
        self._modes = device["modes"]
        self._compiler = device["compiler"]
        self._gate_parameters = device["gate_parameters"]

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
This module contains a class that represents the specifications of
a device available via the API.
"""
from collections.abc import Sequence
import re

import blackbird
from blackbird.error import BlackbirdSyntaxError

import strawberryfields as sf
from strawberryfields.compilers import Ranges


class DeviceSpec:
    """The specifications for a specific hardware device.

    Args:
        target (str): name of the target hardware device
        spec (dict): dictionary representing the raw device specification.
            This dictionary should contain the following key-value pairs:

            - layout (str): string containing the Blackbird circuit layout
            - modes (int): number of modes supported by the target
            - compiler (list): list of supported compilers
            - gate_parameters (dict): parameters for the circuit gates

        connection (strawberryfields.api.Connection): connection over which the
            job is managed
    """

    def __init__(self, target, spec, connection):
        self._target = target
        self._connection = connection
        self._spec = spec

    @property
    def target(self):
        """str: The name of the target hardware device."""
        return self._target

    @property
    def layout(self):
        """str: Returns a string containing the Blackbird circuit layout."""
        return self._spec["layout"]

    @property
    def modes(self):
        """int: Number of modes supported by the device."""
        return self._spec["modes"]

    @property
    def compiler(self):
        """list[str]: A list of strings corresponding to Strawberry Fields compilers supported
        by the hardware device."""
        return self._spec["compiler"]

    @property
    def default_compiler(self):
        """sf.compilers.Compiler: Specified default compiler"""
        if self.compiler:
            return self.compiler[0]

        # For now, use Xunitary compiler by default for devices
        # if the default compiler is not specified.
        return "Xunitary"

    @property
    def gate_parameters(self):
        """dict[str, strawberryfields.compilers.Ranges]: A dictionary of gate parameters
        and allowed ranges.

        The parameter names correspond to those present in the Blackbird circuit layout.

        **Example**

        >>> spec.gate_parameters
        {'squeezing_amplitude_0': x=0, x=1, 'phase_0': x=0, 0≤x≤6.283185307179586}
        """
        gate_parameters = dict()

        for gate_name, param_ranges in self._spec["gate_parameters"].items():
            # convert gate parameter allowed ranges to Range objects
            range_list = [[i] if not isinstance(i, Sequence) else i for i in param_ranges]
            gate_parameters[gate_name] = Ranges(*range_list)

        return gate_parameters

    def layout_is_formatted(self):
        """bool: Whether the device layout is formatted or not."""
        p = re.compile(r"{{\w*}}")
        return not bool(p.search(self.layout))

    def fill_template(self, program):
        """Fill template with parameter values from a program"""
        if self.layout_is_formatted():
            return

        if program.type == "tdm":
            self._spec["layout"] = self._spec["layout"].format(
                target=self.target, tm=program.timebins
            )
        else:
            # TODO: update when `self._spec["layout"]` is returned as an unformatted string
            raise NotImplementedError("Formatting not required or supported for non-TDM programs.")

    def validate_parameters(self, **parameters):
        """Validate gate parameters against the device spec.

        Gate parameters should be passed as keyword arguments, with names
        corresponding to those present in the Blackbird circuit layout.
        """
        # check that all provided parameters are valid
        for p, v in parameters.items():
            if p in self.gate_parameters and v not in self.gate_parameters[p]:
                # parameter is present in the device specifications
                # but the user has provided an invalid value
                raise ValueError(
                    f"{p} has invalid value {v}. Only {self.gate_parameters[p]} allowed."
                )

            if p not in self.gate_parameters:
                raise ValueError(f"Parameter {p} not a valid parameter for this device")

    def create_program(self, **parameters):
        """Create a Strawberry Fields program matching the low-level layout of the
        device.

        Gate arguments should be passed as keyword arguments, with names
        correspond to those present in the Blackbird circuit layout. Parameters not
        present will be assumed to have a value of 0.

        **Example**

        Device specifications can be retrieved from the API by using the
        :class:`~.Connection` class:

        >>> spec.create_program(squeezing_amplitude_0=0.43)
        <strawberryfields.program.Program at 0x7fd37e27ff50>

        Keyword Args:
            Supported parameter values for the specific device

        Returns:
            strawberryfields.program.Program: program compiled to the device
        """
        try:
            bb = blackbird.loads(self.layout)
        except BlackbirdSyntaxError as e:
            raise BlackbirdSyntaxError("Layout is not formatted correctly.") from e
        self.validate_parameters(**parameters)

        # determine parameter value if not provided
        extra_params = set(self.gate_parameters) - set(parameters)

        for p in extra_params:
            # Set parameter value as the first allowed
            # value in the gate parameters dictionary.
            parameters[p] = self.gate_parameters[p].ranges[0].x

        # evaluate the blackbird template
        bb = bb(**parameters)
        prog = sf.io.to_program(bb)
        prog._compile_info = (self, self.default_compiler)
        return prog

    def refresh(self):
        """Refreshes the device specifications"""
        self._spec = self._connection._get_device_dict(self.target)

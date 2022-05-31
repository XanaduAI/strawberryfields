# Copyright 2020-2022 Xanadu Quantum Technologies Inc.

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
This module contains a class that represents a device on the Xanadu Cloud.
"""
import re
from typing import Generator, Iterable, Sequence, Mapping, Any, Optional

import blackbird
from blackbird.error import BlackbirdSyntaxError

from strawberryfields.io import to_program
from strawberryfields.compilers import Ranges


class Device:
    """The representation of a specific remote device.

    Args:
        spec (dict): dictionary representing the raw device specification.
            This dictionary must contain the following key-value pairs:

            - layout (str): string containing the Blackbird circuit layout or XIR manifest
            - modes (int): number of modes supported by the target
            - compiler (list): list of supported compilers
            - gate_parameters (dict): parameters for the circuit gates

        cert (dict, optional): dictionary representing the device certificate
    """

    def __init__(self, spec: Mapping[str, Any], cert: Optional[Mapping[str, Any]] = None) -> None:
        missing_keys = {"target", "layout", "modes", "compiler", "gate_parameters"} - spec.keys()
        if missing_keys:
            raise ValueError(
                f"Device specification is missing the following keys: {sorted(missing_keys)}"
            )

        self._spec = self.validate_target(spec)
        self._certificate = cert

    @staticmethod
    def validate_target(spec: Mapping[str, Any]):
        """Check that the target in the specification is equal to the layout target.

        Returns:
            dict: dictionary representing the raw device specification.
        """
        layout = spec["layout"]
        target = spec["target"]

        groups = re.findall(r"\b(?=\w)target (\w+)", layout or "")
        if len(groups) <= 1:
            if len(groups) == 1 and target != groups[0]:
                raise ValueError(
                    f"Target in specification '{target}' differs from the "
                    f"target in layout '{groups[0]}'."
                )
            return spec

        raise ValueError(f"Layout must have a single target; found {len(groups)}.")

    @property
    def target(self) -> str:
        """str: The name of the target hardware device."""
        return self._spec["target"]

    @property
    def layout(self) -> Optional[str]:
        """str: A string containing the Blackbird circuit layout."""
        return self._spec["layout"]

    @property
    def modes(self) -> int:
        """int: The number of modes supported by the device."""
        return self._spec["modes"]

    @property
    def compiler(self) -> Sequence[str]:
        """list[str]: A list of strings corresponding to Strawberry Fields compilers supported
        by the hardware device."""
        return self._spec["compiler"]

    @property
    def default_compiler(self) -> str:
        """sf.compilers.Compiler: Specified default compiler"""
        if self.compiler:
            return self.compiler[0]

        # For now, use Xunitary compiler by default for devices
        # if the default compiler is not specified.
        return "Xunitary"

    @property
    def gate_parameters(self) -> Optional[Mapping[str, Ranges]]:
        """dict[str, strawberryfields.compilers.Ranges]: A dictionary of gate parameters
        and allowed ranges.

        The parameter names correspond to those present in the Blackbird circuit layout.

        **Example**

        >>> spec.gate_parameters
        {'squeezing_amplitude_0': x=0, x=1, 'phase_0': x=0, 0≤x≤6.283185307179586}
        """
        if self._spec["gate_parameters"] is None:
            return None

        gate_parameters = {}
        for gate_name, param_ranges in self._spec["gate_parameters"].items():
            # convert gate parameter allowed ranges to Range objects
            range_list = [[i] if not isinstance(i, Sequence) else i for i in param_ranges]
            gate_parameters[gate_name] = Ranges(*range_list)

        return gate_parameters

    @property
    def certificate(self) -> Optional[Mapping[str, Any]]:
        """dict[str, Any]: A device certificate containing the current operating
        conditions of the device."""
        return self._certificate

    def validate_parameters(self, **parameters: complex) -> None:
        """Validate the gate parameters against the device specification.

        Gate parameters should be passed as keyword arguments, with names
        corresponding to those present in the Blackbird circuit layout.

        Raises:
            ValueError: if an invalid parameter is passed
        """
        if self.gate_parameters is None:
            return

        def _flatten(iterable: Iterable) -> Generator:
            for it in iterable:
                if isinstance(it, Iterable) and not isinstance(it, str):
                    yield from _flatten(it)
                else:
                    yield it

        # check that all provided parameters are valid
        for p, v in parameters.items():
            if p not in self.gate_parameters:
                raise ValueError(f"Parameter '{p}' not a valid parameter for this device")

            if isinstance(v, Iterable):
                for i in _flatten(v):
                    if i not in self.gate_parameters[p]:
                        raise ValueError(
                            f"'{p}' has invalid value {i}. Only {self.gate_parameters[p]} allowed."
                        )
            elif v not in self.gate_parameters[p]:
                # parameter is present in the device specifications
                # but the user has provided an invalid value
                raise ValueError(
                    f"'{p}' has invalid value {v}. Only {self.gate_parameters[p]} allowed."
                )

    # NOTE: does not work with `TDMProgram` due to template array parameters being stored
    # differently in Blackbird, e.g., an array `s` would be stored as s_0_0, s_0_1, etc.
    def create_program(self, **parameters: complex):
        """Create a Strawberry Fields program matching the low-level Blackbird
        layout of the device.

        Gate arguments should be passed as keyword arguments, with names
        correspond to those present in the Blackbird circuit layout. Parameters not
        present will be assumed to have a value of 0.

        **Example**

        Device specifications can be retrieved from the API by using the
        :class:`xcc.Device` and :class:`xcc.Connection` classes:

        >>> spec.create_program(squeezing_amplitude_0=0.43)
        <strawberryfields.program.Program at 0x7fd37e27ff50>

        Keyword Args:
            Supported parameter values for the specific device

        Returns:
            strawberryfields.program.Program: program compiled to the device
        """
        if not self.layout:
            raise ValueError("Cannot create program. Specification is missing a circuit layout.")

        try:
            bb = blackbird.loads(self.layout)
        except BlackbirdSyntaxError as e:
            raise BlackbirdSyntaxError("Layout is not formatted correctly.") from e
        self.validate_parameters(**parameters)

        if self.gate_parameters:
            # determine parameter value if not provided
            extra_params = set(self.gate_parameters) - set(parameters)

            for p in extra_params:
                # Set parameter value as the first allowed
                # value in the gate parameters dictionary.
                parameters[p] = self.gate_parameters[p].ranges[0].x

        # evaluate the blackbird template
        bb = bb(**parameters)
        prog = to_program(bb)
        prog.compile(compiler=self.default_compiler)

        return prog

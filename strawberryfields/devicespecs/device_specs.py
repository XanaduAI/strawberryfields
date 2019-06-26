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
"""
Backend capabilities
====================

**Module name:** :mod:`strawberryfields.devicespecs.device_specs`

.. currentmodule:: strawberryfields.devicespecs.device_specs

The :class:`DeviceSpecs` class stores information about the capabilities of various execution backends.
This information is used e.g. in :class:`Program` validation.


Classes
-------

.. autosummary::
   DeviceSpecs


Code details
~~~~~~~~~~~~

"""

from typing import List, Set, Dict, Union
import abc

import blackbird
from blackbird.utils import to_DiGraph


class DeviceSpecs(abc.ABC):
    """Abstract base class for describing execution backend capabilities."""

    @property
    @abc.abstractmethod
    def modes(self) -> Union[int, None]:
        """The supported number of modes of the device.

        If the device supports arbitrary number of modes, set this to 0.

        Returns:
            int: number of supported modes
        """

    @property
    @abc.abstractmethod
    def local(self) -> bool:
        """Whether the backend supports local execution.

        Returns:
            bool: ``True`` if the backend supports local execution
        """

    @property
    @abc.abstractmethod
    def remote(self) -> bool:
        """Whether the backend supports remote execution.

        Returns:
            bool: ``True`` if the backend supports remote execution
        """

    @property
    @abc.abstractmethod
    def interactive(self) -> bool:
        """Whether the backend can be used interactively, that is,
        the backend state is not reset between engine executions.

        Returns:
            bool: ``True`` if the backend supports interactive use
        """

    @property
    @abc.abstractmethod
    def primitives(self) -> Set[str]:
        """The primitive set of quantum operations directly supported
        by the backend.

        Returns:
            set[str]: the quantum primitives the backend supports
        """

    @property
    @abc.abstractmethod
    def decompositions(self) -> Dict[str, Dict]:
        """Quantum operations that are not quantum primitives for the
        backend, but are supported via specified decompositions.

        This should be of the form

        .. code-block:: python

            {'operation_name': {'option1': val, 'option2': val,...}}

        For each operation specified in the dictionary, the
        :meth:`~Operation.decompose` method will be called during
        :class:`Program` compilation, with keyword arguments
        given by the dictionary value.

        Returns:
            dict[str, dict]: the quantum operations that are supported
            by the backend via decomposition
        """

    @property
    def parameter_ranges(self) -> Dict[str, List[List[float]]]:
        """Allowed parameter ranges for supported quantum operations.

        This property is optional.

        Returns:
            dict[str, list]: a dictionary mapping an allowed quantum operation
            to a nested list of the form ``[[p0_min, p0_max], [p1_min, p0_max], ...]``.
            where ``pi`` corresponds to the ``i`` th gate parameter.
        """
        return dict()

    @property
    def graph(self):
        """The allowed circuit topology of the backend device as a directed
        acyclic graph.

        This property is optional; if arbitrary topologies are allowed by the device,
        this will simply return ``None``.

        Returns:
            networkx.DiGraph: a directed acyclic graph
        """
        if self.circuit is None:
            return None

        # returned DAG has all parameters set to 0
        bb = blackbird.loads(self.circuit)

        if bb.is_template():
            params = bb.parameters
            kwargs = {p: 0 for p in params}

            # initialize the topology with all template
            # parameters set to zero
            topology = to_DiGraph(bb(**kwargs))

        else:
            topology = to_DiGraph(bb)

        return topology

    @property
    def circuit(self):
        """The Blackbird circuit that will be accepted by the backend device.

        This property is optional. If arbitrary topologies are allowed by the device,
        **do not define this property**. In such a case, it will simply return ``None``.

        If the device expects a specific template for the recieved Blackbird
        script, this method will return the serialized Blackbird circuit in string
        form.

        Returns:
            Union[str, None]: Blackbird program or template representing the circuit
        """
        return None

    @property
    def compile(self):
        """Device-specific compilation method.

        This property is optional. If no special compilation logic is required,
        **do not define this property**. In such a case, it will simply return ``None``.
        """
        return None

# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

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
This module provides a class that represents the result of a quantum computation.
"""

import warnings
from typing import Mapping, Optional

import numpy as np

from strawberryfields.backends import BaseState


class Result:
    """Result of a quantum computation.

    Represents the results of the execution of a quantum program on a local or
    remote backend.

    The returned :class:`~Result` object provides several useful properties
    for accessing the results of your program execution:

    * ``results.state``: The quantum state object contains details and methods
      for manipulation of the final circuit state. Not available for remote
      backends. See :doc:`/introduction/states` for more information regarding available
      state methods.

    * ``results.samples``: Measurement samples from any measurements performed.

    * ``results.ancilla_samples``: Measurement samples from any ancillary states
      used for measurement-based gates.

    **Example:**

    The following example runs an existing Strawberry Fields
    quantum :class:`~.Program` on the Gaussian backend to get
    a ``Result`` object.

    Using this ``Result`` object, the measurement samples
    can be returned, as well as quantum state information.

    >>> eng = sf.Engine("gaussian")
    >>> results = eng.run(prog)
    >>> print(results)
    <Result: shots=1, num_modes=3, contains state=True>
    >>> results.samples
    np.array([[0, 0, 0]])
    >>> results.state.is_pure()
    True

    .. note::

        Only local simulators will return a state object. Remote
        simulators and hardware backends will return
        measurement samples  (:attr:`Result.samples`),
        but the return value of ``Result.state`` will be ``None``.
    """

    def __init__(self, result: Mapping, ancilla_samples: Mapping = None) -> None:
        self._state = None
        self._result = result
        self._ancilla_samples = ancilla_samples

    @property
    def samples(self) -> Optional[np.ndarray]:
        """Measurement samples.

        Returned measurement samples will have shape ``(shots, modes)``.

        Returns:
            array[array[float, int]]: measurement samples returned from
            program execution
        """
        output = self._result.get("output")
        if not output:
            return None

        if len(output) > 1:
            warnings.warn(
                f"Result dictionary has {len(output)} output "
                "entries; returning only the first entry."
            )
        return output[0]

    @property
    def samples_dict(self) -> Mapping:
        """All measurement samples as a dictionary. Only available on simulators.

        Returns a dictionary which associates each mode (keys) with the list of
        measurements outcomes (values), including modes that are being measured
        several times. For multiple shots or batched execution, arrays and
        tensors are stored.

        Returns:
            dict[int, list]: mode index associated with the list of measurement outcomes
        """
        samples_dict = {key: val for key, val in self._result.items() if isinstance(key, int)}
        return samples_dict

    @property
    def ancilla_samples(self) -> Optional[Mapping]:
        """All measurement samples from ancillary modes used for measurement-based
        gates.

        Returns a dictionary which associates each mode (keys) with the
        list of measurements outcomes (values) from all the ancilla-assisted
        gates applied to that mode.

        Returns:
            dict[int, list]: mode index associated with the list of ancilla
            measurement outcomes
        """
        return self._ancilla_samples

    @property
    def state(self) -> Optional[BaseState]:
        """The quantum state object.

        The quantum state object contains details and methods
        for manipulation of the final circuit state.

        See :doc:`/introduction/states` for more information regarding available
        state methods.

        .. note::

            Only local simulators will return a state object. Remote
            simulators and hardware backends will return
            measurement samples (:attr:`Result.samples`),
            but the return value of ``Result.state`` will be ``None``.

        Returns:
            BaseState: quantum state returned from program execution
        """
        return self._state

    def __repr__(self) -> str:
        """String representation."""
        if self.samples.ndim == 2:
            # if samples has dim 2, assume they're from a standard Program
            shots, modes = self.samples.shape

            if self.ancilla_samples is not None:
                ancilla_modes = 0
                for i in self.ancilla_samples.keys():
                    ancilla_modes += len(self.ancilla_samples[i])
                return (
                    f"<Result: shots={shots}, num_modes={modes}, num_ancillae={ancilla_modes}, "
                    f"contains state={self._state is not None}>"
                )

            return "<Result: shots={}, num_modes={}, contains state={}>".format(
                shots, modes, self._state is not None
            )

        if self.samples.ndim == 3:
            # if samples has dim 3, assume they're TDM
            shots, modes, timebins = self.samples.shape
            return "<Result: shots={}, spatial_modes={}, timebins={}, contains state={}>".format(
                shots, modes, timebins, self._state is not None
            )

        return "<Result: contains state={}>".format(self._state is not None)

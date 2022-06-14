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
from typing import Mapping, Optional, List

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

    * ``results.samples_dict``: Measurement samples from any measurements performed,
      sorted by measured mode. Not available for remote backends.

    * ``results.metadata``: Metadata for job results.

    * ``results.ancillae_samples``: Measurement samples from any ancillary states
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

    def __init__(self, result: Mapping, **kwargs) -> None:
        self._state = None
        self._result = result

        self._ancillae_samples = kwargs.get("ancillae_samples", None)
        self._samples_dict = kwargs.get("samples_dict", None)

    @property
    def samples(self) -> Optional[np.ndarray]:
        """Measurement samples.

        Returned measurement samples will have shape ``(shots, modes)``.

        Returns:
            ndarray[ndarray[float, int]]: measurement samples returned from
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
    def samples_dict(self) -> Optional[Mapping[int, List]]:
        """All measurement samples as a dictionary. Only available on simulators.

        Returns a dictionary which associates each mode (keys) with the list of
        measurements outcomes (values), including modes that are being measured
        several times. For multiple shots or batched execution, arrays and
        tensors are stored.

        .. note::

            The samples dictionary may contain samples that are not present in
            the samples array. In the samples array, each time a mode is measured
            the prior measured sample on the same mode is replaced. The samples
            dictionary, on the other hand, keeps _all_ measurements.

        Returns:
            Mapping[int, list]: mode index associated with the list of measurement outcomes
        """
        return self._samples_dict

    @property
    def metadata(self) -> Mapping[str, np.ndarray]:
        """Metadata for the job results.

        The metadata is considered to be everything contained in the raw results
        except for the samples, which is stored under the "output" key.

        Returns:
            Mapping[str, ndarray]: dictionary containing job result metadata
        """
        metadata = {key: val for key, val in self._result.items() if key != "output"}
        return metadata

    @property
    def ancillae_samples(self) -> Optional[Mapping[int, List]]:
        """All measurement samples from ancillary modes used for measurement-based
        gates.

        Returns a dictionary which associates each mode (keys) with the
        list of measurements outcomes (values) from all the ancilla-assisted
        gates applied to that mode.

        Returns:
            Mapping[int, list]: mode index associated with the list of ancilla
            measurement outcomes
        """
        return self._ancillae_samples

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

    @state.setter
    def state(self, state: BaseState) -> None:
        """Set the state on local simulations if not previously set.

        Raises:
            TypeError: if state is already set
            ValueError: if result have ``samples_dict`` (i.e, do not come from a local simulation)
        """
        if self._state:
            raise TypeError("State already set and cannot be changed.")
        # Samples from local simulation jobs will always have a samples_dict containing the same
        # samples organized by measured mode. `self.samples` is either `None` or an array.
        if self.samples is not None and len(self.samples) != 0 and not self.samples_dict:
            raise ValueError("State can only be set for local simulations.")
        if not (isinstance(state, BaseState) or state is None):
            raise TypeError(f"State must be of type 'BaseState', not '{type(state)}'")

        self._state = state

    def __repr__(self) -> str:
        """String representation."""
        if self.samples is not None and self.samples.ndim == 2:
            # if samples has dim 2, assume they're from a standard Program
            shots, modes = self.samples.shape

            if self.ancillae_samples is not None:
                ancillae_modes = 0
                for i in self.ancillae_samples.keys():
                    ancillae_modes += len(self.ancillae_samples[i])
                return (
                    f"<Result: shots={shots}, num_modes={modes}, num_ancillae={ancillae_modes}, "
                    f"contains state={self._state is not None}>"
                )

            return (
                f"<Result: shots={shots}, num_modes={modes}, contains "
                f"state={self._state is not None}>"
            )

        if self.samples is not None and self.samples.ndim == 3:
            # if samples has dim 3, assume they're TDM
            shots, modes, timebins = self.samples.shape
            return (
                f"<Result: shots={shots}, spatial_modes={modes}, timebins={timebins}, "
                f"contains state={self._state is not None}>"
            )

        return f"<Result: contains state={self._state is not None}>"

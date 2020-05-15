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
import numpy as np


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

    **Example:**

    The following example runs an existing Strawberry Fields
    quantum :class:`~.Program` on the Gaussian backend to get
    a ``Result`` object.

    Using this ``Result`` object, the measurement samples
    can be returned, as well as quantum state information.

    >>> eng = sf.Engine("gaussian")
    >>> results = eng.run(prog)
    >>> print(results)
    Result: 3 subsystems
        state: <GaussianState: num_modes=3, pure=True, hbar=2>
        samples: [[0, 0, 0]]
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

    def __init__(self, samples, is_stateful=True):
        self._state = None
        self._is_stateful = is_stateful

        # samples arrives as either a 2-D array (for shots > 1) or a 1-D array (for shots = 1)
        # the latter needs to be converted to a 2-D array with shape (shots, modes)
        if np.ndim(samples) == 1:
            samples = np.array([samples])
        self._samples = samples

    @property
    def samples(self):
        """Measurement samples.

        Returned measurement samples will have shape ``(shots, modes)``.

        Returns:
            array[array[float, int]]: measurement samples returned from
            program execution
        """
        return self._samples

    @property
    def state(self):
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
        if not self._is_stateful:
            raise AttributeError("The state is undefined for a stateless computation.")
        return self._state

    def __repr__(self):
        """String representation."""
        shots, modes = self.samples.shape
        return "<Result: num_modes={}, shots={}, contains state={}>".format(
            modes, shots, self._is_stateful
        )

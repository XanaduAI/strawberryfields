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

    def __init__(self, samples, all_samples=None, is_stateful=True, ancilla_samples=None):
        self._state = None
        self._is_stateful = is_stateful
        self._samples = samples
        self._all_samples = all_samples
        self._ancilla_samples = ancilla_samples

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
    def all_samples(self):
        """All measurement samples.

        Returns a dictionary which associates each mode (keys) with the
        list of measurements outcomes (values). For multiple shots or
        batched execution arrays and tensors are stored.

        Returns:
            dict[int, list]: mode index associated with the list of
            measurement outcomes
        """
        return self._all_samples

    @property
    def ancilla_samples(self):
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
        if self.samples.ndim == 2:
            # if samples has dim 2, assume they're from a standard Program
            shots, modes = self.samples.shape

            if self.ancilla_samples is not None:
                ancilla_modes = 0
                for i in self.ancilla_samples.keys():
                    ancilla_modes += len(self.ancilla_samples[i])
                return (
                    f"<Result: shots={shots}, num_modes={modes}, num_ancillae={ancilla_modes}, "
                    f"contains state={self._is_stateful}>"
                )

            return "<Result: shots={}, num_modes={}, contains state={}>".format(
                shots, modes, self._is_stateful
            )

        if self.samples.ndim == 3:
            # if samples has dim 3, assume they're TDM
            shots, modes, timebins = self.samples.shape
            return "<Result: shots={}, spatial_modes={}, timebins={}, contains state={}>".format(
                shots, modes, timebins, self._is_stateful
            )

        return "<Result: contains state={}>".format(self._is_stateful)

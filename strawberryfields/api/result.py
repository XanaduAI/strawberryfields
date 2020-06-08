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
import itertools as it
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

    @staticmethod
    def combine_samples(samples_list, mode_order):
        """Combine samples measured at different times into one nested array.

        Args:
            samples_list (list[array, tensor]): the sample measurements in a list
            mode_order (list[array]): The nested mode order for the measured modes. Must have the
                same structure as samples_list.

        Returns:
            array: the samples in ascending mode order with shape ``(shots, measured_modes)``
        """

        shots = len(samples_list[0])

        mode_order_flat = np.hstack(mode_order)
        len_modes = len(set(mode_order_flat))

        # remove duplicate mode-measurements and only return that modes last measurement
        if len(mode_order_flat) != len_modes:
            # change to list to allow updates via reference, and split samples into shots
            mode_order_list = [list(m) for m in mode_order]
            samples_shot_list = [[list(s[i]) for s in samples_list] for i in range(shots)]

            combos = it.combinations(range(len(mode_order_list)), 2)

            # go through each combination of mode orders (one per measurement) and delete
            # the first occurence of any duplicates; also delete corresponding sample(s)
            for a, b in it.combinations(mode_order_list, 2):
                intersect, *idx = np.intersect1d(a, b, return_indices=True)
                k = combos.__next__()[0]
                for i, inter in enumerate(intersect):
                    a.remove(inter)
                    for j in range(shots):
                        s_val = samples_shot_list[j][k][idx[i][0]]
                        samples_shot_list[j][k].remove(s_val)

            # reshape into a list of arrays, ignoring empty ones
            mode_order = [np.array(m) for m in mode_order_list if m]
            samples_list = [np.array([s]) for s in zip(*samples_shot_list) if s[0]]

            shapes = np.cumsum([len(m) for m in mode_order])

        # create the correct mode order if not all modes are measured
        mode_order_flat = np.hstack(mode_order)
        if np.max(mode_order_flat) + 1 != len_modes:
            shapes = np.cumsum([len(m) for m in mode_order])
            # argsort twice returns the rank
            mode_order_flat = np.argsort(np.argsort(mode_order_flat))
            mode_order = np.split(mode_order_flat, shapes)[:-1]

        # find the "widest" type in samples_list
        type_list = ["c", "f", "i"]
        while type_list[0] not in [np.array(s).dtype.kind for s in samples_list]:
            del type_list[0]

        ret = np.empty([shots, len_modes], dtype=type_list[0] + "8")

        for m, s in zip(mode_order, samples_list):
            ret[:, m] = s

        return ret

    def __repr__(self):
        """String representation."""
        shots, modes = self.samples.shape
        return "<Result: num_modes={}, shots={}, contains state={}>".format(
            modes, shots, self._is_stateful
        )

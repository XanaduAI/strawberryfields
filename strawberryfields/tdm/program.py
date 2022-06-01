# Copyright 2019-2022 Xanadu Quantum Technologies Inc.

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
This module implements the :class:`.TDMProgram` class which acts as a representation for time-domain quantum circuits.
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init

import itertools
from operator import itemgetter
from collections.abc import Iterable
from functools import reduce

import numpy as np
from strawberryfields import ops
from strawberryfields.program import Program
from strawberryfields.parameters import par_is_symbolic, FreeParameter
from strawberryfields.program_utils import CircuitError


def is_ptype(name: str) -> bool:
    """Checks whether a variable name is a p-type variable.

    p-type variables are used with TDM programs to represent a looped-over parameter, and
    consist of the letter 'p' followed by an integer. For example, 'p0', 'p1', 'p42'.
    """
    return len(name) > 1 and name[0] == "p" and name[1:].isdigit()


def shift_by(l, n):
    """Convenience function to shift a list by a number of steps.

    If ``n`` is positive it shifts to the left.

    Args:
        l (list): list to be shifted
        n (int): size of the shift
    """
    return l[n:] + l[:n]


def _get_modes(cmd, q):
    """Returns the registers from q required by a command.

    Args:
        cmd (.Command): a Strawberryfields command
        q (.RegRef): a register object

    Return:
        modes (tuple[int]): sequence of mode labels
    """
    reg_getter = itemgetter(*[r.ind for r in cmd.reg])
    modes = reg_getter(q)

    if not isinstance(modes, tuple):
        modes = (modes,)

    return modes


def _input_check(args):
    """Checks the input arguments have consistent dimensions.

    Args:
        args (Sequence[Sequence]): sequence of sequences specifying the value of the parameters
    """
    # check if all lists are of equal length
    param_lengths = [len(param) for param in args]
    if len(set(param_lengths)) != 1:
        raise ValueError("Gate-parameter lists must be of equal length.")


def _get_mode_order(num_of_values, modes, N):
    """Get the order in which the modes were measured.

    The mode order is determined by the circuit and the mode-shifting occurring in
    :class:`~.TDMProgram`. For the following circuit, the mode order returned by this
    function would be ``[0, 2, 0, 1]``, duplicated shots number of times:

    >>> prog = sf.TDMProgram(N = [1, 2])

    >>> with prog.context([1, 2], [4, 5]) as (p, q):
    ...     MeasureHomodyne(p[0]) | q[0]
    ...     MeasureHomodyne(p[1]) | q[2]

    """

    all_modes = []
    mode_order = []

    for i, _ in enumerate(N):
        timebin_modes = list(range(sum(N[:i]), sum(N[: i + 1])))
        # shift the timebin_modes if the measured mode isn't the first in the
        # band, so that the measurements start at the correct mode
        shift = modes[i] - sum(N[:i])
        timebin_modes = timebin_modes[shift:] + timebin_modes[:shift]

        # extend the modes by duplicating the list so that the measured mode
        # orders in all bands have the same length
        extended_modes = timebin_modes * (1 + num_of_values // len(timebin_modes))
        all_modes.append(extended_modes[:num_of_values])

        # alternate measurements in the bands and extend/duplicate the resulting
        # list so that it is at least as long as `num_of_values`
        mode_order = [i for j in zip(*all_modes) for i in j]
    return mode_order[:num_of_values]


def reshape_samples(samples_dict, modes, N, timebins):
    """Reshapes the samples dict so that they have the expected correct shape.

    Corrects the :attr:`~.Results.samples_dict` dictionary so that the measured modes are
    the ones defined to be measured in the circuit, instead of being spread over a larger
    number of modes due to the mode-shifting occurring in :class:`~.TDMProgram`.

    The function iterates through samples obtained from the unrolled circuit to populate
    and return a new samples dictionary with the shape ``{spatial mode: (shots,
    timebins)}``. E.g., this unrolled circuit:

    .. code-block:: python

        ...
        MeasureHomodyne(0) | (q[0])  # shot 0, timebin 0, spatial mode 0  (sample 0)
        MeasureHomodyne(0) | (q[2])  # shot 0, timebin 0, spatial mode 2  (sample 1)
        ...
        MeasureHomodyne(0) | (q[0])  # shot 0, timebin 1, spatial mode 0  (sample 2)
        MeasureHomodyne(0) | (q[1])  # shot 0, timebin 1, spatial mode 2  (sample 3)
        ...
        MeasureHomodyne(0) | (q[0])  # shot 1, timebin 0, spatial mode 0  (sample 4)
        MeasureHomodyne(0) | (q[2])  # shot 1, timebin 0, spatial mode 2  (sample 5)
        ...
        MeasureHomodyne(0) | (q[0])  # shot 1, timebin 1, spatial mode 0  (sample 6)
        MeasureHomodyne(0) | (q[1])  # shot 1, timebin 1, spatial mode 2  (sample 7)

    would return the dictionary

    .. code-block:: python

        {
            0: np.array([(sample 0), (sample 2), (sample 4), (sample 6)]),
            2: np.array([(sample 1), (sample 5)]),
            1: np.array([(sample 3), (sample 7)]),
        }

    which would then be reshaped, and returned, as follows:

    .. code-block:: python

        {
            0: np.array([[(sample 0), (sample 2)],
                         [(sample 4), (sample 6)]]),
            2: np.array([[(sample 1), (sample 3)],
                         [(sample 5), (sample 7)]]),
        }

    Args:
        samples_dict (dict[int, list]): the raw measured samples
        modes (Sequence[int]): the modes that are measured in the circuit
        N (Sequence[int]): the number of concurrent modes per belt/spatial modes
        timebins (int): the number of timebins/temporal modes in the program per shot

    Returns:
        dict[int, array]: the re-shaped samples, where each key correspond to a spatial
            mode and the values have shape ``(shots, timebins)``
    """
    # calculate the total number of samples and the order in which they were measured
    num_of_values = len([i for j in samples_dict.values() for i in j])
    mode_order = _get_mode_order(num_of_values, modes, N)
    idx_tracker = {i: 0 for i in mode_order}

    # iterate backwards through `samples_dict` and add them into the correct mode
    new_samples = {}
    timebin_idx = 0
    for i, mode in enumerate(mode_order):
        mode_idx = modes[i % len(N)]

        if mode_idx not in new_samples:
            # create an entry for the new mode with a nested list for each timebin
            new_samples[mode_idx] = [[] for _ in range(timebins)]

        sample = samples_dict[mode][idx_tracker[mode]][0]
        idx_tracker[mode] += 1
        new_samples[mode_idx][timebin_idx].append(sample)

        # populate each spatial mode in one timebin before moving on to the next timebin
        # when each timebin has been filled, move to the next shot
        last_mode_in_timebin = (i + 1) % len(N) == 0
        if last_mode_in_timebin:
            timebin_idx = (timebin_idx + 1) % timebins

    # transpose each value so that it has shape `(shots, timebins)`
    return {k: np.array(v).T for k, v in new_samples.items()}


class TDMProgram(Program):
    r"""Represents a photonic quantum circuit in the time domain encoding.

    The ``TDMProgram`` class provides a context manager for easily defining
    a single time-bin of the time domain algorithm. As with the standard
    :class:`~.Program`, Strawberry Fields operations are appended to the
    time domain program using the Python-embedded Blackbird syntax.

    Once created, time domain programs can be executed on Strawberry Fields'
    Gaussian backend in an efficient manner, or submitted
    to be executed on compatible hardware.

    Args:
        N (int or Sequence[int]): If an integer, the number of concurrent (or 'alive')
            modes in each time bin. Alternatively, a sequence of integers
            may be provided, corresponding to the number of concurrent modes in
            the possibly multiple bands in the circuit.
        name (str): the program name (optional)

    **Example**

    Below, we create a time domain program with 2 concurrent modes:

    >>> import strawberryfields as sf
    >>> from strawberryfields import ops
    >>> prog = sf.TDMProgram(N=2)

    Once created, we can construct the program using the ``prog.context()``
    context manager.

    >>> with prog.context([1, 2], [3, 4]) as (p, q):
    ...     ops.Sgate(0.7, 0) | q[1]
    ...     ops.BSgate(p[0]) | (q[0], q[1])
    ...     ops.MeasureHomodyne(p[1]) | q[0]

    Printing out this program:

    >>> prog.print()
    Sgate(0.7, 0) | (q[1])
    BSgate({p0}, 0) | (q[0], q[1])
    MeasureHomodyne({p1}) | (q[0])

    Note that ``p0`` and ``p1`` are symbolic gate parameters; to access the numeric values,
    we must use the ``prog.parameters`` attribute:

    >>> prog.parameters
    {'p0': [1, 2], 'p1': [3, 4]}

    When we simulate a time-domain program, it is first unrolled by the engine; unrolling involves
    explicitly repeating the single time-bin sequence constructed above, and *shifting* the
    simulated registers. This 'unrolling' procedure is performed automatically by the engine, however, we can
    visualize the unrolled program by calling the :meth:`~.unroll` method.

    >>> prog.unroll(shots=3).print()
    Sgate(0.7, 0) | (q[1])
    BSgate(1, 0) | (q[0], q[1])
    MeasureHomodyne(3) | (q[0])
    Sgate(0.7, 0) | (q[0])
    BSgate(2, 0) | (q[1], q[0])
    MeasureHomodyne(4) | (q[1])
    Sgate(0.7, 0) | (q[1])
    BSgate(1, 0) | (q[0], q[1])
    MeasureHomodyne(3) | (q[0])
    Sgate(0.7, 0) | (q[0])
    BSgate(2, 0) | (q[1], q[0])
    MeasureHomodyne(4) | (q[1])
    Sgate(0.7, 0) | (q[1])
    BSgate(1, 0) | (q[0], q[1])
    MeasureHomodyne(3) | (q[0])
    Sgate(0.7, 0) | (q[0])
    BSgate(2, 0) | (q[1], q[0])
    MeasureHomodyne(4) | (q[1])

    Note the 'shifting' of the measured registers --- this optimization allows
    for time domain algorithms to be simulated in memory much more efficiently:

    >>> eng = sf.Engine("gaussian")
    >>> results = eng.run(prog, shots=3)

    The engine automatically takes this mode shifting into account; returned samples
    will always be transformed to match the modes specified during construction:

    >>> print(results.samples_dict)
    {0: [array([1.26208025]), array([1.53910032]), array([-1.29648336]),
    array([0.75743215]), array([-0.17850101]), array([-1.44751996])]}

    Note that, unlike the standard :class:`~.Program`,
    the TDM context manager has both required and essential arguments:

    * Positional arguments are used to pass sequences of gate arguments
      to apply per time-bin. Within the context, these are accessible via the ``p``
      context variable; ``p[i]`` corresponds to the ``i``-th positional
      arguments.

      All positional arguments corresponding to sequences of gate arguments
      **must be the same length**. This length determines how many time-bins
      are present in the TDM program.

      If the ``p`` variable is not used, the gate argument is assumed to be **constant**
      across all time-bins.

    * ``shift="default"`` *(str or int)*: Defines how the qumode register is shifted at the end of
      each time bin. If set to ``"default"``, the qumode register is shifted such that each measured
      qumode reappears as a fresh mode at the beginning of the subsequent time bin. This is
      equivalent to a measured qumode being removed from the representation (by measurement) and a
      new one being added (by a light source). If set to an integer value, the register will shift
      by a step size of this integer at the end of each time bin.

    .. raw:: html

        <a class="meth-details-header collapse-header" data-toggle="collapse" href="#tutorials" aria-expanded="false" aria-controls="tutorials">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Related tutorials
         </h2>
        </a>
        <div class="collapse" id="tutorials">

    .. gallery-item::
        :description: :doc:`demos/run_time_domain`
        :figure: _static/oneloop.svg

    .. raw:: html

        <div style='clear:both'></div>
        </div>
    """

    def __init__(self, N, name=None):
        # N is a list with the number of qumodes for each spatial mode
        if isinstance(N, int):
            self.N = [N]
        else:
            self.N = N
        self._concurr_modes = sum(self.N)

        super().__init__(num_subsystems=self._concurr_modes, name=name)

        # store the number of shots used when unrolling the circuit (if unrolled; else `None`)
        self._unrolled_shots = None

        self._timebins = 0
        self._spatial_modes = 0
        self._measured_modes = set()

        self.rolled_circuit = []
        # `unrolled_circuit` contains the unrolled single-shot circuit, reusing previously measured
        # modes (doesn't work with Fock measurements)
        self.unrolled_circuit = None
        # `space_unrolled_circuit` contains the space-unrolled single-shot circuit, instead adding
        # new modes for each new measurement (works with Fock measurements)
        self.space_unrolled_circuit = None
        # `_num_added_subsystems` corresponds to the number of subsystems added when space-unrolling
        self._num_added_subsystems = 0
        self.run_options = {}
        """dict[str, Any]: dictionary of default run options, to be passed to the engine upon
        execution of the program. Note that if the ``run_options`` dictionary is passed
        directly to :meth:`~.Engine.run`, it takes precedence over the run options specified
        here.
        """

    @property
    def measured_modes(self):
        """The number of measured modes in the program returned as a list."""
        return list(self._measured_modes)

    @property
    def timebins(self):
        """The number of timebins in the program."""
        return self._timebins

    @property
    def spatial_modes(self):
        """The number of spatial modes in the program."""
        return self._spatial_modes

    @property
    def concurr_modes(self):
        """The number of concurrent modes in the program."""
        return self._concurr_modes

    # pylint: disable=arguments-differ, invalid-overridden-method
    def context(self, *args, shift="default"):
        _input_check(args)
        self.tdm_params = list(args)
        self.shift = shift
        self.loop_vars = self.params(*[f"p{i}" for i, _ in enumerate(args)])
        # if a single parameter list is supplied, only a single free
        # parameter will be created; turn it into a list
        if not isinstance(self.loop_vars, Iterable):
            self.loop_vars = [self.loop_vars]
        return self

    def __enter__(self):
        super().__enter__()
        return self.loop_vars, self.register

    def __exit__(self, ex_type, ex_value, ex_tb):
        super().__exit__(ex_type, ex_value, ex_tb)

        if ex_type is None:
            self._timebins = len(self.tdm_params[0])
            self.rolled_circuit = self.circuit.copy()

            self._spatial_modes = len(self.N)

    @property
    def parameters(self):
        """Return the parameters of the ``TDMProgram`` as a dictionary with the parameter
        name as keys, and the parameter lists as values."""
        return dict(zip([i.name for i in self.loop_vars], self.tdm_params))

    @property
    def is_unrolled(self):
        """Whether the circuit is either unrolled or space-unrolled."""
        return not (self.unrolled_circuit is None and self.space_unrolled_circuit is None)

    def roll(self):
        """Represent the program in a compressed way without unrolling the for loops."""
        if not self.is_unrolled:
            return

        self._unrolled_shots = None
        self.circuit = self.rolled_circuit

        if self.space_unrolled_circuit is not None:
            if self._num_added_subsystems > 0:
                self._delete_subsystems(self.register[-self._num_added_subsystems :])
                self.init_num_subsystems -= self._num_added_subsystems
                self._num_added_subsystems = 0

            self.space_unrolled_circuit = None
        self.unrolled_circuit = None

    def unroll(self, shots=1):
        """Construct program with the register shift and store it in ``self.circuit``.

        Calls the ``_unroll_program`` method which constructs the unrolled single-shot program,
        storing it in `self.unrolled_circuit` when run for the first time, and returns the unrolled
        program.

        Args:
            shots (int): the number of times the circuit should be repeated
        """
        _locked = self.locked
        if self.locked:
            self.locked = False

        if self.unrolled_circuit is not None:
            if self._unrolled_shots == shots:
                self.circuit = self.unrolled_circuit
                return
            self.roll()

        # store the number of shots in the unrolled circuit
        self._unrolled_shots = shots

        if self.space_unrolled_circuit is not None:
            raise ValueError(
                "Program is space-unrolled and cannot be unrolled. Must be rolled (by calling the"
                "'roll()' method) before unrolling."
            )

        self._unroll_program(shots, space=False)
        self.locked = _locked

    def space_unroll(self, shots=1):
        """Construct the space-unrolled program and set it to ``self.circuit``.

        Calls the ``_unroll_program`` method which constructs the space-unrolled single-shot program,
        storing it in `self.space_unrolled_circuit` when run for the first time, and returns the
        space-unrolled program.

        Args:
            shots (int): the number of times the circuit should be repeated
        """
        _locked = self.locked
        if self.locked:
            self.locked = False

        if self.space_unrolled_circuit is not None and self._unrolled_shots == shots:
            self.circuit = self.space_unrolled_circuit
            return
        self.roll()

        # store the number of shots in the unrolled circuit
        self._unrolled_shots = shots

        vac_modes = self.concurr_modes - 1
        self._num_added_subsystems = self.timebins - self.init_num_subsystems + vac_modes
        if self._num_added_subsystems > 0:
            self._add_subsystems(self._num_added_subsystems)

            self.init_num_subsystems += self._num_added_subsystems

        if self.unrolled_circuit is not None:
            raise ValueError(
                "Program is unrolled and cannot be space-unrolled. Must be rolled (by calling the"
                "`roll()` method) before space-unrolling."
            )

        self._unroll_program(shots, space=True)
        self.locked = _locked

    def _unroll_program(self, shots, space):
        """Construct the unrolled program either using space-unrolling or with register shift."""
        # self.circuit should always be rolled here, so we can store `self.rolled_circuit`
        self.rolled_circuit = self.circuit
        assert not self.is_unrolled

        # reset the circuit which is to be recreated below
        self.circuit = []

        q = self.register

        sm = []
        for i, _ in enumerate(self.N):
            start = sum(self.N[:i])
            stop = sum(self.N[:i]) + self.N[i]
            sm.append(slice(start, stop))

        # Above we define a list of slice intervals;
        # each slice interval corresponding to one spatial mode

        # For instance, say
        # N = [5, 4, 9, 1],
        # then this corresponds to
        # 5 concurrent modes in spatial mode A
        # 4 concurrent modes in spatial mode B
        # 9 concurrent modes in spatial mode C
        # 1 concurrent modes in spatial mode D.

        # The entries of sm will then be:
        # slice(0, 6, None)   for spatial mode A
        # slice(6, 10, None)  for spatial mode B
        # slice(10, 19, None) for spatial mode C
        # slice(19, 20, None) for spatial mode D.

        # This helps to define:
        # q[sm[0]] as concurrent modes of spatial mode A
        # q[sm[1]] as concurrent modes of spatial mode B
        # q[sm[2]] as concurrent modes of spatial mode C
        # q[sm[3]] as concurrent modes of spatial mode D.

        for _ in range(shots):
            # save previous mode index of a command to be able to check when modes
            # are looped back to the start (not allowed when space-unrolling)
            previous_mode_index = {}

            for cmd in self.rolled_circuit:
                previous_mode_index[cmd] = 0
                if isinstance(cmd.op, ops.Measurement):
                    self._measured_modes.add(cmd.reg[0].ind)

            for i in range(self.timebins):
                for cmd in self.rolled_circuit:
                    modes = _get_modes(cmd, q)
                    has_looped_back = any(m.ind < previous_mode_index[cmd] for m in modes)
                    valid_application = not space or not has_looped_back
                    if valid_application:
                        self.apply_op(cmd, modes, i)
                        previous_mode_index[cmd] = min(m.ind for m in modes)

                if space:
                    q = shift_by(q, 1)
                elif self.shift == "default":
                    # shift each spatial mode SEPARATELY by one step
                    q_aux = list(q)
                    for j, _ in enumerate(self.N):
                        q_aux[sm[j]] = shift_by(q_aux[sm[j]], 1)
                    q = tuple(q_aux)

                elif isinstance(self.shift, int):
                    q = shift_by(q, self.shift)  # shift at end of each time bin

        # Unrolling the circuit for the first time: storing a copy of the unrolled circuit
        if space:
            self.space_unrolled_circuit = self.circuit.copy()
        else:
            self.unrolled_circuit = self.circuit.copy()

    def apply_op(self, cmd, modes, t):
        """Apply a particular operation on register q at timestep t."""
        params = cmd.op.p.copy()

        for i, _ in enumerate(params):
            if par_is_symbolic(params[i]):
                params[i] = self.parameters[params[i].name][t % self.timebins]

        self.append(cmd.op.__class__(*params), modes)

    def assert_modes(self, device):
        """Check that the number of modes in the program is valid.

        .. note::

            ``device.modes`` must be a dictionary containing the maximum number of allowed
            measurements for the specified target.

        Args:
            device (.strawberryfields.DeviceSpec): device specification object to use
        """
        if self.timebins > device.modes["temporal_max"]:
            raise CircuitError(
                f"This program contains {self.timebins} temporal modes, but the device '{device.target}' "
                f"only supports up to {device.modes['temporal_max']} modes."
            )
        if self.concurr_modes != device.modes["concurrent"]:
            raise CircuitError(
                f"This program contains {self.concurr_modes} concurrent modes, but the device '{device.target}' "
                f"only supports {device.modes['concurrent']} modes."
            )
        if self.spatial_modes != device.modes["spatial"]:
            raise CircuitError(
                f"This program contains {self.spatial_modes} spatial modes, but the device '{device.target}' "
                f"only supports {device.modes['spatial']} modes."
            )

    def get_delays(self):
        """Calculates and returns the loop delays.

        This method calculates the delays present in the circuit by using the modes in which
        beamsplitters are acting on: on TMDPrograms with a single spatial mode, loop crossings
        always correspond to beamsplitters; if no beamsplitters are presents this method returns an
        empty list.

        Returns:
            List[int]: A list where the length represent the number of loops present in the circuit
            and every entry corresponds to the number of modes that will fit at the same time
            in each loop.
        """

        if self.spatial_modes > 1:
            raise NotImplementedError(
                "Calculating delays for programs with more than one spatial mode is not implemented."
            )

        # Extract index of modes on which beamsplitter ops are acting
        bs_modes = []
        for cmd in self.rolled_circuit:
            if isinstance(cmd.op, ops.BSgate):
                bs_modes.append(sorted([cmd.reg[0].ind, cmd.reg[1].ind]))

        # Check if circuit has nested delay loops when there is more than one bs
        # by checking if the ranges between all beamsplitter mode indexes overlap
        if len(bs_modes) > 1:
            bs_mode_ranges = [set(range(mode0, mode1)) for mode0, mode1 in bs_modes]
            modes_intersection = reduce(lambda s1, s2: s1 & s2, bs_mode_ranges)

            if len(modes_intersection) != 0:
                raise NotImplementedError(
                    "Calculating delays for programs with nested loops is not implemented."
                )

        # Calculate delay list
        bs_modes = np.array(sorted(set(itertools.chain(*bs_modes)))[::-1])
        delays = (bs_modes - np.roll(bs_modes, -1))[:-1]

        return list(delays)

    def get_crop_value(self):
        """Calculates and returns the value for where to crop the gate parameters.

        This method returns the number of vacuum modes arriving at the detector before the first
        computational mode for any job run on a single spatial mode TDM program.

        Returns:
            int: crop parameter
        """

        if self.spatial_modes > 1:
            raise NotImplementedError(
                "Cropping vacuum modes for programs with more than one spatial mode is not implemented."
            )

        # arrival time of first computational mode at a respective
        # loop or detector
        arrival_time = 0

        # Extract beamsplitter symbolic p parameters from rolled circuit
        bs_p_params = []
        for cmd in self.rolled_circuit:
            if isinstance(cmd.op, ops.BSgate):
                for p in cmd.op.p:
                    if isinstance(p, FreeParameter):
                        bs_p_params.append(p.name)
                        # BS can have two symbolic parameters,
                        # as soon one is found, append and continue checking the next op
                        break

        for p_param, max_delay in zip(bs_p_params, self.get_delays()):

            # number of cross time bins (open loop) at the beginning of the
            # sequence, counting from the arrival time of the first computational
            # mode
            alpha = self.parameters[p_param]
            start_zeros = next(
                (index for index, value in enumerate(alpha[arrival_time:]) if value != 0),
                len(alpha[arrival_time:]),
            )

            # delay imposed by loop corresponds to start_zeros, but is upper-bounded
            # by max_delay
            delay_imposed_by_loop = min(start_zeros, max_delay)

            # delay imposed by loop i adds to the arrival time of first comp. mode
            # at loop i+1 (or detector)
            arrival_time += delay_imposed_by_loop

        return arrival_time

    def __str__(self):
        s = (
            f"<TDMProgram: concurrent modes={self.concurr_modes}, "
            f"time bins={self.timebins}, "
            f"spatial modes={self.spatial_modes}>"
        )
        return s

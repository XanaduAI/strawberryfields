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
This module implements the :class:`.TDMProgram` class which acts as a representation for time-domain quantum circuits.
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init

from operator import itemgetter
from math import ceil
from collections.abc import Iterable

import numpy as np
import blackbird as bb
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.parameters import par_is_symbolic
from strawberryfields.program_utils import CircuitError


def shift_by(l, n):
    """Convenience function to shift a list by a number of steps.

    If ``n`` is positive it shifts to the left.

    Args:
        l (list): list to be shifted
        n (int): size of the shift
    """
    return l[n:] + l[:n]


def get_modes(cmd, q):
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


def validate_measurements(circuit, N):
    """Validate the TDM program measurements are correct.

    Args:
        circuit (list): a list containing commands or measurements specifying a circuit
        N (list): list giving the number of concurrent modes per band

    Return:
        spatial_modes (int): number of spatial modes
    """
    spatial_modes = 0

    for cmd in circuit:
        if isinstance(cmd.op, ops.Measurement):
            modes = [r.ind for r in cmd.reg]
            spatial_modes += len(modes)

    if not spatial_modes:
        raise ValueError("Must be at least one measurement.")

    if spatial_modes is not len(N):
        raise ValueError("Number of measurement operators must match number of spatial modes.")

    return spatial_modes


def input_check(args):
    """Checks the input arguments have consistent dimensions.

    Args:
        args (Sequence[Sequence]): sequence of sequences specifying the value of the parameters
    """
    # check if all lists are of equal length
    param_lengths = [len(param) for param in args]
    if len(set(param_lengths)) != 1:
        raise ValueError("Gate-parameter lists must be of equal length.")


def _get_mode_order(num_of_values, modes, N, timebins):
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
    for i in range(len(N)):
        timebin_modes = list(range(sum(N[:i]), sum(N[: i + 1])))
        # shift the timebin_modes if the measured mode isn't the first in the
        # band, so that the measurements start at the correct mode
        shift = modes[i] - sum(N[:i])
        timebin_modes = timebin_modes[shift:] + timebin_modes[:shift]

        # extend the modes by duplicating the list so that the measured mode
        # orders in all bands have the same length
        extended_modes = timebin_modes * ceil(1 + timebins // len(timebin_modes))
        all_modes.append(extended_modes[:timebins])

    # alternate measurements in the bands and extend/duplicate the resulting
    # list so that it is at least as long as `num_of_values`
    mode_order = [i for j in zip(*all_modes) for i in j]
    mode_order *= ceil(1 + num_of_values / len(mode_order))

    return mode_order[:num_of_values]


def reshape_samples(all_samples, modes, N, timebins):
    """Reshapes the samples dict so that they have the expected correct shape.

    Corrects the :attr:`~.Results.all_samples` dictionary so that the measured modes are
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
        all_samples (dict[int, list]): the raw measured samples
        modes (Sequence[int]): the modes that are measured in the circuit
        N (Sequence[int]): the number of concurrent modes per belt/spatial modes
        timebins (int): the number of timebins/temporal modes in the program per shot

    Returns:
        dict[int, array]: the re-shaped samples, where each key correspond to a spatial
            mode and the values have shape ``(shots, timebins)``
    """
    # calculate the total number of samples and the order in which they were measured
    num_of_values = len([i for j in all_samples.values() for i in j])
    mode_order = _get_mode_order(num_of_values, modes, N, timebins)

    # iterate backwards through all_samples and add them into the correct mode
    new_samples = dict()
    timebin_idx = 0
    for i, mode in enumerate(mode_order):
        mode_idx = modes[i % len(N)]

        if mode_idx not in new_samples:
            # create an entry for the new mode with a nested list for each timebin
            new_samples[mode_idx] = [[] for _ in range(timebins)]

        sample = all_samples[mode].pop(0)[0]
        new_samples[mode_idx][timebin_idx].append(sample)

        # populate each spatial mode in one timebin before moving on to the next timebin
        # when each timebin has been filled, move to the next shot
        last_mode_in_timebin = (i + 1) % len(N) == 0
        if last_mode_in_timebin:
            timebin_idx = (timebin_idx + 1) % timebins

    # transpose each value so that it has shape `(shots, timebins)`
    return {k: np.array(v).T for k, v in new_samples.items()}


def move_vac_modes(samples, N, crop=False):
    """Moves all measured vacuum modes from the first shot of the
    returned TDM samples array to the end of the last shot.

    Args:
        samples (array[float]): samples as received from ``TDMProgram``, with the
            measured vacuum modes in the first shot
        N (int or Sequence[int]): If an integer, the number of concurrent (or 'alive')
            modes in each time bin. Alternatively, a sequence of integers
            may be provided, corresponding to the number of concurrent modes in
            the possibly multiple bands in the circuit.

    Keyword args:
        crop (bool): whether to remove all the shots containing measured vacuum
            modes at the end

    Returns:
        array[float]: the post-processed samples
    """
    num_of_vac_modes = np.max(N) - 1
    shape = samples.shape

    flat_samples = np.ravel(samples)
    samples = np.append(flat_samples[num_of_vac_modes:], [0] * num_of_vac_modes)
    samples = samples.reshape(shape)

    if crop and num_of_vac_modes != 0:
        # remove the final shots that include vac mode measurements
        num_of_shots_with_vac_modes = -num_of_vac_modes // (np.prod(shape[1:]) + 1)
        samples = samples[:num_of_shots_with_vac_modes]

    return samples


class TDMProgram(sf.Program):
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

    >>> print(results.all_samples)
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

    .. customgalleryitem::
        :tooltip: Simulate massive time-domain quantum systems
        :description: :doc:`demos/run_time_domain`
        :figure: /_static/oneloop.svg

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
        self.concurr_modes = sum(self.N)

        super().__init__(num_subsystems=self.concurr_modes, name=name)

        self.type = "tdm"
        self.timebins = 0
        self.spatial_modes = 0
        self.measured_modes = []
        self.rolled_circuit = None
        # `unrolled_circuit` only contains the unrolled single-shot circuit
        self.unrolled_circuit = None
        self.run_options = {}
        """dict[str, Any]: dictionary of default run options, to be passed to the engine upon
        execution of the program. Note that if the ``run_options`` dictionary is passed
        directly to :meth:`~.Engine.run`, it takes precedence over the run options specified
        here.
        """

    # pylint: disable=arguments-differ, invalid-overridden-method
    def context(self, *args, shift="default"):
        input_check(args)
        self.tdm_params = args
        self.shift = shift
        self.loop_vars = self.params(*[f"p{i}" for i in range(len(args))])
        # if a single parameter list is supplied, only a single free
        # parameter will be created; turn it into a list
        if not isinstance(self.loop_vars, Iterable):
            self.loop_vars = [self.loop_vars]
        return self

    # pylint: disable=too-many-branches
    def compile(self, *, device=None, compiler=None):
        """Compile the time-domain program given a Strawberry Fields photonic hardware device specification.

        Currently, the compilation is simply a check that the program matches the device.

        Args:
            device (~strawberryfields.api.DeviceSpec): device specification object to use for
                program compilation
            compiler (str, ~strawberryfields.compilers.Compiler): Compiler name or compile strategy
                to use. If a device is specified, this overrides the compile strategy specified by
                the hardware :class:`~.DeviceSpec`. If no compiler is passed, the default "TD2"
                compiler is used. Currently, the only other allowed compiler is "gaussian".

        Returns:
            Program: compiled program
        """
        if compiler == "gaussian":
            return super().compile(device=device, compiler=compiler)

        if device is not None:
            device_layout = bb.loads(device.layout)

            if device_layout.programtype["name"] != "tdm":
                raise TypeError(
                    'TDM compiler only supports "tdm" type device specification layouts. '
                    "Received {} type.".format(device_layout.programtype["name"])
                )

            if device.modes is not None:
                self.assert_number_of_modes(device)

            # First check: the gates are in the correct order
            program_gates = [cmd.op.__class__.__name__ for cmd in self.rolled_circuit]
            device_gates = [op["op"] for op in device_layout.operations]
            if device_gates != program_gates:
                raise CircuitError(
                    "The gates or the order of gates used in the Program is incompatible with the device '{}' ".format(
                        device.target
                    )
                )

            # Second check: the gates act on the correct modes
            program_modes = [[r.ind for r in cmd.reg] for cmd in self.rolled_circuit]
            device_modes = [op["modes"] for op in device_layout.operations]
            if program_modes != device_modes:
                raise CircuitError(
                    "Program cannot be used with the device '{}' "
                    "due to incompatible mode ordering.".format(device.target)
                )

            # Third check: the parameters of the gates are valid
            # We will loop over the different operations in the device specification

            for i, operation in enumerate(device_layout.operations):
                # We obtain the name of the parameter(s)
                param_names = operation["args"]

                program_params_len = len(self.rolled_circuit[i].op.p)
                device_params_len = len(param_names)
                # The next if is to make sure we do not flag incorrectly things like Sgate(r,0) being different Sgate(r)
                # This assumes that parameters other than the first one are zero if not explicitly stated.
                if device_params_len < program_params_len:
                    for j in range(1, program_params_len):
                        if self.rolled_circuit[i].op.p[j] != 0:
                            raise CircuitError(
                                "Program cannot be used with the device '{}' "
                                "due to incompatible parameter.".format(device.target)
                            )
                # Now we will check explicitly if the parameters in the program match
                num_symbolic_param = 0  # counts the number of symbolic variables, which are labelled consecutively by the context method

                for k, param_name in enumerate(param_names):
                    # Obtain the value of the corresponding parameter in the program
                    program_param = self.rolled_circuit[i].op.p[k]

                    # make sure that hardcoded parameters in the device layout are correct
                    if not isinstance(param_name, str):
                        if not program_param == param_name:
                            raise CircuitError(
                                "Program cannot be used with the device '{}' "
                                "due to incompatible parameter. Parameter has value '{}' "
                                "while its valid value is '{}'".format(
                                    device.target, program_param, param_name
                                )
                            )
                        continue

                    # Obtain the relevant parameter range from the device
                    param_range = device.gate_parameters[param_name]
                    if sf.parameters.par_is_symbolic(program_param):
                        # If it is a symbolic value go and lookup its corresponding list in self.tdm_params
                        local_p_vals = self.tdm_params[num_symbolic_param]

                        for x in local_p_vals:
                            if not x in param_range:
                                raise CircuitError(
                                    "Program cannot be used with the device '{}' "
                                    "due to incompatible parameter. Parameter has value '{}' "
                                    "while its valid range is '{}'".format(
                                        device.target, x, param_range
                                    )
                                )
                        num_symbolic_param += 1

                    else:
                        # If it is a numerical value check directly
                        if not program_param in param_range:
                            raise CircuitError(
                                "Program cannot be used with the device '{}' "
                                "due to incompatible parameter. Parameter has value '{}' "
                                "while its valid range is '{}'".format(
                                    device.target, program_param, param_range
                                )
                            )
            return self

        raise CircuitError("TDM programs cannot be compiled without a valid device specification.")

    def __enter__(self):
        super().__enter__()
        return self.loop_vars, self.register

    def __exit__(self, ex_type, ex_value, ex_tb):
        super().__exit__(ex_type, ex_value, ex_tb)

        if ex_type is None:
            self.spatial_modes = validate_measurements(self.circuit, self.N)
            self.timebins = len(self.tdm_params[0])
            self.rolled_circuit = self.circuit.copy()

    @property
    def parameters(self):
        """Return the parameters of the ``TDMProgram`` as a dictionary with the parameter
        name as keys, and the parameter lists as values"""
        return dict(zip([i.name for i in self.loop_vars], self.tdm_params))

    def roll(self):
        """Represent the program in a compressed way without rolling the for loops"""
        self.circuit = self.rolled_circuit
        return self

    def unroll(self, shots):
        """Construct program with the register shift

        Constructs the unrolled single-shot program, storing it in `self.unrolled_circuit` when run
        for the first time, and returns the unrolled program including shots.

        Args:
            shots (int): the number of times the circuit should be repeated

        Returns:
            Program: unrolled program (including shots)
        """
        if self.unrolled_circuit is not None:
            self.circuit = self.unrolled_circuit * shots
            return self

        self.circuit = []

        q = self.register

        sm = []
        for i in range(len(self.N)):
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

        for cmd in self.rolled_circuit:
            if isinstance(cmd.op, ops.Measurement):
                self.measured_modes.append(cmd.reg[0].ind)
        for i in range(self.timebins):
            for cmd in self.rolled_circuit:
                self.apply_op(cmd, q, i)

            if self.shift == "default":
                # shift each spatial mode SEPARATELY by one step
                q_aux = list(q)
                for j in range(len(self.N)):
                    q_aux[sm[j]] = shift_by(q_aux[sm[j]], 1)
                q = tuple(q_aux)

            elif isinstance(self.shift, int):
                q = shift_by(q, self.shift)  # shift at end of each time bin

        # Unrolling the circuit for the first time: storing a copy of the unrolled circuit
        self.unrolled_circuit = self.circuit.copy()
        self.circuit = self.unrolled_circuit * shots

        return self

    def apply_op(self, cmd, q, t):
        """Apply a particular operation on register q at timestep t"""
        params = cmd.op.p.copy()

        for i in range(len(params)):
            if par_is_symbolic(params[i]):
                arg_index = int(params[i].name[1:])
                params[i] = self.tdm_params[arg_index][t % self.timebins]

        self.append(cmd.op.__class__(*params), get_modes(cmd, q))

    def assert_number_of_modes(self, device):
        if self.timebins > device.modes["temporal"]["max"]:
            raise CircuitError(
                f"This program contains {self.timebins} temporal modes, but the device '{device.target}' "
                f"only supports up to {device.modes['temporal']['max']} modes."
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

    def __str__(self):
        s = (
            f"<TDMProgram: concurrent modes={self.concurr_modes}, "
            f"time bins={self.timebins}, "
            f"spatial modes={self.spatial_modes}>"
        )
        return s

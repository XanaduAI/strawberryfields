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
This module implements the :class:`.TDMProgram` class which acts as a representation for time-domain quantum circuits.
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init

from operator import itemgetter
from math import ceil
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.parameters import par_is_symbolic


def shift_by(l, n):
    """Convenience function to shift a list by a number of steps.

    If ``n`` is positive it shifts to the left.

    Args:
        l (list): list to be shifted
        n (int): size of the shift
    """
    return l[n:] + l[:n]


def get_modes(cmd, q):
    """Returns the registers from q required by a command"""
    reg_getter = itemgetter(*[r.ind for r in cmd.reg])
    modes = reg_getter(q)

    if not isinstance(modes, tuple):
        modes = (modes,)

    return modes


def validate_measurements(circuit, N):
    """Validate the TDM program measurements are correct"""
    spatial_modes = 0
    measurement_reg = []

    for cmd in circuit:
        if isinstance(cmd.op, ops.Measurement):
            modes = [r.ind for r in cmd.reg]
            measurement_reg.extend(modes)
            spatial_modes += len(modes)

    if not spatial_modes:
        raise ValueError("Must be at least one measurement.")

    if spatial_modes is not len(N):
        raise ValueError("Number of measurement operators must match number of spatial modes.")

    return spatial_modes


def input_check(args, copies):
    """Checks the input arguments have consistent dimensions"""
    # check if all lists are of equal length
    param_lengths = [len(param) for param in args]
    if len(set(param_lengths)) != 1:
        raise ValueError("Gate-parameter lists must be of equal length.")

    # check copies
    if not isinstance(copies, int) or copies < 1:
        raise TypeError("Number of copies must be a positive integer.")


def _get_mode_order(num_of_values, N):
    """Get the order by which the modes where measured"""
    N = N.copy()

    all_modes = []
    for i in range(len(N)):
        ra = list(range(sum(N[:i]), sum(N[: i + 1])))
        all_modes.append(ra * ceil(max(N) / len(ra)))

    mode_order = [i for j in zip(*all_modes) for i in j]
    mode_order *= ceil(num_of_values / len(mode_order))

    return mode_order[:num_of_values]


def reshape_samples(all_samples, modes, N):
    """Reshapes the samples dict so that they have the expected correct shape

    Reshapes the samples dict so that they the measured modes are the ones
    defined to be measured in the circuit instead of being spread over a larger
    number of modes due to the mode-shifting occuring in TDMProgram.

    Args:
        all_samples (dict[int, list]): the raw measured samples
        modes (Sequence[int]): the modes that are measured in the circuit
        N (Sequence[int]): the number of used modes per belt/spatial modes

    Returns:
        new_samples (dict[int, list]): the re-shaped samples
    """
    num_of_values = len([i for j in all_samples.values() for i in j])
    mode_order = _get_mode_order(num_of_values, N)

    # go backwards through all_samples and add them into the correct mode
    new_samples = dict()
    for i, m in enumerate(mode_order):
        idx = modes[i % len(N)]
        if idx not in new_samples:
            new_samples[idx] = []
        new_samples[idx].append(all_samples[m].pop(0))
    return new_samples


class TDMProgram(sf.Program):
    """Represents a photonic quantum circuit in the time domain encoding"""

    def __init__(self, N, name=None):
        # N is a list with the number of qumodes for each spatial mode
        if isinstance(N, int):
            self.N = [N]
        else:
            self.N = N
        self.concurr_modes = sum(self.N)

        self.total_timebins = 0
        self.spatial_modes = 0
        self.measured_modes = []
        super().__init__(num_subsystems=self.concurr_modes, name=name)

    # pylint: disable=arguments-differ, invalid-overridden-method
    def context(self, *args, copies=1, shift="default"):
        input_check(args, copies)
        self.copies = copies
        self.tdm_params = args
        self.shift = shift
        self.loop_vars = self.params(*[f"p{i}" for i in range(len(args))])
        return self

    def __enter__(self):
        super().__enter__()
        return self.loop_vars, self.register

    def __exit__(self, ex_type, ex_value, ex_tb):
        super().__exit__(ex_type, ex_value, ex_tb)
        self.spatial_modes = validate_measurements(self.circuit, self.N)
        self.construct_circuit()

    def construct_circuit(self):
        """Construct program with the register shift"""
        cmds = self.circuit.copy()
        self.circuit = []

        self.timebins = len(self.tdm_params[0])
        self.total_timebins = self.timebins * self.copies

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

        for cmd in cmds:
            if isinstance(cmd.op, ops.Measurement):
                self.measured_modes.append(cmd.reg[0].ind)

        for i in range(self.total_timebins):
            for cmd in cmds:
                self.apply_op(cmd, q, i)

            if self.shift == "default":
                # shift each spatial mode SEPARATELY by one step
                q_aux = list(q)
                # Here the index used to be i in changed to j, check this is fine.
                for j in range(len(self.N)):
                    q_aux[sm[j]] = shift_by(q_aux[sm[j]], 1)
                q = tuple(q_aux)

            elif isinstance(self.shift, int):
                q = shift_by(q, self.shift)  # shift at end of each time bin

    def apply_op(self, cmd, q, t):
        """Apply a particular operation on register q at timestep t"""
        params = cmd.op.p.copy()

        if par_is_symbolic(params[0]):
            arg_index = int(params[0].name[1:])
            params[0] = self.tdm_params[arg_index][t % self.timebins]

        self.append(cmd.op.__class__(*params), get_modes(cmd, q))

    def __str__(self):
        s = (
            f"<TDMProgram: concurrent modes={self.concurr_modes}, "
            f"time bins={self.total_timebins}, "
            f"spatial modes={self.spatial_modes}>"
        )
        return s

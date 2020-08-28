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
import numpy as np
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


def validate_measurements(circuit, shift, N):
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

    # if shift == "end" and list(set(measurement_reg)) != list(range(spatial_modes)):
    #     raise ValueError("Measurements must be on consecutive modes starting from q[0].")

    # if shift == "after" and list(set(measurement_reg)) != [0]:
    #     raise ValueError("Measurements allowed only on q[0].")

    #####################################################################################
    ##### THIS CHANGE ACCOUNTS FOR NEW PROPOSAL: ONE BELT PER SPATIAL MODE ##############
    #####################################################################################
    # N is now a list with number of qumodes for each spatial mode
    if spatial_modes is not len(N):
        raise ValueError("Numer of measurements operators must match number of spatial_modes.")

    return spatial_modes


def input_check(args, copies):
    """Checks the input arguments have consistent dimensions"""
    # check if all lists of equal length
    param_lengths = [len(param) for param in args]
    if len(set(param_lengths)) != 1:
        raise ValueError("Gate-parameter lists must be of equal length.")

    # check copies
    if not isinstance(copies, int) or copies < 1:
        raise TypeError("Number of copies must be a positive integer.")


def reshape_samples(all_samples, modes):
    """Reshapes the samples dict so that they have the expected correct shape"""
    num_of_samples = len([i for j in all_samples.values() for i in j])
    new_samples = dict()
    shifted_modes = modes.copy()

    # walk through all_samples and insert each value in the correct spot in new_samples
    for _ in range(num_of_samples//len(modes)):
        for i, m in enumerate(modes):
            if m not in new_samples:
                new_samples[m] = []
            new_samples[m].append(all_samples[shifted_modes[i]].pop(0))
        # shift the modes one step to the left and repeat
        shifted_modes = [(m+1) % len(all_samples) for m in shifted_modes]
    return new_samples



class TDMProgram(sf.Program):
    """Represents a photonic quantum circuit in the time domain encoding"""

    def __init__(self, N, name=None):
        #####################################################################################
        ##### THIS CHANGE ACCOUNTS FOR NEW PROPOSAL: ONE BELT PER SPATIAL MODE ##############
        #####################################################################################
        # N is now a list with number of qumodes for each spatial mode
        if isinstance(N, int):
            self.N = [N]
        else:
            self.N = N
        self.concurr_modes = sum(self.N)

        self.total_timebins = 0
        self.spatial_modes = 0
        self.measured_modes = []
        super().__init__(num_subsystems=self.concurr_modes, name=name)

    def context(self, *args, copies=1, shift="end"):
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
        self.spatial_modes = validate_measurements(self.circuit, self.shift, self.N)
        self.construct_circuit()

    def construct_circuit(self):
        """Construct program with the register shift"""
        cmds = self.circuit.copy()
        self.circuit = []

        self.timebins = len(self.tdm_params[0])
        self.total_timebins = self.timebins * self.copies

        q = self.register


        #####################################################################################
        ##### THIS CHANGE ACCOUNTS FOR NEW PROPOSAL: ONE BELT PER SPATIAL MODE ##############
        #####################################################################################
        # N is now a list with number of qumodes for each spatial mode

        sm = []
        for i in range(len(self.N)):
            start = sum(self.N[:i])
            stop = sum(self.N[:i])+self.N[i]
            sm.append(slice(start, stop))
        ''' ^ list of slice intervals; each slice interval corresponding to one spatial mode

        For instance, say
        N = [5,4,9,1],
        then this corresponds to
        5 concurrent modes in spatial mode A
        4 concurrent modes in spatial mode B
        9 concurrent modes in spatial mode C
        1 concurrent modes in spatial mode D.

        The entries of sm will then be:
        slice(0, 6, None)   for spatial mode A
        slice(6, 10, None)  for spatial mode B
        slice(10, 19, None) for spatial mode C
        slice(19, 20, None) for spatial mode D.

        This helps to define:
        q[sm[0]] as concurrent modes of spatial mode A
        q[sm[1]] as concurrent modes of spatial mode B
        q[sm[2]] as concurrent modes of spatial mode C
        q[sm[3]] as concurrent modes of spatial mode D.

        '''
        for cmd in cmds:
            if isinstance(cmd.op, ops.Measurement):
                self.measured_modes.append(cmd.reg[0].ind)

        for i in range(self.total_timebins):
            for cmd in cmds:
                self.apply_op(cmd, q, i)

                if self.shift == "after" and isinstance(cmd.op, ops.Measurement):
                    q = shift_by(q, 1)  # shift after each measurement

            if self.shift == "end":
                #####################################################################################
                ##### THIS CHANGE ACCOUNTS FOR NEW PROPOSAL: ONE BELT PER SPATIAL MODE ##############
                #####################################################################################
                # shift each spatial mode SEPARATELY by one step
                q_aux = list(q)
                for i in range(len(self.N)):
                    q_aux[sm[i]] = shift_by(q_aux[sm[i]], 1)
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

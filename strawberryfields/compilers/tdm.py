# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""General state compilers for TDM circuits."""

import re
from typing import List, Optional, Sequence

import numpy as np

import blackbird
from blackbird.listener import is_ptype

import strawberryfields.io as sio
from strawberryfields.ops import BSgate, LossChannel, MeasureFock, Rgate, Operation, Sgate
from strawberryfields.compilers.compiler import Compiler
from strawberryfields.program_utils import CircuitError, Command
from strawberryfields.parameters import FreeParameter
from strawberryfields.logger import create_logger

logger = create_logger(__name__)


class TDM(Compiler):
    """General compiler for Time-Domain Multiplexing (TDM) circuits."""

    short_name = "TDM"
    interactive: bool = False

    _layout = None
    _graph = None

    primitives: set = {
        "Sgate",
        "Rgate",
        "BSgate",
        "MeasureFock",
        "MeasureHomodyne",
    }

    decompositions: dict = {"S2gate": {}}

    def compile(self, seq, registers):
        """TDM-specific circuit compilation method.

        Checks that the compilers has access to the program's circuit layout and, if so,
        compiles the program using the base compiler.

        Args:
            seq (Sequence[Command]): quantum circuit to modify
            registers (Sequence[RegRefs]): quantum registers

        Returns:
            List[Command]: modified circuit

        Raises:
            CircuitError: if the given circuit hasn't been initialized for this compiler
        """
        if not self.circuit:
            raise CircuitError("TDM programs cannot be compiled without a valid circuit layout.")

        return super().compile(seq, registers)


class TD2(TDM):
    """Compiler for 2-loop time-domain circuits with homodyne measurements."""

    short_name = "TD2"
    interactive: bool = False

    _layout = None
    _graph = None

    primitives: set = {
        "Sgate",
        "Rgate",
        "BSgate",
        "MeasureHomodyne",
    }

    decompositions: dict = {"S2gate": {}}


class Borealis(TDM):
    """Compiler for 3-loop time-domain circuits with Fock measurements."""

    short_name = "borealis"
    interactive: bool = False

    _layout = None
    _graph = None

    primitives: set = {
        "Sgate",
        "Rgate",
        "BSgate",
        "MeasureFock",
    }

    decompositions: dict = {}

    # Borealis specific constants
    delays = [1, 6, 36]
    """delay applied by each loop in time bins"""

    phi_range = [-np.pi / 2, np.pi / 2]
    """range of phase shift accessible to the Borealis phase modulators"""

    def _is_loop_offset(self, op: Operation) -> bool:
        """Return whether an operation is a loop offset rotation gate."""
        return (
            isinstance(op, Rgate)
            and isinstance(op.p[0], FreeParameter)
            and not is_ptype(op.p[0].name)
            and "loop" in str(op.p[0])
        )

    def compile(self, seq, registers):
        """Compiles the Borealis circuit by inserting missing phase gates due to loop offsets.

        Args:
            seq (Sequence[Command]): quantum circuit to modify
            registers (Sequence[RegRefs]): quantum registers

        Returns:
            List[Command]: modified circuit

        Raises:
            CircuitError: the given circuit cannot be validated to belong to this circuit class
        """
        # keep track of whether a loop offset has been set by the user (True)
        # or by the compiler (False), for
        self._user_offsets: List[bool] = []

        if self.circuit:
            bb = blackbird.loads(self.circuit)
            program = sio.to_program(bb)
            circuit = program.circuit or []

            for i, cmds in enumerate(zip(circuit, seq)):
                wires_0 = {m.ind for m in cmds[0].reg}
                wires_1 = {m.ind for m in cmds[1].reg}

                ops_not_equal = type(cmds[0].op) != type(cmds[1].op) or wires_0 != wires_1

                # if the operation in the device spec is _not_ a loop offset and differs from the
                # user set value, the topology cannot be made to match the device layout by
                # just inserting loop offsets.
                if self._is_loop_offset(cmds[0].op):
                    if ops_not_equal:
                        seq.insert(i, cmds[0])
                        self._user_offsets.append(False)
                    else:
                        self._user_offsets.append(True)
                elif ops_not_equal:
                    raise CircuitError(
                        "Compilation not possible due to incompatible topologies. Expected loop "
                        f"offset gate or '{type(cmds[0].op).__name__}' on mode(s) {wires_0}, got "
                        f"'{type(cmds[1].op).__name__}' on mode(s) {wires_1}."
                    )

            seq.extend(circuit[len(seq) :])

        # pass the circuit sequence to the general TMD compiler to make sure that
        # it corresponds to the correct device layout in the specification
        return super().compile(seq, registers)

    def update_params(self, program, device):
        """Compensates for the loop offsets in the rotation gates.

        .. note::

            About 50% of the applied phase-shift parameter values for the rotation gates in each
            loop will lay beyond the range of the Borealis phase modulators, and will be offset by pi.
            This is due to hardware limitations and is unfortunately unavoidable. It is possible to
            avoid this warning by manually inserting the correct loop phase offsets obtainable via
            the device certificate:

            .. code::

                >>> eng = RemoteEngine("borealis")
                >>> eng.device.certificate["loop_phases"]
                [0.10124, -0.15121, 3.54901]

            Alternatively, the utility function :meth:`strawberryfields.tdm.utils.full_compile` or
            :meth:`strawberryfields.tdm.utils.make_phases_compatible` can be used to compile the
            gate arguments into hardware applicable ones.

        Args:
            program (.tdm.TDMProgram): TDMProgram containing the circuit and gate parameters device
            device (.Device): device containing the certificate with the loop offsets
        """
        # the three intrinsic loop phases ("offsets")
        phi_loop = device.certificate["loop_phases"]

        # retrieve which loop offsets have been set by the user (True) and which have been
        # inserted by the compiler (False); defaults to False
        user_offsets = getattr(self, "_user_offsets", [False] * len(phi_loop))

        prog_length = len(program.tdm_params[1])

        # assign initial corrections; phase corrections of loop ``loop`` will
        # depend on the corrections applied to loop ``loop - 1``
        corr_previous_loop = np.zeros(prog_length)

        for loop, offset in enumerate(phi_loop):
            if user_offsets[loop]:
                continue

            # correcting for the intrinsic phase applied by the loop; the loop
            # phase is applied once at each roundtrip where a roundtrip lasts
            # ``delay = delays[loop]`` time bins; so at time bin ``j`` the phase to be
            # corrected for has amounted to ``int(j / delay)`` times the initial
            # loop offset
            corr_loop = np.array([offset * int(j / self.delays[loop]) for j in range(prog_length)])

            # the original phase-gate arguments associated to the loop
            phi_corr = np.array(program.tdm_params[1 + 2 * loop], dtype=np.float64)

            # the offset-corrected phase angles depend on the corrections
            # applied at this loop and the corrections applied at the previous
            # loop
            phi_corr += corr_loop - corr_previous_loop

            # make sure the corrected phase-gate arguments are within the
            # interval [-pi, pi]
            phi_corr = np.mod(phi_corr, 2 * np.pi)
            phi_corr = np.where(phi_corr > np.pi, phi_corr - 2 * np.pi, phi_corr)

            # which of the corrected phase-gate arguments are beyond range of
            # the phase modulators?
            corr_low = phi_corr < self.phi_range[0]
            corr_high = phi_corr > self.phi_range[1]

            # add or subtract pi wherever the corrected phase-gate arguments are
            # beyond range of the modulators
            phi_corr = np.where(corr_low, phi_corr + np.pi, phi_corr)
            phi_corr = np.where(corr_high, phi_corr - np.pi, phi_corr)

            # warning if phase-gate arguments had to be changed to bring them
            # within range of modulators
            if loop != 0 and (any(corr_low) or any(corr_high)):
                logger.warning(
                    f"{sum(corr_low) + sum(corr_high)}/{len(phi_corr)} parameter values for the "
                    f"'Rgate' in the loop {loop} are beyond the range of the Borealis phase modulators "
                    "and have been offset by pi. See https://strawberryfields.readthedocs.io/en/"
                    "stable/code/api/strawberryfields.compilers.tdm.Borealis.html#strawberryfields."
                    "compilers.tdm.Borealis.update_params for more details. The new paramameter values "
                    "can be found in the 'Program.parameters' attribute."
                )

            # re-assign phase-gate arguments with offset-corrected values
            program.tdm_params[1 + 2 * loop] = phi_corr.tolist()

            # re-assing loop correction used for loop ``loop + 1``
            corr_previous_loop = corr_loop

        self._replace_loop_offset_params(program, device, user_offsets)

    def add_loss(self, program, device):
        """Adds realistic Borealis loss to circuit."""
        eta_glob = device.certificate["common_efficiency"]
        etas_loop = device.certificate["loop_efficiencies"]
        etas_ch_rel = device.certificate["relative_channel_efficiencies"]

        prog_length = len(program.tdm_params[0])
        reps = int(np.ceil(prog_length / 16))
        etas_ch_rel = np.tile(etas_ch_rel, reps)[:prog_length]

        program.tdm_params.append(etas_ch_rel)
        program.locked = False

        seq = []

        loop = 0
        for i, s in enumerate(program.circuit):
            # apply loss before MeasureFock
            if isinstance(s.op, MeasureFock):
                program.loop_vars.append(program.params(f"p{len(program.tdm_params) - 1}"))
                seq.append(Command(LossChannel(program.loop_vars[-1]), program.circuit[i].reg))

            seq.append(s)

            # apply loss after Sgate and BSgate
            if isinstance(s.op, Sgate):
                seq.append(Command(LossChannel(eta_glob), program.circuit[i].reg))
            if isinstance(s.op, BSgate):
                seq.append(Command(LossChannel(etas_loop[loop]), program.circuit[i].reg[1]))
                loop += 1

        program.circuit = seq
        program.lock()

    def _replace_loop_offset_params(self, program, device, user_offsets):
        """Replaces free parameters in loop offset gates with correct value."""
        for cmd in program.circuit:
            if self._is_loop_offset(cmd.op):
                # check for single integer value embedded in loop offset parameter name,
                # should be of form `loop2_phase`, and extract it (e.g., 2 in example)
                re_loop = re.findall(r"\d+", cmd.op.p[0].name)
                if len(re_loop) != 1 or (len(re_loop) == 1 and not re_loop[0].isdigit()):
                    raise ValueError(
                        "Invalid loop offset parameter name. "
                        f"Expected single integer as part of name, got '{cmd.op.p[0].name}'"
                    )
                loop = int(re_loop[0])
                if not user_offsets[loop]:
                    cmd.op.p[0] = device.certificate["loop_phases"][loop]

# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This package provides utility functions for building and running TDM programs"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import blackbird as bb
import numpy as np

from strawberryfields.logger import create_logger

logger = create_logger(__name__)

pi = np.pi


def get_mode_indices(delays: List[int]) -> Tuple[np.ndarray, int]:
    """Calculates the mode indices for use in a ``TDMProgram``.

    Args:
        delays (list[int]): List of loop delays. E.g. ``delays = [1, 6, 36]`` for Borealis.

    Returns:
        tuple(ndarray, int): the mode indices and number of concurrent (or 'alive')
        modes for the program
    """
    cum_sums = np.cumsum([1] + delays)
    N = sum(delays) + 1
    return N - cum_sums, N


def move_vac_modes(samples: np.ndarray, N: Union[int, List[int]], crop: bool = False) -> np.ndarray:
    """Moves all measured vacuum modes from the first shot of the
    returned TDM samples array to the end of the last shot.

    Args:
        samples (array[float]): samples as received from ``TDMProgram``, with the
            measured vacuum modes in the first shot
        N (int or List[int]): If an integer, the number of concurrent (or 'alive')
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


def random_r(length: int, low: float = 0, high: float = 2 * pi) -> List[float]:
    """Creates a list of random rotation-gate arguments for GBS jobs.

    Args:
        length (int): number of computational modes (squeezed-light pulses)
        low (float): lower bound for random phase-gate arguments
        high (float): upper bound for random phase-gate arguments

    Returns:
        list[float]: list of random ``Rgate`` arguments for a GBS job
    """
    return np.random.uniform(low=low, high=high, size=length).tolist()


def random_bs(length: int, low: float = 0.45, high: float = 0.55) -> List[float]:
    """Creates a list of random beamsplitter arguments for GBS jobs.

    Args:
        length (int): number of computational modes (squeezed-light pulses)
        low (float): lower bound for random beamsplitter transmission values
        high (float): upper bound for random beamsplitter transmission values

    Returns:
        list[float]: list of random ``BSgate`` arguments for a GBS job
    """
    T = np.random.uniform(low=low, high=high, size=length)
    return np.arccos(np.sqrt(T)).tolist()


def to_args_list(
    gate_dict: Dict[str, Any], device: Optional["sf.Device"] = None  # type: ignore
) -> List[List[float]]:
    """Takes the gate arguments as a dictionary and returns them as a list.

    The returned list will be compatible with circuits conforming to the circuit layout
    in the provided device specification, and can be passed to the context when creating
    the corresponding ``TDMProgram``.

    .. note::

        If no device is passed, this function will simply flatten the loops in the ``gate_dict``.
        If a device is passed, the gate order in the returned list will be validated against the
        gate order in the device layout. Any gates with free parameters containing the substring
        "loop" will be skipped.

    Args:
        gate_args (dict): dictionary with the collected arguments for squeezing gate,
            phase gates and beamsplitter gates
        device (sf.Device): the Borealis device containing the supported squeezing parameters

    Returns:
        list: list with the collected arguments for squeezing gate, phase gates
        and beamsplitter gates
    """
    circuit = bb.loads(device.layout) if device else None

    gate_list = []
    if circuit:
        # if a device is passed, assert that the device layout and gate_dict are compatible
        if circuit.operations[0]["op"] == "Sgate":
            gate_list.append(gate_dict["Sgate"])
        else:
            raise ValueError(
                "Incompatible device layout. "
                f"Expected '{circuit.operations[0]['op']}', got 'Sgate'."
            )
    else:
        # else flatten the gates prior to the loops
        for key in gate_dict.keys():
            if key == "loops":
                break
            if key != "crop":
                gate_list.append(gate_dict[key])

    # iterate through the loops and assert that the device layout and gate_dict are compatible,
    # if no device is passed, simply flatten the loops
    idx = 1
    loops = sorted(gate_dict["loops"].items())
    for _, vals in loops:
        for gate, arg in vals.items():
            if circuit:
                if "loop" in str(circuit.operations[idx]["args"][0]):
                    idx += 1
                if circuit.operations[idx]["op"] == gate:
                    gate_list.append(arg)
                else:
                    raise ValueError(
                        "Incompatible device layout. "
                        f"Expected '{circuit.operations[idx]['op']}', got '{gate}'."
                    )
                idx += 1
            else:
                gate_list.append(arg)

    # if no device is passed, flatten the gates following the loops
    if not circuit:
        tmp_list = []
        for key in reversed(list(gate_dict)):
            if key == "loops":
                break
            if key != "crop":
                tmp_list.append(gate_dict[key])
        gate_list.extend(reversed(tmp_list))

    return gate_list


def to_args_dict(gate_list: List[List[float]], device: "sf.Device") -> Dict[str, Any]:  # type: ignore
    """Takes the gate arguments as a list and returns them as a dictionary.

    The returned dictionary will be compatible with circuits conforming to the circuit layout
    in the provided device specification, and can be passed to the helper functions in the TDM
    utils module.

    This function assumes that the loop gates are ordered in pairs of `Rgate` and `BSgate`
    (ignoring loop offsets and measurements), and that it begins with an `Sgate`:

    .. code-block::

        Sgate({s}, 0.0) | 43
        Rgate({r0}) | 43
        BSgate({bs0}, 1.571) | [42, 43]
        Rgate({loop0_phase}) | 43  # offset
        Rgate({r1}) | 42
        BSgate({bs1}, 1.571) | [36, 42]
        Rgate({loop1_phase}) | 42  # offset
        Rgate({r2}) | 36
        BSgate({bs2}, 1.571) | [0, 36]
        Rgate({loop2_phase}) | 36  # offset
        MeasureFock() | 0

    Args:
        gate_list (list): list with the collected arguments for squeezing gate,
            phase gates and beamsplitter gates
        device (sf.Device): the Borealis device containing the supported squeezing parameters

    Raises:
        ValueError: if the device specification layout doesn't agree with the above layout, or if
            the passed gate arguments list contains too few entries

    Returns:
        dict: dictionary with the collected arguments for squeezing gate, phase
        gates and beamsplitter gates
    """
    gates_dict: Dict[str, Any] = {"loops": {}}
    circuit = bb.loads(device.layout)

    circuit_idx = arglist_idx = loop = 0
    while circuit_idx < len(circuit.operations):
        op = circuit.operations[circuit_idx]["op"]
        args = circuit.operations[circuit_idx]["args"]
        is_loop_offset = op == "Rgate" and "loop" in str(args[0])

        if not is_loop_offset and "Measure" not in op:
            # make sure that `gate_list` contains the next value array
            try:
                gate_args = gate_list[arglist_idx]
                gate_args_next = gate_list[arglist_idx + 1]
            except IndexError:
                raise ValueError(f"List of gate argument arrays is incomplete.")

            # the first operation should be an Sgate (outside of the loops)
            if circuit_idx == 0:
                if op == "Sgate":
                    gates_dict["Sgate"] = gate_args
                else:
                    raise ValueError(f"Incompatible device layout. Expected 'Sgate', got '{op}'.")
            elif op == "Rgate" and circuit.operations[circuit_idx + 1]["op"] == "BSgate":
                # the rest of the gates should be Rgate-BSgate pairs in the loops
                gates_dict["loops"][loop] = {}
                gates_dict["loops"][loop]["Rgate"] = gate_args
                gates_dict["loops"][loop]["BSgate"] = gate_args_next
                loop += 1
                circuit_idx += 1
            else:
                raise ValueError(
                    "Incompatible device layout. Expected an 'Rgate' - 'BSgate' pair "
                    f"in each loop, got extra '{op}'."
                )

        circuit_idx += 1
        arglist_idx += 1

    return gates_dict


def loop_phase_from_device(device: "sf.Device") -> List[float]:  # type: ignore
    """Extracts the three loop-phase offsets from the device specification.

    Args:
        device (sf.Device): the device containing the loop offset phases

    Return:
       list[float]: list containing the loop offset phases
    """
    return device.certificate["loop_phases"]


def vacuum_padding(gate_args: Dict[str, Any], delays: List[int] = [1, 6, 36]) -> Dict[str, Any]:
    """Pad the gate arguments with 0 for vacuum modes.

    Adds zeros to the beginning and end of the gate-argument lists in order to make sure:
    - the non-trivial gates start punctual with the arrival of the first (possibly delayed) pulse
    - all loops are emptied from optical pulses at the end of the program
    - all gate lists have the same number of elements

    Args:
        gate_args (dict): dictionary with the collected arguments for squeezing gate,
            phase gates and beamsplitter gates
        delays (list, optional): The delay applied by each loop in time bins. Defaults
            to ``[1, 6, 36]``.

    Returns:
        dict: the input dictionary with the lists of phase-gate and beamsplitter arguments
        sandwiched with appropriate amounts of zeros
    """
    gate_args_out = deepcopy(gate_args)
    arrival_time = 0
    prologue_bins = []

    def start_zeros(alpha):
        for idx, val in enumerate(alpha):
            if val != 0:
                return idx
        return len(alpha)

    for loop, max_delay in zip(sorted(gate_args["loops"]), delays):

        # number of time bins before the first pulse arrives at loop i
        prologue_bins.append(arrival_time)

        # number of cross time bins (open loop) at the beginning of the
        # sequence, counting from the arrival time of the first computational
        # mode
        alpha = gate_args["loops"][loop]["BSgate"]

        # number of zeros at the beginning of the BS-argument list
        initial_zeros = start_zeros(alpha)

        if not initial_zeros == len(alpha):
            # delay imposed by loop corresponds to `initial_zeros`, but is
            # upper-bounded by `max_delay`
            delay_imposed_by_loop = min(start_zeros(alpha), max_delay)
        else:
            # if the list is all zero (cross-state) the delay imposed by the
            # loop is always the maximum delay
            delay_imposed_by_loop = max_delay

        # delay imposed by loop i adds to the arrival time of first comp. mode
        # at loop i+1 (or detector)
        arrival_time += delay_imposed_by_loop

    epilogue_bins = [arrival_time - prologue_bins_i for prologue_bins_i in prologue_bins]

    if isinstance(gate_args["Sgate"], list):
        gate_args_out["Sgate"] = (
            [0] * prologue_bins[0] + gate_args["Sgate"] + [0] * epilogue_bins[0]
        )

    for loop in sorted(gate_args["loops"]):
        gate_args_out["loops"][loop]["Rgate"] = (
            [0] * prologue_bins[loop]
            + gate_args["loops"][loop]["Rgate"]
            + [0] * epilogue_bins[loop]
        )
        gate_args_out["loops"][loop]["BSgate"] = (
            [0] * prologue_bins[loop]
            + gate_args["loops"][loop]["BSgate"]
            + [0] * epilogue_bins[loop]
        )

    gate_args_out["crop"] = arrival_time

    return gate_args_out


def make_squeezing_compatible(gate_args: Dict[str, Any], device: "sf.Device") -> Dict[str, Any]:  # type: ignore
    """Matches the user-defined squeezing values to a discrete set of squeezing
    levels supported by the device.

    Args:
        gate_args (dict): dictionary with the collected arguments for squeezing gate,
            phase gates and beamsplitter gates
        device (sf.Device): the device containing the supported squeezing parameters

    Returns:
        dict: the input dictionary with the ``gate_args["Sgate"]`` replaced by hardware-compatible
        squeezing values
    """
    gate_args_out = deepcopy(gate_args)

    user_values = gate_args["Sgate"]
    prog_length = len(gate_args["loops"][0]["Rgate"])
    crop = gate_args.get("crop", 0)

    allowed_values = device.certificate["squeezing_parameters_mean"]
    allowed_values["zero"] = 0

    if isinstance(user_values, list):
        # calculate the number of squeezed computational modes there are
        num_final_zeros = next(
            (index for index, value in enumerate(user_values[::-1]) if value != 0),
            len(user_values[::-1]),
        )
        comp_modes = len(user_values) - num_final_zeros

        # make sure that comp_modes is not larger than the upper bound implied by `prog_length`
        # and `crop`, and correct the user values such that `comp_modes == prog_length - crop`
        if comp_modes > prog_length - crop:
            comp_modes = prog_length - crop
            user_values[comp_modes:] = [0] * len(user_values[comp_modes:])

        # set to the median value of the non-zero squeezing values
        gate_args_out["Sgate"] = np.append(
            np.ones_like(user_values[:comp_modes]) * np.median(user_values[:comp_modes]),
            user_values[comp_modes:],
        )

        # NOTE: update if/when different squeezing values on different modes is allowed
        allowed_values = np.array(list(allowed_values.values()))

        closest_idx = np.argmin(np.abs(allowed_values - gate_args_out["Sgate"][0]))
        gate_args_out["Sgate"][:comp_modes] = allowed_values[closest_idx]

        if not np.allclose(gate_args["Sgate"], gate_args_out["Sgate"]):
            allowed_value_str = ["{:.3f}".format(av) for av in sorted(allowed_values)]

            logger.warning(
                "Submitted squeezing values have been matched to the closest "
                f"median value supported by hardware: {allowed_value_str}."
            )

        gate_args_out["Sgate"] = gate_args_out["Sgate"].tolist()

    elif isinstance(user_values, str) and user_values in [
        "zero",
        "low",
        "medium",
        "high",
    ]:
        prog_length = len(gate_args["loops"][0]["Rgate"])
        crop = gate_args.get("crop", 0)

        r0 = allowed_values.get(user_values)
        gate_args_out["Sgate"] = [r0] * (prog_length - crop) + [0] * crop
    else:
        raise TypeError(
            f"Invalid squeezing type; must be either a list of squeezing "
            "values or one of the following strings: 'zero', 'low', 'medium' or 'high'."
        )

    return gate_args_out


def make_phases_compatible(
    gate_args: Dict[str, Any],
    device: "sf.Device",  # type: ignore
    delays: List[int] = [1, 6, 36],
    phi_range: List[float] = [-pi / 2, pi / 2],
) -> Dict[str, Any]:
    """Adds a pi offset to the phase-gate arguments that cannot be applied by the Borealis modulators.

    Args:
        gate_args (dict): dictionary with the collected arguments for squeezing gate,
            phase gates and beamsplitter gates
        device (sf.Device): the Borealis device containing the supported squeezing parameters
        delays (list, optional): the delay applied by each loop in time bins. Defaults to [1, 6, 36].
        phi_range (list, optional): The range of phase shift accessible to the Borealis phase modulators.
            Defaults to [-pi / 2, pi / 2].

    Returns:
        dict: the input dictionary with the lists of phase-gate arguments replaced by hardware-
        compatible phase arguments
    """
    gate_args_out = deepcopy(gate_args)
    phi_loop = loop_phase_from_device(device)

    prog_length = len(gate_args["loops"][0]["Rgate"])

    corr_previous_loop = np.zeros(prog_length)
    for loop in sorted(gate_args["loops"]):
        corr_loop = np.array([phi_loop[loop] * int(j / delays[loop]) for j in range(prog_length)])

        # loop 0 is always compensatable because the phases are invariant over
        # pi shifts (unlike loops 1 and 2)
        if loop != 0:
            rgate_args = np.array(gate_args["loops"][loop]["Rgate"])
            phi_corr = (rgate_args + corr_loop - corr_previous_loop) % (2 * pi)  # type: ignore
            phi_corr = np.where(phi_corr > pi, phi_corr - 2 * pi, phi_corr)
            correction_indices = np.where((phi_corr < phi_range[0]) | (phi_corr > phi_range[1]))[0]

            for j in correction_indices:
                gate_args_out["loops"][loop]["Rgate"][j] = (
                    gate_args_out["loops"][loop]["Rgate"][j] + pi
                ) % (2 * pi)

            if len(correction_indices) > 0:
                logger.warning(
                    f"{len(correction_indices)}/{len(rgate_args)} arguments of phase gate {loop} "
                    "have been shifted by pi in order to be compatible with the phase modulators."
                )

        corr_previous_loop = corr_loop

    return gate_args_out


def full_compile(
    gate_args: Dict[str, Any],
    device: "sf.Device",  # type: ignore
    delays: List[int] = [1, 6, 36],
    phi_range: List[float] = [-pi / 2, pi / 2],
    return_list: bool = True,
) -> Union[Dict[str, Any], List[List[float]]]:
    """Makes a user-defined set of gate arguments compatible with ``TDMProgram`` and
    the Borealis hardware.

    Args:
        gate_args (dict): dictionary with the collected arguments for squeezing gate,
            phase gates and beamsplitter gates
        device (sf.Device): the Borealis device containing the supported squeezing parameters
        delays (list, optional): The delay applied by each loop in time bins.
            Defaults to [1, 6, 36].
        phi_range (list, optional): The range of phase shift accessible to the Borealis phase modulators.
            Defaults to [-pi / 2, pi / 2].

    Returns:
        dict: the input dictionary in which the gate-argument lists are properly padded
        with vacuum time bins, compatible with the allowed squeezing values and with the
        range of the Borealis phase modulators
    """

    gate_args_comp = deepcopy(gate_args)
    gate_args_pad = vacuum_padding(gate_args_comp, delays=delays)
    gate_args_pad_sq = make_squeezing_compatible(gate_args_pad, device)
    gate_args_pad_sq_phase = make_phases_compatible(
        gate_args_pad_sq, device, delays=delays, phi_range=phi_range
    )

    # check that number of temporal modes (i.e., length of parameter arrays) ar less than max
    modes = len(gate_args_pad_sq_phase["loops"][0]["Rgate"])
    if modes > device.modes["temporal_max"]:
        raise ValueError(
            f"{modes} modes used; device '{device.target}' supports up to "
            f"{device.modes['temporal_max']} modes."
        )

    if return_list:
        return to_args_list(gate_args_pad_sq_phase, device)
    return gate_args_pad_sq_phase


def borealis_gbs(
    device: "sf.Device",  # type: ignore
    modes: int = 216,
    squeezing: str = "medium",
    open_loops: List[bool] = [True, True, True],
    return_list: bool = True,
) -> Union[Dict[str, Any], List[List[float]]]:
    """Creates a GBS instance in form of a ``gates_args`` dictionary that can be directly
    submitted to a ``TDMProgram``.

    Args:
        device (sf.Device): the Borealis device containing the supported squeezing parameters
        modes (int, optional): The number of computational modes in the GBS instance.
            Defaults to 216.
        squeezing (str, optional): The squeezing level. Defaults to "medium".
        open_loops (list, optional): List of booleans to include the first, second and third
            delay line. Defaults to [True, True, True]
        return_list (bool, optional): return a list of gate-arguments instead of a dictionary

    Returns:
        dict: dictionary of gate arguments that has undergone full compilation and is therefore
        ready to be submitted to the Borealis hardware
    """
    delays = [1, 6, 36]

    def _get_angles(loop, include):
        if include:
            phi = random_r(modes)
            alpha = random_bs(modes)
            alpha[: delays[loop]] = [0] * min(delays[loop], len(alpha))
            return phi, alpha

        return [0] * modes, [pi / 2] * modes

    gate_args = {"Sgate": squeezing, "loops": {0: {}, 1: {}, 2: {}}}
    for loop, include in enumerate(open_loops):
        phi, alpha = _get_angles(loop, include)
        gate_args["loops"][loop]["Rgate"] = phi
        gate_args["loops"][loop]["BSgate"] = alpha

    return full_compile(gate_args, device, return_list=return_list)

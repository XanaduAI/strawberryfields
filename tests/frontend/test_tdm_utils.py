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
r"""Unit tests for tdm/program.py"""
import copy
import inspect

import numpy as np
import pytest
from thewalrus.quantum import photon_number_covmat, photon_number_mean_vector

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.backends.states import BaseGaussianState
from strawberryfields.device import Device
from strawberryfields.tdm import (borealis_gbs, get_mode_indices,
                                  loop_phase_from_device,
                                  make_phases_compatible,
                                  make_squeezing_compatible, random_bs,
                                  random_r, to_args_dict, to_args_list,
                                  vacuum_padding)
from strawberryfields.tdm.utils import full_compile, to_args_list

pi = np.pi

pytestmark = pytest.mark.frontend

# make test deterministic
np.random.seed(42)


def get_layout():
    """Returns the Borealis layout as required for the device spec

    Returns:
        str: the Borealis layout
    """
    return inspect.cleandoc(
        """
        name template_borealis
        version 1.0
        target borealis (shots=100000)
        type tdm (temporal_modes=259, copies=1)

        float array p0[1, 259] =
            {s}
        float array p1[1, 259] =
            {r0}
        float array p2[1, 259] =
            {bs0}
        float array p3[1, 259] =
            {loop0_phase}
        float array p4[1, 259] =
            {r1}
        float array p5[1, 259] =
            {bs1}
        float array p6[1, 259] =
            {loop1_phase}
        float array p7[1, 259] =
            {r2}
        float array p8[1, 259] =
            {bs2}
        float array p9[1, 259] =
            {loop2_phase}


        Sgate({s}, 0.0) | 43
        Rgate({r0}) | 43
        BSgate({bs0}, 1.5707963267948966) | [42, 43]
        Rgate({loop0_phase}) | 43
        Rgate({r1}) | 42
        BSgate({bs1}, 1.5707963267948966) | [36, 42]
        Rgate({loop1_phase}) | 42
        Rgate({r2}) | 36
        BSgate({bs2}, 1.5707963267948966) | [0, 36]
        Rgate({loop2_phase}) | 36
        MeasureFock() | 0
        """
    )


def get_device_spec():
    """Returns the Borealis device spec

    Returns:
        dict: the Borealis device spec
    """
    return {
        "target": "borealis",
        "expected_uptime": {},
        "layout": get_layout(),
        "modes": {"temporal_max": 331, "concurrent": 44, "spatial": 1},
        "supported_languages": ["blackbird:1.0"],
        "compiler": ["borealis"],
        "compiler_default": "borealis",
        "gate_parameters": {
            "s": [0, 0.712, 1.019, 1.212],
            "r0": [[-1.5707963267948966, 1.5707963267948966]],
            "bs0": [[0, 1.5707963267948966]],
            "loop0_phase": [0.478],
            "r1": [[-1.5707963267948966, 1.5707963267948966]],
            "bs1": [[0, 1.5707963267948966]],
            "loop1_phase": [1.337],
            "r2": [[-1.5707963267948966, 1.5707963267948966]],
            "bs2": [[0, 1.5707963267948966]],
            "loop2_phase": [0.112],
        },
    }


def get_device_cert():
    """Returns a dummy Borealis certificate

    Returns:
        dict: a dummy Borealis certificate
    """
    return {
        "target": "borealis",
        "finished_at": "2022-02-03T15:00:59.641616+00:00",
        "loop_phases": [0.478, 1.337, 0.112],
        "schmidt_number": 1.333,
        "common_efficiency": 0.551,
        "loop_efficiencies": [0.941, 0.914, 0.876],
        "squeezing_parameters_mean": {"low": 0.712, "high": 1.019, "medium": 1.212},
        "relative_channel_efficiencies": [
            0.976,
            0.934,
            0.967,
            1,
            0.921,
            0.853,
            0.919,
            0.939,
            0.877,
            0.942,
            0.919,
            0.987,
            0.953,
            0.852,
            0.952,
            0.945,
        ],
    }


def get_device():
    """Builds a dummy Borealis ``sf.Device`` from device spec and certificate

    Returns:
        obj: a dummy ``sf.Device`` object
    """
    device_spec = get_device_spec()
    device_cert = get_device_cert()
    return Device(spec=device_spec, cert=device_cert)


def are_gate_args_legal(
    gate_args,
    device,
    delays=[1, 6, 36],
    assert_arg_length=True,
    assert_squeezing=True,
    assert_phase_gates=True,
    assert_beamsplitters=True,
):
    """Validates certain keys of a ``gate_args`` dict

    Args:
        gate_args (dict): dictionary with the collected arguments for squeezing
            gate, phase gates and beamsplitter gates
        device (sf.Device): the Borealis device
        delays (list, optional): the delay applied by each loop in time bins.
            Defaults to [1, 6, 36].
        assert_arg_length (bool, optional): Should the function check that all
            gate-argument lists are of equal length? Defaults to True.
        assert_squeezing (bool, optional): Should the function check squeezing
            values? Defaults to True.
        assert_phase_gates (bool, optional): Should the function check that all
            phase-gate arguments are within allowed range? Defaults to True.
        assert_beamsplitters (bool, optional): Should the function check that
            all beamsplitter arguments are within allowed range? Defaults to True.
    """

    # are all gate lists of the same length?
    if assert_arg_length:
        args_list = to_args_list(gate_args, device)
        args_lengths = [len(gate_args) for gate_args in args_list]
        assert len(set(args_lengths)) == 1

    # are all squeezing values allowed?
    if assert_squeezing:
        r = gate_args["Sgate"]
        allowed_values = device.certificate["squeezing_parameters_mean"].values()
        allowed_sq_values_list = [0] + [*allowed_values]
        assert all([r_i in allowed_sq_values_list for r_i in r])

    # will the phase gates be hardware-compatible after loop-offset
    # compensation?
    gate_parameters = device._spec["gate_parameters"]
    if assert_phase_gates:
        # the gate arguments
        phi_0 = gate_args["loops"][0]["Rgate"]
        phi_1 = gate_args["loops"][1]["Rgate"]
        phi_2 = gate_args["loops"][2]["Rgate"]

        # the supported ranges
        phi_0_min, phi_0_max = gate_parameters["r0"][0]
        phi_1_min, phi_1_max = gate_parameters["r1"][0]
        phi_2_min, phi_2_max = gate_parameters["r2"][0]

        loop_phases = loop_phase_from_device(device)

        # get the actual phase-gate arguments applied under consideration of the
        # loop offsets
        phi_0_actual, phi_1_actual, phi_2_actual = compensate_loop_offsets(
            phi_args=[phi_0, phi_1, phi_2],
            phi_loop=loop_phases,
            delays=delays,
            phi_range=[phi_0_min, phi_0_max],
        )

        # are Rgate arguments within allowed range?
        assert all(phi_0_min <= phi <= phi_0_max for phi in phi_0_actual)
        assert all(phi_1_min <= phi <= phi_1_max for phi in phi_1_actual)
        assert all(phi_2_min <= phi <= phi_2_max for phi in phi_2_actual)

    if assert_beamsplitters:
        # the gate arguments
        alpha_0 = gate_args["loops"][0]["BSgate"]
        alpha_1 = gate_args["loops"][1]["BSgate"]
        alpha_2 = gate_args["loops"][2]["BSgate"]

        # the supported ranges
        alpha_0_min, alpha_0_max = gate_parameters["bs0"][0]
        alpha_1_min, alpha_1_max = gate_parameters["bs1"][0]
        alpha_2_min, alpha_2_max = gate_parameters["bs2"][0]

        # are BSgate arguments within allowed range?
        assert all(alpha_0_min <= alpha <= alpha_0_max for alpha in alpha_0)
        assert all(alpha_1_min <= alpha <= alpha_1_max for alpha in alpha_1)
        assert all(alpha_2_min <= alpha <= alpha_2_max for alpha in alpha_2)


def compensate_loop_offsets(phi_args, phi_loop, delays, phi_range):
    """Helper function to absorb the intrinsic loop phases into the
    programmable gate arguments. An equivalent function exists in
    ``compiler.tdm``. This one here is only used for validation purposes.

    Args:
        phi_args (list): a nested list including lists of phase-gate arguments
         -- one for each loop
        phi_loop (list): the three intrinsic loop phases
        delays (list, optional): the delay applied by each loop in time bins.
        phi_range (list): lower and upper bound of the phase range supported by
            the phase-gate modulators

    Returns:
        list: a nested list including lists of phase-gate arguments -- one for
            each loop
    """

    prog_length = len(phi_args[0])

    # assign initial corrections; phase corrections of loop ``loop`` will
    # depend on the corrections applied to loop ``loop - 1``
    corr_previous_loop = np.zeros(prog_length)

    phi_args_out = []

    for loop, offset in enumerate(phi_loop):

        # correcting for the intrinsic phase applied by the loop; the loop
        # phase is applied once at each roundtrip where a roundtrip lasts
        # ``delay = delays[loop]`` time bins; so at time bin ``j`` the phase to be
        # corrected for has amounted to ``int(j / delay)`` times the initial
        # loop offset
        corr_loop = np.array([offset * int(j / delays[loop]) for j in range(prog_length)])

        # the original phase-gate arguments associated to the loop (specify
        # ``dtype=float`` because if ``phi_args[loops]`` is an integer array it
        # is unsuited for float additions down below)
        phi_corr = np.array(phi_args[loop], dtype=float)

        # the offset-corrected phase angles depend on the corrections
        # applied at this loop and the corrections applied at the previous
        # loop
        phi_corr += corr_loop - corr_previous_loop  # type: ignore

        # make sure the corrected phase-gate arguments are within the
        # interval [-pi, pi]
        phi_corr = np.mod(phi_corr, 2 * np.pi)
        phi_corr = np.where(phi_corr > np.pi, phi_corr - 2 * np.pi, phi_corr)

        # loop 0 is always compansatable because the phases are invariant over
        # pi shifts (unlike loops 1 and 2)
        if loop == 0:
            # which of the corrected phase-gate arguments are beyond range of
            # the phase modulators?
            corr_low = phi_corr < phi_range[0]
            corr_high = phi_corr > phi_range[1]

            # add or subtract pi wherever the corrected phase-gate arguments are
            # beyond range of the modulators
            phi_corr = np.where(corr_low, phi_corr + np.pi, phi_corr)
            phi_corr = np.where(corr_high, phi_corr - np.pi, phi_corr)

        phi_args_out.append(phi_corr.tolist())

        # re-assing loop correction used for loop ``loop + 1``
        corr_previous_loop = corr_loop

    return phi_args_out


def get_random_gate_args(modes=234):
    """Returns a dictionary with random gate arguments for testing purposes

    Args:
        modes (int, optional): Number of computational modes. Defaults to 234.

    Returns:
        dict: dictionary with the collected arguments for squeezing gate,
            phase gates and beamsplitter gates
    """
    random_s = np.random.uniform(low=0, high=1, size=modes)
    if isinstance(random_s, np.ndarray):
        random_s = random_s.tolist()
    else:
        random_s = [random_s]

    return {
        "Sgate": random_s,
        "loops": {
            0: {"Rgate": random_r(length=modes), "BSgate": random_bs(length=modes)},
            1: {"Rgate": random_r(length=modes), "BSgate": random_bs(length=modes)},
            2: {"Rgate": random_r(length=modes), "BSgate": random_bs(length=modes)},
        },
    }


def get_unitary(gate_args, device, phi_loop=None, delays=[1, 6, 36]):
    """Computes the unitary corresponding to a given set of gates

    Args:
        gate_args (_type_): dictionary with the collected arguments for
            squeezing gate, phase gates and beamsplitter gates
        device (sf.Device): the Borealis device
        phi_loop (list, optional): list containing the three loop offsets.
            Defaults to None.
        delays (list, optional): the delay applied by each loop in time bins.

    Returns:
        np.ndarray: a unitary matrix
    """
    args_list = to_args_list(gate_args, device)

    n, N = get_mode_indices(delays)

    prog = sf.TDMProgram(N)
    with prog.context(*args_list) as (p, q):
        for i in range(len(delays)):
            ops.Rgate(p[2 * i + 1]) | q[n[i]]
            ops.BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i]], q[n[i + 1]])
            if phi_loop is not None:
                ops.Rgate(phi_loop[i]) | q[n[i]]
    prog.space_unroll()
    prog = prog.compile(compiler="passive")

    # unitary matrix
    assert prog.circuit
    U = prog.circuit[0].op.p[0]

    return U


def get_photon_number_moments(gate_args, device, phi_loop=None, delays=[1, 6, 36]):
    """Computes first and second moment of the photon number distribution
    obtained from a single

    Args:
        gate_args (_type_): dictionary with the collected arguments for
            squeezing gate, phase gates and beamsplitter gates
        device (sf.Device): the Borealis device
        phi_loop (list, optional): list containing the three loop offsets.
            Defaults to None.
        delays (list, optional): the delay applied by each loop in time bins.

    Returns:
        tuple: two np.ndarrays with the mean photon numbers and photon-number
            covariance matrix, respectively
    """
    args_list = to_args_list(gate_args, device)

    n, N = get_mode_indices(delays)

    prog = sf.TDMProgram(N)
    with prog.context(*args_list) as (p, q):
        ops.Sgate(p[0]) | q[n[0]]
        for i in range(len(delays)):
            ops.Rgate(p[2 * i + 1]) | q[n[i]]
            ops.BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i]], q[n[i + 1]])
            if phi_loop is not None:
                ops.Rgate(phi_loop[i]) | q[n[i]]
    prog.space_unroll()
    eng = sf.Engine("gaussian")
    results = eng.run(prog)

    assert isinstance(results.state, BaseGaussianState)

    # quadrature mean vector and covariance matrix
    mu = results.state.means()
    cov_q = results.state.cov()

    # photon-number mean vector and covariance matrix
    mean_n = photon_number_mean_vector(mu, cov_q)
    cov_n = photon_number_covmat(mu, cov_q)

    return mean_n, cov_n


def get_crop_value(gate_args, delays=[1, 6, 36]):
    """Obtains the number of vacuum modes arriving at the detector before the first
    computational mode for any job run on Borealis.

    Args:
        gate_args (dict): dictionary with the collected arguments for squeezing gate,
            phase gates and beamsplitter gates
        delays (list, optional): The delay applied by each loop in time bins.
            Defaults to [1, 6, 36].

    Returns:
        int: crop parameter value
    """

    # arrival time of first computational mode at a respective
    # loop or detector
    arrival_time = 0

    for loop, max_delay in zip(sorted(gate_args["loops"]), delays):

        # number of cross time bins (open loop) at the beginning of the
        # sequence, counting from the arrival time of the first computational
        # mode
        alpha = gate_args["loops"][loop]["BSgate"]
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


### test the Borealis helper functions in tdm.utils


def test_loop_phase_from_device():
    """Tests if loop phases are correctly obtained from ``sf.Device``"""
    device_spec = get_device_spec()
    device_cert = get_device_cert()

    loop_phases = np.random.uniform(low=0, high=2 * pi, size=3).tolist()

    device_cert["loop_phases"] = loop_phases

    device = Device(spec=device_spec, cert=device_cert)

    loop_phases_out = loop_phase_from_device(device=device)

    assert np.allclose(loop_phases, loop_phases_out)


@pytest.mark.parametrize("delays", [[1, 6, 36], [9, 8, 7]])
@pytest.mark.parametrize("init_cross", [[1, 6, 36], [0, 1, 2]])
def test_vacuum_padding(delays, init_cross):
    """Tests the front- and back-padding of the gate arrays to correctly
    account for waiting times implied by the lengths of the delay lines
    and beamsplitter settings


    Args:
        delays (list): the delay applied by each loop in time bins
        init_cross (list): the number of initial time bins in which the three
            beamsplitters are in the cross state (T=1)
    """
    modes = 123

    gate_args = get_random_gate_args(modes=modes)

    # set the initial cross states of each BS
    gate_args["loops"][0]["BSgate"][: init_cross[0]] = [0] * init_cross[0]
    gate_args["loops"][1]["BSgate"][: init_cross[1]] = [0] * init_cross[1]
    gate_args["loops"][2]["BSgate"][: init_cross[2]] = [0] * init_cross[2]

    r = gate_args["Sgate"]

    phi_0 = gate_args["loops"][0]["Rgate"]
    alpha_0 = gate_args["loops"][0]["BSgate"]

    phi_1 = gate_args["loops"][1]["Rgate"]
    alpha_1 = gate_args["loops"][1]["BSgate"]

    phi_2 = gate_args["loops"][2]["Rgate"]
    alpha_2 = gate_args["loops"][2]["BSgate"]

    # the time bins awaited by each loop before the first pulse arrives
    prologue_bins_0 = 0
    prologue_bins_1 = prologue_bins_0 + min(delays[0], init_cross[0])
    prologue_bins_2 = prologue_bins_1 + min(delays[1], init_cross[1])

    total_pad = prologue_bins_2 + min(delays[2], init_cross[2])

    # the time bins awaited by each loop after the final pulse has been coupled
    # into the loop
    epilogue_bins_0 = total_pad - prologue_bins_0
    epilogue_bins_1 = total_pad - prologue_bins_1
    epilogue_bins_2 = total_pad - prologue_bins_2

    gate_args_out = vacuum_padding(gate_args=gate_args, delays=delays)

    r_out = gate_args_out["Sgate"]

    phi_0_out = gate_args_out["loops"][0]["Rgate"]
    alpha_0_out = gate_args_out["loops"][0]["BSgate"]

    phi_1_out = gate_args_out["loops"][1]["Rgate"]
    alpha_1_out = gate_args_out["loops"][1]["BSgate"]

    phi_2_out = gate_args_out["loops"][2]["Rgate"]
    alpha_2_out = gate_args_out["loops"][2]["BSgate"]

    crop_out = gate_args_out["crop"]

    # same length for all argument lists
    are_gate_args_legal(
        gate_args=gate_args_out,
        device=get_device(),
        delays=delays,
        assert_arg_length=True,
        assert_squeezing=False,
        assert_beamsplitters=False,
        assert_phase_gates=False,
    )

    # are the squeezed modes protected from interference with vacuum? in other
    # words, is the total unitary a block-diagonal of the identity (acting
    # exclusively on vacuum) and the nontrivial unitary (acting exclusively on
    # pulses)?
    if delays == init_cross:
        U = get_unitary(gate_args_out, device=None)
        assert np.allclose(U[:crop_out, :crop_out], np.eye(crop_out))

    # number of zeros at the beginning
    assert phi_0_out[:prologue_bins_0] == [0] * prologue_bins_0
    assert alpha_0_out[:prologue_bins_0] == [0] * prologue_bins_0
    assert phi_1_out[:prologue_bins_1] == [0] * prologue_bins_1
    assert alpha_1_out[:prologue_bins_1] == [0] * prologue_bins_1
    assert phi_2_out[:prologue_bins_2] == [0] * prologue_bins_2
    assert alpha_2_out[:prologue_bins_2] == [0] * prologue_bins_2

    # nontrivial gates acting on computational modes
    assert r_out[:modes] == r
    assert phi_0_out[prologue_bins_0 : prologue_bins_0 + modes] == phi_0
    assert alpha_0_out[prologue_bins_0 : prologue_bins_0 + modes] == alpha_0
    assert phi_1_out[prologue_bins_1 : prologue_bins_1 + modes] == phi_1
    assert alpha_1_out[prologue_bins_1 : prologue_bins_1 + modes] == alpha_1
    assert phi_2_out[prologue_bins_2 : prologue_bins_2 + modes] == phi_2
    assert alpha_2_out[prologue_bins_2 : prologue_bins_2 + modes] == alpha_2

    # number of zeros at the end
    assert r_out[modes:] == [0] * total_pad
    assert phi_0_out[-epilogue_bins_0:] == [0] * epilogue_bins_0
    assert alpha_0_out[-epilogue_bins_0:] == [0] * epilogue_bins_0
    assert phi_1_out[-epilogue_bins_1:] == [0] * epilogue_bins_1
    assert alpha_1_out[-epilogue_bins_1:] == [0] * epilogue_bins_1
    assert phi_2_out[-epilogue_bins_2:] == [0] * epilogue_bins_2
    assert alpha_2_out[-epilogue_bins_2:] == [0] * epilogue_bins_2

    # crop value
    assert crop_out == total_pad


@pytest.mark.parametrize("sq_input", ["user_array", "zero", "low", "medium", "high"])
def test_make_squeezing_compatible(sq_input):
    """Tests whether squeezing values are correctly matched to the ones allowed
    by the hardware

    Args:
        sq_input (str): specifies the user input for the squeezing
    """

    # the allowed squeezing values
    sq_zero = 0
    sq_low = np.random.uniform(low=0.100, high=0.399)
    sq_medium = np.random.uniform(low=0.400, high=0.699)
    sq_high = np.random.uniform(low=0.700, high=1.000)

    modes = 300
    gate_args = get_random_gate_args(modes=modes)

    # replace Sgate parameter array with string if user input is string
    if sq_input != "user_array":
        gate_args["Sgate"] = sq_input

    device_spec = get_device_spec()
    device_cert = get_device_cert()

    # update the ``device_cert`` with the random values obtained above
    device_cert["squeezing_parameters_mean"]["low"] = sq_low
    device_cert["squeezing_parameters_mean"]["medium"] = sq_medium
    device_cert["squeezing_parameters_mean"]["high"] = sq_high

    device = Device(spec=device_spec, cert=device_cert)

    gate_args_out = make_squeezing_compatible(gate_args=gate_args, device=device)
    r_out = gate_args_out["Sgate"]

    # are the new squeezing values hardware-compatible?
    are_gate_args_legal(
        gate_args=gate_args_out,
        device=device,
        delays=None,
        assert_arg_length=False,
        assert_squeezing=True,
        assert_beamsplitters=False,
        assert_phase_gates=False,
    )

    # if a global squeezing level has been defined (instead of a squeezing
    # array), have the values been assigned correctly?
    if sq_input != "user_array":
        allowed_values_dict = {
            "zero": sq_zero,
            "low": sq_low,
            "medium": sq_medium,
            "high": sq_high,
        }
        assert r_out == [allowed_values_dict[sq_input]] * modes


def test_make_phases_compatible():
    """Tests if user-defined phases have been successfully changed such that
    they are hardware-compatible _after_ loop-offset compensation
    """

    modes = 216
    gate_args = get_random_gate_args(modes=modes)

    device_spec = get_device_spec()
    device_cert = get_device_cert()

    # the loop phases
    loop_phases = np.random.uniform(low=0, high=2 * pi, size=3).tolist()

    # update the ``device_cert`` with the random values obtained above
    device_cert["loop_phases"] = loop_phases

    device = Device(spec=device_spec, cert=device_cert)

    gate_args_out = make_phases_compatible(gate_args=gate_args, device=device)

    # the compatible phase-gate arguments
    phi_0_out = gate_args_out["loops"][0]["Rgate"]
    phi_1_out = gate_args_out["loops"][1]["Rgate"]
    phi_2_out = gate_args_out["loops"][2]["Rgate"]

    # the phase range supported by the modulators
    phi_min = device_spec["gate_parameters"]["r0"][0][0]
    phi_max = device_spec["gate_parameters"]["r0"][0][1]

    # obtain the actual phases after loop-offset compensation
    phi_0_actual, phi_1_actual, phi_2_actual = compensate_loop_offsets(
        phi_args=[phi_0_out, phi_1_out, phi_2_out],
        phi_loop=loop_phases,
        delays=[1, 6, 36],
        phi_range=[phi_min, phi_max],
    )

    # save actual (loop-offset compensated) phases to new ``gate_args`` dict
    gate_args_actual = copy.deepcopy(gate_args_out)
    gate_args_actual["loops"][0]["Rgate"] = phi_0_actual
    gate_args_actual["loops"][1]["Rgate"] = phi_1_actual
    gate_args_actual["loops"][2]["Rgate"] = phi_2_actual

    # photon-number statistics obtained by gate arguments that have been made
    # hardware compatible (without the presence of loop offsets)
    mean_n_wished, cov_n_wished = get_photon_number_moments(gate_args_out, device)

    # photon-number statistics obtained by loop-offset compensated phases in the
    # presence of loop offsets
    mean_n_actual, cov_n_actual = get_photon_number_moments(
        gate_args_actual, device, phi_loop=loop_phases
    )

    # are the offset-compensated phase-gate arguments hardware-compatible?
    are_gate_args_legal(
        gate_args=gate_args_out,
        device=device,
        delays=[1, 6, 36],
        assert_arg_length=False,
        assert_squeezing=False,
        assert_beamsplitters=False,
        assert_phase_gates=True,
    )

    # do we arrive at the same photon-number statistics after loop-offset
    # compensation?
    assert np.allclose(mean_n_wished, mean_n_actual)
    assert np.allclose(cov_n_wished, cov_n_actual)


@pytest.mark.parametrize("sq_input", ["user_array", "zero", "low", "medium", "high"])
@pytest.mark.parametrize("delays", [[1, 6, 36], [9, 8, 7]])
@pytest.mark.parametrize("init_cross", [[1, 6, 36], [0, 1, 2]])
def test_full_compile(sq_input, delays, init_cross):
    """Tests if an arbitrary set of gates is compiled successfully

    Args:
        sq_input (str): specifies the user input for the squeezing
        delays (list): the delay applied by each loop in time bins.
        init_cross (list): the number of initial time bins in which the three
            beamsplitters are in the cross state (T=1)
    """
    modes = 123
    gate_args = get_random_gate_args(modes=modes)

    device = get_device()

    if sq_input == "user_array":
        gate_args["Sgate"] = [np.random.random()] * modes
    else:
        gate_args["Sgate"] = sq_input

    # set the initial cross states of each BS
    gate_args["loops"][0]["BSgate"][: init_cross[0]] = [0] * init_cross[0]
    gate_args["loops"][1]["BSgate"][: init_cross[1]] = [0] * init_cross[1]
    gate_args["loops"][2]["BSgate"][: init_cross[2]] = [0] * init_cross[2]

    gate_args_out = full_compile(
        gate_args=gate_args, device=device, delays=delays, return_list=False
    )

    # validate every aspect of ``gate_args_out``
    are_gate_args_legal(
        gate_args=gate_args_out,
        device=device,
        delays=delays,
        assert_arg_length=True,
        assert_squeezing=True,
        assert_beamsplitters=True,
        assert_phase_gates=True,
    )


@pytest.mark.parametrize(
    "open_loops",
    [
        [True, True, True],
        [True, False, False],
        [False, True, True],
        [True, False, True],
    ],
)
@pytest.mark.parametrize("return_list", [True, False])
def test_borealis_gbs(open_loops, return_list):
    """Tests if the returned dict is hardware-compatible and correctly
    implements a GBS instance

    Args:
        return_list (bool): ``borealis_gbs`` returns a list of gate arguments
            instead of a dict
    """
    device = get_device()
    gate_args = borealis_gbs(device=device, open_loops=open_loops, return_list=return_list)

    if return_list:
        assert isinstance(gate_args, list)
        gate_args = to_args_dict(gate_args, device)
    assert isinstance(gate_args, dict)

    crop = gate_args.get("crop", get_crop_value(gate_args))

    # validate every aspect of ``gate_args``
    are_gate_args_legal(
        gate_args=gate_args,
        device=device,
        delays=[1, 6, 36],
        assert_arg_length=True,
        assert_squeezing=True,
        assert_beamsplitters=True,
        assert_phase_gates=True,
    )

    # are the squeezed modes protected from interference with vacuum? in other
    # words, is the total unitary a block-diagonal of the identity (acting
    # exclusively on vacuum) and the nontrivial unitary (acting exclusively on
    # pulses)?
    U = get_unitary(gate_args=gate_args, device=device)
    assert np.allclose(U[:crop, :crop], np.eye(crop))


def test_to_args_dict():
    """Tests that a list of gate arguments can be converted into a dictionary
    following the device layout."""
    gate_list = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    expected_dict = {
        "Sgate": [0.0],
        "loops": {
            0: {"Rgate": [1.0], "BSgate": [2.0]},
            1: {"Rgate": [3.0], "BSgate": [4.0]},
            2: {"Rgate": [5.0], "BSgate": [6.0]},
        },
    }
    gate_dict = to_args_dict(gate_list, device=get_device())
    assert gate_dict == expected_dict


@pytest.mark.parametrize("device", [None, get_device()])
def test_to_args_list(device):
    """Tests that a dictionary of gate arguments can be converted into a list
    with or without a device."""
    gate_dict = {
        "Sgate": [0.0],
        "loops": {
            0: {"Rgate": [1.0], "BSgate": [2.0]},
            1: {"Rgate": [3.0], "BSgate": [4.0]},
            2: {"Rgate": [5.0], "BSgate": [6.0]},
        },
    }
    expected_list = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    gate_list = to_args_list(gate_dict, device=device)
    assert gate_list == expected_list


def test_to_args_list_different_order():
    """Tests that a dictionary of gate arguments can be converted into a list with a device,
    when the dictionary's keys and loops are scrambled.

    Note, the Rgate-BSgate pairs are positional, and must be in the same order
    as in the device layout (if passed).
    """
    gate_dict = {
        "loops": {
            1: {"Rgate": [3], "BSgate": [4]},
            0: {"Rgate": [1], "BSgate": [2]},
            2: {"Rgate": [5], "BSgate": [6]},
        },
        "Sgate": [0],
    }
    expected_list = [[0], [1], [2], [3], [4], [5], [6]]
    gate_list = to_args_list(gate_dict, device=get_device())
    assert gate_list == expected_list


def test_to_args_list_no_device():
    """Tests that a dictionary of gate arguments can be converted into a list without a device.

    Note, the non-loop gates are positional, and will be flattened in the order
    that they're declared, while the loop gates are ordered by the loop index.
    """
    gate_dict = {
        "BSgate": [0],
        "loops": {
            1: {"Rgate": [3], "BSgate": [4]},
            0: {"Rgate": [1], "BSgate": [2]},
            2: {"Rgate": [5], "BSgate": [6]},
        },
        "Rgate": [7],
        "Sgate": [8],
    }
    expected_list = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    gate_list = to_args_list(gate_dict)
    assert gate_list == expected_list

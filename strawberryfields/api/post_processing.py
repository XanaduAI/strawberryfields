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
This module provides functions for post-processing samples.
"""
import numpy as np


def number_expectation_pnr(photon_number_samples, modes=None):
    """The expectation value of the number operator from PNR samples.

    **Example:**

    .. code-block:: python

        samples = np.array([[2, 0, None],
                            [2, 2, None],
                            [2, 0, None],
                            [0, 0, None]])

    Getting the expectation value for these PNR samples:

    >>> number_expectation_pnr(samples)
    1.0

    Args:
        photon_number_samples (ndarray): the photon number samples with a shape
            of (shots, modes)
        modes (Sequence): a flat sequence containing indices of modes to get
            the expectation value for

    Returns:
        float: the expectation value from the samples
    """
    product_across_modes = _checks_and_get_product(photon_number_samples, modes=modes)
    expval = product_across_modes.mean()
    return expval


def number_variance_pnr(photon_number_samples, modes=None):
    """The variance of the number operator from PNR samples.

    **Example:**

    .. code-block:: python

        samples = np.array([[2, 0, None],
                            [2, 2, None],
                            [2, 0, None],
                            [0, 0, None]])

    Getting the variance for these PNR samples:

    >>> number_expectation_pnr(samples)
    3.0

    Args:
        photon_number_samples (ndarray): the photon number samples with a shape
            of (shots, modes)
        modes (Sequence): a flat sequence containing indices of modes to get
            the variance for

    Returns:
        float: the variance from the samples
    """
    product_across_modes = _checks_and_get_product(photon_number_samples, modes=modes)
    variance = product_across_modes.var()
    return variance


def _checks_and_get_product(photon_number_samples, modes=None):
    """Input validation checks followed by getting the product across modes.

    Checks include data types checks and dimension of the input samples.

    Args:
        photon_number_samples (ndarray): the photon number samples with a shape
            of (shots, modes)
        modes (Sequence): a flat sequence containing indices of modes to get
            the expectation value for

    Returns:
        ndarray: product of the samples across modes
    """
    _check_samples(photon_number_samples)

    num_modes = photon_number_samples.shape[1]

    if modes is None:
        modes = np.arange(num_modes)

    _check_modes(photon_number_samples, modes)

    modes = np.array(modes)
    selected_samples = photon_number_samples[:, modes]
    return np.prod(selected_samples, axis=1)


def _check_samples(samples):
    """Validation check for the input samples.

    Checks include data types checks and dimension of the input samples.

    Args:
        samples (ndarray): the photon number samples with a shape of (shots,
            modes)
    """
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise Exception("Samples needs to be represented as a two dimensional numpy array.")


def _check_modes(samples, modes):
    """Validation checks for the input modes.

    Checks include data types checks, checks for the dimension of the inputs
    and the validity of the modes specified.

    Args:
        photon_number_samples (ndarray): the photon number samples with a shape
            of (shots, modes)
        modes (Sequence): the input modes to get the expectation value for, a
            flattened sequence
    """
    num_modes = samples.shape[1]
    flattened_sequence_indices_msg = (
        "The input modes need to be specified as a flattened sequence of indices!"
    )

    modes = np.array(modes)

    # Checking if modes is a valid flattened sequence of indices
    try:
        non_index_modes = modes[(np.mod(modes, 1) != 0) | (modes < 0)]
    except:
        raise Exception(flattened_sequence_indices_msg)

    if modes.ndim != 1 or non_index_modes.size > 0:
        raise Exception(flattened_sequence_indices_msg)

    # Checking if valid modes were specified
    largest_valid_index = num_modes - 1
    out_of_bounds_modes = modes[modes > largest_valid_index]
    if out_of_bounds_modes.size > 0:
        raise Exception(
            "Cannot specify mode indices {} for a {} mode system!".format(
                out_of_bounds_modes, num_modes
            )
        )

    # TODO: remove when the returned samples no longer contain Nones
    indices_for_no_measurement = np.argwhere(samples == None)
    modes_not_measured = set(indices_for_no_measurement[:, 1])
    invalid_modes_specified = modes_not_measured.intersection(set(modes))
    if len(invalid_modes_specified) > 0:
        raise Exception(
            "Modes {} were specified for post-processing, but no samples were found (they were not measured)!".format(
                invalid_modes_specified
            )
        )


number_expectation = number_expectation_pnr
number_variance = number_variance_pnr
shorthands = ["number_expectation", "number_variance"]

__all__ = ["number_expectation_pnr", "number_variance_pnr"] + shorthands

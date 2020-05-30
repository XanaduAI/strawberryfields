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

def number_expectation(photon_number_samples, modes=None):
    """
    Args:
        photon_number_samples (ndarray): the photon number samples with a shape
        of (shots, modes) modes=None
        modes (Sequence): the input modes to get the expectation value for, a
        flattened sequence
    """
    if modes is None:
        modes = np.arange(num_modes)

    _check_samples_modes(photon_number_samples, modes)

    shots = samples.shape[0]

    # TODO: need to add support for TF & TF batching!
    flattened_samples_without_nones = photon_number_samples[photon_number_samples != None]
    samples_without_nones = flattened_samples_without_nones.reshape((shots,-1))

    elementwise_product = np.prod(samples_without_nones, axis=1)
    expval = elementwise_product.mean()
    return expval

    """
    def number_variance(photon_number_samples, modes=None):
        expval = number_expectation(photon_number_samples, modes=modes)
        variance = number_expectation(photon_number_samples, modes=modes)
    """

def _check_samples_modes(samples, modes):
    """Validation checks for the input samples and modes.
    
    Checks include data types checks, checks for the shape of the inputs
    and the validity of the modes specified.

    Args:
        photon_number_samples (ndarray): the photon number samples with a shape
            of (shots, modes) 
        modes (Sequence): the input modes to get the expectation value for, a
            flattened sequence
    """
    # Validation checks for samples 
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise Exception("Samples needs to be represented as a two dimensional numpy array.")

    num_modes = samples.shape[1] - 1

    flattened_sequence_indices_msg = "The input modes need to be specified as a flattened sequence of indices!"
    # Validation checks for modes
    try:
        # Check if a rugged sequence was inputted
        modes = np.array(modes)
    except:
        raise Exception(flattened_sequence_indices_msg)

    # Checking if modes is a valid flattened sequence of indices
    try:
        non_index_modes = modes[(np.mod(modes, 1) != 0) | (modes<0)]
    except:
        raise Exception(flattened_sequence_indices_msg)

    if modes.ndim != 1 or non_index_modes.size > 0:
        raise Exception(flattened_sequence_indices_msg)

    # Checking if valid modes were specified
    invalid_modes = modes[modes>num_modes]
    if invalid_modes.size > 0:
        raise Exception("Cannot specify mode(s) {} for a {} mode system!".format(invalid_modes, num_modes))

    indices_for_no_measurement = np.argwhere(samples == None)
    modes_not_measured = np.unique(indices_for_no_measurement[:,1])
    if modes_not_measured.size > 0:
        raise Exception("Modes {} were specified for post-processing, but no samples were found (they were not measured)!".format(modes_not_measured))

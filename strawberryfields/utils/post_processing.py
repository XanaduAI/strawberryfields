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


def _product_for_modes(samples, modes=None):
    """Getting the product of samples across modes.

    Args:
        samples (ndarray): the photon number samples with a shape of ``(shots, modes)``
        modes (Sequence): indices of modes to compute the expectation value for

    Returns:
        ndarray: product of the samples across modes
    """
    modes = np.array(modes)
    selected_samples = samples[:, modes]
    return np.prod(selected_samples, axis=1)


def _check_samples(samples):
    """Validation check for the input samples.

    Checks include data types checks and dimension of the input samples.

    Args:
        samples (array): the photon number samples with a shape of ``(shots, modes)``
    """
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("Samples needs to be represented as a two dimensional NumPy array.")


def _check_modes(samples, modes):
    """Validation checks for the input modes.

    Checks include data types checks, checks for the dimension of the inputs
    and the validity of the modes specified.

    Args:
        samples (array): the photon number samples with a shape of ``(shots, modes)``
        modes (Sequence): the input modes to get the expectation value for
    """
    num_modes = samples.shape[1]
    flattened_sequence_indices_msg = (
        "The input modes need to be specified as a flattened sequence of non-negative integers!"
    )

    modes = np.array(modes)

    # Extracting non index modes while also checking that the type of the
    # objects in the array is correct E.g. a TypeError is raised for an array
    # containing lists
    try:
        non_index_modes = modes[(np.mod(modes, 1) != 0) | (modes < 0)]
    except TypeError:
        raise TypeError(flattened_sequence_indices_msg)

    if modes.ndim != 1 or non_index_modes.size > 0 or len(modes) == 0:
        raise ValueError(flattened_sequence_indices_msg)

    # Checking if valid modes were specified
    largest_valid_index = num_modes - 1
    out_of_bounds_modes = modes[modes > largest_valid_index]
    if out_of_bounds_modes.size > 0:
        raise ValueError(
            f"Cannot specify mode indices {out_of_bounds_modes} for a {num_modes} mode system."
        )


def samples_expectation(samples, modes=None):
    r"""Uses samples obtained by a measurement operator to return the
    expectation value of the operator.

    In case samples were obtained for multiple modes, it is assumed that the
    same measurement operator was used for each mode.

    **Using PNR samples to obtain number expectation**

    If applied to a single mode, this simply corresponds to mean photon number
    :math:`\langle n_i\rangle`.

    For multiple modes, the expectation value of the tensor product of number
    operator, :math:`\langle n_{i_0} \otimes n_{i_1} \otimes \cdots \otimes
    n_{i_m}\rangle` is returned.

    **Example:**

    >>> samples = np.array([[2, 0], [2, 2], [2, 0], [0, 0]])
    >>> samples_expectation(samples)
    1.0

    **Using homodyne samples to obtain expectation of the quadrature operators**

    Assuming that samples were obtained after measuring either the \hat{x} or
    \hat{p} operator.

    If applied to a single mode, this simply corresponds to the expectation
    value of the given quadrature operator. For example for samples obtained by
    measuring in the position basis, for a single mode we have :math:`\langle
    \hat{x}_i\rangle`,


    For multiple modes, the expectation value of the tensor product of the
    momentum operator is obtained: :math:`\langle \hat{x}_{i_0} \otimes
    \hat{x}_{i_1} \otimes \cdots \otimes \hat{x}_{i_m}\rangle`.

    **Example:**

    >>> samples = np.array([[1.23, 0], [12.32, 0.32], [0.3222, 6.34], [0, 3.543]])
    >>> samples_expectation(samples)
    1.4962870000000001

    Args:
        samples (array): samples with a shape of ``(shots, modes)``
        modes (Sequence): indices of modes to compute the expectation value over

    Returns:
        float: the expectation value from the samples
    """
    _check_samples(samples)

    num_modes = samples.shape[1]

    if modes is None:
        modes = np.arange(num_modes)
    else:
        _check_modes(samples, modes)

    product_across_modes = _product_for_modes(samples, modes)
    expval = product_across_modes.mean()
    return expval


def samples_variance(samples, modes=None):
    r"""Uses samples obtained by a measurement operator to return the
    expectation value of the operator.

    In case samples were obtained for multiple modes, it is assumed that the
    same measurement operator was used for each mode.

    **Using PNR samples to obtain variance of the number operator**

    If applied to a single mode, this simply corresponds to mean photon number
    :math:`Var[ n_i ]`.

    For multiple modes, the expectation value of the tensor product of number
    operator, :math:`Var[ n_{i_0} \otimes n_{i_1} \otimes \cdots \otimes
    n_{i_m} ]` is returned.

    **Example:**

    >>> samples = np.array([[2, 0], [2, 2], [2, 0], [0, 0]])
    >>> samples_variance(samples)
    3.0

    **Using homodyne samples to obtain variance of the quadrature operator**

    Assuming that samples were obtained after measuring either the \hat{x} or \hat{p}
    operator.

    If applied to a single mode, this simply corresponds to the expectation
    value of the given quadrature operator. For example for samples obtained by
    measuring in the position basis, for a single mode we have :math:`Var[
    \hat{x}_i]`

    For multiple modes, the variance of the tensor product of the momentum
    operator is obtained: :math:`Var[\hat{x}_{i_0} \otimes \hat{x}_{i_1}
    \otimes \cdots \otimes \hat{x}_{i_m}]`.

    **Example:**

    >>> samples = np.array([[1.23, 0], [12.32, 0.32], [0.3222, 6.34], [0, 3.543]])
    >>> samples_variance(samples)
    2.6899595015070004

    Args:
        homodyne_samples (ndarray): the homodyne samples with a shape of
            (shots, modes)
        modes (Sequence): indices of modes to get the variance for

    Returns:
        float: the variance from the samples
    """
    _check_samples(samples)

    num_modes = samples.shape[1]

    if modes is None:
        modes = np.arange(num_modes)
    else:
        _check_modes(samples, modes)

    product_across_modes = _product_for_modes(samples, modes)
    variance = product_across_modes.var()
    return variance


def all_fock_probs_pnr(photon_number_samples):
    r"""The Fock state probabilities for the specified modes obtained from PNR
    samples.

    Measured modes that are not specified are traced over. If either all
    the modes or no modes were specified, the marginal probabilities are
    returned.

    The Fock basis cutoff truncation is determined by the maximum Fock state
    with a non-zero probability.

    **Example:**

    >>> samples = np.array([[2, 0], [2, 2], [2, 0], [0, 0]])
    >>> probs = all_fock_probs_pnr(samples)

    2 out of 4 shots had the ``(2, 0)`` outcome, that is
    :math:`|\braketD{2,0}{\psi}|^2=0.5`. We can check if the probability for
    this state is correct by indexing into ``probs``:

    >>> probs[(2, 0)]
    0.5

    We can further check the entire array of probabilities:

    >>> probs
    [[0.25 0.   0.  ]
    [0.   0.   0.  ]
    [0.5  0.   0.25]]

    Args:
        photon_number_samples (array): the photon number samples with a shape
            of ``(shots, modes)``

    Returns:
        array: array of dimension :math:`\underbrace{D\times D\times
        D\cdots\times D}_{\text{num modes}}` containing the Fock state
        probabilities, where :math:`D` is the Fock basis cutoff truncation
    """
    _check_samples(photon_number_samples)

    num_modes = photon_number_samples.shape[1]

    shots = photon_number_samples.shape[0]
    num_modes = photon_number_samples.shape[1]
    cutoff = np.max(photon_number_samples)
    bins = [list(range(cutoff + 2)) for i in range(num_modes)]
    H, _ = np.histogramdd(photon_number_samples, bins=bins)
    probabilities = H / shots
    return probabilities


__all__ = [
    "samples_expectation",
    "samples_variance",
    "all_fock_probs_pnr",
]

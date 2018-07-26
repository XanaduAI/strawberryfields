# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Key mathematical operations used for tensorflow backend
======================

Contains the basic linear algebraic utilities needed by the tensorflow simulator backend
Hyperlinks: :class:`Circuit`

.. currentmodule:: strawberryfields.backends.tfbackend.ops

Contents
----------------------
.. autosummary::

"""
# pylint: disable=too-many-arguments

from string import ascii_lowercase as indices
from functools import lru_cache

import tensorflow as tf
import numpy as np
from scipy.special import binom, factorial

from ..shared_ops import generate_bs_factors, load_bs_factors, save_bs_factors, squeeze_parity

def_type = tf.complex64
max_num_indices = len(indices)

###################################################################

# Helper functions:

def _numer_safe_power(base, exponent):
    """gives the desired behaviour of 0**0=1"""
    if exponent == 0:
        return tf.ones_like(base, dtype=def_type)

    return base ** exponent

def mixed(pure_state, batched=False):
    """Converts the state from pure to mixed"""
    #todo: In the fock backend mixing is done by ops.mix(), maybe the functions should be named identically?
    if not batched:
        pure_state = tf.expand_dims(pure_state, 0) # add in fake batch dimension
    batch_offset = 1
    num_modes = len(pure_state.shape) - batch_offset
    max_num = (max_num_indices - batch_offset) // 2
    if num_modes > max_num:
        raise ValueError("Converting state from pure to mixed currently only supported for {} modes.".format(max_num))
    else:
        # eqn: 'abc...xyz,ABC...XYZ->aAbBcC...xXyYzZ' (lowercase belonging to 'bra' side, uppercase belonging to 'ket' side)
        batch_index = indices[:batch_offset]
        bra_indices = indices[batch_offset: batch_offset + num_modes]
        ket_indices = indices[batch_offset + num_modes : batch_offset + 2 * num_modes]
        eqn_lhs = batch_index + bra_indices + "," + batch_index + ket_indices
        eqn_rhs = "".join(bdx + kdx for bdx, kdx in zip(bra_indices, ket_indices))
        eqn = eqn_lhs + "->" + batch_index + eqn_rhs
        mixed_state = tf.einsum(eqn, pure_state, tf.conj(pure_state))
    if not batched:
        mixed_state = tf.squeeze(mixed_state, 0) # drop fake batch dimension
    return mixed_state

def batchify_indices(idxs, batch_size):
    """adds batch indices to the index numbering"""
    return [(bdx,) + idxs[i] for bdx in range(batch_size) for i in range(len(idxs))]

def unravel_index(ind, tensor_shape):
    """no official tensorflow implementation for this yet;
    this is based on one proposed in https://github.com/tensorflow/tensorflow/issues/2075
    """
    ind = tf.expand_dims(tf.cast(ind, tf.int64), 0)
    tensor_shape = tf.expand_dims(tf.cast(tensor_shape, tf.int64), 1)
    strides = tf.cumprod(tensor_shape, reverse=True)
    strides_shifted = tf.cumprod(tensor_shape, exclusive=True, reverse=True)
    unraveled_coords = (ind % strides) // strides_shifted
    return tf.transpose(unraveled_coords)

@lru_cache()
def get_prefac_tensor(D, directory, save):
    """Equivalent to the functionality of shared_ops the bs_factors functions from shared_ops,
    but caches the return value as a tensor. This allows us to re-use the same prefactors and save
    space on the computational graph."""
    try:
        prefac = load_bs_factors(D, directory)
    except FileNotFoundError:
        prefac = generate_bs_factors(D)
        if save:
            save_bs_factors(prefac, directory)
    prefac = tf.expand_dims(tf.cast(prefac[:D, :D, :D, :D, :D], def_type), 0)
    return prefac

###################################################################

# Matrices:
# These return a matrix of shape [D, D]
# which transforms from the input (truncated) Fock basis to the
# output (truncated) Fock basis.

def squeezed_vacuum_vector(r, theta, D, batched=False, eps=1e-32):
    """returns the ket representing a single mode squeezed vacuum state"""
    if batched:
        batch_size = r.shape[0].value
    r = tf.cast(r, def_type)
    theta = tf.cast(theta, def_type)
    c1 = tf.cast(tf.stack([tf.sqrt(1 / tf.cosh(r)) * np.sqrt(factorial(k)) / factorial(k / 2.) for k in range(0, D, 2)], axis=-1), def_type)
    c2 = tf.stack([(-0.5 * tf.exp(1j * theta) * tf.cast(tf.tanh(r + eps), def_type)) ** (k / 2.) for k in range(0, D, 2)], axis=-1)
    even_coeffs = c1 * c2
    ind = [(k,) for k in np.arange(0, D, 2)]
    shape = [D]
    if batched:
        ind = batchify_indices(ind, batch_size)
        shape = [batch_size] + shape
    output = tf.scatter_nd(ind, tf.reshape(even_coeffs, [-1]), shape)
    return output

def squeezer_matrix(r, theta, D, batched=False):
    """creates the single mode squeeze matrix"""
    r = tf.cast(r, tf.float64)
    if not batched:
        r = tf.expand_dims(r, 0) # introduce artificial batch dimension
    r = tf.reshape(r, [-1, 1, 1, 1])
    theta = tf.cast(theta, def_type)
    theta = tf.reshape(theta, [-1, 1, 1, 1])

    rng = np.arange(D)
    n = np.reshape(rng, [-1, D, 1, 1])
    m = np.reshape(rng, [-1, 1, D, 1])
    k = np.reshape(rng, [-1, 1, 1, D])

    phase = tf.exp(1j * theta * (n-m) / 2)
    signs = squeeze_parity(D).reshape([1, D, 1, D])
    mask = np.logical_and((m + n) % 2 == 0, k <= np.minimum(m, n)) # kills off terms where the sum index k goes past min(m,n)
    k_terms = signs * \
                        tf.pow(tf.sinh(r) / 2, mask * (n + m - 2 * k) / 2) * mask / \
                        tf.pow(tf.cosh(r), (n + m + 1) / 2) * \
                        tf.exp(0.5 * tf.lgamma(tf.cast(m + 1, tf.float64)) + \
                               0.5 * tf.lgamma(tf.cast(n + 1, tf.float64)) - \
                               tf.lgamma(tf.cast(k + 1, tf.float64)) -
                               tf.lgamma(tf.cast((m - k) / 2 + 1, tf.float64)) - \
                               tf.lgamma(tf.cast((n - k) / 2 + 1, tf.float64))
                              )
    output = tf.reduce_sum(phase * tf.cast(k_terms, def_type), axis=-1)

    if not batched:
        # remove extra batch dimension
        output = tf.squeeze(output, 0)
    return output

def phase_shifter_matrix(theta, D, batched=False):
    """creates the single mode phase shifter matrix"""
    if batched:
        batch_size = theta.shape[0].value
    theta = tf.cast(theta, def_type)
    shape = [D, D]
    if batched:
        shape = [batch_size] + shape
    zero_matrix = tf.zeros(shape=shape, dtype=def_type)
    diag = [tf.exp(1j * theta * k) for k in np.arange(D, dtype=np.complex64)]
    if batched:
        diag = tf.stack(diag, axis=1)
    diag_matrix = tf.matrix_set_diag(zero_matrix, diag)
    return diag_matrix

def kerr_interaction_matrix(kappa, D, batched=False):
    """creates the single mode Kerr interaction matrix"""
    coeffs = [tf.exp(1j * kappa * n ** 2) for n in range(D)]
    if batched:
        coeffs = tf.stack(coeffs, axis=1)
    output = tf.matrix_diag(coeffs)
    return output

def cross_kerr_interaction_matrix(kappa, D, batched=False):
    """creates the two mode cross-Kerr interaction matrix"""
    coeffs = [tf.exp(1j * kappa * n1 * n2) for n1 in range(D) for n2 in range(D)]
    if batched:
        coeffs = tf.stack(coeffs, axis=1)
    output = tf.matrix_diag(coeffs)
    if batched:
        output = tf.transpose(tf.reshape(output, [-1] + [D]*4), [0, 1, 3, 2, 4])
    else:
        output = tf.transpose(tf.reshape(output, [D]*4), [0, 2, 1, 3])
    return output

def cubic_phase_matrix(gamma, D, hbar, batched=False, method="self_adjoint_eig"):
    """creates the single mode cubic phase matrix"""
    a, ad = ladder_ops(D)
    x = np.sqrt(hbar / 2) * tf.cast(a + ad, def_type)
    x3 = x @ x @ x
    if batched:
        x3 = tf.expand_dims(x3, 0)
        gamma = tf.reshape(gamma, [-1, 1, 1])
    H0 = gamma / (3 * hbar) * x3
    lambdas, U = tf.self_adjoint_eig(H0)
    transpose_list = [1, 0]
    if batched:
        transpose_list = [0, 2, 1]
    if method == "self_adjoint_eig":
        # This seems to work as long as the matrix has dimension 18x18 or smaller
        # For larger matrices, Tensorflow returns the error: 'Self-adjoint eigen decomposition was not successful.'
        V = U @ tf.matrix_diag(tf.exp(1j * lambdas)) @ tf.conj(tf.transpose(U, transpose_list))
    # below version works for matrices larger than 18x18, but
    # expm was not added until TF v1.5, while
    # as of TF v1.5, gradient of expm is not implemented
    #elif method == "expm":
    #    V = tf.linalg.expm(1j * H0)
    else:
        raise ValueError("'method' must be either 'self_adjoint_eig' or 'expm'.")
    return V

def loss_superop(T, D, batched=False):
    """creates the single mode loss channel matrix"""
    # indices are abcd, corresponding to action |a><b| (.) |c><d|
    if not batched:
        T = tf.expand_dims(T, 0) # introduce artificial batch dimension
    T = tf.reshape(T, [-1, 1, 1, 1, 1, 1])
    T = tf.cast(T, tf.float64)
    rng = np.arange(D)
    a = np.reshape(rng, [-1, D, 1, 1, 1, 1])
    b = np.reshape(rng, [-1, 1, D, 1, 1, 1])
    c = np.reshape(rng, [-1, 1, 1, D, 1, 1])
    d = np.reshape(rng, [-1, 1, 1, 1, D, 1])
    l = np.reshape(rng, [-1, 1, 1, 1, 1, D])
    a_mask = np.where(a == b - l, 1, 0)
    d_mask = np.where(d == c - l, 1, 0)
    mask = (l <= np.minimum(b, c)) * a_mask * d_mask

    exponent = tf.abs((b + c) / 2 - l) * mask
    T_numer = tf.pow((1 - T), l) * tf.pow(T, exponent)
    fact_numer = np.sqrt(factorial(b) * factorial(c))
    fact_denom = np.sqrt(factorial(b - l) * factorial(c - l)) * factorial(l)
    fact_denom_masked = np.where(fact_denom > 0, fact_denom, 1)
    factors = mask * fact_numer / fact_denom_masked
    l_terms = T_numer * factors
    output = tf.cast(tf.reduce_sum(l_terms, -1), def_type)
    if not batched:
        output = tf.squeeze(output, 0) # drop artificial batch dimension
    return output

def displacement_matrix(alpha, D, batched=False):
    """creates the single mode displacement matrix"""
    if batched:
        batch_size = alpha.shape[0].value
    alpha = tf.cast(alpha, def_type)
    idxs = [(j, k) for j in range(D) for k in range(j)]
    values = [alpha ** (j-k) * tf.cast(tf.sqrt(binom(j, k) / factorial(j-k)), def_type)
              for j in range(D) for k in range(j)]
    values = tf.stack(values, axis=-1)
    dense_shape = [D, D]
    vals = [1.0] * D
    ind = [(idx, idx) for idx in range(D)]
    if batched:
        dense_shape = [batch_size] + dense_shape
        vals = vals * batch_size
        ind = batchify_indices(ind, batch_size)
    eye_diag = tf.SparseTensor(ind, vals, dense_shape)
    signs = [(-1) ** (j-k) for j in range(D) for k in range(j)]
    if batched:
        idxs = batchify_indices(idxs, batch_size)
        signs = signs * batch_size
    sign_lower_diag = tf.cast(tf.SparseTensor(idxs, signs, dense_shape), tf.float32)
    sign_matrix = tf.sparse_add(eye_diag, sign_lower_diag)
    sign_matrix = tf.cast(tf.sparse_tensor_to_dense(sign_matrix), def_type)
    lower_diag = tf.scatter_nd(idxs, tf.reshape(values, [-1]), dense_shape)
    E = tf.cast(tf.eye(D), def_type) + lower_diag
    E_prime = tf.conj(E) * sign_matrix
    if batched:
        eqn = 'aik,ajk->aij' # pylint: disable=bad-whitespace
    else:
        eqn = 'ik,jk->ij' # pylint: disable=bad-whitespace
    prefactor = tf.expand_dims(tf.expand_dims(tf.cast(tf.exp(-0.5 * tf.abs(alpha) ** 2), def_type), -1), -1)
    D_alpha = prefactor * tf.einsum(eqn, E, E_prime)
    return D_alpha

def beamsplitter_matrix(t, r, D, batched=False, save=False, directory=None):
    """creates the two mode beamsplitter matrix"""
    if not batched:
        # put in a fake batch dimension for broadcasting convenience
        t = tf.expand_dims(t, -1)
        r = tf.expand_dims(r, -1)
    t = tf.cast(tf.reshape(t, [-1, 1, 1, 1, 1, 1]), def_type)
    r = tf.cast(tf.reshape(r, [-1, 1, 1, 1, 1, 1]), def_type)
    mag_t = tf.cast(t, tf.float32)
    mag_r = tf.abs(r)
    phase_r = tf.atan2(tf.imag(r), tf.real(r))

    rng = tf.range(D, dtype=tf.float32)
    N = tf.reshape(rng, [1, -1, 1, 1, 1, 1])
    n = tf.reshape(rng, [1, 1, -1, 1, 1, 1])
    M = tf.reshape(rng, [1, 1, 1, -1, 1, 1])
    k = tf.reshape(rng, [1, 1, 1, 1, 1, -1])
    n_minus_k = n - k
    N_minus_k = N - k
    M_minus_n_plus_k = M - n + k
    # need to deal with 0*(-n) for n integer
    n_minus_k = tf.where(tf.greater(n, k), n_minus_k, tf.zeros_like(n_minus_k))
    N_minus_k = tf.where(tf.greater(N, k), N_minus_k, tf.zeros_like(N_minus_k))
    M_minus_n_plus_k = tf.where(tf.greater(M_minus_n_plus_k, 0), M_minus_n_plus_k, tf.zeros_like(M_minus_n_plus_k))

    powers = tf.cast(tf.pow(mag_t, k) * tf.pow(mag_r, n_minus_k) * tf.pow(mag_r, N_minus_k) * tf.pow(mag_t, M_minus_n_plus_k), def_type)
    phase = tf.exp(1j * tf.cast(phase_r * (n - N), def_type))

    # load parameter-independent prefactors
    prefac = get_prefac_tensor(D, directory, save)

    if prefac.graph != phase.graph:
        # if cached prefactors live on another graph, we'll have to reload them into this graph.
        # In future versions, if 'copy_variable_to_graph' comes out of contrib, consider using that
        get_prefac_tensor.cache_clear()
        prefac = get_prefac_tensor(D, directory, save)

    BS_matrix = tf.reduce_sum(phase * powers * prefac, -1)

    if not batched:
        # drop artificial batch index
        BS_matrix = tf.squeeze(BS_matrix, [0])
    return BS_matrix

###################################################################

# Input states:
# Creates input states on a single mode

def fock_state(n, D, pure=True, batched=False):
    """creates a single mode input Fock state"""
    if not isinstance(n, (np.ndarray, int)):
        raise ValueError("'n' is expected to be either an int or a numpy array")
    if batched:
        batch_size = n.shape[0]
        idxs = [(b, f) for (b, f) in zip(range(batch_size), n)]
        values = [1.0] * batch_size
        shape = [batch_size, D]
    else:
        idxs = [(n,)]
        values = [1.0]
        shape = [D]
    fock_sparse = tf.scatter_nd(idxs, values, shape)
    fock = tf.cast(fock_sparse, def_type)
    if not pure:
        fock = mixed(fock, batched)
    return fock

def coherent_state(alpha, D, pure=True, batched=False):
    """creates a single mode input coherent state"""
    coh = tf.stack([tf.exp(-0.5 * tf.conj(alpha) * alpha) * _numer_safe_power(alpha, n) / tf.cast(np.sqrt(factorial(n)), def_type) for n in range(D)], axis=-1)
    if not pure:
        coh = mixed(coh, batched)
    return coh

def squeezed_vacuum(r, theta, D, pure=True, batched=False):
    """creates a single mode input squeezed vacuum state"""
    squeezed = squeezed_vacuum_vector(r, theta, D, batched=batched)
    if not pure:
        squeezed = mixed(squeezed, batched)
    return squeezed

def displaced_squeezed(alpha, r, phi, D, pure=True, batched=False, eps=1e-12):
    """creates a single mode input displaced squeezed state"""
    alpha = tf.cast(alpha, def_type)
    r = tf.cast(r, def_type) + eps # to prevent nans if r==0, we add an epsilon (default is miniscule)
    phi = tf.cast(phi, def_type)

    phase = tf.exp(1j * phi)
    sinh = tf.sinh(r)
    cosh = tf.cosh(r)
    tanh = tf.tanh(r)

    # create Hermite polynomials
    gamma = alpha * cosh + tf.conj(alpha) * phase * sinh
    hermite_arg = gamma / tf.sqrt(phase * tf.sinh(2 * r))

    prefactor = tf.expand_dims(tf.exp(-0.5 * alpha * tf.conj(alpha) - 0.5 * tf.conj(alpha) ** 2 * phase * tanh), -1)
    coeff = tf.stack([_numer_safe_power(0.5 * phase * tanh, n / 2.) / tf.sqrt(factorial(n) * cosh)
                      for n in range(D)], axis=-1)
    hermite_terms = tf.stack([tf.cast(H(n, hermite_arg), def_type) for n in range(D)], axis=-1)
    squeezed_coh = prefactor * coeff * hermite_terms

    if not pure:
        squeezed_coh = mixed(squeezed_coh, batched)
    return squeezed_coh

def thermal_state(nbar, D):
    """creates a single mode input thermal state.
    Note that the batch dimension is determined by argument nbar.
    """
    nbar = tf.cast(nbar, def_type)
    coeffs = tf.stack([_numer_safe_power(nbar, n) / _numer_safe_power(nbar + 1, n + 1) for n in range(D)], axis=-1)
    thermal = tf.matrix_diag(coeffs)
    return thermal

###################################################################

# Generic Gate implementations:
# These apply the given matrix to the specified mode(s) of in_modes

def single_mode_gate(matrix, mode, in_modes, pure=True, batched=False):
    """basic form:
    'ab,cde...b...xyz->cde...a...xyz' (pure state)
    'ab,ef...bc...xyz,cd->ef...ad...xyz' (mixed state)
    """
    if batched:
        batch_offset = 1
    else:
        batch_offset = 0
    batch_index = indices[:batch_offset]
    left_gate_str = indices[batch_offset : batch_offset + 2] # |a><b|
    num_indices = len(in_modes.shape)
    if pure:
        num_modes = num_indices - batch_offset
        mode_size = 1
    else:
        right_gate_str = indices[batch_offset + 2 : batch_offset + 4] # |c><d|
        num_modes = (num_indices - batch_offset) // 2
        mode_size = 2
    max_len = len(indices) - 2 * mode_size - batch_offset
    if num_modes == 0:
        raise ValueError("'in_modes' must have at least one mode")
    if num_modes > max_len:
        raise NotImplementedError("The max number of supported modes for this operation is currently {}".format(max_len))
    if mode < 0 or mode >= num_modes:
        raise ValueError("'mode' argument is not compatible with number of in_modes")
    else:
        other_modes_indices = indices[batch_offset + 2 * mode_size : batch_offset + (1 + num_modes) * mode_size]
        if pure:
            eqn_lhs = "{},{}{}{}{}".format(batch_index + left_gate_str, batch_index, other_modes_indices[:mode * mode_size], left_gate_str[1], other_modes_indices[mode * mode_size:])
            eqn_rhs = "".join([batch_index, other_modes_indices[:mode * mode_size], left_gate_str[0], other_modes_indices[mode * mode_size:]])
        else:
            eqn_lhs = "{},{}{}{}{}{},{}".format(batch_index + left_gate_str, batch_index, other_modes_indices[:mode * mode_size], left_gate_str[1], right_gate_str[0], other_modes_indices[mode * mode_size:], batch_index + right_gate_str)
            eqn_rhs = "".join([batch_index, other_modes_indices[:mode * mode_size], left_gate_str[0], right_gate_str[1], other_modes_indices[mode * mode_size:]])

    eqn = eqn_lhs + "->" + eqn_rhs
    einsum_inputs = [matrix, in_modes]
    if not pure:
        transposed_axis = [0, 2, 1] if batched else [1, 0]
        einsum_inputs.append(tf.transpose(tf.conj(matrix), transposed_axis))
    output = tf.einsum(eqn, *einsum_inputs)
    return output

def two_mode_gate(matrix, mode1, mode2, in_modes, pure=True, batched=False):
    """basic form:
    'abcd,efg...b...d...xyz->efg...a...c...xyz' (pure state)
    'abcd,ij...be...dg...xyz,efgh->ij...af...ch...xyz' (mixed state)
    """
    # pylint: disable=too-many-branches,too-many-statements
    if batched:
        batch_offset = 1
    else:
        batch_offset = 0
    batch_index = indices[:batch_offset]
    left_gate_str = indices[batch_offset : batch_offset + 4] # |a><b| |c><d|
    num_indices = len(in_modes.shape)
    if pure:
        num_modes = num_indices - batch_offset
        mode_size = 1
    else:
        right_gate_str = indices[batch_offset + 4 : batch_offset + 8] # |e><f| |g><h|
        num_modes = (num_indices - batch_offset) // 2
        mode_size = 2
    max_len = (len(indices) - 4) // mode_size - batch_offset

    if num_modes == 0:
        raise ValueError("'in_modes' must have at least one mode")
    if num_modes > max_len:
        raise NotImplementedError("The max number of supported modes for this operation is currently {}".format(max_len))
    else:
        min_mode = min(mode1, mode2)
        max_mode = max(mode1, mode2)
        if min_mode < 0 or max_mode >= num_modes or mode1 == mode2:
            raise ValueError("One or more mode numbers are incompatible")
        else:
            other_modes_indices = indices[batch_offset + 4 * mode_size : batch_offset + 4 * mode_size + mode_size * (num_modes - 2)]
            # build equation
            if mode1 == min_mode:
                lhs_min_mode_indices = left_gate_str[1]
                lhs_max_mode_indices = left_gate_str[3]
                rhs_min_mode_indices = left_gate_str[0]
                rhs_max_mode_indices = left_gate_str[2]
            else:
                lhs_min_mode_indices = left_gate_str[3]
                lhs_max_mode_indices = left_gate_str[1]
                rhs_min_mode_indices = left_gate_str[2]
                rhs_max_mode_indices = left_gate_str[0]
            if not pure:
                if mode1 == min_mode:
                    lhs_min_mode_indices += right_gate_str[0]
                    lhs_max_mode_indices += right_gate_str[2]
                    rhs_min_mode_indices += right_gate_str[1]
                    rhs_max_mode_indices += right_gate_str[3]
                else:
                    lhs_min_mode_indices += right_gate_str[2]
                    lhs_max_mode_indices += right_gate_str[0]
                    rhs_min_mode_indices += right_gate_str[3]
                    rhs_max_mode_indices += right_gate_str[1]
            eqn_lhs = "{},{}{}{}{}{}{}".format(batch_index + left_gate_str,
                                               batch_index,
                                               other_modes_indices[:min_mode * mode_size],
                                               lhs_min_mode_indices,
                                               other_modes_indices[min_mode * mode_size : (max_mode - 1) * mode_size],
                                               lhs_max_mode_indices,
                                               other_modes_indices[(max_mode - 1) * mode_size:])
            if not pure:
                eqn_lhs += "," + batch_index + right_gate_str
            eqn_rhs = "".join([batch_index,
                               other_modes_indices[:min_mode * mode_size],
                               rhs_min_mode_indices,
                               other_modes_indices[min_mode * mode_size: (max_mode - 1) * mode_size],
                               rhs_max_mode_indices,
                               other_modes_indices[(max_mode - 1) * mode_size:]
                              ])
            eqn = eqn_lhs + "->" + eqn_rhs
            einsum_inputs = [matrix, in_modes]
            if not pure:
                if batched:
                    transpose_list = [0, 2, 1, 4, 3]
                else:
                    transpose_list = [1, 0, 3, 2]
                einsum_inputs.append(tf.conj(tf.transpose(matrix, transpose_list)))
            output = tf.einsum(eqn, *einsum_inputs)
            return output

def single_mode_superop(superop, mode, in_modes, pure=True, batched=False):
    """rho_out = S[rho_in]
    (state is always converted to mixed to apply superop)
    basic form:
    abcd,ef...klbcmn...yz->ef...kladmn...yz
    """

    if batched:
        batch_offset = 1
    else:
        batch_offset = 0

    max_len = (len(indices) - 2) // 2 - batch_offset
    if pure:
        num_modes = len(in_modes.shape) - batch_offset
    else:
        num_modes = (len(in_modes.shape) - batch_offset) // 2
    if num_modes > max_len:
        raise NotImplementedError("The max number of supported modes for this operation is currently {}".format(max_len))
    else:
        if pure:
            in_modes = mixed(in_modes, batched)

        # create equation
        batch_index = indices[:batch_offset]
        superop_indices = indices[batch_offset : batch_offset + 4]
        state_indices = indices[batch_offset + 4 : batch_offset + 4 + 2 * num_modes]
        left_unchanged_indices = state_indices[:2 * mode]
        right_unchanged_indices = state_indices[2 * mode : 2 * (num_modes - 1)]
        eqn_lhs = ",".join([batch_index + superop_indices, batch_index + left_unchanged_indices + superop_indices[1:3] + right_unchanged_indices])
        eqn_rhs = "".join([batch_index, left_unchanged_indices + superop_indices[0] + superop_indices[3] + right_unchanged_indices])
        eqn = "->".join([eqn_lhs, eqn_rhs])
        new_state = tf.einsum(eqn, superop, in_modes)
        return new_state

###################################################################

# Quantum Gate/Channel functions:
# Primary quantum optical gates implemented as transformations on 'in_modes'

def phase_shifter(theta, mode, in_modes, D, pure=True, batched=False):
    """returns phase shift unitary matrix on specified input modes"""
    matrix = phase_shifter_matrix(theta, D, batched=batched)
    output = single_mode_gate(matrix, mode, in_modes, pure, batched)
    return output

def displacement(alpha, mode, in_modes, D, pure=True, batched=False):
    """returns displacement unitary matrix on specified input modes"""
    matrix = displacement_matrix(alpha, D, batched)
    output = single_mode_gate(matrix, mode, in_modes, pure, batched)
    return output

def squeezer(r, theta, mode, in_modes, D, pure=True, batched=False):
    """returns squeezer unitary matrix on specified input modes"""
    matrix = squeezer_matrix(r, theta, D, batched)
    output = single_mode_gate(matrix, mode, in_modes, pure, batched)
    return output

def kerr_interaction(kappa, mode, in_modes, D, pure=True, batched=False):
    """returns Kerr unitary matrix on specified input modes"""
    matrix = kerr_interaction_matrix(kappa, D, batched)
    output = single_mode_gate(matrix, mode, in_modes, pure, batched)
    return output

def cross_kerr_interaction(kappa, mode1, mode2, in_modes, D, pure=True, batched=False):
    """returns cross-Kerr unitary matrix on specified input modes"""
    matrix = cross_kerr_interaction_matrix(kappa, D, batched)
    output = two_mode_gate(matrix, mode1, mode2, in_modes, pure, batched)
    return output

def cubic_phase(gamma, mode, in_modes, D, hbar=2, pure=True, batched=False, method="self_adjoint_eig"):
    """returns cubic phase unitary matrix on specified input modes"""
    matrix = cubic_phase_matrix(gamma, D, hbar, batched, method=method)
    output = single_mode_gate(matrix, mode, in_modes, pure, batched)
    return output

def beamsplitter(t, r, mode1, mode2, in_modes, D, pure=True, batched=False):
    """returns beamsplitter unitary matrix on specified input modes"""
    matrix = beamsplitter_matrix(t, r, D, batched)
    output = two_mode_gate(matrix, mode1, mode2, in_modes, pure, batched)
    return output

def loss_channel(T, mode, in_modes, D, pure=True, batched=False):
    """returns loss channel matrix on specified input modes"""
    superop = loss_superop(T, D, batched)
    output = single_mode_superop(superop, mode, in_modes, pure, batched)
    return output

###################################################################

# Other useful functions for organizing modes

def combine_single_modes(modes_list, batched=False):
    """Group together a list of single modes (each having dim=1 or dim=2) into a composite mode system."""
    if batched:
        batch_offset = 1
    else:
        batch_offset = 0
    num_modes = len(modes_list)
    if num_modes <= 1:
        raise ValueError("'modes_list' must have at least two modes")

    dims = np.array([len(mode.shape) - batch_offset for mode in modes_list])
    if min(dims) < 1 or max(dims) > 2:
        raise ValueError("Each mode in 'modes_list' can only have dim=1 or dim=2")

    if np.all(dims == 1):
        # All modes are represented as pure states.
        # Can return combined state also as pure state.
        # basic form:
        # 'a,b,c,...,x,y,z->abc...xyz'
        max_num = max_num_indices - batch_offset
        if num_modes > max_num:
            raise NotImplementedError("The max number of supported modes for this operation with pure states is currently {}".format(max_num))
        batch_index = indices[:batch_offset]
        out_str = indices[batch_offset : batch_offset + num_modes]
        modes_str = ",".join([batch_index + idx for idx in out_str])
        eqn = "{}->{}".format(modes_str, batch_index + out_str)
        einsum_inputs = modes_list
    else:
        # Some modes are mixed.
        # Return combined state as mixed.
        # basic form:
        # e.g., if first mode is pure and second is mixed...
        # 'a,b,cd,...->abcd...'
        # where (a,b) will belong to the first mode (bra & ket)
        # and cd will belong to the second mode (density matrix)
        max_num = (max_num_indices - batch_offset) // 2
        batch_index = indices[:batch_offset]
        if num_modes > max_num:
            raise NotImplementedError("The max number of supported modes for this operation with mixed states is currently {}".format(max_num))
        mode_idxs = [indices[slice(batch_offset + idx, batch_offset + idx + 2)] for idx in range(0, 2 * num_modes, 2)] # each mode gets a pair of consecutive indices
        eqn_rhs = batch_index + "".join(mode_idxs)
        eqn_idxs = [batch_index + m if dims[idx] == 2 else ",".join(m) for idx, m in enumerate(mode_idxs)]
        eqn_lhs = ",".join(eqn_idxs)
        eqn = eqn_lhs + "->" + eqn_rhs
        einsum_inputs = []
        for idx, mode in enumerate(modes_list):
            if dims[idx] == 1:
                new_inputs = [mode, tf.conj(mode)]
            elif dims[idx] == 2:
                new_inputs = [mode]
            einsum_inputs += new_inputs
    combined_modes = tf.einsum(eqn, *einsum_inputs)
    return combined_modes

def replace_mode(replacement, mode, system, state_is_pure, batched=False):
    """Replace the subsystem 'mode' of 'system' with new state 'replacement'. Argument 'state_is_pure' indicates whether

    This is just a simple wrapper for replace_modes()
    """
    # deprecated: This method has become obsolete and is superseeded by replace_modes()
    if isinstance(mode, int):
        mode = [mode]
    replace_modes(replacement, mode, system, state_is_pure, batched)

def replace_modes(replacement, modes, system, system_is_pure, batched=False):
    """Replace the subsystem 'mode' of 'system' with new state 'replacement'. Argument 'system_is_pure' indicates whether
    'system' is represented by a pure state or a density matrix.
    Note: Does not check if this replacement is physically valid (i.e., if 'replacement' is a valid state)
    Note: expects the shape of both replacement and system to match the batched parameter
    Note: modes does not need to be ordered.
    """
    #todo: write a better docstring
    if isinstance(modes, int):
        modes = [modes]

    if batched:
        batch_offset = 1
    else:
        batch_offset = 0

    num_modes = len(system.shape) - batch_offset
    if not system_is_pure:
        num_modes = int(num_modes/2)

    replacement_is_pure = bool(len(replacement.shape) - batch_offset == len(modes))

    # we take care of sorting modes below
    if len(modes) == num_modes:
        # we are replacing all the modes
        revised_modes = replacement
        revised_modes_pure = replacement_is_pure
    else:
        # we are replacing a subset, so have to trace
        # make both system and replacement mixed
        # todo: For performance the partial trace could be done directly from the pure state. This would of course require a better partial trace function...
        if system_is_pure:
            system = mixed(system, batched)

        #mix the replacement if it is pure
        if replacement_is_pure:
            replacement = mixed(replacement, batched)

        # partial trace out modes
        # todo: We are tracing out the modes one by one in descending order (to not screw up things). This is quite inefficient.
        reduced_state = system
        for mode in sorted(modes, reverse=True):
            reduced_state = partial_trace(reduced_state, mode, False, batched)
        # append mode and insert state (There is also insert_state(), but it seemed unnecessarily complicated to try to generalize this function, which does a lot of manual list comprehension, to the multi mode case than to write the two liner below)
        # todo: insert_state() could be rewritten to take full advantage of the high level functions of tf. Before doing that different implementatinos should be benchmarked to compare speed and memory requirements, as in practice these methods will be perfomance critical.
        if not batched:
            #todo: remove the hack in the line below and enable the line with axes=0 instead, if ever we change the dependency of SF to tensorflow>=1.6
            #revised_modes = tf.tensordot(reduced_state, replacement, axes=0)
            revised_modes = tf.tensordot(tf.expand_dims(reduced_state, 0), tf.expand_dims(replacement, 0), axes=[[0], [0]])
        else:
            batch_size = reduced_state.shape[0].value
            #todo: remove the hack in the line below and enabled the line with axes=0 instead, if ever we change the dependency of SF to tensorflow>=1.6
            #revised_modes = tf.stack([tf.tensordot(reduced_state[b], replacement[b], axes=0) for b in range(batch_size)])
            revised_modes = tf.stack([tf.tensordot(tf.expand_dims(reduced_state[b], 0), tf.expand_dims(replacement[b], 0), axes=[[0], [0]]) for b in range(batch_size)])
        revised_modes_pure = False

    # unless the preparation was meant to go into the last modes in the standard order, we need to swap indices around
    if modes != list(range(num_modes-len(modes), num_modes)):
        mode_permutation = [x for x in range(num_modes) if x not in modes] + modes
        revised_modes = reorder_modes(revised_modes, mode_permutation, revised_modes_pure, batched)

    return revised_modes

def reorder_modes(state, mode_permutation, pure, batched):
    """Reorder the indices of states according to the given mode_permutation, needs to know whether the state is pure and/or batched."""
    if pure:
        scale = 1
        index_permutation = mode_permutation
    else:
        scale = 2
        index_permutation = [scale*x+i for x in mode_permutation for i in (0, 1)] #two indices per mode if we have pure states

    if batched:
        index_permutation = [0] + [i+1 for i in index_permutation]

    index_permutation = np.argsort(index_permutation)

    reordered_modes = tf.transpose(state, index_permutation)

    return reordered_modes

def insert_state(state, system, state_is_pure, mode=None, batched=False):
    """
    Append a new mode (at slot 'mode') to system and initialize it in 'state'.
    If 'mode' is not specified or is greater than the largest current mode number, the new mode is added to the end.
    If an integer within [0,...,N-1] is given for 'mode' (where N is the number of modes) in 'system,'
    then the new state is put in the corresponding mode, and all following modes are shifted to the right by one.
    """
    # pylint: disable=too-many-branches
    num_indices = len(system.shape)
    if batched:
        batch_offset = 1
    else:
        batch_offset = 0
    if state_is_pure:
        num_modes = num_indices - batch_offset
    else:
        num_modes = (num_indices - batch_offset) // 2

    if mode is None or mode >= num_modes:
        mode = num_modes

    if len(system.shape) == 0: # pylint: disable=len-as-condition
        # no modes in system
        # pylint: disable=no-else-return
        if len(state.shape) - batch_offset == 1:
            if state_is_pure:
                return state
            else:
                return mixed(state)
        elif len(state.shape) - batch_offset == 2:
            return state
        else:
            raise ValueError("'state' must have dim={} or dim={}".format(1 - batch_offset, 2 - batch_offset))
    else:
        # modes in system
        if len(state.shape) == batch_offset + 1 and state_is_pure:
            # everything is pure
            # basic form:
            # 'ab...ln...yz,m->ab...lmn...yz' ('m' indices belong to bra of mode being inserted)
            mode_size = 1
        else:
            # everything is mixed
            # basic form:
            # 'abcd...klop...wxyz,mn->abcd...klmnop...wxyz' ('mn' indices belong to bra/ket of mode being inserted)
            mode_size = 2

        batch_index = indices[:batch_offset]
        left_part = indices[batch_offset : batch_offset + mode * mode_size]
        middle_part = indices[batch_offset + mode * mode_size : batch_offset + (mode + 1) * mode_size]
        right_part = indices[batch_offset + (mode + 1) * mode_size : batch_offset + (num_modes + 1) * mode_size]
        eqn_lhs = batch_index + left_part + right_part + "," + batch_index + middle_part
        eqn_rhs = batch_index + left_part + middle_part + right_part
        eqn = eqn_lhs + '->' + eqn_rhs
        revised_modes = tf.einsum(eqn, system, state)

    return revised_modes

def partial_trace(system, mode, state_is_pure, batched=False):
    """
    Trace out subsystem 'mode' from 'system'.
    This operation always returns a mixed state, since we do not know in advance if a mode is entangled with others.
    """
    num_indices = len(system.shape)
    if batched:
        batch_offset = 1
    else:
        batch_offset = 0
    if state_is_pure:
        num_modes = num_indices - batch_offset
    else:
        num_modes = (num_indices - batch_offset) // 2

    if state_is_pure:
        system = mixed(system, batched)

    # tensorflow trace implementation
    # requires subsystem to be traced out to be at end
    # i.e., ab...klmnop...yz goes to ab...klop...yzmn
    indices_list = [m for m in range(batch_offset + 2 * num_modes)]
    dim_list = indices_list[ : batch_offset + 2 * mode] + indices_list[batch_offset + 2 * (mode + 1):] + indices_list[batch_offset + 2 * mode: batch_offset + 2 * (mode + 1)]
    permuted_sys = tf.transpose(system, dim_list)
    reduced_state = tf.trace(permuted_sys)
    return reduced_state

def reduced_density_matrix(system, mode, state_is_pure, batched=False):
    """
    Trace out all subsystems except 'mode' from 'system'.
    This operation always returns a mixed state, since we do not know in advance if a mode is entangled with others.
    """
    if state_is_pure:
        reduced_state = mixed(system, batched)
    else:
        reduced_state = system
    num_indices = len(reduced_state.shape)
    if batched:
        batch_offset = 1
    else:
        batch_offset = 0
    num_modes = (num_indices - batch_offset) // 2 # always mixed
    for m in range(num_modes):
        if m != mode:
            reduced_state = partial_trace(reduced_state, m, False, batched)
    return reduced_state

def conditional_state(system, projector, mode, state_is_pure, batched=False):
    """Compute the (unnormalized) conditional state of 'system' after applying ket 'projector' to 'mode'."""
    # basic_form (pure states): abc...ijk...xyz,j-> abc...ik...xyz
    # basic_form (mixed states): abcd...ijklmn...wxyz,k,l-> abcd...ijmn...wxyz
    mode = int(mode)
    if not batched:
        # add in fake batch dimension (TF 1.5 gives unexplainable errors if we do it without this)
        system = tf.expand_dims(system, 0)
        projector = tf.expand_dims(projector, 0)
    num_indices = len(system.shape)
    batch_offset = 1
    if state_is_pure:
        mode_size = 1
    else:
        mode_size = 2
    num_modes = (num_indices - batch_offset) // mode_size
    max_num = (max_num_indices - batch_offset) // num_modes
    if num_modes > max_num:
        raise ValueError("Conditional state projection currently only supported for {} modes.".format(max_num))

    batch_index = indices[:batch_offset]
    mode_indices = indices[batch_offset : batch_offset + num_modes * mode_size]
    projector_indices = mode_indices[:mode_size]
    free_mode_indices = mode_indices[mode_size : num_modes * mode_size]
    state_lhs = batch_index + free_mode_indices[:mode * mode_size] + projector_indices + free_mode_indices[mode * mode_size:]
    projector_lhs = batch_index + projector_indices[0]
    if mode_size == 2:
        projector_lhs += "," + batch_index + projector_indices[1]
    eqn_lhs = ",".join([state_lhs, projector_lhs])
    eqn_rhs = batch_index + free_mode_indices
    eqn = eqn_lhs + '->' + eqn_rhs
    einsum_args = [system, tf.conj(projector)]
    if not state_is_pure:
        einsum_args.append(projector)
    cond_state = tf.einsum(eqn, *einsum_args)
    if not batched:
        cond_state = tf.squeeze(cond_state, 0) # drop fake batch dimension
    return cond_state

###################################################################

# Helpful auxiliary functions for ops

def ladder_ops(D):
    """returns the matrix representation of the annihilation and creation operators"""
    ind = [(i - 1, i) for i in range(1, D)]
    updates = [np.sqrt(i) for i in range(1, D)]
    shape = [D, D]
    a = tf.scatter_nd(ind, updates, shape)
    ad = tf.transpose(tf.conj(a), [1, 0])
    return a, ad

def H(n, x):
    """Explicit expression for Hermite polynomials."""
    prefactor = factorial(n)
    terms = tf.reduce_sum(tf.stack([_numer_safe_power(-1, m) / (factorial(m) * factorial(n - 2 * m)) *
                                    _numer_safe_power(2 * x, n - 2 * m)
                                    for m in range(int(np.floor(n / 2)) + 1)], axis=0), axis=0)
    return prefactor * terms

def H_n_plus_1(H_n, H_n_m1, n, x):
    """Recurrent definition of Hermite polynomials."""
    H_n_p1 = 2 * x * H_n - 2 * n * H_n_m1
    return H_n_p1

###################################################################

# Helper functions

def _check_for_eval(kwargs):
    """
    Helper function to check keyword arguments for user-supplied information about how numerical evaluation should proceed,
    what session to use, and what the feed dictionary should be.
    """
    if "feed_dict" in kwargs:
        feed_dict = kwargs["feed_dict"]
    else:
        feed_dict = {}
    if "session" in kwargs:
        # evaluate tensors in supplied session
        session = kwargs["session"]
        evaluate_results = True
        close_session = False
    elif "eval" in kwargs and kwargs["eval"] is False:
        # don't evaluate tensors
        session = None
        evaluate_results = False
        close_session = False
    else:
        # evaluate tensors in temporary session
        session = tf.Session()
        evaluate_results = True
        close_session = True
    return evaluate_results, session, feed_dict, close_session

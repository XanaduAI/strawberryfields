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
import warnings

from string import ascii_lowercase as indices
from string import ascii_letters as indices_full

import tensorflow as tf
import numpy as np
from scipy.special import factorial
from scipy.linalg import expm

from thewalrus.fock_gradients import displacement as displacement_tw
from thewalrus.fock_gradients import grad_displacement as grad_displacement_tw
from thewalrus.fock_gradients import squeezing as squeezing_tw
from thewalrus.fock_gradients import grad_squeezing as grad_squeezing_tw
from thewalrus.fock_gradients import beamsplitter as beamsplitter_tw
from thewalrus.fock_gradients import grad_beamsplitter as grad_beamsplitter_tw
from thewalrus.fock_gradients import mzgate as mzgate_tw
from thewalrus.fock_gradients import grad_mzgate as grad_mzgate_tw
from thewalrus.fock_gradients import two_mode_squeezing as two_mode_squeezing_tw
from thewalrus.fock_gradients import grad_two_mode_squeezing as grad_two_mode_squeezing_tw
from thewalrus._hermite_multidimensional import hermite_multidimensional as gaussian_gate_tw
from thewalrus._hermite_multidimensional import (
    grad_hermite_multidimensional as grad_gaussian_gate_tw,
)
from thewalrus.symplectic import is_symplectic, sympmat

# With TF 2.1+, the legacy tf.einsum was renamed to _einsum_v1, while
# the replacement tf.einsum introduced the bug. This try-except block
# will dynamically patch TensorFlow versions where _einsum_v1 exists, to make it the
# default einsum implementation.
#
# For more details, see https://github.com/tensorflow/tensorflow/issues/37307
try:
    from tensorflow.python.ops.special_math_ops import _einsum_v1

    tf.einsum = _einsum_v1
except ImportError:
    pass

max_num_indices = len(indices)

###################################################################

# Warning message format


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """User warning formatter"""
    # pylint: disable=unused-argument
    return "{}:{}: {}: {}\n".format(filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


class WarnOnlyOnce:
    """Warning class to only warn the user once per program"""

    warnings = set()

    @classmethod
    def warn(cls, message):
        """Redefine the warn function to check if the warning message has already displayed"""
        h = hash(message)
        if h not in cls.warnings:
            warnings.warn(message)
            cls.warnings.add(h)


###################################################################

# Helper functions:
def _numer_safe_power(base, exponent, dtype=tf.complex64):
    """gives the desired behaviour of 0**0=1"""
    if exponent == 0:
        return tf.ones_like(base, dtype)

    return base ** exponent


def mix(pure_state, batched=False):
    """Converts the state from pure to mixed"""
    if not batched:
        pure_state = tf.expand_dims(pure_state, 0)  # add in fake batch dimension

    batch_offset = 1
    num_modes = len(pure_state.shape) - batch_offset
    max_num = (max_num_indices - batch_offset) // 2

    if num_modes > max_num:
        raise ValueError(
            "Converting state from pure to mixed currently only supported for {} modes.".format(
                max_num
            )
        )

    # eqn: 'abc...xyz,ABC...XYZ->aAbBcC...xXyYzZ' (lowercase belonging to 'bra' side, uppercase belonging to 'ket' side)
    batch_index = indices[:batch_offset]
    bra_indices = indices[batch_offset : batch_offset + num_modes]
    ket_indices = indices[batch_offset + num_modes : batch_offset + 2 * num_modes]
    eqn_lhs = batch_index + bra_indices + "," + batch_index + ket_indices
    eqn_rhs = "".join(bdx + kdx for bdx, kdx in zip(bra_indices, ket_indices))
    eqn = eqn_lhs + "->" + batch_index + eqn_rhs
    mixed_state = tf.einsum(eqn, pure_state, tf.math.conj(pure_state))

    if not batched:
        mixed_state = tf.squeeze(mixed_state, 0)  # drop fake batch dimension
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
    strides = tf.math.cumprod(tensor_shape, reverse=True)
    strides_shifted = tf.math.cumprod(tensor_shape, exclusive=True, reverse=True)
    unraveled_coords = (ind % strides) // strides_shifted
    return tf.transpose(unraveled_coords)


###################################################################

# Matrices:
# These return a matrix of shape [cutoff, cutoff]
# which transforms from the input (truncated) Fock basis to the
# output (truncated) Fock basis.


def squeezed_vacuum_vector(r, theta, cutoff, batched=False, eps=1e-32, dtype=tf.complex64):
    """returns the ket representing a single mode squeezed vacuum state"""
    if batched:
        batch_size = r.shape[0]
    r = tf.cast(r, dtype)
    theta = tf.cast(theta, dtype)
    c1 = tf.cast(
        tf.stack(
            [
                tf.sqrt(1 / tf.cosh(r)) * np.sqrt(factorial(k)) / factorial(k / 2.0)
                for k in range(0, cutoff, 2)
            ],
            axis=-1,
        ),
        dtype,
    )
    c2 = tf.stack(
        [
            (-0.5 * tf.exp(1j * theta) * tf.cast(tf.tanh(r + eps), dtype)) ** (k / 2.0)
            for k in range(0, cutoff, 2)
        ],
        axis=-1,
    )
    even_coeffs = c1 * c2
    ind = [(k,) for k in np.arange(0, cutoff, 2)]
    shape = [cutoff]
    if batched:
        ind = batchify_indices(ind, batch_size)
        shape = [batch_size] + shape
    output = tf.scatter_nd(ind, tf.reshape(even_coeffs, [-1]), shape)
    return output


@tf.custom_gradient
def single_squeezing_matrix(r, phi, cutoff, dtype=tf.complex64.as_numpy_dtype):
    """creates a single-mode squeezing matrix"""
    r = r.numpy()
    phi = phi.numpy()
    gate = squeezing_tw(r, phi, cutoff, dtype)

    # NOTE: tested independently in thewalrus
    def grad(dy):  # pragma: no cover
        Dr, Dphi = grad_squeezing_tw(gate, r, phi)
        grad_r = tf.math.real(tf.reduce_sum(dy * tf.math.conj(Dr)))
        grad_phi = tf.math.real(tf.reduce_sum(dy * tf.math.conj(Dphi)))
        return grad_r, grad_phi, None

    return gate, grad


def squeezer_matrix(r, phi, cutoff, batched=False, dtype=tf.complex64):
    """creates a single mode squeezing matrix accounting for batching"""
    r = tf.cast(r, dtype)
    phi = tf.cast(phi, dtype)
    if batched:
        return tf.stack(
            [
                single_squeezing_matrix(r_, phi_, cutoff, dtype=dtype.as_numpy_dtype)
                for r_, phi_ in tf.transpose([r, phi])
            ]
        )
    return single_squeezing_matrix(r, phi, cutoff, dtype=dtype.as_numpy_dtype)


def phase_shifter_matrix(theta, cutoff, batched=False, dtype=tf.complex64):
    """creates the single mode phase shifter matrix"""
    if batched:
        batch_size = theta.shape[0]
    theta = tf.cast(theta, dtype)
    shape = [cutoff, cutoff]
    if batched:
        shape = [batch_size] + shape
    zero_matrix = tf.zeros(shape=shape, dtype=dtype)
    diag = [tf.exp(1j * theta * k) for k in np.arange(cutoff, dtype=np.complex64)]
    if batched:
        diag = tf.stack(diag, axis=1)
    diag_matrix = tf.linalg.set_diag(zero_matrix, diag)
    return diag_matrix


def kerr_interaction_matrix(kappa, cutoff, batched=False):
    """creates the single mode Kerr interaction matrix"""
    coeffs = [tf.exp(1j * kappa * n ** 2) for n in range(cutoff)]
    if batched:
        coeffs = tf.stack(coeffs, axis=1)
    output = tf.linalg.diag(coeffs)
    return output


def cross_kerr_interaction_matrix(kappa, cutoff, batched=False):
    """creates the two mode cross-Kerr interaction matrix"""
    coeffs = [tf.exp(1j * kappa * n1 * n2) for n1 in range(cutoff) for n2 in range(cutoff)]
    if batched:
        coeffs = tf.stack(coeffs, axis=1)
    output = tf.linalg.diag(coeffs)
    if batched:
        output = tf.transpose(tf.reshape(output, [-1] + [cutoff] * 4), [0, 1, 3, 2, 4])
    else:
        output = tf.transpose(tf.reshape(output, [cutoff] * 4), [0, 2, 1, 3])
    return output


def cubic_phase_matrix(
    gamma, cutoff, hbar, batched=False, method="self_adjoint_eig", dtype=tf.complex64
):
    """creates the single mode cubic phase matrix"""
    a, ad = ladder_ops(cutoff)
    x = np.sqrt(hbar / 2) * tf.cast(a + ad, dtype)
    x3 = x @ x @ x
    if batched:
        x3 = tf.expand_dims(x3, 0)
        gamma = tf.reshape(gamma, [-1, 1, 1])
    H0 = gamma / (3 * hbar) * x3
    lambdas, U = tf.linalg.eigh(H0)
    transpose_list = [1, 0]
    if batched:
        transpose_list = [0, 2, 1]
    if method == "self_adjoint_eig":
        # This seems to work as long as the matrix has dimension 18x18 or smaller
        # For larger matrices, TensorFlow returns the error: 'Self-adjoint eigen decomposition was not successful.'
        V = U @ tf.linalg.diag(tf.exp(1j * lambdas)) @ tf.math.conj(tf.transpose(U, transpose_list))
    # below version works for matrices larger than 18x18, but
    # expm was not added until TF v1.5, while
    # as of TF v1.5, gradient of expm is not implemented
    # elif method == "expm":
    #    V = tf.linalg.expm(1j * H0)
    else:
        raise ValueError("'method' must be either 'self_adjoint_eig' or 'expm'.")
    return V


def loss_superop(T, cutoff, batched=False, dtype=tf.complex64):
    """creates the single mode loss channel matrix"""
    # indices are abcd, corresponding to action |a><b| (.) |c><d|
    if not batched:
        T = tf.expand_dims(T, 0)  # introduce artificial batch dimension
    T = tf.reshape(T, [-1, 1, 1, 1, 1, 1])
    T = tf.cast(T, tf.float64)
    rng = np.arange(cutoff)
    a = np.reshape(rng, [-1, cutoff, 1, 1, 1, 1])
    b = np.reshape(rng, [-1, 1, cutoff, 1, 1, 1])
    c = np.reshape(rng, [-1, 1, 1, cutoff, 1, 1])
    d = np.reshape(rng, [-1, 1, 1, 1, cutoff, 1])
    l = np.reshape(rng, [-1, 1, 1, 1, 1, cutoff])
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
    output = tf.cast(tf.reduce_sum(l_terms, -1), dtype)
    if not batched:
        output = tf.squeeze(output, 0)  # drop artificial batch dimension
    return output


@tf.custom_gradient
def single_displacement_matrix(r, phi, cutoff, dtype=tf.complex64.as_numpy_dtype):
    """creates a single mode displacement matrix"""
    r = r.numpy()
    phi = phi.numpy()
    gate = displacement_tw(r, phi, cutoff, dtype)

    # NOTE: tested when testing the gate and its gradients; also tested independently in thewalrus
    def grad(dy):  # pragma: no cover
        Dr, Dphi = grad_displacement_tw(gate, r, phi)
        grad_r = tf.math.real(tf.reduce_sum(dy * tf.math.conj(Dr)))
        grad_phi = tf.math.real(tf.reduce_sum(dy * tf.math.conj(Dphi)))
        return grad_r, grad_phi, None

    return gate, grad


def displacement_matrix(r, phi, cutoff, batched=False, dtype=tf.complex64):
    """creates a single mode displacement matrix accounting for batching"""
    r = tf.cast(r, dtype)
    phi = tf.cast(phi, dtype)
    if batched:
        return tf.stack(
            [
                single_displacement_matrix(r_, phi_, cutoff, dtype=dtype.as_numpy_dtype)
                for r_, phi_ in tf.transpose([r, phi])
            ]
        )
    return single_displacement_matrix(r, phi, cutoff, dtype=dtype.as_numpy_dtype)


@tf.custom_gradient
def single_beamsplitter_matrix(theta, phi, cutoff, dtype=tf.complex64.as_numpy_dtype):
    """creates a single mode beamsplitter matrix"""
    theta = theta.numpy()
    phi = phi.numpy()

    gate = beamsplitter_tw(theta, phi, cutoff, dtype)
    gate = np.transpose(gate, [0, 2, 1, 3])

    def grad(dy):
        Dtheta, Dphi = grad_beamsplitter_tw(np.transpose(gate, [0, 2, 1, 3]), theta, phi)
        Dtheta = np.transpose(Dtheta, [0, 2, 1, 3])
        Dphi = np.transpose(Dphi, [0, 2, 1, 3])
        grad_theta = tf.math.real(tf.reduce_sum(dy * tf.math.conj(Dtheta)))
        grad_phi = tf.math.real(tf.reduce_sum(dy * tf.math.conj(Dphi)))
        return grad_theta, grad_phi, None

    return gate, grad


def beamsplitter_matrix(theta, phi, cutoff, batched=False, dtype=tf.complex64):
    """creates a single mode beamsplitter matrix accounting for batching"""
    theta = tf.cast(theta, dtype)
    phi = tf.cast(phi, dtype)
    if batched:
        return tf.stack(
            [
                single_beamsplitter_matrix(theta_, phi_, cutoff, dtype=dtype.as_numpy_dtype)
                for theta_, phi_ in tf.transpose([theta, phi])
            ]
        )
    return tf.convert_to_tensor(
        single_beamsplitter_matrix(theta, phi, cutoff, dtype=dtype.as_numpy_dtype)
    )


@tf.custom_gradient
def single_mzgate_matrix(phi_in, phi_ex, cutoff, dtype=tf.complex64.as_numpy_dtype):
    """creates a single mode mzgate matrix"""
    phi_in = phi_in.numpy()
    phi_ex = phi_ex.numpy()

    gate = mzgate_tw(phi_in, phi_ex, cutoff, dtype)
    gate = np.transpose(gate, [0, 2, 1, 3])

    def grad(dy):
        Dtheta, Dphi = grad_mzgate_tw(np.transpose(gate, [0, 2, 1, 3]), phi_in, phi_ex)
        Dtheta = np.transpose(Dtheta, [0, 2, 1, 3])
        Dphi = np.transpose(Dphi, [0, 2, 1, 3])
        grad_theta = tf.math.real(tf.reduce_sum(dy * tf.math.conj(Dtheta)))
        grad_phi = tf.math.real(tf.reduce_sum(dy * tf.math.conj(Dphi)))
        return grad_theta, grad_phi, None

    return gate, grad


def mzgate_matrix(phi_in, phi_ex, cutoff, batched=False, dtype=tf.complex64):
    """creates a single mode mzgate matrix accounting for batching"""
    phi_in = tf.cast(phi_in, dtype)
    phi_ex = tf.cast(phi_ex, dtype)
    if batched:
        return tf.stack(
            [
                single_mzgate_matrix(phi_in_, phi_ex_, cutoff, dtype=dtype.as_numpy_dtype)
                for phi_in_, phi_ex_ in tf.transpose([phi_in, phi_ex])
            ]
        )
    return tf.convert_to_tensor(
        single_mzgate_matrix(phi_in, phi_ex, cutoff, dtype=dtype.as_numpy_dtype)
    )


@tf.custom_gradient
def single_two_mode_squeezing_matrix(theta, phi, cutoff, dtype=tf.complex64.as_numpy_dtype):
    """creates a single mode two-mode squeezing matrix"""
    theta = theta.numpy()
    phi = phi.numpy()

    gate = two_mode_squeezing_tw(theta, phi, cutoff, dtype)
    gate = np.transpose(gate, [0, 2, 1, 3])

    def grad(dy):
        Dr, Dphi = grad_two_mode_squeezing_tw(np.transpose(gate, [0, 2, 1, 3]), theta, phi)
        Dr = np.transpose(Dr, [0, 2, 1, 3])
        Dphi = np.transpose(Dphi, [0, 2, 1, 3])
        grad_r = tf.math.real(tf.reduce_sum(dy * tf.math.conj(Dr)))
        grad_phi = tf.math.real(tf.reduce_sum(dy * tf.math.conj(Dphi)))
        return grad_r, grad_phi, None

    return gate, grad


def two_mode_squeezer_matrix(theta, phi, cutoff, batched=False, dtype=tf.complex64):
    """creates a single mode two-mode squeezing matrix accounting for batching"""
    theta = tf.cast(theta, dtype)
    phi = tf.cast(phi, dtype)
    if batched:
        return tf.stack(
            [
                single_two_mode_squeezing_matrix(theta_, phi_, cutoff, dtype=dtype.as_numpy_dtype)
                for theta_, phi_ in tf.transpose([theta, phi])
            ]
        )
    return tf.convert_to_tensor(
        single_two_mode_squeezing_matrix(theta, phi, cutoff, dtype=dtype.as_numpy_dtype)
    )


def choi_trick(S, d, dtype=tf.complex128):
    """Transforms the parameter from S,d to R, y, C corresponding to the parameters of the multidimensional hermite polynomials"""
    S = tf.cast(S, dtype=dtype)
    d = tf.cast(d, dtype=dtype)
    m = S.shape[0] // 2
    alpha = (d[:m] + 1j * d[m:]) / 2
    choi_r = np.arcsinh(1.0)
    ch = np.diag([np.cosh(choi_r)] * m)
    sh = np.diag([np.sinh(choi_r)] * m)
    zh = np.zeros([m, m])
    Schoi = np.block([[ch, sh, zh, zh], [sh, ch, zh, zh], [zh, zh, ch, -sh], [zh, zh, -sh, ch]])
    Schoi = tf.convert_to_tensor(Schoi, dtype=dtype)
    zh = tf.zeros([m, m], dtype=dtype)
    Sxx = S[:m, :m]
    Sxp = S[:m, m:]
    Spx = S[m:, :m]
    Spp = S[m:, m:]
    idl = tf.eye(m, dtype=dtype)
    S_exp = (
        tf.concat(
            [
                tf.concat([Sxx, zh, Sxp, zh], 1),
                tf.concat([zh, idl, zh, zh], 1),
                tf.concat([Spx, zh, Spp, zh], 1),
                tf.concat([zh, zh, zh, idl], 1),
            ],
            0,
        )
        @ Schoi
    )
    choi_cov = 0.5 * S_exp @ tf.transpose(S_exp)
    idl = tf.eye(2 * m, dtype=dtype)
    Rot = tf.cast(tf.math.sqrt(0.5), dtype=dtype) * tf.concat(
        [tf.concat([idl, 1j * idl], 1), tf.concat([idl, -1j * idl], 1)], 0
    )
    sigma = Rot @ choi_cov @ tf.transpose(tf.math.conj(Rot))
    zh = tf.zeros([2 * m, 2 * m], dtype=dtype)
    X = tf.concat([tf.concat([zh, idl], 1), tf.concat([idl, zh], 1)], 0)
    sigma_Q = sigma + 0.5 * tf.eye(4 * m, dtype=dtype)
    A_mat = X @ (tf.eye(4 * m, dtype=dtype) - tf.linalg.inv(sigma_Q))
    choi_r = tf.cast(tf.math.asinh(1.0), dtype=dtype)
    E = tf.linalg.diag(
        tf.concat([tf.ones([m], dtype=dtype), tf.ones([m], dtype=dtype) / tf.math.tanh(choi_r)], 0)
    )
    R = -tf.math.conj(E @ A_mat[: 2 * m, : 2 * m] @ E)
    y = tf.concat(
        [
            tf.linalg.matvec(R[:m, :m], tf.math.conj(alpha)) + tf.transpose(alpha),
            tf.linalg.matvec(R[m:, :m], tf.math.conj(alpha)),
        ],
        0,
    )
    alpha_tilt = tf.concat([tf.cast(alpha, dtype=dtype), tf.zeros(m, dtype=dtype)], 0)
    zeta = alpha_tilt + tf.linalg.matvec(tf.cast(R, dtype=dtype), tf.math.conj(alpha_tilt))
    C = tf.math.sqrt(
        tf.math.sqrt(tf.linalg.det(tf.eye(m, dtype=dtype) - R[:m, :m] @ tf.math.conj(R[:m, :m])))
    ) * tf.exp(-0.5 * tf.reduce_sum(tf.math.conj(alpha_tilt) * zeta))
    return R, y, C


@tf.custom_gradient
def single_gaussian_gate_matrix(R, y, C, cutoff, dtype=tf.complex128):
    """creates a N-mode gaussian gate matrix"""
    gate = tf.numpy_function(gaussian_gate_tw, [R, cutoff, y, C, True, True, True], tf.complex128)
    N = y.shape[-1] // 2
    transpose_list = [x for pair in zip(range(N), range(N, 2 * N)) for x in pair]

    def grad(dL_dG_conj):
        WarnOnlyOnce.warn(
            "Warning: gradients of a symplectic matrix cannot be used for gradient descent directly. Use sf.backends.tfbackend.update_symplectic(S, gradS) for the optimization step."
        )
        dL_dG_conj = tf.transpose(dL_dG_conj, [transpose_list.index(i) for i in range(2 * N)])
        dG_dC, dG_dR, dG_dy = tf.numpy_function(
            grad_gaussian_gate_tw, [gate, R, y, C], [tf.complex128] * 3
        )
        dL_dC_conj = tf.reduce_sum(dL_dG_conj * tf.math.conj(dG_dC))
        dL_dy_conj = tf.reduce_sum(
            dL_dG_conj[..., None] * tf.math.conj(dG_dy), axis=list(range(len(dL_dG_conj.shape)))
        )
        dL_dR_conj = tf.reduce_sum(
            dL_dG_conj[..., None, None] * tf.math.conj(dG_dR),
            axis=list(range(len(dL_dG_conj.shape))),
        )
        return dL_dR_conj, dL_dy_conj, dL_dC_conj, None

    return tf.cast(tf.transpose(gate, transpose_list), dtype), grad


def update_symplectic(S, dS, lr):
    """returns the updated syplectic matrix S according to its geodesic.
    S (Tensor): symplectic matrix to be updated.
    dS (Tensor): euclidean gradient of S.
    lr (float): learning rate.
    """
    Jmat = sympmat(S.shape[1] // 2)
    Z = np.matmul(np.transpose(S), dS)
    Y = 0.5 * (Z + np.linalg.multi_dot([Jmat, Z.T, Jmat]))
    S.assign(S @ expm(-lr * np.transpose(Y)) @ expm(-lr * (Y - np.transpose(Y))), read_value=False)


def gaussian_gate_matrix(S, d, cutoff: int, batched=False, dtype=tf.complex128):
    """creates a N-mode gaussian gate matrix accounting for batching"""
    S = tf.cast(S, dtype)
    d = tf.cast(d, dtype)
    if batched:
        return tf.stack(
            [
                single_gaussian_gate_matrix(*choi_trick(S_, d_), cutoff, dtype=dtype.as_numpy_dtype)
                for S_, d_ in zip(S, d)
            ]
        )
    return tf.convert_to_tensor(
        single_gaussian_gate_matrix(
            *choi_trick(S, d, dtype=dtype), cutoff, dtype=dtype.as_numpy_dtype
        )
    )


###################################################################

# Input states:
# Creates input states on a single mode


def fock_state(n, cutoff, pure=True, batched=False, dtype=tf.complex64):
    """creates a single mode input Fock state"""
    if not isinstance(n, (np.ndarray, int)):
        raise ValueError("'n' is expected to be either an int or a numpy array")
    if batched:
        batch_size = n.shape[0]
        idxs = list(zip(range(batch_size), n))
        values = [1.0] * batch_size
        shape = [batch_size, cutoff]
    else:
        idxs = [(n,)]
        values = [1.0]
        shape = [cutoff]
    fock_sparse = tf.scatter_nd(idxs, values, shape)
    fock = tf.cast(fock_sparse, dtype)
    if not pure:
        fock = mix(fock, batched)
    return fock


def coherent_state(r, phi, cutoff, pure=True, batched=False, dtype=tf.complex64):
    """creates a single mode input coherent state"""
    alpha = tf.cast(r, dtype) * tf.exp(1j * tf.cast(phi, dtype))
    coh = tf.stack(
        [
            tf.cast(tf.exp(-0.5 * tf.abs(r) ** 2), dtype)
            * _numer_safe_power(alpha, n, dtype)
            / tf.cast(np.sqrt(factorial(n)), dtype)
            for n in range(cutoff)
        ],
        axis=-1,
    )
    if not pure:
        coh = mix(coh, batched)
    return coh


def squeezed_vacuum(r, theta, cutoff, pure=True, batched=False, dtype=tf.complex64):
    """creates a single mode input squeezed vacuum state"""
    squeezed = squeezed_vacuum_vector(r, theta, cutoff, batched=batched, dtype=dtype)
    if not pure:
        squeezed = mix(squeezed, batched)
    return squeezed


def displaced_squeezed(
    r_d, phi_d, r_s, phi_s, cutoff, pure=True, batched=False, eps=1e-12, dtype=tf.complex64
):
    """creates a single mode input displaced squeezed state"""
    alpha = tf.cast(r_d, dtype) * tf.exp(1j * tf.cast(phi_d, dtype))
    r_s = (
        tf.cast(r_s, dtype) + eps
    )  # to prevent nans if r==0, we add an epsilon (default is miniscule)
    phi_s = tf.cast(phi_s, dtype)

    phase = tf.exp(1j * phi_s)
    sinh = tf.sinh(r_s)
    cosh = tf.cosh(r_s)
    tanh = tf.tanh(r_s)

    # create Hermite polynomials
    gamma = alpha * cosh + tf.math.conj(alpha) * phase * sinh
    hermite_arg = gamma / tf.sqrt(phase * tf.sinh(2 * r_s))

    prefactor = tf.expand_dims(
        tf.exp(-0.5 * alpha * tf.math.conj(alpha) - 0.5 * tf.math.conj(alpha) ** 2 * phase * tanh),
        -1,
    )
    coeff = tf.stack(
        [
            _numer_safe_power(0.5 * phase * tanh, n / 2.0, dtype) / tf.sqrt(factorial(n) * cosh)
            for n in range(cutoff)
        ],
        axis=-1,
    )
    hermite_terms = tf.stack(
        [tf.cast(H(n, hermite_arg, dtype), dtype) for n in range(cutoff)], axis=-1
    )
    squeezed_coh = prefactor * coeff * hermite_terms

    if not pure:
        squeezed_coh = mix(squeezed_coh, batched)
    return squeezed_coh


def thermal_state(nbar, cutoff, dtype=tf.complex64):
    """creates a single mode input thermal state.
    Note that the batch dimension is determined by argument nbar.
    """
    nbar = tf.cast(nbar, dtype)
    coeffs = tf.stack(
        [
            _numer_safe_power(nbar, n, dtype) / _numer_safe_power(nbar + 1, n + 1, dtype)
            for n in range(cutoff)
        ],
        axis=-1,
    )
    thermal = tf.linalg.diag(coeffs)
    return thermal


##################################################################

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
    left_gate_str = indices[batch_offset : batch_offset + 2]  # |a><b|
    num_indices = len(in_modes.shape)
    if pure:
        num_modes = num_indices - batch_offset
        mode_size = 1
    else:
        right_gate_str = indices[batch_offset + 2 : batch_offset + 4]  # |c><d|
        num_modes = (num_indices - batch_offset) // 2
        mode_size = 2
    max_len = len(indices) - 2 * mode_size - batch_offset
    if num_modes == 0:
        raise ValueError("'in_modes' must have at least one mode")
    if num_modes > max_len:
        raise NotImplementedError(
            "The max number of supported modes for this operation is currently {}".format(max_len)
        )
    if mode < 0 or mode >= num_modes:
        raise ValueError("'mode' argument is not compatible with number of in_modes")

    other_modes_indices = indices[
        batch_offset + 2 * mode_size : batch_offset + (1 + num_modes) * mode_size
    ]
    if pure:
        eqn_lhs = "{},{}{}{}{}".format(
            batch_index + left_gate_str,
            batch_index,
            other_modes_indices[: mode * mode_size],
            left_gate_str[1],
            other_modes_indices[mode * mode_size :],
        )
        eqn_rhs = "".join(
            [
                batch_index,
                other_modes_indices[: mode * mode_size],
                left_gate_str[0],
                other_modes_indices[mode * mode_size :],
            ]
        )
    else:
        eqn_lhs = "{},{}{}{}{}{},{}".format(
            batch_index + left_gate_str,
            batch_index,
            other_modes_indices[: mode * mode_size],
            left_gate_str[1],
            right_gate_str[0],
            other_modes_indices[mode * mode_size :],
            batch_index + right_gate_str,
        )
        eqn_rhs = "".join(
            [
                batch_index,
                other_modes_indices[: mode * mode_size],
                left_gate_str[0],
                right_gate_str[1],
                other_modes_indices[mode * mode_size :],
            ]
        )

    eqn = eqn_lhs + "->" + eqn_rhs
    einsum_inputs = [matrix, in_modes]
    if not pure:
        transposed_axis = [0, 2, 1] if batched else [1, 0]
        einsum_inputs.append(tf.transpose(tf.math.conj(matrix), transposed_axis))
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
    left_gate_str = indices[batch_offset : batch_offset + 4]  # |a><b| |c><d|
    num_indices = len(in_modes.shape)
    if pure:
        num_modes = num_indices - batch_offset
        mode_size = 1
    else:
        right_gate_str = indices[batch_offset + 4 : batch_offset + 8]  # |e><f| |g><h|
        num_modes = (num_indices - batch_offset) // 2
        mode_size = 2
    max_len = (len(indices) - 4) // mode_size - batch_offset

    if num_modes == 0:
        raise ValueError("'in_modes' must have at least one mode")
    if num_modes > max_len:
        raise NotImplementedError(
            "The max number of supported modes for this operation is currently {}".format(max_len)
        )

    min_mode = min(mode1, mode2)
    max_mode = max(mode1, mode2)
    if min_mode < 0 or max_mode >= num_modes or mode1 == mode2:
        raise ValueError("One or more mode numbers are incompatible")

    other_modes_indices = indices[
        batch_offset + 4 * mode_size : batch_offset + 4 * mode_size + mode_size * (num_modes - 2)
    ]
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
    eqn_lhs = "{},{}{}{}{}{}{}".format(
        batch_index + left_gate_str,
        batch_index,
        other_modes_indices[: min_mode * mode_size],
        lhs_min_mode_indices,
        other_modes_indices[min_mode * mode_size : (max_mode - 1) * mode_size],
        lhs_max_mode_indices,
        other_modes_indices[(max_mode - 1) * mode_size :],
    )
    if not pure:
        eqn_lhs += "," + batch_index + right_gate_str
    eqn_rhs = "".join(
        [
            batch_index,
            other_modes_indices[: min_mode * mode_size],
            rhs_min_mode_indices,
            other_modes_indices[min_mode * mode_size : (max_mode - 1) * mode_size],
            rhs_max_mode_indices,
            other_modes_indices[(max_mode - 1) * mode_size :],
        ]
    )
    eqn = eqn_lhs + "->" + eqn_rhs
    einsum_inputs = [matrix, in_modes]
    if not pure:
        if batched:
            transpose_list = [0, 2, 1, 4, 3]
        else:
            transpose_list = [1, 0, 3, 2]
        einsum_inputs.append(tf.math.conj(tf.transpose(matrix, transpose_list)))
    output = tf.einsum(eqn, *einsum_inputs)
    return output


def n_mode_gate(matrix, modes, in_modes, pure=True, batched=False):
    """basic form:
    'abcd,efg...b...d...xyz->efg...a...c...xyz' (pure state)
    'abcd,ij...be...dg...xyz,efgh->ij...af...ch...xyz' (mixed state)
    """
    # pylint: disable=too-many-branches,too-many-statements
    # "a": reserved for batching
    # "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ": 51 letters left
    # matrix : out_1 in_1 ... out_n in_n ...
    # modes : Tuple(0,1,2,3,...)
    # in_modes : input state
    in_modes = tf.cast(in_modes, dtype=matrix.dtype)
    if pure:
        num_modes = len(in_modes.shape) - batched
    else:
        num_modes = (len(in_modes.shape) - batched) // 2
    if num_modes == 0:
        raise ValueError("'in_modes' must have at least one mode")
    if len(indices_full) - 2 * num_modes - batched < 0:
        raise NotImplementedError("Exceed the max number of supported modes for this operation")
    if min(modes) < 0 or max(modes) >= num_modes:
        raise ValueError("'mode' argument is not compatible with number of in_modes")
    if pure:
        index_in_modes = indices_full[1 : 1 + num_modes]
        index_in_state = ""
        index_in_state_rest = ""
        for i in range(num_modes):
            if i in modes:
                index_in_state += indices_full[i + 1]
            else:
                index_in_state_rest += indices_full[i + 1]
        num_mode_op = len(modes)
        index_output_op = indices_full[1 + num_modes : 1 + num_modes + num_mode_op]
        index_op = index_output_op + index_in_state
        index_op_new = ""
        for i in range(num_mode_op):
            index_op_new += index_op[i] + index_op[i + num_mode_op]
        index_output = ""
        j = 0
        for i in range(num_modes):
            if i in modes:
                index_output += index_output_op[j]
                j += 1
            else:
                index_output += indices_full[i + 1]
        if batched:
            eqn = "a" + index_op_new + "," + "a" + index_in_modes + "->" + "a" + index_output
        else:
            eqn = index_op_new + "," + index_in_modes + "->" + index_output
    else:
        index_in_modes = indices_full[1 : 1 + 2 * num_modes]
        index_in_state = ""
        index_in_state_rest = ""
        for i in range(num_modes):
            if i in modes:
                index_in_state += indices_full[2 * i + 1] + indices_full[2 * i + 2]
            else:
                index_in_state_rest += indices_full[2 * i + 1] + indices_full[2 * i + 2]
        index_in_state = index_in_state[::2] + index_in_state[1::2]
        num_mode_op = len(modes)
        index_output_op = indices_full[1 + 2 * num_modes : 1 + 2 * num_modes + num_mode_op]
        index_output_op_conj = indices_full[
            1 + 2 * num_modes + num_mode_op : 1 + 2 * num_modes + 2 * num_mode_op
        ]
        index_op = index_output_op + index_in_state[:num_mode_op]
        index_op_conj = index_in_state[num_mode_op:] + index_output_op_conj
        index_op_new = ""
        index_op_conj_new = ""
        for i in range(num_mode_op):
            index_op_new += index_op[i] + index_op[i + num_mode_op]
            index_op_conj_new += index_op_conj[i] + index_op_conj[i + num_mode_op]
        index_output = ""
        j = 0
        for i in range(num_modes):
            if i in modes:
                index_output += index_output_op[j] + index_output_op_conj[j]
                j += 1
            else:
                index_output += indices_full[2 * i + 1] + indices_full[2 * i + 2]
        if batched:
            eqn = (
                "a"
                + index_op_new
                + ","
                + "a"
                + index_in_modes
                + ","
                + "a"
                + index_op_conj_new
                + "->"
                + "a"
                + index_output
            )
        else:
            eqn = (
                index_op_new + "," + index_in_modes + "," + index_op_conj_new + "->" + index_output
            )

    if batched:
        transpose_list = [0]
        N = np.arange(1, len(matrix.shape))
    else:
        transpose_list = []
        N = np.arange(len(matrix.shape))
    for x, y in zip(N[1::2], N[::2]):
        transpose_list.append(x)
        transpose_list.append(y)
    einsum_inputs = [matrix, in_modes]
    if not pure:
        einsum_inputs.append(tf.math.conj(tf.transpose(matrix, transpose_list)))
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
        raise NotImplementedError(
            "The max number of supported modes for this operation is currently {}".format(max_len)
        )

    if pure:
        in_modes = mix(in_modes, batched)

    # create equation
    batch_index = indices[:batch_offset]
    superop_indices = indices[batch_offset : batch_offset + 4]
    state_indices = indices[batch_offset + 4 : batch_offset + 4 + 2 * num_modes]
    left_unchanged_indices = state_indices[: 2 * mode]
    right_unchanged_indices = state_indices[2 * mode : 2 * (num_modes - 1)]
    eqn_lhs = ",".join(
        [
            batch_index + superop_indices,
            batch_index + left_unchanged_indices + superop_indices[1:3] + right_unchanged_indices,
        ]
    )
    eqn_rhs = "".join(
        [
            batch_index,
            left_unchanged_indices
            + superop_indices[0]
            + superop_indices[3]
            + right_unchanged_indices,
        ]
    )
    eqn = "->".join([eqn_lhs, eqn_rhs])
    new_state = tf.einsum(eqn, superop, in_modes)
    return new_state


###################################################################

# Quantum Gate/Channel functions:
# Primary quantum optical gates implemented as transformations on 'in_modes'


def phase_shifter(theta, mode, in_modes, cutoff, pure=True, batched=False, dtype=tf.complex64):
    """returns phase shift unitary matrix on specified input modes"""
    matrix = phase_shifter_matrix(theta, cutoff, batched=batched, dtype=dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure, batched)
    return output


def displacement(r, phi, mode, in_modes, cutoff, pure=True, batched=False, dtype=tf.complex64):
    """returns displacement unitary matrix on specified input modes"""
    r = tf.cast(r, dtype)
    phi = tf.cast(phi, dtype)
    matrix = displacement_matrix(r, phi, cutoff, batched, dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure, batched)
    return output


def squeezer(r, theta, mode, in_modes, cutoff, pure=True, batched=False, dtype=tf.complex64):
    """returns squeezer unitary matrix on specified input modes"""
    r = tf.cast(r, dtype)
    theta = tf.cast(theta, dtype)
    matrix = squeezer_matrix(r, theta, cutoff, batched, dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure, batched)
    return output


def kerr_interaction(kappa, mode, in_modes, cutoff, pure=True, batched=False):
    """returns Kerr unitary matrix on specified input modes"""
    matrix = kerr_interaction_matrix(kappa, cutoff, batched)
    output = single_mode_gate(matrix, mode, in_modes, pure, batched)
    return output


def cross_kerr_interaction(kappa, mode1, mode2, in_modes, cutoff, pure=True, batched=False):
    """returns cross-Kerr unitary matrix on specified input modes"""
    matrix = cross_kerr_interaction_matrix(kappa, cutoff, batched)
    output = two_mode_gate(matrix, mode1, mode2, in_modes, pure, batched)
    return output


def cubic_phase(
    gamma,
    mode,
    in_modes,
    cutoff,
    hbar=2,
    pure=True,
    batched=False,
    method="self_adjoint_eig",
    dtype=tf.complex64,
):
    """returns cubic phase unitary matrix on specified input modes"""
    matrix = cubic_phase_matrix(gamma, cutoff, hbar, batched, method=method, dtype=dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure, batched)
    return output


def beamsplitter(
    theta, phi, mode1, mode2, in_modes, cutoff, pure=True, batched=False, dtype=tf.complex64
):
    """returns beamsplitter unitary matrix on specified input modes"""
    theta = tf.cast(theta, dtype)
    phi = tf.cast(phi, dtype)
    matrix = beamsplitter_matrix(theta, phi, cutoff, batched, dtype)
    output = two_mode_gate(matrix, mode1, mode2, in_modes, pure, batched)
    return output


def mzgate(
    phi_in, phi_ex, mode1, mode2, in_modes, cutoff, pure=True, batched=False, dtype=tf.complex64
):
    """returns mzgate unitary matrix on specified input modes"""
    phi_in = tf.cast(phi_in, dtype)
    phi_ex = tf.cast(phi_ex, dtype)
    matrix = mzgate_matrix(phi_in, phi_ex, cutoff, batched, dtype)
    output = two_mode_gate(matrix, mode1, mode2, in_modes, pure, batched)
    return output


def two_mode_squeeze(
    r, theta, mode1, mode2, in_modes, cutoff, pure=True, batched=False, dtype=tf.complex64
):
    """returns two mode squeeze unitary matrix on specified input modes"""
    matrix = two_mode_squeezer_matrix(r, theta, cutoff, batched, dtype)
    output = two_mode_gate(matrix, mode1, mode2, in_modes, pure, batched)
    return output


def gaussian_gate(S, d, modes, in_modes, cutoff, pure=True, batched=False, dtype=tf.complex64):
    """returns gaussian gate unitary matrix on specified input modes"""
    S = tf.cast(S, dtype)
    d = tf.cast(d, dtype)
    if batched:
        for S_, d_ in zip(S, d):
            if (S_.shape[0]) != d_.shape[0]:
                raise ValueError("The matrix S and the vector d do not have compatible dimensions")
            if not is_symplectic(S_.numpy(), rtol=1e-05, atol=1e-06):
                raise ValueError("The matrix S is not symplectic")
    else:
        if (S.shape[0]) != d.shape[0]:
            raise ValueError("The matrix S and the vector d do not have compatible dimensions")
        if not is_symplectic(S.numpy(), rtol=1e-05, atol=1e-06):
            raise ValueError("The matrix S is not symplectic")
    matrix = gaussian_gate_matrix(S, d, cutoff, batched, dtype)
    output = n_mode_gate(matrix, modes, in_modes=in_modes, pure=pure, batched=batched)
    return output


def loss_channel(T, mode, in_modes, cutoff, pure=True, batched=False, dtype=tf.complex64):
    """returns loss channel matrix on specified input modes"""
    superop = loss_superop(T, cutoff, batched, dtype)
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
            raise NotImplementedError(
                "The max number of supported modes for this operation with pure states is currently {}".format(
                    max_num
                )
            )
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
            raise NotImplementedError(
                "The max number of supported modes for this operation with mixed states is currently {}".format(
                    max_num
                )
            )
        mode_idxs = [
            indices[slice(batch_offset + idx, batch_offset + idx + 2)]
            for idx in range(0, 2 * num_modes, 2)
        ]  # each mode gets a pair of consecutive indices
        eqn_rhs = batch_index + "".join(mode_idxs)
        eqn_idxs = [
            batch_index + m if dims[idx] == 2 else ",".join(m) for idx, m in enumerate(mode_idxs)
        ]
        eqn_lhs = ",".join(eqn_idxs)
        eqn = eqn_lhs + "->" + eqn_rhs
        einsum_inputs = []
        for idx, mode in enumerate(modes_list):
            if dims[idx] == 1:
                new_inputs = [mode, tf.math.conj(mode)]
            elif dims[idx] == 2:
                new_inputs = [mode]
            einsum_inputs += new_inputs
    combined_modes = tf.einsum(eqn, *einsum_inputs)
    return combined_modes


def replace_modes(replacement, modes, system, system_is_pure, batched=False):
    """Replace the subsystem 'mode' of 'system' with new state 'replacement'. Argument 'system_is_pure' indicates whether
    'system' is represented by a pure state or a density matrix.
    Note: Does not check if this replacement is physically valid (i.e., if 'replacement' is a valid state)
    Note: expects the shape of both replacement and system to match the batched parameter
    Note: modes does not need to be ordered.
    """
    # todo: write a better docstring
    if isinstance(modes, int):
        modes = [modes]

    if batched:
        batch_offset = 1
    else:
        batch_offset = 0

    num_modes = len(system.shape) - batch_offset
    if not system_is_pure:
        num_modes = int(num_modes / 2)

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
            system = mix(system, batched)

        # mix the replacement if it is pure
        if replacement_is_pure:
            replacement = mix(replacement, batched)

        # partial trace out modes
        # todo: We are tracing out the modes one by one in descending order (to not screw up things). This is quite inefficient.
        reduced_state = system
        for mode in sorted(modes, reverse=True):
            reduced_state = partial_trace(reduced_state, mode, False, batched)
        # append mode and insert state (There is also insert_state(), but it seemed unnecessarily complicated to try to generalize this function, which does a lot of manual list comprehension, to the multi mode case than to write the two liner below)
        # todo: insert_state() could be rewritten to take full advantage of the high level functions of tf. Before doing that different implementatinos should be benchmarked to compare speed and memory requirements, as in practice these methods will be perfomance critical.
        idx1 = (slice(None, None),) * batched + (Ellipsis,) + (None,) * (replacement.ndim - batched)
        idx2 = (
            (slice(None, None),) * batched + (None,) * (reduced_state.ndim - batched) + (Ellipsis,)
        )
        revised_modes = reduced_state[idx1] * replacement[idx2]
        revised_modes_pure = False

    # unless the preparation was meant to go into the last modes in the standard order, we need to swap indices around
    if modes != list(range(num_modes - len(modes), num_modes)):
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
        index_permutation = [
            scale * x + i for x in mode_permutation for i in (0, 1)
        ]  # two indices per mode if we have pure states

    if batched:
        index_permutation = [0] + [i + 1 for i in index_permutation]

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

    if len(system.shape) == 0:  # pylint: disable=len-as-condition
        # no modes in system
        # pylint: disable=no-else-return
        if len(state.shape) - batch_offset == 1:
            if state_is_pure:
                return state
            else:
                return mix(state)
        elif len(state.shape) - batch_offset == 2:
            return state
        else:
            raise ValueError(
                "'state' must have dim={} or dim={}".format(1 - batch_offset, 2 - batch_offset)
            )
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
        middle_part = indices[
            batch_offset + mode * mode_size : batch_offset + (mode + 1) * mode_size
        ]
        right_part = indices[
            batch_offset + (mode + 1) * mode_size : batch_offset + (num_modes + 1) * mode_size
        ]
        eqn_lhs = batch_index + left_part + right_part + "," + batch_index + middle_part
        eqn_rhs = batch_index + left_part + middle_part + right_part
        eqn = eqn_lhs + "->" + eqn_rhs
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
        system = mix(system, batched)

    # tensorflow trace implementation
    # requires subsystem to be traced out to be at end
    # i.e., ab...klmnop...yz goes to ab...klop...yzmn
    indices_list = list(range(batch_offset + 2 * num_modes))
    dim_list = (
        indices_list[: batch_offset + 2 * mode]
        + indices_list[batch_offset + 2 * (mode + 1) :]
        + indices_list[batch_offset + 2 * mode : batch_offset + 2 * (mode + 1)]
    )
    permuted_sys = tf.transpose(system, dim_list)
    reduced_state = tf.linalg.trace(permuted_sys)
    return reduced_state


def reduced_density_matrix(system, modes, state_is_pure, batched=False):
    """
    Trace out all subsystems except those specified in ``modes`` from ``system``. ``modes`` can be either an int or a list.
    This operation always returns a mixed state, since we do not know in advance if a mode is entangled with others.
    """
    if isinstance(modes, int):
        modes = [modes]

    if state_is_pure:
        reduced_state = mix(system, batched)
    else:
        reduced_state = system
    num_indices = len(reduced_state.shape)
    if batched:
        batch_offset = 1
    else:
        batch_offset = 0
    num_modes = (num_indices - batch_offset) // 2  # always mixed
    removed_cnt = 0
    for m in range(num_modes):
        if m not in modes:
            reduced_state = partial_trace(reduced_state, m - removed_cnt, False, batched)
            removed_cnt += 1
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
        raise ValueError(
            "Conditional state projection currently only supported for {} modes.".format(max_num)
        )

    batch_index = indices[:batch_offset]
    mode_indices = indices[batch_offset : batch_offset + num_modes * mode_size]
    projector_indices = mode_indices[:mode_size]
    free_mode_indices = mode_indices[mode_size : num_modes * mode_size]
    state_lhs = (
        batch_index
        + free_mode_indices[: mode * mode_size]
        + projector_indices
        + free_mode_indices[mode * mode_size :]
    )
    projector_lhs = batch_index + projector_indices[0]
    if mode_size == 2:
        projector_lhs += "," + batch_index + projector_indices[1]
    eqn_lhs = ",".join([state_lhs, projector_lhs])
    eqn_rhs = batch_index + free_mode_indices
    eqn = eqn_lhs + "->" + eqn_rhs
    einsum_args = [system, tf.math.conj(projector)]
    if not state_is_pure:
        einsum_args.append(projector)
    cond_state = tf.einsum(eqn, *einsum_args)
    if not batched:
        cond_state = tf.squeeze(cond_state, 0)  # drop fake batch dimension
    return cond_state


###################################################################

# Helpful auxiliary functions for ops


def ladder_ops(cutoff):
    """returns the matrix representation of the annihilation and creation operators"""
    ind = [(i - 1, i) for i in range(1, cutoff)]
    updates = [np.sqrt(i) for i in range(1, cutoff)]
    shape = [cutoff, cutoff]
    a = tf.scatter_nd(ind, updates, shape)
    ad = tf.transpose(tf.math.conj(a), [1, 0])
    return a, ad


def H(n, x, dtype=tf.complex64):
    """Explicit expression for Hermite polynomials."""
    prefactor = factorial(n)
    terms = tf.reduce_sum(
        tf.stack(
            [
                _numer_safe_power(-1, m, dtype)
                / (factorial(m) * factorial(n - 2 * m))
                * _numer_safe_power(2 * x, n - 2 * m, dtype)
                for m in range(int(np.floor(n / 2)) + 1)
            ],
            axis=0,
        ),
        axis=0,
    )
    return prefactor * terms


def H_n_plus_1(H_n, H_n_m1, n, x):
    """Recurrent definition of Hermite polynomials."""
    H_n_p1 = 2 * x * H_n - 2 * n * H_n_m1
    return H_n_p1

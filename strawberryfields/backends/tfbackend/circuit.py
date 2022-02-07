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
TensorFlow backend proper
======================

Contains most of the code for managing the simulator state and offloading
operations to the utilities in ops.

Hyperlinks: :class:`Circuit`

.. currentmodule:: strawberryfields.backends.tfbackend.circuit

Contents
----------------------
.. autosummary::
     Circuit

"""
# pylint: disable=too-many-arguments,too-many-statements,too-many-branches,protected-access,attribute-defined-outside-init

from itertools import product
from string import ascii_lowercase as indices

import numpy as np
from scipy.special import factorial
import tensorflow as tf

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

from . import ops


class Circuit:
    """Base class for representing and operating on a collection of
    CV quantum optics modes in the Fock basis.
    The modes are initialized in the (multimode) vacuum state,
    using the Fock representation with given cutoff_dim.
    The state of the modes is manipulated by calling the various methods."""

    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    def __init__(self, num_modes, cutoff_dim, pure, batch_size, dtype):
        self._hbar = 2
        self.reset(
            num_subsystems=num_modes,
            pure=pure,
            cutoff_dim=cutoff_dim,
            batch_size=batch_size,
            dtype=dtype,
        )

    def _make_vac_states(self, cutoff_dim):
        """Make vacuum state tensors for the underlying graph"""
        one = tf.cast([1.0], self._dtype)
        v = tf.scatter_nd([[0]], one, [cutoff_dim])
        self._single_mode_pure_vac = v
        self._single_mode_mixed_vac = tf.einsum("i,j->ij", v, v)
        if self._batched:
            self._single_mode_pure_vac = tf.stack([self._single_mode_pure_vac] * self._batch_size)
            self._single_mode_mixed_vac = tf.stack([self._single_mode_mixed_vac] * self._batch_size)

    def _update_state(self, new_state):
        """Helper function to update the state history and the current state"""
        # pylint: disable=attribute-defined-outside-init
        self._state_history.append(new_state)
        self._state = new_state

    def _valid_modes(self, modes):
        # todo: this method should probably be moved into BaseBackend and then maybe
        # overridden and expended in the subclasses to avoid code duplication and
        # missing out on conditions.
        if isinstance(modes, int):
            modes = [modes]

        for mode in modes:
            if mode < 0:
                raise ValueError("Specified mode number(s) cannot be negative.")
            if mode >= self._num_modes:
                raise ValueError(
                    "Specified mode number(s) are not compatible with number of modes."
                )

        if len(modes) != len(set(modes)):
            raise ValueError("The specified modes cannot appear multiple times.")

        return True

    def _replace_and_update(self, replacement, modes):
        """
        Helper function for replacing a mode, updating the state history,
        and possibly setting circuit's state_is_pure variable to a new value.

        Expects replacement to be batched if self._batched.
        """
        if isinstance(modes, int):
            modes = [modes]

        if self._batched:
            batch_offset = 1
        else:
            batch_offset = 0

        num_modes = len(self._state.shape) - batch_offset
        if not self._state_is_pure:
            num_modes = int(num_modes / 2)

        new_state = ops.replace_modes(
            replacement, modes, self._state, self._state_is_pure, self._batched
        )
        self._update_state(new_state)

        # update purity depending on whether we have replaced all modes or a subset
        if len(modes) == num_modes:
            replacement_is_pure = bool(len(replacement.shape) - batch_offset == len(modes))
            self._state_is_pure = replacement_is_pure
        else:
            self._state_is_pure = False

    def _maybe_batch(self, param, convert_to_tensor=True):
        """Helper function to broadcast a param to the correct shape (if necessary) when working in batch mode. If param is not a scalar,
        it will raise an exception if param's batch size is not equal to the circuit's batch size."""
        shape_err = False
        if convert_to_tensor:
            p = tf.convert_to_tensor(param)
            if not self._batched:
                broadcast_p = p
            else:
                if p.shape.ndims == 0:
                    # scalar
                    broadcast_p = tf.stack([p] * self._batch_size)
                elif p.shape.ndims == 1:
                    # vector
                    if p.shape.dims[0].value == self._batch_size:
                        broadcast_p = p
                    else:
                        shape_err = True
                else:
                    shape_err = True
        else:
            p = np.array(param)
            if not self._batched:
                broadcast_p = p
            else:
                if len(p.shape) == 0:  # pylint: disable=len-as-condition
                    # scalar
                    broadcast_p = np.concatenate([np.expand_dims(p, 0)] * self._batch_size)
                elif len(p.shape) == 1:
                    # vector
                    if p.shape[0] == self._batch_size:
                        broadcast_p = p
                    else:
                        shape_err = True
                else:
                    shape_err = True

        if shape_err:
            raise ValueError(
                "Parameter can be either a scalar or a vector of length {}.".format(
                    self._batch_size
                )
            )
        return broadcast_p

    def _check_incompatible_batches(self, *params):
        """Helper function for verifying that all the params from a list have the same batch size. Only does something
        when the circuit is running in batched mode."""
        if self._batched:
            for idx, p in enumerate(params):
                param_batch_size = p.shape.dims[0].value
                if idx == 0:
                    ref_batch_size = param_batch_size
                else:
                    if param_batch_size != ref_batch_size:
                        raise ValueError("Parameters have incompatible batch sizes.")

    def del_mode(self, modes_list):
        """Remove the modes in modes_list from the circuit."""
        pure = self._state_is_pure
        for mode in modes_list:
            reduced_state = ops.partial_trace(self._state, mode, pure, self._batched)
            pure = False
        self._update_state(reduced_state)
        self._state_is_pure = False
        self._num_modes -= len(modes_list)

    def add_mode(self, num_modes):
        """Append M modes (initialized in vacuum states) to the circuit."""
        vac = self._single_mode_pure_vac if self._state_is_pure else self._single_mode_mixed_vac
        new_state = self._state
        for _ in range(num_modes):
            new_state = ops.insert_state(vac, new_state, self._state_is_pure, batched=self._batched)
        self._update_state(new_state)
        self._num_modes += num_modes

    def reset(self, **kwargs):
        r"""
        Resets the state of the circuit to have all modes in vacuum.
        If a keyword arg is not present, the corresponding parameter is unchanged.

        Keyword Args:
            num_subsystems (int): sets the number of modes in the reset circuit.
            pure (bool): if True, the reset circuit will represent its state as a pure state. If False, the representation will be mixed.
            cutoff_dim (int): new Fock space cutoff dimension to use.
            batch_size (None, int): None means no batching. An integer value >= 2 sets the batch size to use.
            dtype (tf.DType): (optional) complex Tensorflow Tensor type representation, either ``tf.complex64`` (default) or ``tf.complex128``.
        """
        if "pure" in kwargs:
            pure = kwargs["pure"]
            if not isinstance(pure, bool):
                raise ValueError("Argument 'pure' must be either True or False")
            self._state_is_pure = pure

        if "num_subsystems" in kwargs:
            ns = kwargs["num_subsystems"]
            if not isinstance(ns, int):
                raise ValueError("Argument 'num_subsystems' must be a positive integer")
            self._num_modes = ns

        if "cutoff_dim" in kwargs:
            cutoff_dim = kwargs["cutoff_dim"]
            if not isinstance(cutoff_dim, int) or cutoff_dim < 1:
                raise ValueError("Argument 'cutoff_dim' must be a positive integer")
            self._cutoff_dim = cutoff_dim

        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
            if batch_size is not None:
                if not isinstance(batch_size, int) or batch_size < 2:
                    raise ValueError("Argument 'batch_size' must be either None or an integer > 1")
            self._batch_size = batch_size
            self._batched = batch_size is not None

        self._dtype = kwargs.get("dtype", tf.complex64)
        if self._dtype not in (tf.complex64, tf.complex128):
            raise ValueError("Argument 'dtype' must be a complex Tensorflow DType")

        self._state_history = []
        self._cache = {}

        self._make_vac_states(self._cutoff_dim)
        single_mode_vac = self._single_mode_pure_vac if pure else self._single_mode_mixed_vac
        if self._num_modes == 1:
            vac = single_mode_vac
        else:
            vac = ops.combine_single_modes([single_mode_vac] * self._num_modes, self._batch_size)
        vac = tf.identity(vac, name="Vacuum")
        self._update_state(vac)

    def prepare_vacuum_state(self, mode):
        """
        Traces out the state in 'mode' and replaces it with a vacuum state.
        """
        if self._valid_modes(mode):
            if self._state_is_pure:
                state = self._single_mode_pure_vac
            else:
                state = self._single_mode_mixed_vac
            self._replace_and_update(state, mode)

    def prepare_fock_state(self, n, mode):
        """
        Traces out the state in 'mode' and replaces it with a Fock state defined by n.
        """
        if self._valid_modes(mode):
            n = self._maybe_batch(n, convert_to_tensor=False)
            fock_state = ops.fock_state(
                n,
                cutoff=self._cutoff_dim,
                pure=self._state_is_pure,
                batched=self._batched,
                dtype=self._dtype,
            )
            self._replace_and_update(fock_state, mode)

    def prepare_coherent_state(self, r, phi, mode):
        """
        Traces out the state in 'mode' and replaces it with a coherent state defined by alpha=r exp(i phi).
        """
        if self._valid_modes(mode):
            r = self._maybe_batch(r)
            phi = self._maybe_batch(phi)
            coherent_state = ops.coherent_state(
                r,
                phi,
                cutoff=self._cutoff_dim,
                pure=self._state_is_pure,
                batched=self._batched,
                dtype=self._dtype,
            )
            self._replace_and_update(coherent_state, mode)

    def prepare_squeezed_state(self, r, theta, mode):
        """
        Traces out the state in 'mode' and replaces it with a squeezed state defined by r and theta.
        """
        if self._valid_modes(mode):
            r = self._maybe_batch(r)
            theta = self._maybe_batch(theta)
            self._check_incompatible_batches(r, theta)
            squeezed_state = ops.squeezed_vacuum(
                r,
                theta,
                cutoff=self._cutoff_dim,
                pure=self._state_is_pure,
                batched=self._batched,
                dtype=self._dtype,
            )
            self._replace_and_update(squeezed_state, mode)

    def prepare_displaced_squeezed_state(self, r_d, phi_d, r_s, phi_s, mode):
        """
        Traces out the state in 'mode' and replaces it with a displaced squeezed state defined by alpha, r and theta.
        """
        if self._valid_modes(mode):
            r_d = self._maybe_batch(r_d)
            r_s = self._maybe_batch(r_s)
            phi_d = self._maybe_batch(phi_d)
            phi_s = self._maybe_batch(phi_s)
            self._check_incompatible_batches(r_d, phi_d, r_s, phi_s)
            displaced_squeezed = ops.displaced_squeezed(
                r_d,
                phi_d,
                r_s,
                phi_s,
                cutoff=self._cutoff_dim,
                pure=self._state_is_pure,
                batched=self._batched,
                dtype=self._dtype,
            )
            self._replace_and_update(displaced_squeezed, mode)

    def prepare_multimode(self, state, modes, input_state_is_pure=False):
        r"""Prepares a given mode or list of modes in the given state.

        After the preparation the system is in a mixed product state,
        with the specified modes replaced by state.

        The given state can be either in tensor form or in matrix/vector form and
        can be a batch of states or a single state. This method needs to know whether
        input_state_is_pure to distinguish between a batch of pure states and a mixed state.

        If modes is not ordered, the subsystems of the input are
        reordered to reflect that, i.e., if modes=[3,1], then the first mode
        of state ends up in mode 3 and the second mode of state ends up in
        mode 1 of the output state.

        If modes is None, it is attempted to prepare state in all modes.
        The reduced state on all other modes remains unchanged and
        the final state is product with respect to the partition into
        the modes in modes and the complement.

        Args:
            state (array): vector, matrix, or tensor representation of the ket state or
                density matrix state (or a batch of such states) in the fock basis to prepare
            modes (list[int] or non-negative int): The mode(s) into which state is
                to be prepared. Needs not be ordered.
        """
        if self._valid_modes(modes):
            if isinstance(modes, int):
                modes = [modes]

            n_modes = len(modes)
            if input_state_is_pure:
                input_is_batched = len(state.shape) > n_modes or (
                    len(state.shape) == 2 and state.shape[1] == self._cutoff_dim**n_modes
                )
            else:
                input_is_batched = len(state.shape) % 2 == 1

            pure_shape = tuple([self._cutoff_dim] * n_modes)
            mixed_shape = tuple([self._cutoff_dim] * (2 * n_modes))
            pure_shape_as_vector = tuple([self._cutoff_dim**n_modes])
            mixed_shape_as_matrix = tuple([self._cutoff_dim**n_modes] * 2)
            if input_is_batched:
                pure_shape = (self._batch_size,) + pure_shape
                mixed_shape = (self._batch_size,) + mixed_shape
                pure_shape_as_vector = (self._batch_size,) + pure_shape_as_vector
                mixed_shape_as_matrix = (self._batch_size,) + mixed_shape_as_matrix

            # reshape to support input both as tensor and vector/matrix
            if state.shape == pure_shape_as_vector:
                state = tf.reshape(state, pure_shape)
            elif state.shape == mixed_shape_as_matrix:
                state = tf.reshape(state, mixed_shape)

            state = tf.cast(tf.convert_to_tensor(state), self._dtype)
            # batch state now if not already batched and self._batched
            if self._batched and not input_is_batched:
                state = tf.stack([state] * self._batch_size)
            self._replace_and_update(state, modes)

    def prepare_thermal_state(self, nbar, mode):
        """
        Prepares the thermal state with mean photon nbar in the specified mode.
        """
        if self._valid_modes(mode):
            nbar = self._maybe_batch(nbar)
            thermal = ops.thermal_state(nbar, cutoff=self._cutoff_dim, dtype=self._dtype)
            self._replace_and_update(thermal, mode)

    def phase_shift(self, theta, mode):
        """
        Apply the phase-shift operator to the specified mode.
        """
        theta = self._maybe_batch(theta)
        new_state = ops.phase_shifter(
            theta,
            mode,
            self._state,
            self._cutoff_dim,
            self._state_is_pure,
            self._batched,
            dtype=self._dtype,
        )
        self._update_state(new_state)

    def displacement(self, r, phi, mode):
        """
        Apply the displacement operator to the specified mode.
        """
        r = self._maybe_batch(r)
        phi = self._maybe_batch(phi)
        new_state = ops.displacement(
            r,
            phi,
            mode,
            self._state,
            self._cutoff_dim,
            self._state_is_pure,
            self._batched,
            dtype=self._dtype,
        )
        self._update_state(new_state)

    def squeeze(self, r, theta, mode):
        """
        Apply the single-mode squeezing operator to the specified mode.
        """
        r = self._maybe_batch(r)
        theta = self._maybe_batch(theta)
        self._check_incompatible_batches(r, theta)
        new_state = ops.squeezer(
            r,
            theta,
            mode,
            self._state,
            self._cutoff_dim,
            self._state_is_pure,
            self._batched,
            dtype=self._dtype,
        )
        self._update_state(new_state)

    def beamsplitter(self, theta, phi, mode1, mode2):
        """
        Apply a beamsplitter operator to the two specified modes.
        """
        theta = self._maybe_batch(theta)
        phi = self._maybe_batch(phi)
        self._check_incompatible_batches(theta, phi)
        new_state = ops.beamsplitter(
            theta,
            phi,
            mode1,
            mode2,
            self._state,
            self._cutoff_dim,
            self._state_is_pure,
            self._batched,
            dtype=self._dtype,
        )
        self._update_state(new_state)

    def mzgate(self, phi_in, phi_ex, mode1, mode2):
        """
        Apply a MZ-gate operator to the two specified modes.
        """
        phi_in = self._maybe_batch(phi_in)
        phi_ex = self._maybe_batch(phi_ex)
        self._check_incompatible_batches(phi_in, phi_ex)
        new_state = ops.mzgate(
            phi_in,
            phi_ex,
            mode1,
            mode2,
            self._state,
            self._cutoff_dim,
            self._state_is_pure,
            self._batched,
            dtype=self._dtype,
        )
        self._update_state(new_state)

    def two_mode_squeeze(self, r, phi, mode1, mode2):
        """
        Apply a two-mode squeezing operator to the two specified modes.
        """
        r = self._maybe_batch(r)
        phi = self._maybe_batch(phi)
        self._check_incompatible_batches(r, phi)
        new_state = ops.two_mode_squeeze(
            r,
            phi,
            mode1,
            mode2,
            self._state,
            self._cutoff_dim,
            self._state_is_pure,
            self._batched,
            dtype=self._dtype,
        )
        self._update_state(new_state)

    def kerr_interaction(self, kappa, mode):
        """
        Apply the Kerr interaction operator to the specified mode.
        """
        k = tf.cast(kappa, self._dtype)
        k = self._maybe_batch(k)
        new_state = ops.kerr_interaction(
            k, mode, self._state, self._cutoff_dim, self._state_is_pure, self._batched
        )
        self._update_state(new_state)

    def cross_kerr_interaction(self, kappa, mode1, mode2):
        """
        Apply the cross-Kerr interaction operator to the specified mode.
        """
        k = tf.cast(kappa, self._dtype)
        k = self._maybe_batch(k)
        new_state = ops.cross_kerr_interaction(
            k, mode1, mode2, self._state, self._cutoff_dim, self._state_is_pure, self._batched
        )
        self._update_state(new_state)

    def cubic_phase(self, gamma, mode):
        """
        Apply the cubic phase operator to the specified mode.
        """
        g = tf.cast(gamma, self._dtype)
        g = self._maybe_batch(g)
        new_state = ops.cubic_phase(
            g,
            mode,
            self._state,
            self._cutoff_dim,
            self._hbar,
            self._state_is_pure,
            self._batched,
            dtype=self._dtype,
        )
        self._update_state(new_state)

    def gaussian_gate(self, S, d, modes):
        """
        Apply the N-mode gaussian gates to the specified mode.
        """
        self._batched = len(d.shape) == 2
        self._batch_size = d.shape[0] if self._batched else None
        new_state = ops.gaussian_gate(
            S,
            d,
            modes,
            in_modes=self._state,
            cutoff=self._cutoff_dim,
            pure=self._state_is_pure,
            batched=self._batched,
            dtype=self._dtype,
        )
        self._update_state(new_state)

    def loss(self, T, mode):
        """
        Apply a loss channel  to the specified mode.
        """
        T = tf.cast(T, self._dtype)
        T = self._maybe_batch(T)
        new_state = ops.loss_channel(
            T,
            mode,
            self._state,
            self._cutoff_dim,
            self._state_is_pure,
            self._batched,
            dtype=self._dtype,
        )
        self._update_state(new_state)
        self._state_is_pure = False  # loss output always in mixed state representation

    def vacuum_element(self):
        """Compute the fidelity with the (multi-mode) vacuum state."""
        if self._batched:
            vac_component = tf.reshape(self._state, [self._batch_size, -1])[:, 0]
        else:
            vac_component = tf.reshape(self._state, [-1])[0]

        if self._state_is_pure:
            return tf.abs(vac_component) ** 2
        return vac_component

    def measure_fock(self, modes, select=None, **kwargs):
        """
        Measures 'modes' in the Fock basis and updates remaining modes conditioned on this result.
        After measurement, the states in 'modes' are reset to the vacuum.

        Args:
            modes (Sequence[int]): which modes to measure (in increasing order).
            select (Sequence[int]): user-specified measurement value (used instead of random sampling)

        Returns:
            tuple[int]: The Fock number measurement results for each mode.
        """
        # pylint: disable=unused-argument

        # allow integer (non-list) arguments
        # not part of the API, but provided for convenience
        if isinstance(modes, int):
            modes = [modes]
        if isinstance(select, int):
            select = [select]

        # convert lists to np arrays
        if isinstance(modes, list):
            modes = np.array(modes)
        if isinstance(select, list):
            select = np.array(select)

        # check for valid 'modes' argument
        if (
            len(modes) == 0 or len(modes) > self._num_modes or len(modes) != len(set(modes))
        ):  # pylint: disable=len-as-condition
            raise ValueError("Specified modes are not valid.")
        if np.any(modes != sorted(modes)):
            raise ValueError("'modes' must be sorted in increasing order.")

        # check for valid 'select' argument
        if select is not None:
            if np.any(select == None):  # pylint: disable=singleton-comparison
                raise NotImplementedError(
                    "Post-selection lists must only contain numerical values."
                )
            if self._batched:
                num_meas_modes = len(modes)
                # in this case, select must either be:
                # np array of shape (M,), or
                # np array of shape (B,M)
                # where B is the batch_size and M is the number of measured modes
                shape_err = False
                if len(select.shape) == 1:
                    # non-batched list, must broadcast
                    if select.shape[0] != num_meas_modes:
                        shape_err = True
                    else:
                        select = np.vstack([select] * self._batch_size)
                elif len(select.shape) == 2:
                    # batch of lists, no need to broadcast
                    if select.shape != (self._batch_size, num_meas_modes):
                        shape_err = True
                else:
                    shape_err = True
                if shape_err:
                    raise ValueError("The shape of 'select' is incompatible with 'modes'.")
            else:
                # in this case, select should be a vector
                if select.shape != modes.shape:
                    raise ValueError("'select' must be have the same shape as 'modes'")

        # carry out the operation
        num_reduced_state_modes = len(modes)
        reduced_state = self._state
        if self._state_is_pure:
            mode_size = 1
        else:
            mode_size = 2
        if self._batched:
            batch_size = self._batch_size
            batch_offset = 1
        else:
            batch_size = 1
            batch_offset = 0

        if select is not None:
            # just use the supplied measurement results
            meas_result = tf.convert_to_tensor([[i] for i in select], dtype=tf.int64)
        else:
            # compute and sample measurement result
            if self._state_is_pure and len(modes) == self._num_modes:
                # in this case, measure directly on the pure state
                probs = tf.abs(self._state) ** 2
                logprobs = tf.math.log(probs)
                sample = tf.random.categorical(tf.reshape(logprobs, [batch_size, -1]), 1)
                sample_tensor = tf.squeeze(sample)
            else:
                # otherwise, trace out unmeasured modes and sample using diagonal of reduced state
                removed_ctr = 0
                red_state_is_pure = self._state_is_pure
                for m in range(self._num_modes):
                    if m not in modes:
                        new_mode_idx = m - removed_ctr
                        reduced_state = ops.partial_trace(
                            reduced_state, new_mode_idx, red_state_is_pure, self._batched
                        )
                        red_state_is_pure = False
                        removed_ctr += 1
                # go from bra_A,ket_A,bra_B,ket_B,... -> bra_A,bra_B,ket_A,ket_B,... since this is what diag_part expects
                # workaround for getting multi-index diagonal since tensorflow doesn't support getting diag of more than one subsystem at once
                if num_reduced_state_modes > 1:
                    state_indices = np.arange(batch_offset + 2 * num_reduced_state_modes)
                    batch_index = state_indices[:batch_offset]
                    bra_indices = state_indices[batch_offset::2]
                    ket_indices = state_indices[batch_offset + 1 :: 2]
                    transpose_list = np.concatenate([batch_index, bra_indices, ket_indices])
                    reduced_state_reshuffled = tf.transpose(reduced_state, transpose_list)
                else:
                    reduced_state_reshuffled = reduced_state
                diag_indices = [self._cutoff_dim**num_reduced_state_modes] * 2
                if self._batched:
                    diag_indices = [self._batch_size] + diag_indices
                diag_tensor = tf.reshape(reduced_state_reshuffled, diag_indices)
                diag_entries = tf.linalg.diag_part(diag_tensor)
                # hack so we can use tf.random.categorical for sampling
                logprobs = tf.math.log(tf.cast(diag_entries, tf.float64))
                sample = tf.random.categorical(tf.reshape(logprobs, [batch_size, -1]), 1)
                # sample is a single integer; we need to convert it to the corresponding [n0,n1,n2,...]
                sample_tensor = tf.squeeze(sample)

            # sample_val is a single integer for each batch entry;
            # we need to convert it to the corresponding [n0,n1,n2,...]
            meas_result = ops.unravel_index(
                sample_tensor, [self._cutoff_dim] * num_reduced_state_modes
            )

            if self._batched:
                meas_shape = meas_result.shape
                meas_result = tf.reshape(meas_result, (meas_shape[0], 1, meas_shape[1]))

        # project remaining modes into conditional state
        if len(modes) == self._num_modes:
            # in this case, all modes were measured and we can put everything in vacuum by reseting
            self.reset(pure=self._state_is_pure)
        else:
            # only some modes were measured: put unmeasured modes in conditional state, while reseting measured modes to vac
            fock_state = tf.one_hot(meas_result[0], depth=self._cutoff_dim, dtype=self._dtype)
            conditional_state = self._state
            for idx, mode in enumerate(modes):
                if self._batched:
                    f = fock_state[:, idx]
                else:
                    f = fock_state[idx]
                conditional_state = ops.conditional_state(
                    conditional_state, f, mode, self._state_is_pure, batched=self._batched
                )

            if self._state_is_pure:
                norm = tf.norm(tf.reshape(conditional_state, [batch_size, -1]), axis=1)
            else:
                # calculate norm of conditional_state
                # use a cheap hack since tensorflow doesn't allow einsum equation for trace:
                r = conditional_state
                for _ in range(self._num_modes - num_reduced_state_modes - 1):
                    r = ops.partial_trace(r, 0, False, self._batched)
                norm = tf.linalg.trace(r)

            # for broadcasting
            norm_reshape = [1] * len(conditional_state.shape[batch_offset:])
            if self._batched:
                norm_reshape = [self._batch_size] + norm_reshape

            normalized_conditional_state = conditional_state / tf.reshape(norm, norm_reshape)

            # reset measured modes into vacuum
            single_mode_vac = (
                self._single_mode_pure_vac if self._state_is_pure else self._single_mode_mixed_vac
            )
            if len(modes) == 1:
                meas_modes_vac = single_mode_vac
            else:
                meas_modes_vac = ops.combine_single_modes(
                    [single_mode_vac] * len(modes), self._batched
                )
            batch_index = indices[:batch_offset]
            meas_mode_indices = indices[batch_offset : batch_offset + mode_size * len(modes)]
            conditional_indices = indices[
                batch_offset + mode_size * len(modes) : batch_offset + mode_size * self._num_modes
            ]
            eqn_lhs = batch_index + meas_mode_indices + "," + batch_index + conditional_indices
            eqn_rhs = ""
            meas_ctr = 0
            cond_ctr = 0
            for m in range(self._num_modes):
                if m in modes:
                    # use measured_indices
                    eqn_rhs += meas_mode_indices[mode_size * meas_ctr : mode_size * (meas_ctr + 1)]
                    meas_ctr += 1
                else:
                    # use conditional indices
                    eqn_rhs += conditional_indices[
                        mode_size * cond_ctr : mode_size * (cond_ctr + 1)
                    ]
                    cond_ctr += 1
            eqn = eqn_lhs + "->" + batch_index + eqn_rhs
            new_state = tf.einsum(eqn, meas_modes_vac, normalized_conditional_state)

            self._update_state(new_state)

        return meas_result

    def measure_homodyne(self, phi, mode, select=None, **kwargs):
        """
            Measures 'modes' in the basis of quadrature eigenstates (rotated by phi)
            and updates remaining modes conditioned on this result.
            After measurement, the states in 'modes' are reset to the vacuum.

            Args:
                phi (float): phase angle of quadrature to measure
                mode (int): which mode to measure.
                select (float): user-specified measurement value (used instead of random sampling)

        Returns:
            float, list[float]: The measured value, or a list of measured values when running in batch mode.
        """

        if not isinstance(mode, int):
            raise ValueError("Specified modes are not valid.")
        if mode < 0 or mode >= self._num_modes:
            raise ValueError("Specified modes are not valid.")

        m_omega_over_hbar = 1 / self._hbar
        if self._state_is_pure:
            mode_size = 1
        else:
            mode_size = 2
        if self._batched:
            batch_offset = 1
            batch_size = self._batch_size
        else:
            batch_offset = 0
            batch_size = 1

        phi = tf.cast(phi, self._dtype)
        phi = self._maybe_batch(phi)

        if select is not None:
            meas_result = self._maybe_batch(select)
            homodyne_sample = tf.cast(meas_result, tf.float64, name="Meas_result")
        else:
            # create reduced state on mode to be measured
            reduced_state = ops.reduced_density_matrix(
                self._state, mode, self._state_is_pure, self._batched
            )

            # rotate to homodyne basis
            # pylint: disable=invalid-unary-operand-type
            reduced_state = ops.phase_shifter(
                -phi, 0, reduced_state, self._cutoff_dim, False, self._batched, self._dtype
            )

            # create pdf for homodyne measurement
            # We use the following quadrature wavefunction for the Fock states:
            # \psi_n(x) = 1/sqrt[2^n n!](\frac{m \omega}{\pi \hbar})^{1/4}
            #             \exp{-\frac{m \omega}{2\hbar} x^2} H_n(\sqrt{\frac{m \omega}{\pi}} x)
            # where H_n(x) is the (physicists) nth Hermite polynomial
            q_mag = kwargs.get("max", 10)
            num_bins = kwargs.get("kwargs", 100000)
            if "q_tensor" in self._cache:
                # use cached q_tensor
                q_tensor = self._cache["q_tensor"]
            else:
                q_tensor = tf.constant(np.linspace(-q_mag, q_mag, num_bins))
                self._cache["q_tensor"] = q_tensor
            x = np.sqrt(m_omega_over_hbar) * q_tensor
            if "hermite_polys" in self._cache:
                # use cached polynomials
                hermite_polys = self._cache["hermite_polys"]
            else:
                H0 = 0 * x + 1.0
                H1 = 2 * x
                hermite_polys = [H0, H1]
                Hn = H1
                Hn_m1 = H0
                for n in range(1, self._cutoff_dim - 1):
                    Hn_p1 = ops.H_n_plus_1(Hn, Hn_m1, n, x)
                    hermite_polys.append(Hn_p1)
                    Hn_m1 = Hn
                    Hn = Hn_p1
                self._cache["hermite_polys"] = hermite_polys

            number_state_indices = list(product(range(self._cutoff_dim), repeat=2))
            terms = [
                1
                / np.sqrt(2**n * factorial(n) * 2**m * factorial(m))
                * hermite_polys[n]
                * hermite_polys[m]
                for n, m in number_state_indices
            ]
            hermite_matrix = tf.scatter_nd(
                number_state_indices, terms, [self._cutoff_dim, self._cutoff_dim, num_bins]
            )
            hermite_terms = tf.multiply(
                tf.expand_dims(reduced_state, -1),
                tf.expand_dims(tf.cast(hermite_matrix, self._dtype), 0),
            )
            rho_dist = (
                tf.cast(tf.reduce_sum(hermite_terms, axis=[1, 2]), tf.float64)
                * (m_omega_over_hbar / np.pi) ** 0.5
                * tf.exp(-(x**2))
                * (q_tensor[1] - q_tensor[0])
            )  # Delta_q for normalization (only works if the bins are equally spaced)

            # use tf.random.categorical to sample
            logprobs = tf.math.log(rho_dist)
            samples_idx = tf.random.categorical(logprobs, 1)
            homodyne_sample = tf.gather(q_tensor, samples_idx)
            homodyne_sample = tf.squeeze(homodyne_sample)

        meas_result = tf.identity(homodyne_sample, name="Meas_result")

        # project remaining modes into conditional state
        if self._num_modes == 1:
            # in this case, all modes were measured and we we put everything into vacuum
            self.reset(pure=self._state_is_pure)
        else:
            # only some modes were measured: put unmeasured modes in conditional state, while reseting measured modes to vac
            inf_squeezed_vac = tf.convert_to_tensor(
                [
                    (-0.5) ** (m // 2) * np.sqrt(factorial(m)) / factorial(m // 2)
                    if m % 2 == 0
                    else 0.0
                    for m in range(self._cutoff_dim)
                ],
                dtype=self._dtype,
            )
            if self._batched:
                inf_squeezed_vac = tf.tile(tf.expand_dims(inf_squeezed_vac, 0), [batch_size, 1])
            displacement_size = tf.stack(
                tf.convert_to_tensor(meas_result * np.sqrt(m_omega_over_hbar / 2))
            )
            quad_eigenstate = ops.displacement(
                tf.math.abs(displacement_size),
                tf.math.angle(displacement_size),
                0,
                inf_squeezed_vac,
                self._cutoff_dim,
                True,
                self._batched,
                self._dtype,
            )
            homodyne_eigenstate = ops.phase_shifter(
                phi, 0, quad_eigenstate, self._cutoff_dim, True, self._batched, self._dtype
            )

            conditional_state = ops.conditional_state(
                self._state, homodyne_eigenstate, mode, self._state_is_pure, batched=self._batched
            )

            # normalize
            if self._state_is_pure:
                norm = tf.norm(tf.reshape(conditional_state, [batch_size, -1]), axis=1)
            else:
                # calculate norm of conditional_state
                # cheap hack since tensorflow doesn't allow einsum equation for trace:
                r = conditional_state
                for _ in range(self._num_modes - 2):
                    r = ops.partial_trace(r, 0, False, self._batched)
                norm = tf.linalg.trace(r)

            # for broadcasting
            norm_reshape = [1] * len(conditional_state.shape[batch_offset:])
            if self._batched:
                norm_reshape = [self._batch_size] + norm_reshape

            normalized_conditional_state = conditional_state / tf.reshape(norm, norm_reshape)

            # reset measured modes into vacuum
            meas_mode_vac = (
                self._single_mode_pure_vac if self._state_is_pure else self._single_mode_mixed_vac
            )
            batch_index = indices[:batch_offset]
            meas_mode_indices = indices[batch_offset : batch_offset + mode_size]
            conditional_indices = indices[
                batch_offset + mode_size : batch_offset + mode_size * self._num_modes
            ]
            eqn_lhs = batch_index + meas_mode_indices + "," + batch_index + conditional_indices
            eqn_rhs = ""
            meas_ctr = 0
            cond_ctr = 0
            for m in range(self._num_modes):
                if m == mode:
                    # use measured_indices
                    eqn_rhs += meas_mode_indices[mode_size * meas_ctr : mode_size * (meas_ctr + 1)]
                    meas_ctr += 1
                else:
                    # use conditional indices
                    eqn_rhs += conditional_indices[
                        mode_size * cond_ctr : mode_size * (cond_ctr + 1)
                    ]
                    cond_ctr += 1
            eqn = eqn_lhs + "->" + batch_index + eqn_rhs
            new_state = tf.einsum(eqn, meas_mode_vac, normalized_conditional_state)

            self._update_state(new_state)

        # `meas_result` will always be a single value since multiple shots is not supported
        if self.batched:
            return tf.reshape(meas_result, (len(meas_result), 1, 1))
        return tf.cast([[meas_result]], self._dtype)

    @property
    def num_modes(self):
        """Number of modes in the circuit"""
        return self._num_modes

    @property
    def cutoff_dim(self):
        """Circuit cutoff dimension"""
        return self._cutoff_dim

    @property
    def state_is_pure(self):
        """Returns true if the circuit state is pure"""
        return self._state_is_pure

    @property
    def hbar(self):
        """Returns the value of hbar circuit is initialised with"""
        return self._hbar

    @property
    def batched(self):
        """Returns True if the circuit is batched"""
        return self._batched

    @property
    def batch_size(self):
        """Returns the batch size"""
        return self._batch_size

    @property
    def dtype(self):
        """Returns the circuit dtype"""
        return self._dtype

    @property
    def state(self):
        """Returns the circuit state"""
        return tf.identity(self._state, name="State")

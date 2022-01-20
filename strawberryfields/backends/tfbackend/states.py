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
"""TensorFlow states module"""
from string import ascii_lowercase as indices

import numpy as np
import tensorflow as tf
from scipy.special import factorial

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

from strawberryfields.backends.states import BaseFockState
from .ops import ladder_ops, phase_shifter_matrix, reduced_density_matrix


class FockStateTF(BaseFockState):
    r"""Class for the representation of quantum states in the Fock basis using the TFBackend.

    Args:
            state_data (array): the state representation in the Fock basis
            num_modes (int): the number of modes in the state
            pure (bool): True if the state is a pure state, false if the state is mixed
            cutoff_dim (int): the Fock basis truncation size
            batched (bool): (optional) default False means no batching
            mode_names (Sequence): (optional) this argument contains a list providing mode names
                    for each mode in the state
            dtype (tf.DType): (optional) complex Tensorflow Tensor type representation, either
                    ``tf.complex64`` (default) or ``tf.complex128``
    """

    def __init__(
        self,
        state_data,
        num_modes,
        pure,
        cutoff_dim,
        batched=False,
        mode_names=None,
        dtype=tf.complex64,
    ):
        # pylint: disable=too-many-arguments
        state_data = tf.convert_to_tensor(
            state_data, name="state_data"
        )  # convert everything to a Tensor so we have the option to do symbolic evaluations
        super().__init__(state_data, num_modes, pure, cutoff_dim, mode_names)
        self._batched = batched
        self._dtype = dtype
        self._str = (
            "<FockStateTF: num_modes={}, cutoff={}, pure={}, batched={}, hbar={}, dtype={}>".format(
                self.num_modes, self.cutoff_dim, self._pure, self._batched, self._hbar, self._dtype
            )
        )

    def trace(self, **kwargs):
        r"""
        Computes the trace of the state. May be numerical or symbolic.

        Returns:
            float/Tensor: the numerical value, or an unevaluated Tensor object, for the trace.
        """
        s = self.data
        if not self.batched:
            s = tf.expand_dims(s, 0)  # insert fake batch dimension
        if self.is_pure:
            flat_state = tf.reshape(s, [-1, self.cutoff_dim ** self.num_modes])
            tr = tf.reduce_sum(flat_state * tf.math.conj(flat_state), 1)
        else:
            # hack because tensorflow einsum doesn't allow partial trace
            tr = s
            for _ in range(self.num_modes):
                tr = tf.linalg.trace(tr)
        if not self.batched:
            tr = tf.squeeze(tr, 0)  # drop fake batch dimension

        return tr

    def fock_prob(self, n, **kwargs):
        r"""
        Compute the probabilities of a specific Fock-basis matrix element for the state. May be numerical or symbolic.

        Args:
            n (Sequence[int]): the Fock state :math:`\ket{\vec{n}}` that we want to measure the probability of

        Returns:
            float/Tensor: the numerical values, or an unevaluated Tensor object, for the Fock-state probabilities.
        """

        if len(n) != self.num_modes:
            raise ValueError("List length should be equal to number of modes")

        if max(n) >= self.cutoff_dim:
            raise ValueError("Can't get distribution beyond truncation level")

        state = self.data

        if not self.batched:
            state = tf.expand_dims(state, 0)  # add fake batch dimension

        # put batch dimension at end to match behaviour of tf.gather_nd
        if self.is_pure:
            flat_idx = np.ravel_multi_index(n, [self.cutoff_dim] * self.num_modes)
            flat_state = tf.reshape(state, [-1, self.cutoff_dim ** self.num_modes])
            prob = tf.abs(flat_state[:, flat_idx]) ** 2
        else:
            doubled_n = [n[i // 2] for i in range(len(n) * 2)]
            flat_idx = np.ravel_multi_index(doubled_n, [self.cutoff_dim] * (2 * self.num_modes))
            flat_state = tf.reshape(state, [-1, self.cutoff_dim ** (2 * self.num_modes)])
            prob = flat_state[:, flat_idx]

        if not self.batched:
            prob = tf.squeeze(prob, 0)  # drop fake batch dimension

        prob = tf.math.real(prob)
        prob = tf.identity(prob, name="fock_prob")

        return prob

    def all_fock_probs(self, **kwargs):
        r"""
        Compute the probabilities of all possible Fock-basis states for the state. May be numerical or symbolic.

        For example, in the case of 3 modes, this method allows the Fock state probability
        :math:`|\braketD{0,2,3}{\psi}|^2` to be returned via

        .. code-block:: python

            probs = state.all_fock_probs()
            probs[0,2,3]

        Args:

        Returns:
            array/Tensor: the numerical values, or an unevaluated Tensor object, for the Fock-basis probabilities.
        """

        state = self.data
        if not self.batched:
            state = tf.expand_dims(state, 0)  # add fake batch dimension

        if self.is_pure:
            probs = tf.abs(state) ** 2
        else:
            # convert to index scheme compatible with tf.linalg.diag_part
            rng = np.arange(1, 2 * self.num_modes + 1, 2)
            perm = np.concatenate([[0], rng, rng + 1])
            perm_state = tf.transpose(state, perm)
            probs = tf.map_fn(
                tf.linalg.tensor_diag_part, perm_state
            )  # diag_part doesn't work with batches, need to use map_fn

        if not self.batched:
            probs = tf.squeeze(probs, 0)  # drop fake batch dimension

        probs = tf.identity(probs, name="all_fock_probs")

        return probs

    def fidelity(self, other_state, mode, **kwargs):
        r"""
        Compute the fidelity of the reduced state (on the specified mode) with the state. May be numerical or symbolic.

        Args:
            other_state (array): state vector (ket) to compute the fidelity with respect to
            mode (int): which subsystem to use for the fidelity computation

        Returns:
            float/Tensor: the numerical value, or an unevaluated Tensor object, for the fidelity.
        """
        rho = self.reduced_dm([mode])  # don't pass kwargs yet

        if not self.batched:
            rho = tf.expand_dims(rho, 0)  # add fake batch dimension
        if len(other_state.shape) == 1:
            other_state = tf.expand_dims(other_state, 0)  # add batch dimension for state

        other_state = tf.cast(other_state, self.dtype)
        state_dm = tf.einsum("bi,bj->bij", tf.math.conj(other_state), other_state)
        flat_state_dm = tf.reshape(state_dm, [1, -1])
        flat_rho = tf.reshape(rho, [-1, self.cutoff_dim ** 2])

        f = tf.math.real(
            tf.reduce_sum(flat_rho * flat_state_dm, axis=1)
        )  # implements a batched tr(rho|s><s|)

        if not self.batched:
            f = tf.squeeze(f, 0)  # drop fake batch dimension

        f = tf.identity(f, name="fidelity")

        return f

    def fidelity_coherent(self, alpha_list, **kwargs):
        r"""
        Compute the fidelity of the state with the coherent states specified by alpha_list. May be numerical or symbolic.

        Args:
            alpha_list (Sequence[complex]): list of coherence parameter values, one for each mode

        Returns:
            float/Tensor: the numerical value, or an unevaluated Tensor object, for the fidelity :math:`\bra{\vec{\alpha}}\rho\ket{\vec{\alpha}}`.
        """
        if not hasattr(alpha_list, "__len__"):
            alpha_list = [alpha_list]

        if len(alpha_list) != self.num_modes:
            raise ValueError("The number of alpha values must match the number of modes.")

        max_indices = (len(indices) - 1) // 2
        if len(alpha_list) > max_indices:
            raise ValueError("Length of `alpha_list` exceeds supported number of modes.")

        s = self.data
        if not self.batched:
            s = tf.expand_dims(s, 0)  # introduce fake batch dimension

        coh = lambda a, dim: [
            np.exp(-0.5 * np.abs(a) ** 2) * (a) ** n / np.sqrt(factorial(n)) for n in range(dim)
        ]
        multi_cohs_list = [
            coh(a, self.cutoff_dim) for a in alpha_list
        ]  # shape is: [num_modes, cutoff_dim]
        eqn = ",".join(indices[: self._modes]) + "->" + indices[: self._modes]
        multi_cohs_vec = np.einsum(
            eqn, *multi_cohs_list
        )  # tensor product of specified coherent states
        flat_multi_cohs = np.reshape(
            multi_cohs_vec, [1, self.cutoff_dim ** self.num_modes]
        )  # flattened tensor product; shape is: [1, cutoff_dim * num_modes]

        if self.is_pure:
            flat_state = tf.reshape(s, [-1, self.cutoff_dim ** self.num_modes])
            ovlap = tf.reduce_sum(flat_multi_cohs.conj() * flat_state, axis=1)
            f = tf.abs(ovlap) ** 2
        else:
            batch_index = indices[0]
            free_indices = indices[1:]
            bra_indices = free_indices[: self.num_modes]
            ket_indices = free_indices[self.num_modes : 2 * self.num_modes]
            eqn = (
                bra_indices
                + ","
                + batch_index
                + "".join(bra_indices[idx] + ket_indices[idx] for idx in range(self.num_modes))
                + ","
                + ket_indices
                + "->"
                + batch_index
            )
            f = tf.einsum(
                eqn,
                tf.convert_to_tensor(np.conj(multi_cohs_vec), dtype=self.dtype),
                s,
                tf.convert_to_tensor(multi_cohs_vec, dtype=self.dtype),
            )
        if not self.batched:
            f = tf.squeeze(f, 0)  # drop fake batch dimension

        f = tf.identity(f, name="fidelity_coherent")

        return f

    def fidelity_vacuum(self, **kwargs):
        r"""
        Compute the fidelity of the state with the vacuum state. May be numerical or symbolic.

        Args:

        Returns:
            float/Tensor: the numerical value, or an unevaluated Tensor object, for the fidelity :math:`\bra{\vec{0}}\rho\ket{\vec{0}}`.

        """
        mode_size = 1 if self.is_pure else 2
        if self.batched:
            vac_elem = tf.math.real(
                tf.reshape(self.data, [-1, self.cutoff_dim ** (self.num_modes * mode_size)])[:, 0]
            )
        else:
            vac_elem = tf.abs(tf.reshape(self.data, [-1])[0]) ** 2

        v = tf.identity(vac_elem, name="fidelity_vacuum")

        return v

    def is_vacuum(self, tol=0.0, **kwargs):  # pylint:disable=unused-argument
        r"""
        Computes a boolean which indicates whether the state is the vacuum state. May be numerical or symbolic.

        Args:
            tol: numerical tolerance. If the state has fidelity with vacuum within tol, then this method returns True.

        Returns:
            bool/Tensor: the boolean value, or an unevaluated Tensor object, for whether the state is in vacuum.
        """
        fidel = self.fidelity_vacuum()  # dont pass on kwargs yet
        is_vac = tf.less_equal(1 - fidel, tol, name="is_vacuum")
        return is_vac

    def reduced_dm(self, modes, **kwargs):
        r"""
        Computes the reduced density matrix representation of the state. May be numerical or symbolic.

        Args:
            modes (int or Sequence[int]): specifies the mode(s) to return the reduced
                                density matrix for.

        Returns:
            array/Tensor: the numerical value, or an unevaluated Tensor object, for the density matrix.
        """
        if modes == list(range(self.num_modes)):
            # reduced state is full state
            return self.dm(**kwargs)

        if isinstance(modes, int):
            modes = [modes]
        if modes != sorted(modes):
            raise ValueError("The specified modes cannot be duplicated.")
        if len(modes) > self.num_modes:
            raise ValueError(
                "The number of specified modes cannot " "be larger than the number of subsystems."
            )

        reduced = self.dm(**kwargs)
        reduced = reduced_density_matrix(reduced, modes, False, batched=self.batched)

        s = tf.identity(reduced, name="reduced_density_matrix")
        return s

    def quad_expectation(self, mode, phi=0.0, **kwargs):
        r"""
        Compute the expectation value of the quadrature operator :math:`\hat{x}_\phi` for the reduced state on the specified mode. May be numerical or symbolic.

        Args:
            mode (int): which subsystem to take the expectation value of
            phi (float): rotation angle for the quadrature operator

        Returns:
                float/Tensor: the numerical value, or an unevaluated Tensor object, for the expectation value
        """
        rho = self.reduced_dm([mode])  # don't pass kwargs yet

        phi = tf.convert_to_tensor(phi)
        if self.batched and len(phi.shape) == 0:  # pylint: disable=len-as-condition
            phi = tf.expand_dims(phi, 0)
        larger_cutoff = self.cutoff_dim + 1  # start one dimension higher to avoid truncation errors
        R = phase_shifter_matrix(phi, larger_cutoff, batched=self.batched, dtype=self.dtype)
        if not self.batched:
            R = tf.expand_dims(R, 0)  # add fake batch dimension

        a, ad = ladder_ops(larger_cutoff)
        x = np.sqrt(self._hbar / 2.0) * (a + ad)
        x = tf.expand_dims(tf.cast(x, self.dtype), 0)  # add batch dimension to x
        quad = tf.math.conj(R) @ x @ R
        quad2 = (quad @ quad)[:, : self.cutoff_dim, : self.cutoff_dim]
        quad = quad[
            :, : self.cutoff_dim, : self.cutoff_dim
        ]  # drop highest dimension; remaining array gives correct truncated x**2

        if not self.batched:
            rho = tf.expand_dims(rho, 0)  # add fake batch dimension

        flat_rho = tf.reshape(rho, [-1, self.cutoff_dim ** 2])
        flat_quad = tf.reshape(quad, [1, self.cutoff_dim ** 2])
        flat_quad2 = tf.reshape(quad2, [1, self.cutoff_dim ** 2])

        e = tf.math.real(
            tf.reduce_sum(flat_rho * flat_quad, axis=1)
        )  # implements a batched tr(rho * x)
        e2 = tf.math.real(
            tf.reduce_sum(flat_rho * flat_quad2, axis=1)
        )  # implements a batched tr(rho * x ** 2)
        v = e2 - e ** 2

        if not self.batched:
            e = tf.squeeze(e, 0)  # drop fake batch dimension
            v = tf.squeeze(v, 0)  # drop fake batch dimension

        e = tf.identity(e, name="quad_expectation")
        v = tf.identity(v, name="quad_variance")

        return e, v

    def mean_photon(self, mode, **kwargs):
        r"""
        Compute the mean photon number for the reduced state on the specified mode. May be numerical or symbolic.

        Args:
            mode (int): which subsystem to take the mean photon number of

        Returns:
            tuple(float/Tensor): tuple containing the numerical value, or an unevaluated Tensor object, for the mean photon number and variance.
        """
        rho = self.reduced_dm([mode])  # don't pass kwargs yet

        if not self.batched:
            rho = tf.expand_dims(rho, 0)  # add fake batch dimension

        n = np.diag(np.arange(self.cutoff_dim))
        flat_n = np.reshape(n, [1, self.cutoff_dim ** 2])

        flat_rho = tf.reshape(rho, [-1, self.cutoff_dim ** 2])

        nbar = tf.math.real(
            tf.reduce_sum(flat_rho * flat_n, axis=1)
        )  # implements a batched tr(rho * n)
        nbarSq = tf.math.real(
            tf.reduce_sum(flat_rho * flat_n ** 2, axis=1)
        )  # implements a batched tr(rho * n^2)
        var = nbarSq - nbar ** 2
        if not self.batched:
            nbar = tf.squeeze(nbar, 0)  # drop fake batch dimension
            var = tf.squeeze(var, 0)  # drop fake batch dimension

        nbar = tf.identity(nbar, name="mean_photon")
        var = tf.identity(var, name="mean_photon_variance")

        return nbar, var

    def ket(self, **kwargs):
        r"""
        Computes the ket representation of the state. May be numerical or symbolic.

        Args:

        Returns:
            array/Tensor: the numerical value, or an unevaluated Tensor object, for the ket.
        """
        if not self.is_pure:
            return None

        s = tf.identity(self.data, name="ket")
        return s

    def dm(self, **kwargs):
        r"""
        Computes the density matrix representation of the state. May be numerical or symbolic.

        Args:

        Returns:
            array/Tensor: the numerical value, or an unevaluated Tensor object, for the density matrix.
        """
        if self.is_pure:
            if self.batched:
                batch_index = indices[0]
                free_indices = indices[1:]
            else:
                batch_index = ""
                free_indices = indices
            ket = self.data
            left_str = [batch_index] + [free_indices[i] for i in range(0, 2 * self.num_modes, 2)]
            right_str = [batch_index] + [free_indices[i] for i in range(1, 2 * self.num_modes, 2)]
            out_str = [batch_index] + [free_indices[: 2 * self.num_modes]]
            einstr = "".join(left_str + [","] + right_str + ["->"] + out_str)
            s = tf.einsum(einstr, ket, tf.math.conj(ket))
        else:
            s = tf.identity(self.data, name="density_matrix")

        return s

    def wigner(self, mode, xvec, pvec):
        r"""Calculates the discretized Wigner function of the specified mode.

        .. warning::

            Calculation of the Wigner function is currently only supported if
            ``eval=True`` and ``batched=False``.

        .. note::

            This code is a modified version of the 'iterative' method of the
            `wigner function provided in QuTiP <http://qutip.org/docs/4.0.2/apidoc/functions.html?highlight=wigner#qutip.wigner.wigner>`_,
            which is released under the BSD license, with the following
            copyright notice:

            Copyright (C) 2011 and later, P.D. Nation, J.R. Johansson,
            A.J.G. Pitchford, C. Granade, and A.L. Grimsmo. All rights reserved.

        Args:
            mode (int): the mode to calculate the Wigner function for
            xvec (array): array of discretized :math:`x` quadrature values
            pvec (array): array of discretized :math:`p` quadrature values

        Returns:
            array: 2D array of size [len(xvec), len(pvec)], containing reduced Wigner function
            values for specified x and p values.
        """
        if not self.batched:
            return super().wigner(mode, xvec, pvec)

        raise NotImplementedError(
            "Calculation of the Wigner function is currently " "only supported when batched=False"
        )

    def poly_quad_expectation(self, A, d=None, k=0, phi=0, **kwargs):  # pragma: no cover
        r"""The multi-mode expectation values and variance of arbitrary 2nd order polynomials
        of quadrature operators.

        .. warning::

            Calculation of multi-mode quadratic expectation values is currently only supported if
            ``eval=True`` and ``batched=False``.

        An arbitrary 2nd order polynomial of quadrature operators over $N$ modes can always
        be written in the following form:

        .. math:: P(\mathbf{r}) = \mathbf{r}^T A\mathbf{r} + \mathbf{r}^T \mathbf{d} + k I

        where:

        * :math:`A\in\mathbb{R}^{2N\times 2N}` is a symmetric matrix
          representing the quadratic coefficients,
        * :math:`\mathbf{d}\in\mathbb{R}^{2N}` is a real vector representing
          the linear coefficients,
        * :math:`k\in\mathbb{R}` represents the constant term, and
        * :math:`\mathbf{r} = (\x_1,\dots,\x_N,\p_1,\dots,\p_N)` is the vector
          of quadrature operators in :math:`xp`-ordering.

        This method returns the expectation value of this second-order polynomial,

        .. math:: \langle P(\mathbf{r})\rangle,

        as well as the variance

        .. math:: \Delta P(\mathbf{r})^2 = \langle P(\mathbf{r})^2\rangle - \braket{P(\mathbf{r})}^2

        Args:
            A (array): a real symmetric 2Nx2N NumPy array, representing the quadratic
                coefficients of the second order quadrature polynomial.
            d (array): a symmetric length-2N NumPy array, representing the linear
                coefficients of the second order quadrature polynomial. Defaults to the zero vector.
            k (float): the constant term. Default 0.
            phi (float): quadrature angle, clockwise from the positive :math:`x` axis. If provided,
                the vector of quadrature operators :math:`\mathbf{r}` is first rotated
                by angle :math:`\phi` in the phase space.

        Returns:
            tuple (float, float): expectation value and variance
        """

        if not self.batched:
            return super().poly_quad_expectation(A, d, k, phi, **kwargs)

        raise NotImplementedError(
            "Calculation of multi-mode quadratic expectation values is currently "
            "only supported when batched=False."
        )

    @property
    def batched(self):
        """The number of batches."""
        return self._batched

    @property
    def dtype(self):
        """The circuit dtype"""
        return self._dtype

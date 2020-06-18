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
"""Fock backend proper"""
# pylint: disable=too-many-arguments,len-as-condition,attribute-defined-outside-init
# pylint: disable=too-many-branches,too-many-locals,too-many-public-methods

import copy
import string
import numbers
from itertools import product

import numpy as np
from numpy import sqrt, pi
from scipy.special import factorial as bang
from numba import jit
from numba.typed import List

from . import ops

def_type = np.complex128
indices = string.ascii_lowercase
MAX_MODES = len(indices) - 3


class Circuit:
    """
    Class implementing a basic simulator for a collection of modes
    in the fock basis.
    """

    def __init__(self, num, trunc, pure=True, do_checks=False):
        r"""Class initializer.

        Args:
            num (non-negative int): Number of modes in the register.
            trunc (positive int): Truncation parameter.  Fock states up to |trunc-1> are representable.
            pure (bool, optional): Whether states are pure (True) or mixed (False)
            do_checks (bool, optional): Whether arguments are to be checked first
            mode (str, optional): Whether to use BLAS or einsum for matrix operations.
        """

        # Check validity
        if num < 0:
            raise ValueError("Number of modes must be non-negative -- got {}".format(num))
        if num > MAX_MODES:
            raise ValueError("Fock simulator has a maximum of {} modes".format(MAX_MODES))
        if trunc <= 0:
            raise ValueError("Truncation must be positive -- got {}".format(trunc))

        self._num_modes = num
        self._hbar = 2
        self._checks = do_checks
        self.reset(pure=pure, cutoff_dim=trunc)

    def _apply_channel(self, kraus_ops, modes):
        """Master channel application function. Applies a channel represented by
        Kraus operators.

        .. note::
                Always results in a mixed state.

        Args:
            kraus_ops (list<array>): A list of Kraus operators
            modes (list<non-negative int>): The modes to apply the channel to
        """

        if self._pure:
            self._state = ops.mix(self._state, self._num_modes)
            self._pure = False

        if len(kraus_ops) == 0:
            self._state = np.zeros(
                [self._trunc for i in range(self._num_modes * 2)], dtype=ops.def_type
            )
        else:
            states = [self.apply_gate_BLAS(k, modes, pure=False) for k in kraus_ops]
            self._state = sum(states)

    def reset(self, pure=None, cutoff_dim=None, num_subsystems=None):
        """Resets the simulation state.

        Args:
            pure (bool, optional): Sets the purity setting. Default is unchanged.
            cutoff_dim (int): New Hilbert space truncation dimension.
            num_subsystems (int, optional): Sets the number of modes in the reset
                circuit. Default is unchanged.
        """
        if pure is not None:
            if not isinstance(pure, bool):
                raise ValueError("Argument 'pure' must be either True or False")
            self._pure = pure

        if num_subsystems is not None:
            if not isinstance(num_subsystems, int):
                raise ValueError("Argument 'num_subsystems' must be a positive integer")
            self._num_modes = num_subsystems

        if cutoff_dim is not None:
            if not isinstance(cutoff_dim, int) or cutoff_dim < 1:
                raise ValueError("Argument 'cutoff_dim' must be a positive integer")
            self._trunc = cutoff_dim

        if self._pure:
            self._state = ops.vacuumState(self._num_modes, self._trunc)
        else:
            self._state = ops.vacuumStateMixed(self._num_modes, self._trunc)

    def apply_gate_BLAS(self, mat, modes, **kwargs):
        """Applies a specified gate to the state.

        Gate application based on custom indexing and matrix multiplication.
        Assumes the input matrix has shape (out1, in1, ...).

        This implementation uses indexing and BLAS. As per stack overflow,
        einsum doesn't actually use BLAS but rather a c implementation. In theory
        if reshaping is efficient this should be faster.

        Args:
            mat (array[complex]): The numeric operator to be applied to the state, of shape ``[trunc]*(2*n)``
            modes (list[int]): The list of modes to which the operator is applied on

        Keyword args:
            state (array[complex]): The state that the operator is applied to
            pure (bool): Whether the state is pure or mixed
            n (int): The total number of modes

        Returns:
            array[complex]: The state after application of the two-mode operation
        """
        # pylint: disable=too-many-branches

        state = kwargs.get("state", self._state)
        pure = kwargs.get("pure", self._pure)
        n = kwargs.get("n", self._num_modes)

        size = len(modes)
        dim = self._trunc ** size
        stshape = [self._trunc for i in range(size)]

        # Apply the following matrix transposition:
        # |m1><m1| |m2><m2| ... |mn><mn| -> |m1>|m2>...|mn><m1|<m2|...<mn|
        transpose_list = [2 * i for i in range(size)] + [2 * i + 1 for i in range(size)]
        matview = np.transpose(mat, transpose_list).reshape((dim, dim))

        # checks if matview is diagonal (for example, Kerr/rotation gates) and, if so, use faster contractions
        diag = np.all(matview == np.diag(np.diagonal(matview)))

        if pure:
            if n == 1:
                if diag:
                    mat_diag = mat.diagonal()
                    return np.multiply(mat_diag, state)
                return np.dot(mat, state)

            # Transpose the state into the following form:
            # |psi> |mode[0]> |mode[1]> ... |mode[n]>
            transpose_list = [i for i in range(n) if not i in modes] + modes
            view = np.transpose(state, transpose_list)

            # Apply matrix to each substate
            ret = np.zeros([self._trunc for i in range(n)], dtype=def_type)
            for i in product(*([range(self._trunc) for j in range(n - size)])):
                if diag:
                    mat_diag = matview.diagonal()
                    ret[i] = np.multiply(mat_diag, view[i].ravel()).reshape(stshape)
                else:
                    ret[i] = np.dot(matview, view[i].ravel()).reshape(stshape)

            # "untranspose" the return matrix ret
            untranspose_list = [0] * len(transpose_list)
            for i in range(len(transpose_list)):  # pylint: disable=consider-using-enumerate
                untranspose_list[transpose_list[i]] = i

            return np.transpose(ret, untranspose_list)

        # otherwise, the state is mixed
        if n == 1:
            if diag:
                mat_diag = mat.diagonal().reshape(-1, 1)
                return np.multiply(mat_diag, np.multiply(state, mat_diag.conj().T))
            return np.dot(mat, np.dot(state, mat.conj().T))

        # Transpose the state into the following form:
        # |psi><psi||mode[0]>|mode[1]>...|mode[n]><mode[0]|<mode[1]|...<mode[n]|
        transpose_list = [i for i in range(n * 2) if not i // 2 in modes]
        transpose_list = transpose_list + [2 * i for i in modes] + [2 * i + 1 for i in modes]
        view = np.transpose(state, transpose_list)

        # Apply matrix to each substate
        ret = np.zeros([self._trunc for i in range(n * 2)], dtype=def_type)
        for i in product(*([range(self._trunc) for j in range((n - size) * 2)])):
            if diag:
                mat_diag = matview.diagonal().reshape(-1, 1)
                ret[i] = np.multiply(
                    mat_diag, np.multiply(view[i].reshape((dim, dim)), mat_diag.conj().T)
                ).reshape(stshape + stshape)
            else:
                ret[i] = np.dot(
                    matview, np.dot(view[i].reshape((dim, dim)), matview.conj().T)
                ).reshape(stshape + stshape)

        # "untranspose" the return matrix ret
        untranspose_list = [0] * len(transpose_list)
        for i in range(len(transpose_list)):  # pylint: disable=consider-using-enumerate
            untranspose_list[transpose_list[i]] = i

        return np.transpose(ret, untranspose_list)

    def apply_twomode_gate(self, mat, modes, gate="BSgate"):
        """Applies a two-mode gate to the state.

        Applies the specified two-mode gate to the state using custom tensor contractions and
        the Numba compiler for faster application. Currently, only the beamsplitter and the
        two-mode squeeze gate are supported.

        Args:
            mat (array[complex]): The numeric operator to be applied to the state, of shape `[trunc]*(2*n)`
            modes (list[int]): The list of modes to which the operator is applied on
            gate (str): The gate that is being applied. This argument determines the selection rules that
                are used. Options are ``"BSgate"`` and ``"S2gate"``.

        Returns:
            array[complex]: The state after application of the two-mode operation
        """
        if self._pure:
            t1 = modes[0]
            t2 = modes[1]

            # moves the modes on which the gate should be applied to the
            # front indices of the state, for mode 1 and 2 respectively
            # e.g. from [i1, i2, i3] --> [i2, i3, i1] if the gate is applied to modes 2 and 3
            switch_list_1 = np.arange(self._num_modes)
            switch_list_2 = np.arange(self._num_modes)
            switch_list_1[[0, t1]] = switch_list_1[[t1, 0]]
            switch_list_2[[1, t2]] = switch_list_2[[t2, 1]]

            self._state = self._state.transpose(switch_list_1)
            self._state = self._state.transpose(switch_list_2)

            if gate == "BSgate":
                self._state = self._apply_BS(mat, self._state, self._trunc)
            elif gate == "S2gate":
                self._state = self._apply_S2(mat, self._state, self._trunc)
            else:
                raise NotImplementedError(
                    "Currently, selection rules are only implemented for the BSgate "
                    "and the S2gate. The {} gate is not supported".format(gate)
                )

            self._state = self._state.transpose(switch_list_2)
            ret = self._state.transpose(switch_list_1)
        else:
            t1 = 2 * modes[0]
            t2 = 2 * modes[1]

            # moves the modes on which the gate should be applied to the
            # front indices of the state, for mode 1 and 2 respectively
            # e.g. from [i1, j1, i2, j2, i3, j3] --> [i2, j2, i3, j3, i1, j1]
            # if the gate is applied to modes 2 and 3
            switch_list_1 = np.arange(2 * self._num_modes)
            switch_list_2 = np.arange(2 * self._num_modes)
            switch_list_1[[0, 1, t1, t1 + 1]] = switch_list_1[[t1, t1 + 1, 0, 1]]
            switch_list_2[[0, 1, t2, t2 + 1]] = switch_list_2[[t2, t2 + 1, 0, 1]]

            # puts the modes on which the gate should be applied together
            # e.g. from [i2, j2, i3, j3, i1, j1] --> [i2, i3, j2, j3, i1, j1]
            # if the gate is applied to modes 2 and 3 (which are already switched to the first two modes)
            transpose_list = np.arange(2 * self._num_modes)
            transpose_list[[t1 + 1, t2]] = transpose_list[[t2, t1 + 1]]

            self._state = self._state.transpose(transpose_list)
            self._state = self._state.transpose(switch_list_1)

            if gate == "BSgate":
                self._state = self._apply_BS(mat, self._state, self._trunc)
                self._state = self._state.transpose(switch_list_1)

                self._state = self._state.transpose(switch_list_2)
                self._state = self._apply_BS(mat.conj(), self._state, self._trunc)

            elif gate == "S2gate":
                self._state = self._apply_S2(mat, self._state, self._trunc)
                self._state = self._state.transpose(switch_list_1)

                self._state = self._state.transpose(switch_list_2)
                self._state = self._apply_S2(mat.conj(), self._state, self._trunc)

            else:
                raise NotImplementedError

            self._state = self._state.transpose(switch_list_2)
            ret = self._state.transpose(transpose_list)

        return ret

    # ignored in covtest (doesn't work well with the jit decorator)
    @staticmethod
    @jit(nopython=True)
    def _apply_BS(mat, state, trunc):  # pragma: no cover
        r"""Applies the BS gate to the first bra in state.

        The beamsplitter matrix elements :math:`B_{ij}^{kl}` satisfy the selection
        rules

        .. math::
            B_{ij}^{kl} \propto \delta_{i+j}^{k+l}

        This implies that one can contract pure states (without loss of generality
        we assume only two modes) using one less loop as follows:

        .. math::
            c'_{i, j} = \sum\limits_{k = \text{max}(1+i+j-n, 0)}
                    ^{\text{min}(i+j, n-1) + 1}
            B_{ik}^{j, i+j-k} c_{k, i+j-k}

        where :math:`c_{kl}` is the tensor representing the two mode state and
        :math:`n` is the cutoff in Fock space. Similarly for a mixed state it is
        possible to do the update of the state using two less summations.
        """
        ret = np.zeros_like(state, dtype=np.complex128)
        for i in range(trunc):
            for j in range(trunc):
                for k in range(max(1 + i + j - trunc, 0), min(i + j, trunc - 1) + 1):
                    ret[i, j] += mat[i, k, j, i + j - k] * state[k, i + j - k]
        return ret

    # ignored in covtest (doesn't work well with the jit decorator)
    @staticmethod
    @jit(nopython=True)
    def _apply_S2(mat, state, trunc):  # pragma: no cover
        r"""Applies the S2 gate to the first bra in state.

        The two-mode squeeze matrix elements :math:`T_{ij}^{kl}` satisfy the selection
        rules

        .. math::
            T_{ij}^{kl} \propto \delta_{i-j}^{k-l}

        This implies that one can contract pure states (without loss of generality
        we assume only two modes) using one less loop as follows:

        .. math::
            c'_{i, k} = \sum\limits_{k = \text{max}(i-j, 0)}^{\text{min}(i-j, 0) + n}
            T_{ij}^{k, k+j-i} c_{j, k+j-i}

        where :math:`c_{jl}` is the tensor representing the two mode state and
        :math:`n` is the cutoff in Fock space. Similarly for a mixed state it is
        possible to do the update of the state using two less summations.
        """
        ret = np.zeros_like(state, dtype=np.complex128)
        for i in range(trunc):
            for j in range(trunc):
                for k in range(max(i - j, 0), trunc + min(i - j, 0)):
                    ret[i, k] += mat[i, j, k, k + j - i] * state[j, k + j - i]
        return ret

    def norm(self):
        """returns the norm of the state"""
        if self._pure:
            return sqrt(np.vdot(self._state, self._state).real)
        return ops.trace(self._state, self._num_modes)

    def alloc(self, n=1):
        """allocate a number of modes at the end of the state."""
        # base_shape = [self._trunc for i in range(n)]
        if self._pure:
            vac = ops.vacuumState(n, self._trunc)
        else:
            vac = ops.vacuumStateMixed(n, self._trunc)

        self._state = ops.tensor(self._state, vac, self._num_modes, self._pure)
        self._num_modes = self._num_modes + n

    def dealloc(self, modes):
        """Traces out and deallocates the modes in `modes`"""
        if self._pure:
            self._state = ops.mix(self._state, self._num_modes)
            self._pure = False

        self._state = ops.partial_trace(self._state, self._num_modes, modes)
        self._num_modes = self._num_modes - len(modes)

    def prepare_multimode(self, state, modes):
        r"""
        Prepares a given mode or list of modes in the given state.

        After the preparation the system is in a mixed product state,
        with the specified modes replaced by state.
        The given state can be either in tensor form or in matrix/vector form.
        If modes is not ordered, the subsystems of the input are
        reordered to reflect that, i.e., if modes=[3,1], then the first mode
        of state ends up in mode 3 and the second mode of state ends up in
        mode 1 of the output state.
        If modes is None, it is attempted to prepare state in all modes.
        The reduced state on all other modes remains unchanged and
        the final state is product with respect to the partition into
        the modes in modes and the complement.

        Args:
            state (array): vector, matrix, or tensor representation of the ket state or dm state in the fock basis to prepare
            modes (list[int] or non-negative int): The mode(s) into which state is to be prepared. Needs not be ordered.
        """
        if isinstance(modes, int):
            modes = [modes]

        n_modes = len(modes)
        pure_shape = tuple([self._trunc] * n_modes)
        mixed_shape = tuple([self._trunc] * (2 * n_modes))
        pure_shape_as_vector = tuple([self._trunc ** n_modes])
        mixed_shape_as_matrix = tuple([self._trunc ** n_modes] * 2)

        # Do consistency checks
        if self._checks:
            if (
                state.shape != pure_shape
                and state.shape != mixed_shape
                and state.shape != pure_shape_as_vector
                and state.shape != mixed_shape_as_matrix
            ):
                raise ValueError("Incorrect shape for state preparation")
            if len(modes) != len(set(modes)):
                raise ValueError("The specified modes cannot appear multiple times.")

        # reshape to support input both as tensor and vector/matrix
        if state.shape == pure_shape_as_vector:
            state = state.reshape(pure_shape)
        elif state.shape == mixed_shape_as_matrix:
            state = state.reshape(mixed_shape)

        if self._num_modes == n_modes:
            # Hack for marginally faster state preparation
            self._state = state.astype(ops.def_type)
            self._pure = bool(state.shape == pure_shape)
        else:
            if self._pure:
                self._state = ops.mix(self._state, self._num_modes)
                self._pure = False

            if state.shape == pure_shape:
                state = ops.mix(state, len(modes))

            # Take the partial trace
            # todo: For performance the partial trace could be done directly from the pure state. This would of course require a better partial trace function...
            reduced_state = ops.partial_trace(self._state, self._num_modes, modes)

            # Insert state at the end (I know there is also tensor() from ops but it has extra aguments wich only confuse here)
            self._state = np.tensordot(reduced_state, state, axes=0)

            # unless the preparation was meant to go into the last modes in the standard order, we need to swap indices around
        if modes != list(range(self._num_modes - len(modes), self._num_modes)):
            mode_permutation = [x for x in range(self._num_modes) if x not in modes] + modes
            if self._pure:
                scale = 1
                index_permutation = mode_permutation
            else:
                scale = 2
                index_permutation = [
                    scale * x + i for x in mode_permutation for i in (0, 1)
                ]  # two indices per mode if we have pure states
            index_permutation = np.argsort(index_permutation)

            self._state = np.transpose(self._state, index_permutation)

    def prepare(self, state, mode):
        r"""
        Prepares a given mode in a given state.

        This is a simple wrapper for prepare_multimode(), see there for more details.

        Args:
            state (array): vector, matrix, or tensor representation of the ket state or dm state in the fock basis to prepare
            modes (list[int] or non-negative int or None): The mode(s) into which state is to be prepared. Needs not be ordered.
        """
        if isinstance(mode, int):
            mode = [mode]
        self.prepare_multimode(state, mode)

    def prepare_mode_fock(self, n, mode):
        """
        Prepares a mode in a fock state.
        """

        if self._pure:
            self.prepare(ops.fockState(n, self._trunc), mode)
        else:
            st = ops.fockState(n, self._trunc)
            self.prepare(np.outer(st, st.conjugate()), mode)

    def prepare_mode_coherent(self, r, phi, mode):
        """
        Prepares a mode in a coherent state.
        """
        if self._pure:
            self.prepare(ops.coherentState(r, phi, self._trunc), mode)
        else:
            st = ops.coherentState(r, phi, self._trunc)
            self.prepare(np.outer(st, st.conjugate()), mode)

    def prepare_mode_squeezed(self, r, theta, mode):
        """
        Prepares a mode in a squeezed state.
        """
        if self._pure:
            self.prepare(ops.squeezedState(r, theta, self._trunc), mode)
        else:
            st = ops.squeezedState(r, theta, self._trunc)
            self.prepare(np.outer(st, st.conjugate()), mode)

    def prepare_mode_displaced_squeezed(self, r_d, phi_d, r_s, phi_s, mode):
        """
        Prepares a mode in a displaced squeezed state.
        """
        if self._pure:
            self.prepare(ops.displacedSqueezed(r_d, phi_d, r_s, phi_s, self._trunc), mode)
        else:
            st = ops.displacedSqueezed(r_d, phi_d, r_s, phi_s, self._trunc)
            self.prepare(np.outer(st, st.conjugate()), mode)

    def prepare_mode_thermal(self, nbar, mode):
        """
        Prepares a mode in a thermal state.
        """
        st = ops.thermalState(nbar, self._trunc)
        self.prepare(st, mode)

    def phase_shift(self, theta, mode):
        """
        Applies a phase shifter.
        """
        mat = ops.phase(theta, self._trunc)
        self._state = self.apply_gate_BLAS(mat, [mode])

    def displacement(self, r, phi, mode):
        """
        Applies a displacement gate.
        """
        mat = ops.displacement(r, phi, self._trunc)
        self._state = self.apply_gate_BLAS(mat, [mode])

    def beamsplitter(self, theta, phi, mode1, mode2):
        """
        Applies a beamsplitter.
        """
        mat = ops.beamsplitter(theta, phi, self._trunc)
        self._state = self.apply_twomode_gate(mat, [mode1, mode2], gate="BSgate")

    def squeeze(self, r, theta, mode):
        """
        Applies a squeezing gate.
        """
        mat = ops.squeezing(r, theta, self._trunc)
        self._state = self.apply_gate_BLAS(mat, [mode])

    def two_mode_squeeze(self, r, theta, mode1, mode2):
        """
        Applies a two-mode squeezing gate.
        """
        mat = ops.two_mode_squeeze(r, theta, self._trunc)
        self._state = self.apply_twomode_gate(mat, [mode1, mode2], gate="S2gate")

    def kerr_interaction(self, kappa, mode):
        """
        Applies a Kerr interaction gate.
        """
        mat = ops.kerr(kappa, self._trunc)
        self._state = self.apply_gate_BLAS(mat, [mode])

    def cross_kerr_interaction(self, kappa, mode1, mode2):
        """
        Applies a cross-Kerr interaction gate.
        """
        mat = ops.cross_kerr(kappa, self._trunc)
        self._state = self.apply_gate_BLAS(mat, [mode1, mode2])

    def cubic_phase_shift(self, gamma, mode):
        """
        Applies a cubic phase shift gate.
        """
        mat = ops.cubicPhase(gamma, self._hbar, self._trunc)
        self._state = self.apply_gate_BLAS(mat, [mode])

    def is_vacuum(self, tol):
        """
        Tests whether the system is in the vacuum state.
        """
        # fid = <0|\rho|0>
        if self._pure:
            fid = np.abs(self._state.flat[0]) ** 2
        else:
            fid = self._state.flat[0]
        return np.abs(fid - 1) <= tol

    def get_state(self):
        """
        Returns the state of the system in the fock basis along with its purity.
        """
        return self._state, self._pure

    def loss(self, T, mode):
        """
        Applies a loss channel to the state.
        """
        self._apply_channel(ops.lossChannel(T, self._trunc), [mode])

    def measure_fock(self, modes, select=None):
        """
        Measures a list of modes.
        """
        # pylint: disable=singleton-comparison
        if select is not None and np.any(np.array(select) == None):
            raise NotImplementedError("Post-selection lists must only contain numerical values.")

        # Make sure the state is mixed
        if self._pure:
            state = ops.mix(self._state, self._num_modes)
        else:
            state = self._state

        if select is not None:
            # perform post-selection

            # make sure modes and select are the same length
            if len(select) != len(modes):
                raise ValueError(
                    "When performing post-selection, the number of "
                    "selected values (including None) must match the number of measured modes"
                )

            # make sure the select values are all integers or nones
            if not all(isinstance(s, int) or s is None for s in select):
                raise TypeError("The post-select list elements either be integers or None")

            # modes to measure
            measure = [i for i, s in zip(modes, select) if s is None]

            # modes already post-selected:
            selected = [i for i, s in zip(modes, select) if s is not None]
            select_values = [s for s in select if s is not None]

            # project out postselected modes
            self._state = ops.project_reset(
                selected, select_values, self._state, self._pure, self._num_modes, self._trunc
            )

            if self.norm() == 0:
                raise ZeroDivisionError("Measurement has zero probability.")

            self._state = self._state / self.norm()

        else:
            # no post-selection; modes to measure are the modes provided
            measure = modes

        if len(measure) > 0:
            # sampling needs to be performed
            # Compute distribution by tracing out modes not measured, then computing the diagonal
            unmeasured = [i for i in range(self._num_modes) if i not in measure]
            reduced = ops.partial_trace(state, self._num_modes, unmeasured)
            dist = np.ravel(ops.diagonal(reduced, len(measure)).real)
            # kill spurious tiny values (which are sometimes negative)
            dist = dist * ~np.isclose(dist, 0.0)

            # Make a random choice
            if sum(dist) != 1:
                # WARNING: distribution is not normalized, could hide errors
                i = np.random.choice(list(range(len(dist))), p=dist / sum(dist))
            else:
                i = np.random.choice(list(range(len(dist))), p=dist)

            permuted_outcome = ops.unIndex(i, len(measure), self._trunc)

            # Permute the outcome to match the order of the modes in 'measure'
            permutation = np.argsort(measure)
            outcome = [0] * len(measure)
            for i in range(len(measure)):
                outcome[permutation[i]] = permuted_outcome[i]

            # Project the state onto the measurement outcome & reset in vacuum
            self._state = ops.project_reset(
                measure, outcome, self._state, self._pure, self._num_modes, self._trunc
            )

            if self.norm() == 0:
                raise ZeroDivisionError("Measurement has zero probability.")

            self._state = self._state / self.norm()

        # include post-selected values in measurement outcomes
        if select is not None:
            outcome = copy.copy(select)

        # `outcome` will always be a 1-d sequence since multiple shots is not supported
        return np.array([outcome])

    def measure_homodyne(self, phi, mode, select=None, **kwargs):
        """
        Performs a homodyne measurement on a mode.
        """
        m_omega_over_hbar = 1 / self._hbar

        # Make sure the state is mixed for reduced density matrix
        if self._pure:
            state = ops.mix(self._state, self._num_modes)
        else:
            state = self._state

        if select is not None:
            meas_result = select
            if isinstance(meas_result, numbers.Number):
                homodyne_sample = float(meas_result)
            else:
                raise TypeError("Selected measurement result must be of numeric type.")
        else:
            # Compute reduced density matrix
            unmeasured = [i for i in range(self._num_modes) if not i == mode]
            reduced = ops.partial_trace(state, self._num_modes, unmeasured)

            # Rotate to measurement basis
            reduced = self.apply_gate_BLAS(
                ops.phase(-phi, self._trunc), [0], state=reduced, pure=False, n=1
            )

            # Create pdf. Same as tf implementation, but using
            # the recursive relation H_0(x) = 1, H_1(x) = 2x, H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)
            q_mag = kwargs.get("max", 10)
            num_bins = kwargs.get("num_bins", 100000)

            q_tensor, Hvals = ops.hermiteVals(q_mag, num_bins, m_omega_over_hbar, self._trunc)
            H_matrix = np.zeros((self._trunc, self._trunc, num_bins))
            for n, m in product(range(self._trunc), repeat=2):
                H_matrix[n][m] = 1 / sqrt(2 ** n * bang(n) * 2 ** m * bang(m)) * Hvals[n] * Hvals[m]
            H_terms = np.expand_dims(reduced, -1) * np.expand_dims(H_matrix, 0)
            rho_dist = (
                np.sum(H_terms, axis=(1, 2))
                * (m_omega_over_hbar / pi) ** 0.5
                * np.exp(-m_omega_over_hbar * q_tensor ** 2)
                * (q_tensor[1] - q_tensor[0])
            )  # Delta_q for normalization (only works if the bins are equally spaced)

            # Sample from rho_dist. This is a bit different from tensorflow due to how
            # numpy treats multinomial sampling. In particular, numpy returns a
            # histogram of the samples whereas tensorflow gives the list of samples.
            # Numpy also does not use the log probabilities
            probs = rho_dist.flatten().real
            probs /= np.sum(probs)

            # Due to floating point precision error, values in the calculated probability distribution
            # may have a very small negative value of -epsilon. The following sets
            # these small negative values to 0.
            probs[np.abs(probs) < 1e-10] = 0

            sample_hist = np.random.multinomial(1, probs)
            sample_idx = list(sample_hist).index(1)
            homodyne_sample = q_tensor[sample_idx]

        # Project remaining modes into the conditional state
        inf_squeezed_vac = np.array(
            [
                (-0.5) ** (n // 2) * sqrt(bang(n)) / bang(n // 2) if n % 2 == 0 else 0.0 + 0.0j
                for n in range(self._trunc)
            ],
            dtype=ops.def_type,
        )
        alpha = homodyne_sample * sqrt(m_omega_over_hbar / 2)

        composed = np.dot(
            ops.phase(phi, self._trunc),
            ops.displacement(np.abs(alpha), np.angle(alpha), self._trunc),
        )
        eigenstate = self.apply_gate_BLAS(composed, [0], state=inf_squeezed_vac, pure=True, n=1)

        vac_state = np.array(
            [1.0 + 0.0j if i == 0 else 0.0 + 0.0j for i in range(self._trunc)], dtype=ops.def_type
        )
        projector = np.outer(vac_state, eigenstate.conj())

        self._state = self.apply_gate_BLAS(projector, [mode])

        # Normalize
        self._state = self._state / self.norm()

        # `homodyne_sample` will always be a single value since multiple shots is not supported
        return np.array([[homodyne_sample]])

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
r"""
This module provides abstract base classes which represent the quantum state
returned by a simulator backend via :class:`.Engine`.
"""
import abc
import string
from itertools import chain
from copy import copy

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from scipy.special import factorial
from scipy.integrate import simps

import thewalrus.quantum as twq

import strawberryfields as sf

from .shared_ops import rotation_matrix as _R
from .shared_ops import changebasis

indices = string.ascii_lowercase


class BaseState(abc.ABC):
    r"""Abstract base class for the representation of quantum states."""
    # pylint: disable=too-many-public-methods
    EQ_TOLERANCE = 1e-10

    def __init__(self, num_modes, mode_names=None):
        self._modes = num_modes
        self._hbar = sf.hbar  # always use the global frontend hbar value for state objects
        self._data = None
        self._pure = None

        if mode_names is None:
            self._modemap = {i: "mode {}".format(i) for i in range(num_modes)}
        else:
            self._modemap = {i: "{}".format(j) for i, j in zip(range(num_modes), mode_names)}

        self._str = "<BaseState: num_modes={}, pure={}, hbar={}>".format(
            self.num_modes, self._pure, self._hbar
        )

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._str

    @property
    def data(self):
        r"""Returns the underlying numerical (or symbolic) representation of the state.
        The form of this data differs for different backends."""
        return self._data

    @property
    def hbar(self):
        r"""Returns the value of :math:`\hbar` used in the generation of the state.

        The value of :math:`\hbar` is a convention chosen in the definition of
        :math:`\x` and :math:`\p`. See :ref:`opcon` for more details.

        Returns:
            float: :math:`\hbar` value.
        """
        return self._hbar

    @property
    def is_pure(self):
        r"""Checks whether the state is a pure state.

        Returns:
            bool: True if and only if the state is pure.
        """
        return self._pure

    @property
    def num_modes(self):
        r"""Gets the number of modes that the state represents.

        Returns:
            int: the number of modes in the state
        """
        return self._modes

    @property
    def mode_names(self):
        r"""Returns a dictionary mapping the mode index to mode names.

        The mode names are determined from the initialization argument
        ``mode_names``. If these were not supplied, the names are generated automatically based
        on the mode indices.

        Returns:
            dict: dictionary of the form ``{i:"mode name",...}``
        """
        return self._modemap

    @property
    def mode_indices(self):
        r"""Returns a dictionary mapping the mode names to mode indices.

        The mode names are determined from the initialization argument
        ``mode_names``. If these were not supplied, the names are generated automatically based
        on the mode indices.

        Returns:
            dict: dictionary of the form ``{"mode name":i,...}``
        """
        return {v: k for k, v in self._modemap.items()}

    @abc.abstractmethod
    def ket(self, **kwargs):
        r"""The numerical state vector for the quantum state in the Fock basis.
        Note that if the state is mixed, this method returns None.

        Keyword Args:
            cutoff (int): Specifies where to truncate the returned density matrix (default value is 10).
                Note that the cutoff argument only applies for Gaussian representation;
                states represented in the Fock basis will use their own internal cutoff dimension.

        Returns:
            array/None: the numerical state vector. Returns None if the state is mixed.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def dm(self, **kwargs):
        r"""The numerical density matrix for the quantum state in the Fock basis.

        Keyword Args:
            cutoff (int): Specifies where to truncate the returned density matrix (default value is 10).
                Note that the cutoff argument only applies for Gaussian representation;
                states represented in the Fock basis will use their own internal cutoff dimension.

        Returns:
            array: the numerical density matrix in the Fock basis
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reduced_dm(self, modes, **kwargs):
        r"""Returns a reduced density matrix in the Fock basis.

        Args:
            modes (int or Sequence[int]): specifies the mode(s) to return the reduced density matrix for.

        Keyword Args:
            cutoff (int): Specifies where to truncate the returned density matrix (default value is 10).
                Note that the cutoff argument only applies for Gaussian representation;
                states represented in the Fock basis will use their own internal cutoff dimension.

        Returns:
            array: the reduced density matrix for the specified modes
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fock_prob(self, n, **kwargs):
        r"""Probability of a particular Fock basis state.

        Computes the probability :math:`|\braket{\vec{n}|\psi}|^2` of measuring
        the given multi-mode Fock state based on the state :math:`\ket{\psi}`.

        .. warning::

            Computing the Fock probabilities of states has exponential scaling
            in the Gaussian representation (for example states output by a
            Gaussian backend as a :class:`~.BaseGaussianState`).
            This shouldn't affect small-scale problems, where only a few Fock
            basis state probabilities need to be calculated, but will become
            evident in larger scale problems.

        Args:
            n (Sequence[int]): the Fock state :math:`\ket{\vec{n}}` that we want to measure the probability of

        Keyword Args:
            cutoff (int): Specifies where to truncate the computation (default value is 10).
                Note that the cutoff argument only applies for Gaussian representation;
                states represented in the Fock basis will use their own internal cutoff dimension.

        Returns:
            float: measurement probability
        """
        raise NotImplementedError

    @abc.abstractmethod
    def all_fock_probs(self, **kwargs):
        r"""Probabilities of all possible Fock basis states for the current circuit state.

        For example, in the case of 3 modes, this method allows the Fock state probability
        :math:`|\braketD{0,2,3}{\psi}|^2` to be returned via

        .. code-block:: python

            probs = state.all_fock_probs()
            probs[0,2,3]

        Returns:
            array: array of dimension :math:`\underbrace{D\times D\times D\cdots\times D}_{\text{num modes}}`
                containing the Fock state probabilities, where :math:`D` is the Fock basis cutoff truncation

        Keyword Args:
            cutoff (int): Specifies where to truncate the computation (default value is 10).
                Note that the cutoff argument only applies for Gaussian representation;
                states represented in the Fock basis will use their own internal cutoff dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mean_photon(self, mode, **kwargs):
        """Returns the mean photon number of a particular mode.

        Args:
            mode (int): specifies the mode
            **kwargs:

                  * **cutoff** (*int*): (default 10) Fock basis trunction for calculation of
                    mean photon number.
                    Note that the cutoff argument only applies for Gaussian representation;
                    states represented in the Fock basis will use their own internal cutoff dimension.

        Returns:
            tuple: the mean photon number and variance
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fidelity(self, other_state, mode, **kwargs):
        r"""Fidelity of the reduced state in the specified mode with a user supplied state.
        Note that this method only supports single-mode states.

        Args:
            other_state: a pure state vector array represented in the Fock basis (for Fock backends)
                or a Sequence ``(mu, cov)`` containing the means and covariance matrix (for Gaussian backends)

        Returns:
            The fidelity of the circuit state with ``other_state``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fidelity_vacuum(self, **kwargs):
        """The fidelity of the state with the vacuum state.

        Returns:
            float: the fidelity of the state with the vacuum
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fidelity_coherent(self, alpha_list, **kwargs):
        r"""The fidelity of the state with a product of coherent states.

        The fidelity is defined by

        .. math:: \bra{\vec{\alpha}}\rho\ket{\vec{\alpha}}

        Args:
            alpha_list (Sequence[complex]): list of coherent state parameters, one for each mode

        Returns:
            float: the fidelity value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def wigner(self, mode, xvec, pvec):
        r"""Calculates the discretized Wigner function of the specified mode.

        Args:
            mode (int): the mode to calculate the Wigner function for
            xvec (array): array of discretized :math:`x` quadrature values
            pvec (array): array of discretized :math:`p` quadrature values

        Returns:
            array: 2D array of size [len(xvec), len(pvec)], containing reduced Wigner function
            values for specified x and p values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def quad_expectation(self, mode, phi=0, **kwargs):
        r"""The :math:`\x_{\phi}` operator expectation values and variance for the specified mode.

        The :math:`\x_{\phi}` operator is defined as follows,

        .. math:: \x_{\phi} = \cos\phi~\x + \sin\phi~\p

        with corresponding expectation value

        .. math:: \bar{x_{\phi}}=\langle x_{\phi}\rangle = \text{Tr}(\x_{\phi}\rho_{mode})

        and variance

        .. math:: \Delta x_{\phi}^2 = \langle x_{\phi}^2\rangle - \braket{x_{\phi}}^2

        Args:
            mode (int): the requested mode
            phi (float): quadrature angle, clockwise from the positive :math:`x` axis.

                * :math:`\phi=0` corresponds to the :math:`x` expectation and variance (default)
                * :math:`\phi=\pi/2` corresponds to the :math:`p` expectation and variance

        Returns:
            tuple (float, float): expectation value and variance
        """
        raise NotImplementedError

    @abc.abstractmethod
    def poly_quad_expectation(self, A, d=None, k=0, phi=0, **kwargs):
        r"""The multi-mode expectation values and variance of arbitrary 2nd order polynomials
        of quadrature operators.

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

        .. math:: \Delta P(\mathbf{r})^2 = \braket{P(\mathbf{r})^2} - \braket{P(\mathbf{r})}^2

        Args:
            A (array): a real symmetric 2Nx2N NumPy array, representing the quadratic
                coefficients of the second order quadrature polynomial.
            d (array): a real length-2N NumPy array, representing the linear
                coefficients of the second order quadrature polynomial. Defaults to the zero vector.
            k (float): the constant term. Default 0.
            phi (float): quadrature angle, clockwise from the positive :math:`x` axis. If provided,
                the vectori of quadrature operators :math:`\mathbf{r}` is first rotated
                by angle :math:`\phi` in the phase space.


        Returns:
            tuple (float, float): expectation value and variance
        """
        raise NotImplementedError

    @abc.abstractmethod
    def number_expectation(self, modes):
        r"""
        Calculates the expectation value of the product of the number operators of the modes.

        This method computes the analytic expectation value
        :math:`\langle \hat{n}_{i_0} \hat{n}_{i_1}\dots \hat{n}_{i_{N-1}}\rangle`
        for a (sub)set of modes :math:`[i_0, i_1, \dots, i_{N-1}]` of the system.

        Args:
            modes (list): list of modes for which one wants the expectation of the product of their number operator.

        Return:
            tuple[float, float]: the expectation value and variance

        **Example**

        Consider the following program:

        .. code-block:: python

            prog = sf.Program(3)

            with prog.context as q:
                ops.Sgate(0.5) | q[0]
                ops.Sgate(0.5) | q[1]
                ops.Sgate(0.5) | q[2]
                ops.BSgate(np.pi/3, 0.1) |  (q[0], q[1])
                ops.BSgate(np.pi/3, 0.1) |  (q[1], q[2])

        Executing this on the Fock backend,

        >>> eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
        >>> state = eng.run(prog).state

        we can compute the expectation value :math:`\langle \hat{n}_0\hat{n}_2\rangle`:

        >>> state.number_expectation([0, 2])[0]
        0.07252895071309405

        Executing the same program on the Gaussian backend,

        >>> eng = sf.Engine("gaussian")
        >>> state = eng.run(prog).state
        >>> state.number_expectation([0, 2])[0]
        0.07566984755267293

        This slight difference in value compared to the result from the Fock backend above
        is due to the finite Fock basis truncation when using the Fock backend; this can be
        avoided by increasing the value of ``cutoff_dim``.

        .. warning:: This method only supports at most two modes in the Gaussian backend.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parity_expectation(self, modes):
        """Calculates the expectation value of a product of parity operators acting on given modes"""
        raise NotImplementedError

    def p_quad_values(self, mode, xvec, pvec):

        r"""Calculates the discretized p-quadrature probability distribution of the specified mode.

        Args:
            mode (int): the mode to calculate the p-quadrature probability values of
            xvec (array): array of discretized :math:`x` quadrature values
            pvec (array): array of discretized :math:`p` quadrature values

        Returns:
            array: 1D array of size len(pvec), containing reduced p-quadrature
            probability values for a specified range of x and p.
        """

        W = self.wigner(mode, xvec, pvec)
        y = []
        for i in range(0, len(pvec)):
            res = simps(W[i, : len(xvec)], xvec)
            y.append(res)
        return np.array(y)

    def x_quad_values(self, mode, xvec, pvec):

        r"""Calculates the discretized x-quadrature probability distribution of the specified mode.

        Args:
            mode (int): the mode to calculate the x-quadrature probability values of
            xvec (array): array of discretized :math:`x` quadrature values
            pvec (array): array of discretized :math:`p` quadrature values

        Returns:
            array: 1D array of size len(xvec), containing reduced x-quadrature
            probability values for a specified range of x and p.
        """

        W = self.wigner(mode, xvec, pvec)
        y = []
        for i in range(0, len(xvec)):
            res = simps(W[: len(pvec), i], pvec)
            y.append(res)
        return np.array(y)


class BaseFockState(BaseState):
    r"""Class for the representation of quantum states in the Fock basis.

    Args:
        state_data (array): the state representation in the Fock basis
        num_modes (int): the number of modes in the state
        pure (bool): True if the state is a pure state, false if the state is mixed
        cutoff_dim (int): the Fock basis truncation size
        mode_names (Sequence): (optional) this argument contains a list providing mode names
            for each mode in the state
    """

    def __init__(self, state_data, num_modes, pure, cutoff_dim, mode_names=None):
        # pylint: disable=too-many-arguments

        super().__init__(num_modes, mode_names)

        self._data = state_data
        self._cutoff = cutoff_dim
        self._pure = pure
        self._basis = "fock"

        self._str = "<FockState: num_modes={}, cutoff={}, pure={}, hbar={}>".format(
            self.num_modes, self._cutoff, self._pure, self._hbar
        )

    def __eq__(self, other):
        """Equality operator for BaseFockState.

        Returns True if other BaseFockState is close to self.
        This is done by comparing the dm attribute - if within
        the EQ_TOLERANCE, True is returned.

        Args:
            other (BaseFockState): BaseFockState to compare against.
        """
        if not isinstance(other, type(self)):
            return False

        if self.num_modes != other.num_modes:
            return False

        if self.data.shape != other.data.shape:
            return False

        if np.allclose(self.dm(), other.dm(), atol=self.EQ_TOLERANCE, rtol=0):
            return True

        return False

    @property
    def cutoff_dim(self):
        r"""The numerical truncation of the Fock space used by the underlying state.
        Note that a cutoff of D corresponds to the Fock states :math:`\{|0\rangle,\dots,|D-1\rangle\}`

        Returns:
            int: the cutoff dimension
        """
        return self._cutoff

    def ket(self, **kwargs):
        # pylint: disable=unused-argument
        if self._pure:
            return self.data

        return None  # pragma: no cover

    def dm(self, **kwargs):
        # pylint: disable=unused-argument
        if self._pure:
            left_str = [indices[i] for i in range(0, 2 * self._modes, 2)]
            right_str = [indices[i] for i in range(1, 2 * self._modes, 2)]
            out_str = [indices[: 2 * self._modes]]
            einstr = "".join(left_str + [","] + right_str + ["->"] + out_str)
            rho = np.einsum(einstr, self.ket(), self.ket().conj())
            return rho

        return self.data

    def trace(self, **kwargs):
        r"""Trace of the density operator corresponding to the state.

        For pure states the trace corresponds to the squared norm of the ket vector.

        For physical states this should always be 1, any deviations from this value are due
        to numerical errors and Hilbert space truncation artefacts.

        Returns:
          float: trace of the state
        """
        # pylint: disable=unused-argument
        if self.is_pure:
            return np.vdot(self.ket(), self.ket()).real  # <s|s>

        # need some extra steps to trace over multimode matrices
        eqn_indices = [
            [indices[idx]] * 2 for idx in range(self._modes)
        ]  # doubled indices [['i','i'],['j','j'], ... ]
        eqn = "".join(
            chain.from_iterable(eqn_indices)
        )  # flatten indices into a single string 'iijj...'
        return np.einsum(eqn, self.dm()).real

    def all_fock_probs(self, **kwargs):
        r"""Probabilities of all possible Fock basis states for the current circuit state.

        For example, in the case of 3 modes, this method allows the Fock state probability
        :math:`|\braketD{0,2,3}{\psi}|^2` to be returned via

        .. code-block:: python

            probs = state.all_fock_probs()
            probs[0,2,3]

        Returns:
            array: array of dimension :math:`\underbrace{D\times D\times D\cdots\times D}_{\text{num modes}}`
                containing the Fock state probabilities, where :math:`D` is the Fock basis cutoff truncation
        """
        # pylint: disable=unused-argument
        if self._pure:
            s = np.ravel(self.ket())  # into 1D array
            return np.reshape((s * s.conj()).real, [self._cutoff] * self._modes)

        s = self.dm()
        num_axes = len(s.shape)
        evens = list(range(0, num_axes, 2))
        odds = list(range(1, num_axes, 2))
        flat_size = np.prod([s.shape[k] for k in range(0, num_axes, 2)])
        transpose_list = evens + odds
        probs = np.diag(np.reshape(np.transpose(s, transpose_list), [flat_size, flat_size])).real

        return np.reshape(probs, [self._cutoff] * self._modes)

    # =====================================================
    # the following methods are overwritten from BaseState

    def reduced_dm(self, modes, **kwargs):
        # pylint: disable=unused-argument
        if modes == list(range(self._modes)):
            # reduced state is full state
            return self.dm()  # pragma: no cover

        if isinstance(modes, int):
            modes = [modes]
        if modes != sorted(modes):
            raise ValueError("The specified modes cannot be duplicated.")

        if len(modes) > self._modes:
            raise ValueError(
                "The number of specified modes cannot be larger than the number of subsystems."
            )

        # reduce rho down to specified subsystems
        keep_indices = indices[: 2 * len(modes)]
        trace_indices = indices[2 * len(modes) : len(modes) + self._modes]

        ind = [i * 2 for i in trace_indices]
        ctr = 0

        for m in range(self._modes):
            if m in modes:
                ind.insert(m, keep_indices[2 * ctr : 2 * (ctr + 1)])
                ctr += 1

        indStr = "".join(ind) + "->" + keep_indices
        return np.einsum(indStr, self.dm())

    def fock_prob(self, n, **kwargs):
        # pylint: disable=unused-argument
        if len(n) != self._modes:
            raise ValueError("List length should be equal to number of modes")

        if max(n) >= self._cutoff:
            raise ValueError("Can't get distribution beyond truncation level")

        if self._pure:
            return np.abs(self.ket()[tuple(n)]) ** 2

        return self.dm()[tuple([n[i // 2] for i in range(len(n) * 2)])].real

    def mean_photon(self, mode, **kwargs):
        # pylint: disable=unused-argument
        n = np.arange(self._cutoff)
        probs = np.diagonal(self.reduced_dm(mode))
        mean = np.sum(n * probs).real
        var = np.sum(n ** 2 * probs).real - mean ** 2
        return mean, var

    def fidelity(self, other_state, mode, **kwargs):
        # pylint: disable=unused-argument
        max_indices = len(indices) // 2

        if self.num_modes > max_indices:
            raise Exception("fidelity method can only support up to {} modes".format(max_indices))

        left_indices = indices[:mode]
        eqn_left = "".join([i * 2 for i in left_indices])
        reduced_dm_indices = indices[mode : mode + 2]
        right_indices = indices[mode + 2 : self._modes + 1]
        eqn_right = "".join([i * 2 for i in right_indices])
        eqn = "".join([eqn_left, reduced_dm_indices, eqn_right]) + "->" + reduced_dm_indices
        rho_reduced = np.einsum(eqn, self.dm())

        return np.dot(np.conj(other_state), np.dot(rho_reduced, other_state)).real

    def fidelity_vacuum(self, **kwargs):
        # pylint: disable=unused-argument
        alpha = np.zeros(self._modes)
        return self.fidelity_coherent(alpha)

    def fidelity_coherent(self, alpha_list, **kwargs):
        # pylint: disable=too-many-locals,unused-argument
        if self.is_pure:
            mode_size = 1
            s = self.ket()
        else:
            mode_size = 2
            s = self.dm()

        if not hasattr(alpha_list, "__len__"):
            alpha_list = [alpha_list]  # pragma: no cover

        if len(alpha_list) != self._modes:
            raise ValueError("The number of alpha values must match the number of modes.")

        coh = lambda a, dim: np.array(
            [np.exp(-0.5 * np.abs(a) ** 2) * (a) ** n / np.sqrt(factorial(n)) for n in range(dim)]
        )

        if self._modes == 1:
            multi_cohs_vec = coh(alpha_list[0], self._cutoff)
        else:
            multi_cohs_list = [
                coh(alpha_list[idx], dim) for idx, dim in enumerate(s.shape[::mode_size])
            ]
            eqn = ",".join(indices[: self._modes]) + "->" + indices[: self._modes]
            multi_cohs_vec = np.einsum(
                eqn, *multi_cohs_list
            )  # tensor product of specified coherent states

        if self.is_pure:
            ovlap = np.vdot(multi_cohs_vec, s)
            return np.abs(ovlap) ** 2

        bra_indices = indices[: 2 * self._modes : 2]
        ket_indices = indices[1 : 2 * self._modes : 2]
        new_eqn_lhs = ",".join([bra_indices, ket_indices])
        new_eqn_rhs = "".join(bra_indices[idx] + ket_indices[idx] for idx in range(self._modes))
        outer_prod_eqn = new_eqn_lhs + "->" + new_eqn_rhs
        multi_coh_matrix = np.einsum(outer_prod_eqn, multi_cohs_vec, np.conj(multi_cohs_vec))

        return np.vdot(s, multi_coh_matrix).real

    def wigner(self, mode, xvec, pvec):
        r"""Calculates the discretized Wigner function of the specified mode.

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
        rho = self.reduced_dm(mode)
        Q, P = np.meshgrid(xvec, pvec)
        A = (Q + P * 1.0j) / (2 * np.sqrt(self._hbar / 2))

        Wlist = np.array([np.zeros(np.shape(A), dtype=complex) for k in range(self._cutoff)])

        # Wigner function for |0><0|
        Wlist[0] = np.exp(-2.0 * np.abs(A) ** 2) / np.pi

        # W = rho(0,0)W(|0><0|)
        W = np.real(rho[0, 0]) * np.real(Wlist[0])

        for n in range(1, self._cutoff):
            Wlist[n] = (2.0 * A * Wlist[n - 1]) / np.sqrt(n)
            W += 2 * np.real(rho[0, n] * Wlist[n])

        for m in range(1, self._cutoff):
            temp = copy(Wlist[m])
            # Wlist[m] = Wigner function for |m><m|
            Wlist[m] = (2 * np.conj(A) * temp - np.sqrt(m) * Wlist[m - 1]) / np.sqrt(m)

            # W += rho(m,m)W(|m><m|)
            W += np.real(rho[m, m] * Wlist[m])

            for n in range(m + 1, self._cutoff):
                temp2 = (2 * A * Wlist[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
                temp = copy(Wlist[n])
                # Wlist[n] = Wigner function for |m><n|
                Wlist[n] = temp2

                # W += rho(m,n)W(|m><n|) + rho(n,m)W(|n><m|)
                W += 2 * np.real(rho[m, n] * Wlist[n])

        return W / (self._hbar)

    def quad_expectation(self, mode, phi=0, **kwargs):
        a = np.diag(np.sqrt(np.arange(1, self._cutoff + 5)), 1)
        x = np.sqrt(self._hbar / 2) * (a + a.T)
        p = -1j * np.sqrt(self._hbar / 2) * (a - a.T)

        xphi = np.cos(phi) * x + np.sin(phi) * p
        xphisq = np.dot(xphi, xphi)

        # truncate down
        xphi = xphi[: self._cutoff, : self._cutoff]
        xphisq = xphisq[: self._cutoff, : self._cutoff]

        rho = self.reduced_dm(mode)

        mean = np.trace(np.dot(xphi, rho)).real
        var = np.trace(np.dot(xphisq, rho)).real - mean ** 2

        return mean, var

    def poly_quad_expectation(self, A, d=None, k=0, phi=0, **kwargs):
        # pylint: disable=too-many-branches,too-many-statements

        if A is None:
            A = np.zeros([2 * self._modes, 2 * self._modes])

        if A.shape != (2 * self._modes, 2 * self._modes):
            raise ValueError("Matrix of quadratic coefficients A must be of size 2Nx2N.")

        if not np.allclose(A.T, A):
            raise ValueError("Matrix of quadratic coefficients A must be symmetric.")

        if d is None:
            linear_coeff = np.zeros([2 * self._modes])
        else:
            linear_coeff = d.copy()
            linear_coeff[self._modes :] = -d[self._modes :]

        if linear_coeff.shape != (2 * self._modes,):
            raise ValueError("Vector of linear coefficients d must be of length 2N.")

        # expand the cutoff dimension in approximating the x and p
        # operators in the Fock basis, to reduce numerical inaccuracy.
        worksize = 1
        dim = self._cutoff + worksize

        # construct the x and p operators
        a = np.diag(np.sqrt(np.arange(1, dim)), 1)
        x_ = np.sqrt(self._hbar / 2) * (a + a.T)
        p_ = -1j * np.sqrt(self._hbar / 2) * (a - a.T)

        if phi != 0:
            # rotate the quadrature operators
            x = np.cos(phi) * x_ - np.sin(phi) * p_
            p = np.sin(phi) * x_ + np.cos(phi) * p_
        else:
            x = x_
            p = p_

        def expand_dims(op, n, modes):
            """Expand quadrature operator to act on nth mode"""
            I = np.identity(dim)
            allowed_indices = zip(indices[: 2 * modes : 2], indices[1 : 2 * modes : 2])
            ind = ",".join(a + b for a, b in allowed_indices)
            ops = [I] * n + [op] + [I] * (modes - n - 1)
            # the einsum 'ij,kl,mn->ijklmn' (for 3 modes)
            return np.einsum(ind, *ops)

        # determine modes with quadratic expectation values
        nonzero = np.concatenate(
            [
                np.mod(A.nonzero()[0], self._modes),
                np.mod(linear_coeff.nonzero()[0], self._modes),
            ]
        )
        ex_modes = list(set(nonzero))
        num_modes = len(ex_modes)

        if not ex_modes:
            # only a constant term was provided
            return k, 0.0

        # There are non-zero elements of A and/or d
        # therefore there are quadratic and/or linear terms.
        # find the reduced density matrix
        rho = self.reduced_dm(modes=ex_modes)

        # generate vector of quadrature operators
        # this array will have shape [2*num_modes] + [dim]*(2*num_modes)
        r = np.empty([2 * num_modes] + [dim] * (2 * num_modes), dtype=np.complex128)
        for n in range(num_modes):
            r[n] = expand_dims(x, n, num_modes)
            r[num_modes + n] = expand_dims(p, n, num_modes)

        # reduce the size of A so that we only consider modes
        # which we need to calculate the expectation value for
        rows = ex_modes + [i + self._modes for i in ex_modes]
        quad_coeffs = A[:, rows][rows]
        quad_coeffs[num_modes:, :num_modes] = -quad_coeffs[num_modes:, :num_modes]
        quad_coeffs[:num_modes, num_modes:] = -quad_coeffs[:num_modes, num_modes:]

        # Compute the polynomial
        #
        # For 3 modes, this gives the einsum (with brackets denoting modes):
        # 'a(bc)(de)(fg),a(ch)(ei)(gj)->(bh)(di)(fj)' applied to r, A@r
        #
        # a corresponds to the index in the vector of quadrature operators
        # r = (x_1,...,x_n,p_1,...,p_n), and the remaining indices ijklmn
        # are the elements of the operator acting on a 3 mode density matrix.
        #
        # So, in effect, matrix of quadratic coefficients A acts only on index a,
        # this index is then summed, and then each mode of r, A@r undergoes
        # matrix multiplication
        ind1 = indices[: 2 * num_modes + 1]
        ind2 = ind1[0] + "".join(
            [
                str(i) + str(j)
                for i, j in zip(ind1[2::2], indices[2 * num_modes + 1 : 3 * num_modes + 1])
            ]
        )
        ind3 = "".join([str(i) + str(j) for i, j in zip(ind1[1::2], ind2[2::2])])
        ind = "{},{}->{}".format(ind1, ind2, ind3)

        if np.allclose(quad_coeffs, 0.0):
            poly_op = np.zeros([dim] * (2 * num_modes), dtype=np.complex128)
        else:
            # Einsum above applied to to r,Ar
            # This einsum sums over all quadrature operators, and also applies matrix
            # multiplication between the same mode of each operator
            poly_op = np.einsum(ind, r, np.tensordot(quad_coeffs, r, axes=1)).conj()

        # add linear term
        rows = np.flip(np.array(rows).reshape([2, -1]), axis=1).flatten()
        poly_op += r.T @ linear_coeff[rows]

        # add constant term
        if k != 0:
            poly_op += k * expand_dims(np.eye(dim), 0, num_modes)

        # calculate Op^2
        ind = "{},{}->{}".format(ind1[1:], ind2[1:], ind3)
        poly_op_sq = np.einsum(ind, poly_op, poly_op)

        # truncate down
        sl = tuple([slice(0, self._cutoff)] * (2 * num_modes))
        poly_op = poly_op[sl]
        poly_op_sq = poly_op_sq[sl]

        ind1 = ind1[:-1]
        ind2 = "".join([str(j) + str(i) for i, j in zip(ind1[::2], ind1[1::2])])
        ind = "{},{}".format(ind1, ind2)

        # calculate expectation value, Tr(Op @ rho)
        # For 3 modes, this gives the einsum '(ab)(cd)(ef),(ba)(dc)(fe)->'
        mean = np.einsum(ind, poly_op, rho).real

        # calculate variance Tr(Op^2 @ rho) - Tr(Op @ rho)^2
        var = np.einsum(ind, poly_op_sq, rho).real - mean ** 2

        return mean, var

    def diagonal_expectation(self, modes, values):
        """Calculates the expectation value of an operator that is diagonal in the number basis"""
        if len(modes) != len(set(modes)):
            raise ValueError("There can be no duplicates in the modes specified.")

        cutoff = self._cutoff  # Fock space cutoff.
        num_modes = self._modes  # number of modes in the state.

        traced_modes = tuple(item for item in range(num_modes) if item not in modes)
        if self.is_pure:
            # state is a tensor of probability amplitudes
            ps = np.abs(self.ket()) ** 2
            ps = ps.sum(axis=traced_modes)
            for _ in modes:
                ps = np.tensordot(values, ps, axes=1)
            return float(ps)

        # state is a tensor of density matrix elements in the SF convention
        ps = np.real(self.dm())
        traced_modes = list(traced_modes)
        traced_modes.sort(reverse=True)
        for mode in traced_modes:
            ps = np.tensordot(np.identity(cutoff), ps, axes=((0, 1), (2 * mode, 2 * mode + 1)))
        for _ in range(len(modes)):
            ps = np.tensordot(np.diag(values), ps, axes=((0, 1), (0, 1)))
        return float(ps)

    def number_expectation(self, modes):
        """Calculates the expectation value of a product of number operators acting on given modes"""
        cutoff = self._cutoff
        values = np.arange(cutoff)
        mean = self.diagonal_expectation(modes, values)
        var = self.diagonal_expectation(modes, values ** 2) - mean ** 2
        return mean, var

    def parity_expectation(self, modes):
        cutoff = self._cutoff
        values = (-1) ** np.arange(cutoff)
        return self.diagonal_expectation(modes, values)


class BaseGaussianState(BaseState):
    r"""Class for the representation of quantum states using the Gaussian formalism.

    Note that this class uses the Gaussian representation convention

    .. math:: \bar{\mathbf{r}} = (\bar{x}_1,\bar{x}_2,\dots,\bar{x}_N,\bar{p}_1,\dots,\bar{p}_N)

    Args:
        state_data (tuple(mu, cov)): A tuple containing the vector of means array ``mu`` and the
            covariance matrix array ``cov``, in terms of the complex displacement.
        num_modes (int): the number of modes in the state
        pure (bool): True if the state is a pure state, false if the state is mixed
        mode_names (Sequence): (optional) this argument contains a list providing mode names
            for each mode in the state
    """
    # pylint: disable=too-many-public-methods

    def __init__(self, state_data, num_modes, mode_names=None):
        super().__init__(num_modes, mode_names)

        self._data = state_data

        # vector of means and covariance matrix, using frontend x,p scaling
        self._mu = self._data[0] * np.sqrt(self._hbar / 2)
        self._cov = self._data[1] * (self._hbar / 2)
        # complex displacements of the Gaussian state
        self._alpha = self._mu[: self._modes] + 1j * self._mu[self._modes :]
        self._alpha /= np.sqrt(2 * self._hbar)

        self._pure = (
            np.abs(np.linalg.det(self._cov) - (self._hbar / 2) ** (2 * self._modes))
            < self.EQ_TOLERANCE
        )

        self._basis = "gaussian"
        self._str = "<GaussianState: num_modes={}, pure={}, hbar={}>".format(
            self.num_modes, self._pure, self._hbar
        )

    def __eq__(self, other):
        """Equality operator for BaseGaussianState.

        Returns True if other BaseGaussianState is close to self.
        This is done by comparing the means vector and cov matrix.
        If both are within the EQ_TOLERANCE, True is returned.

        Args:
            other (BaseGaussianState): BaseGaussianState to compare against.
        """
        # pylint: disable=protected-access
        if not isinstance(other, type(self)):
            return False

        if self.num_modes != other.num_modes:
            return False

        if np.allclose(self._mu, other._mu, atol=self.EQ_TOLERANCE, rtol=0) and np.allclose(
            self._cov, other._cov, atol=self.EQ_TOLERANCE, rtol=0
        ):
            return True

        return False

    def means(self):
        r"""The vector of means describing the Gaussian state.

        For a :math:`N` mode state, this has the form

        .. math::
            \bar{\mathbf{r}} = \left(\bar{x}_0,\dots,\bar{x}_{N-1},\bar{p}_0,\dots,\bar{p}_{N-1}\right)

        where :math:`\bar{x}_i` and :math:`\bar{p}_i` refer to the mean
        position and momentum quadrature of mode :math:`i` respectively.

        Returns:
          array: a length :math:`2N` array containing the vector of means.
        """
        return self._mu

    def cov(self):
        r"""The covariance matrix describing the Gaussian state.

        The diagonal elements of the covariance matrix correspond to the
        variance in the position and momentum quadratures:

        .. math::
            \mathbf{V}_{ii} = \begin{cases}
                (\Delta x_i)^2, & 0\leq i\leq N-1\\
                (\Delta p_{i-N})^2, & N\leq i\leq 2(N-1)
            \end{cases}

        where :math:`\Delta x_i` and :math:`\Delta p_i` refer to the
        position and momentum quadrature variance of mode :math:`i` respectively.

        Note that if the covariance matrix is purely diagonal, then this
        corresponds to squeezing :math:`z=re^{i\phi}` where :math:`\phi=0`,
        and :math:`\Delta x_i = e^{-2r}`, :math:`\Delta p_i = e^{2r}`.

        Returns:
          array: the :math:`2N\times 2N` covariance matrix.
        """
        return self._cov

    def reduced_gaussian(self, modes):
        r"""Returns the vector of means and the covariance matrix of the specified modes.

        Args:
            modes (int of Sequence[int]): indices of the requested modes

        Returns:
            tuple (means, cov): where means is an array containing the vector of means,
            and cov is a square array containing the covariance matrix.
        """
        if modes == list(range(self._modes)):
            # reduced state is full state
            return self._mu, self._cov

        # reduce rho down to specified subsystems
        if isinstance(modes, int):
            modes = [modes]

        if modes != sorted(modes):
            raise ValueError("The specified modes cannot be duplicated.")

        if len(modes) > self._modes:
            raise ValueError(
                "The number of specified modes cannot " "be larger than the number of subsystems."
            )

        ind = np.concatenate([np.array(modes), np.array(modes) + self._modes])
        rows = ind.reshape(-1, 1)
        cols = ind.reshape(1, -1)

        mu = self._mu[ind]
        cov = self._cov[rows, cols]

        return mu, cov

    def is_coherent(self, mode, tol=1e-10):
        r"""Returns True if the Gaussian state of a particular mode is a coherent state.

        Args:
            mode (int): the specified mode
            tol (float): the numerical precision in determining if squeezing is not present

        Returns:
            bool: True if and only if the state is a coherent state.
        """
        mu, cov = self.reduced_gaussian([mode])  # pylint: disable=unused-variable
        cov /= self._hbar / 2
        return np.allclose(cov, np.identity(2), atol=tol, rtol=0)

    def displacement(self, modes=None):
        r"""Returns the displacement parameter :math:`\alpha` of the modes specified.

        Args:
            modes (int or Sequence[int]): modes specified

        Returns:
            Sequence[complex]: sequence of complex displacements :math:`\alpha`
            corresponding to the list of specified modes
        """
        if modes is None:
            modes = list(range(self._modes))
        elif isinstance(modes, int):  # pragma: no cover
            modes = [modes]

        return self._alpha[list(modes)]

    def is_squeezed(self, mode, tol=1e-6):
        r"""Returns True if the Gaussian state of a particular mode is a squeezed state.

        Args:
            mode (int): the specified mode
            tol (float): the numerical precision in determining if squeezing is present

        Returns:
           bool: True if and only if the state is a squeezed state.
        """
        mu, cov = self.reduced_gaussian([mode])  # pylint: disable=unused-variable
        cov /= self._hbar / 2
        return np.any(np.abs(cov - np.identity(2)) > tol)

    def squeezing(self, modes=None):
        r"""Returns the squeezing parameters :math:`(r,\phi)` of the modes specified.

        Args:
            modes (int or Sequence[int]): modes specified

        Returns:
            List[(float, float)]: sequence of tuples containing the squeezing
            parameters :math:`(r,\phi)` of the specified modes.
        """
        if modes is None:
            modes = list(range(self._modes))
        elif isinstance(modes, int):  # pragma: no cover
            modes = [modes]

        res = []
        for i in modes:
            mu, cov = self.reduced_gaussian([i])  # pylint: disable=unused-variable
            cov /= self._hbar / 2
            tr = np.trace(cov)

            r = np.arccosh(tr / 2) / 2

            if cov[0, 1] == 0.0:
                phi = 0
            else:
                phi = -np.arcsin(2 * cov[0, 1] / np.sqrt((tr - 2) * (tr + 2)))

            res.append((r, phi))

        return res

    # =====================================================
    # the following methods are overwritten from BaseState

    def wigner(self, mode, xvec, pvec):
        mu, cov = self.reduced_gaussian([mode])

        X, P = np.meshgrid(xvec, pvec)
        grid = np.empty(X.shape + (2,))
        grid[:, :, 0] = X
        grid[:, :, 1] = P
        mvn = multivariate_normal(mu, cov, allow_singular=True)

        return mvn.pdf(grid)

    def quad_expectation(self, mode, phi=0, **kwargs):
        # pylint: disable=unused-argument
        mu, cov = self.reduced_gaussian([mode])
        rot = _R(phi)

        muphi = rot.T @ mu
        covphi = rot.T @ cov @ rot
        return (muphi[0], covphi[0, 0])

    def poly_quad_expectation(self, A, d=None, k=0, phi=0, **kwargs):
        if A is None:
            A = np.zeros([2 * self._modes, 2 * self._modes])

        if A.shape != (2 * self._modes, 2 * self._modes):
            raise ValueError("Matrix of quadratic coefficients A must be of size 2Nx2N.")

        if not np.allclose(A.T, A):
            raise ValueError("Matrix of quadratic coefficients A must be symmetric.")

        if d is not None:
            if d.shape != (2 * self._modes,):
                raise ValueError("Vector of linear coefficients d must be of length 2N.")
        else:
            d = np.zeros([2 * self._modes])

        # determine modes with quadratic expectation values
        nonzero = np.concatenate(
            [np.mod(A.nonzero()[0], self._modes), np.mod(d.nonzero()[0], self._modes)]
        )
        ex_modes = list(set(nonzero))

        # reduce the size of A so that we only consider modes
        # which we need to calculate the expectation value for
        rows = ex_modes + [i + self._modes for i in ex_modes]
        num_modes = len(ex_modes)
        quad_coeffs = A[:, rows][rows]

        if not ex_modes:
            # only a constant term was provided
            return k, 0.0

        mu = self._mu
        cov = self._cov

        if phi != 0:
            # rotate all modes of the covariance matrix and vector of means
            R = _R(phi)
            C = changebasis(self._modes)
            rot = C.T @ block_diag(*([R] * self._modes)) @ C

            mu = rot.T @ mu
            cov = rot.T @ cov @ rot

        # transform to the expectation of a quadratic on a normal distribution with zero mean
        # E[P(r)]_(mu,cov) = E(Q(r+mu)]_(0,cov)
        #                  = E[rT.A.r + rT.(2A.mu+d) + (muT.A.mu+muT.d+cI)]_(0,cov)
        #                  = E[rT.A.r + rT.d' + k']_(0,cov)
        d2 = 2 * A @ mu + d
        k2 = mu.T @ A @ mu + mu.T @ d + k

        # expectation value E[P(r)]_{mu=0} = tr(A.cov) + muT.A.mu + muT.d + k|_{mu=0}
        #                                  = tr(A.cov) + k
        mean = np.trace(A @ cov) + k2
        # variance Var[P(r)]_{mu=0} = 2tr(A.cov.A.cov) + 4*muT.A.cov.A.mu + dT.cov.d|_{mu=0}
        #                           = 2tr(A.cov.A.cov) + dT.cov.d
        var = 2 * np.trace(A @ cov @ A @ cov) + d2.T @ cov @ d2

        # Correction term to account for incorrect symmetric ordering in the variance.
        # This occurs because Var[S(P(r))] = Var[P(r)] - Σ_{m1, m2} |hbar*A_{(m1, m1+N),(m2, m2+N)}|,
        # where m1, m2 are all possible mode numbers, and N is the total number of modes.
        # Therefore, the correction term is the sum of the determinants of 2x2 submatrices of A.
        modes = np.arange(2 * num_modes).reshape(2, -1).T
        var -= np.sum(
            [np.linalg.det(self._hbar * quad_coeffs[:, m][n]) for m in modes for n in modes]
        )

        return mean, var

    def number_expectation(self, modes):
        if len(modes) != len(set(modes)):
            raise ValueError("There can be no duplicates in the modes specified.")

        mu = self._mu
        cov = self._cov

        mean = twq.photon_number_expectation(mu, cov, modes, hbar=self._hbar).real
        mean2 = twq.photon_number_squared_expectation(mu, cov, modes, hbar=self._hbar).real
        var = mean2 - mean ** 2

        return mean, var

    def parity_expectation(self, modes):
        if len(modes) != len(set(modes)):
            raise ValueError("There can be no duplicates in the modes specified.")

        mu = self.means()
        cov = self.cov()
        num = np.exp(-(0.5) * (mu @ (np.linalg.inv(cov) @ mu)))
        parity = ((self.hbar / 2) ** len(modes)) * num / (np.sqrt(np.linalg.det(cov)))

        return parity

    def ket(self, **kwargs):
        cutoff = kwargs.get("cutoff", 10)
        mu = self._mu
        cov = self._cov

        if self._pure:
            return twq.state_vector(
                mu, cov, hbar=self._hbar, normalize=True, cutoff=cutoff, check_purity=False
            )

        return None  # pragma: no cover

    def dm(self, **kwargs):
        cutoff = kwargs.get("cutoff", 10)
        return self.reduced_dm(list(range(self._modes)), cutoff=cutoff)

    def reduced_dm(self, modes, **kwargs):
        if isinstance(modes, int):
            modes = [modes]

        if modes != sorted(modes):
            raise ValueError("The specified modes cannot be duplicated.")

        if len(modes) > self._modes:
            raise ValueError(
                "The number of specified modes cannot be larger than the number of subsystems."
            )

        cutoff = kwargs.get("cutoff", 10)
        mu, cov = self.reduced_gaussian(modes)  # pylint: disable=unused-variable

        if self.is_pure:
            psi = twq.state_vector(
                mu, cov, hbar=self._hbar, normalize=True, cutoff=cutoff, check_purity=False
            )
            rho = np.outer(psi, psi.conj())
            return rho

        return twq.density_matrix(mu, cov, hbar=self._hbar, normalize=True, cutoff=cutoff)

    def mean_photon(self, mode, **kwargs):
        mu, cov = self.reduced_gaussian([mode])
        mean = (np.trace(cov) + mu.T @ mu) / (2 * self._hbar) - 1 / 2
        var = (np.trace(cov @ cov) + 2 * mu.T @ cov @ mu) / (2 * self._hbar ** 2) - 1 / 4
        return mean, var

    def fidelity(self, other_state, mode, **kwargs):
        if isinstance(mode, int):
            mode = [mode]

        mu1, cov1 = other_state
        mu2, cov2 = self.reduced_gaussian(mode)
        return twq.fidelity(mu1, cov1, mu2, cov2, hbar=self._hbar)

    def fidelity_vacuum(self, **kwargs):
        alpha = np.zeros(len(self._alpha))
        return self.fidelity_coherent(alpha)

    def fidelity_coherent(self, alpha_list, **kwargs):
        if len(alpha_list) != self._modes:
            raise ValueError("alpha_list must be same length as the number of modes")

        if not isinstance(alpha_list, np.ndarray):
            alpha_list = np.array(alpha_list)

        mu = np.concatenate([alpha_list.real, alpha_list.imag]) * np.sqrt(2 * self._hbar)
        cov = np.identity(2 * self._modes) * self._hbar / 2
        return self.fidelity([mu, cov], list(range(self._modes)))

    def fock_prob(self, n, **kwargs):
        if len(n) != self._modes:
            raise ValueError("Fock state must be same length as the number of modes")

        cutoff = kwargs.get("cutoff", 10)
        if sum(n) >= cutoff:
            raise ValueError("Cutoff argument must be larger than the sum of photon numbers")

        if self.is_pure:
            return (
                np.abs(
                    twq.pure_state_amplitude(
                        self._mu, self._cov, n, hbar=self._hbar, check_purity=False
                    )
                )
                ** 2
            )

        return twq.density_matrix_element(self._mu, self._cov, n, n, hbar=self._hbar).real

    def all_fock_probs(self, **kwargs):
        cutoff = kwargs.get("cutoff", 10)
        mu = self._mu
        cov = self._cov
        return twq.probabilities(mu, cov, cutoff, hbar=self._hbar)

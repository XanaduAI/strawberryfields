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

r"""
.. _state_class:

Quantum states API
========================================================

**Module name:** :mod:`strawberryfields.backends.states`

.. currentmodule:: strawberryfields.backends.states

This module provides classes which represent the quantum state
returned by a simulator backend and the quantum engine:

.. code-block:: python3

    state = eng.run('backend')

Base quantum state
-------------------------------

An abstract base class for the representation of quantum states. This class should not be instantiated on its
own, instead all states will be represented by one of the inheriting subclasses.

This class contains all methods that should be supported by all
inheriting classes.

.. note::
    In the following, keyword arguments are denoted ``**kwargs``, and allow additional
    options to be passed to the underlying State class - these are documented where
    available. For more details on relevant keyword arguments, please
    consult the backend documentation directly.

.. currentmodule:: strawberryfields.backends.states.BaseState

.. autosummary::
    data
    hbar
    is_pure
    num_modes
    mode_names
    mode_indices
    reduced_dm
    fock_prob
    mean_photon
    fidelity
    fidelity_vacuum
    fidelity_coherent
    wigner
    quad_expectation


Base Gaussian state
-------------------------------

Class for the representation of quantum states using the Gaussian formalism.
This class extends the class :class:`~.BaseState` with additional methods
unique to Gaussian states.

Note that backends using the Gaussian state representation may extend this class with
additional methods particular to the backend, for example :class:`~.GaussianState`
in the :ref:`gaussian_backend`.

.. currentmodule:: strawberryfields.backends.states.BaseGaussianState

.. autosummary::
    means
    cov
    reduced_gaussian
    is_coherent
    is_squeezed
    displacement
    squeezing


Base Fock state
-------------------------------

Class for the representation of quantum states in the Fock basis.
This class extends the class :class:`~.BaseState` with additional methods
unique to states in the Fock-basis representation.

Note that backends using Fock-basis representation may extend this class with
additional methods particular to the backend, for example :class:`~.FockStateTF`
in the :ref:`Tensorflow_backend`.

.. currentmodule:: strawberryfields.backends.states.BaseFockState

.. autosummary::
    cutoff_dim
    ket
    dm
    trace
    all_fock_probs

.. currentmodule:: strawberryfields.backends.states


Code details
~~~~~~~~~~~~

"""
import abc
import string
from itertools import chain
from copy import copy

import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import factorial

from .shared_ops import rotation_matrix as _R

indices = string.ascii_lowercase

class BaseState(abc.ABC):
    r"""Abstract base class for the representation of quantum states."""
    EQ_TOLERANCE = 1e-10

    def __init__(self, num_modes, hbar=2, mode_names=None):
        self._modes = num_modes
        self._hbar = hbar
        self._data = None
        self._pure = None

        if mode_names is None:
            self._modemap = {i:"mode {}".format(i) for i in range(num_modes)}
        else:
            self._modemap = {i:'{}'.format(j) for i, j in zip(range(num_modes), mode_names)}

        self._str = "<BaseState: num_modes={}, pure={}, hbar={}>".format(
            self.num_modes, self._pure, self._hbar)

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
    def reduced_dm(self, modes, **kwargs):
        r"""Returns a reduced density matrix in the Fock basis.

        Args:
            modes (int or Sequence[int]): specifies the mode(s) to return the reduced density matrix for.
            **kwargs:

                  * **cutoff** (*int*): (default 10) specifies where to truncate the returned density matrix.
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

        Args:
            n (Sequence[int]): the Fock state :math:`\ket{\vec{n}}` that we want to measure the probability of
            **kwargs:

                  * **cutoff** (*int*): (default 10) specifies the fock basis truncation when calculating
                    of the fock basis probabilities.
                    Note that the cutoff argument only applies for Gaussian representation;
                    states represented in the Fock basis will use their own internal cutoff dimension.

        Returns:
            float: measurement probability
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
            float: the mean photon number
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


class BaseFockState(BaseState):
    r"""Class for the representation of quantum states in the Fock basis.

    Args:
        state_data (array): the state representation in the Fock basis
        num_modes (int): the number of modes in the state
        pure (bool): True if the state is a pure state, false if the state is mixed
        cutoff_dim (int): the Fock basis truncation size
        hbar (float): (default 2) The value of :math:`\hbar` in the definition of :math:`\x` and :math:`\p` (see :ref:`opcon`)
        mode_names (Sequence): (optional) this argument contains a list providing mode names
            for each mode in the state
    """

    def __init__(self, state_data, num_modes, pure, cutoff_dim, hbar=2., mode_names=None):
        # pylint: disable=too-many-arguments

        super().__init__(num_modes, hbar, mode_names)

        self._data = state_data
        self._cutoff = cutoff_dim
        self._pure = pure
        self._basis = 'fock'

        self._str = "<FockState: num_modes={}, cutoff={}, pure={}, hbar={}>".format(
            self.num_modes, self._cutoff, self._pure, self._hbar)

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

        if np.all(np.abs(self.dm() - other.dm()) <= self.EQ_TOLERANCE):
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
        r"""The numerical state vector for the quantum state.
        Note that if the state is mixed, this method returns None.

        Returns:
            array/None: the numerical state vector. Returns None if the state is mixed.
        """
        # pylint: disable=unused-argument
        if self._pure:
            return self.data

        return None # pragma: no cover

    def dm(self, **kwargs):
        r"""The numerical density matrix for the quantum state.

        Returns:
            array: the numerical density matrix in the Fock basis
        """
        # pylint: disable=unused-argument
        if self._pure:
            left_str = [indices[i] for i in range(0, 2 * self._modes, 2)]
            right_str = [indices[i] for i in range(1, 2 * self._modes, 2)]
            out_str = [indices[: 2 * self._modes]]
            einstr = ''.join(left_str + [','] + right_str + ['->'] + out_str)
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
        eqn_indices = [[indices[idx]] * 2 for idx in range(self._modes)] #doubled indices [['i','i'],['j','j'], ... ]
        eqn = "".join(chain.from_iterable(eqn_indices)) # flatten indices into a single string 'iijj...'
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
            s = np.ravel(self.ket()) # into 1D array
            return np.reshape((s * s.conj()).real, [self._cutoff]*self._modes)

        s = self.dm()
        num_axes = len(s.shape)
        evens = [k for k in range(0, num_axes, 2)]
        odds = [k for k in range(1, num_axes, 2)]
        flat_size = np.prod([s.shape[k] for k in range(0, num_axes, 2)])
        transpose_list = evens + odds
        probs = np.diag(np.reshape(np.transpose(s, transpose_list), [flat_size, flat_size])).real

        return np.reshape(probs, [self._cutoff]*self._modes)

    #=====================================================
    # the following methods are overwritten from BaseState

    def reduced_dm(self, modes, **kwargs):
        # pylint: disable=unused-argument
        if modes == list(range(self._modes)):
            # reduced state is full state
            return self.dm() # pragma: no cover

        if isinstance(modes, int):
            modes = [modes]
        if modes != sorted(modes):
            raise ValueError("The specified modes cannot be duplicated.")

        if len(modes) > self._modes:
            raise ValueError("The number of specified modes cannot "
                             "be larger than the number of subsystems.")

        # reduce rho down to specified subsystems
        keep_indices = indices[: 2 * len(modes)]
        trace_indices = indices[2 * len(modes) : len(modes) + self._modes]

        ind = [i * 2 for i in trace_indices]
        ctr = 0

        for m in range(self._modes):
            if m in modes:
                ind.insert(m, keep_indices[2 * ctr : 2 * (ctr + 1)])
                ctr += 1

        indStr = ''.join(ind) + '->' + keep_indices
        return np.einsum(indStr, self.dm())

    def fock_prob(self, n, **kwargs):
        # pylint: disable=unused-argument
        if len(n) != self._modes:
            raise ValueError("List length should be equal to number of modes")

        elif max(n) >= self._cutoff:
            raise ValueError("Can't get distribution beyond truncation level")

        if self._pure:
            return np.abs(self.ket()[tuple(n)])**2

        return self.dm()[tuple([n[i//2] for i in range(len(n)*2)])].real

    def mean_photon(self, mode, **kwargs):
        # pylint: disable=unused-argument
        n = np.arange(self._cutoff)
        probs = np.diagonal(self.reduced_dm(mode))
        return np.sum(n*probs).real

    def fidelity(self, other_state, mode, **kwargs):
        # pylint: disable=unused-argument
        max_indices = len(indices) // 2

        if self.num_modes > max_indices:
            raise Exception("fidelity method can only support up to {} modes".format(max_indices))

        left_indices = indices[:mode]
        eqn_left = "".join([i*2 for i in left_indices])
        reduced_dm_indices = indices[mode:mode + 2]
        right_indices = indices[mode + 2:self._modes + 1]
        eqn_right = "".join([i*2 for i in right_indices])
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
            alpha_list = [alpha_list] # pragma: no cover

        if len(alpha_list) != self._modes:
            raise ValueError("The number of alpha values must match the number of modes.")

        coh = lambda a, dim: np.array(
            [np.exp(-0.5 * np.abs(a) ** 2) * (a) ** n / np.sqrt(factorial(n)) for n in range(dim)]
        )

        if self._modes == 1:
            multi_cohs_vec = coh(alpha_list[0], self._cutoff)
        else:
            multi_cohs_list = [coh(alpha_list[idx], dim) for idx, dim in enumerate(s.shape[::mode_size])]
            eqn = ",".join(indices[:self._modes]) + "->" + indices[:self._modes]
            multi_cohs_vec = np.einsum(eqn, *multi_cohs_list) # tensor product of specified coherent states

        if self.is_pure:
            ovlap = np.vdot(multi_cohs_vec, s)
            return np.abs(ovlap) ** 2

        bra_indices = indices[:2 * self._modes:2]
        ket_indices = indices[1:2 * self._modes:2]
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
        A = (Q + P * 1.0j) / (2*np.sqrt(self._hbar/2))

        Wlist = np.array([np.zeros(np.shape(A), dtype=complex) for k in range(self._cutoff)])

        # Wigner function for |0><0|
        Wlist[0] = np.exp(-2.0 * np.abs(A)**2) / np.pi

        # W = rho(0,0)W(|0><0|)
        W = np.real(rho[0, 0]) * np.real(Wlist[0])

        for n in range(1, self._cutoff):
            Wlist[n] = (2.0 * A * Wlist[n - 1]) / np.sqrt(n)
            W += 2 * np.real(rho[0, n] * Wlist[n])

        for m in range(1, self._cutoff):
            temp = copy(Wlist[m])
            # Wlist[m] = Wigner function for |m><m|
            Wlist[m] = (2 * np.conj(A) * temp - np.sqrt(m)
                        * Wlist[m - 1]) / np.sqrt(m)

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
        a = np.diag(np.sqrt(np.arange(1, self._cutoff+5)), 1)
        x = np.sqrt(self._hbar/2) * (a + a.T)
        p = -1j * np.sqrt(self._hbar/2) * (a - a.T)

        xphi = np.cos(phi)*x + np.sin(phi)*p
        xphisq = np.dot(xphi, xphi)

        # truncate down
        xphi = xphi[:self._cutoff, :self._cutoff]
        xphisq = xphisq[:self._cutoff, :self._cutoff]

        rho = self.reduced_dm(mode)

        mean = np.trace(np.dot(xphi, rho)).real
        var = np.trace(np.dot(xphisq, rho)).real - mean**2

        return mean, var


class BaseGaussianState(BaseState):
    r"""Class for the representation of quantum states using the Gaussian formalism.

    Note that this class uses the Gaussian representation convention

    .. math:: \bar{\mathbf{r}} = (\bar{x}_1,\bar{x}_2,\dots,\bar{x}_N,\bar{p}_1,\dots,\bar{p}_N)

    Args:
        state_data (tuple(mu, cov)): A tuple containing the vector of means array ``mu`` and the
            covariance matrix array ``cov``, in terms of the complex displacement.
        num_modes (int): the number of modes in the state
        pure (bool): True if the state is a pure state, false if the state is mixed
        hbar (float): (default 2) The value of :math:`\hbar` in the definition of :math:`\x` and :math:`\p` (see :ref:`opcon`)
        mode_names (Sequence): (optional) this argument contains a list providing mode names
            for each mode in the state
    """
    def __init__(self, state_data, num_modes, hbar=2., mode_names=None):
        super().__init__(num_modes, hbar, mode_names)

        self._data = state_data

        # vector of means and covariance matrix
        self._mu = self._data[0]
        self._cov = self._data[1]

        # complex displacements of the Gaussian state
        self._alpha = self._mu[:self._modes] + 1j*self._mu[self._modes:]
        self._alpha /= np.sqrt(2*self._hbar)

        self._pure = np.abs(np.linalg.det(self._cov) - (self._hbar/2)**(2*self._modes)) < self.EQ_TOLERANCE

        self._basis = 'gaussian'
        self._str = "<GaussianState: num_modes={}, pure={}, hbar={}>".format(
            self.num_modes, self._pure, self._hbar)

    def __eq__(self, other):
        """Equality operator for BaseGaussianState.

        Returns True if other BaseGaussianState is close to self.
        This is done by comparing the means vector and cov matrix.
        If both are within the EQ_TOLERANCE, True is returned.

        Args:
            other (BaseGaussianState): BaseGaussianState to compare against.
        """
        #pylint: disable=protected-access
        if not isinstance(other, type(self)):
            return False

        if self.num_modes != other.num_modes:
            return False

        if np.all(np.abs(self._mu - other._mu) <= self.EQ_TOLERANCE) and \
                np.all(np.abs(self._cov - other._cov) <= self.EQ_TOLERANCE):
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
        r""" Returns the vector of means and the covariance matrix of the specified modes.

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
            raise ValueError("The number of specified modes cannot "
                             "be larger than the number of subsystems.")

        ind = np.concatenate([np.array(modes), np.array(modes)+self._modes])
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
        mu, cov = self.reduced_gaussian([mode]) # pylint: disable=unused-variable
        cov /= self._hbar/2
        return np.all(np.abs(cov - np.identity(2)) < tol)

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
        elif isinstance(modes, int): # pragma: no cover
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
        mu, cov = self.reduced_gaussian([mode]) # pylint: disable=unused-variable
        cov /= self._hbar/2
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
        elif isinstance(modes, int): # pragma: no cover
            modes = [modes]

        res = []
        for i in modes:
            mu, cov = self.reduced_gaussian([i]) # pylint: disable=unused-variable
            cov /= self._hbar/2
            tr = np.trace(cov)

            r = np.arccosh(tr/2)/2

            if cov[0, 1] == 0.:
                phi = 0
            else:
                phi = -np.arcsin(2*cov[0, 1] / np.sqrt((tr-2)*(tr+2)))

            res.append((r, phi))

        return res

    #=====================================================
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

        muphi = np.dot(rot.T, mu)
        covphi = np.dot(rot.T, np.dot(cov, rot))
        return (muphi[0], covphi[0, 0])

    @abc.abstractmethod
    def reduced_dm(self, modes, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def fock_prob(self, n, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def mean_photon(self, mode, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def fidelity(self, other_state, mode, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def fidelity_vacuum(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def fidelity_coherent(self, alpha_list, **kwargs):
        raise NotImplementedError

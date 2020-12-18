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
This module defines and implements the Python-embedded quantum programming language
for continuous-variable (CV) quantum systems.
The syntax is modeled after ProjectQ :cite:`projectq2016`.
"""
from collections.abc import Sequence
import copy
import warnings

import numpy as np

from scipy.linalg import block_diag
import scipy.special as ssp

import strawberryfields as sf
import strawberryfields.program_utils as pu
import strawberryfields.decompositions as dec
from .backends.states import BaseFockState, BaseGaussianState
from .backends.shared_ops import changebasis
from .program_utils import Command, RegRef, MergeFailure
from .parameters import par_regref_deps, par_str, par_evaluate, par_is_symbolic, par_funcs as pf

# pylint: disable=abstract-method
# pylint: disable=protected-access
# pylint: disable=arguments-differ  # Measurement._apply introduces the "shots" argument

# numerical tolerances
_decomposition_merge_tol = 1e-13
_decomposition_tol = (
    1e-13  # TODO this tolerance is used for various purposes and is not well-defined
)


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """User warning formatter"""
    # pylint: disable=unused-argument
    return "{}:{}: {}: {}\n".format(filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


def _seq_to_list(s):
    "Converts a Sequence or a single object into a list."
    if not isinstance(s, Sequence):
        s = [s]
    return list(s)


class Operation:
    """Abstract base class for quantum operations acting on one or more subsystems.

    :attr:`Operation.measurement_deps` is a set containing the :class:`.RegRef`
    the :class:`Operation` depends on through its parameters.
    In the quantum circuit diagram notation it corresponds to the vertical double lines of classical
    information entering the :class:`Operation` that originate in the measurement of a subsystem.

    This abstract base class may be initialised with parameters; see the
    :class:`~strawberryfields.parameters.Parameter` class for more details.

    Args:
        par (Sequence[Any]): Operation parameters. An empty sequence if no parameters
            are required.
    """

    # default: one-subsystem operation
    #: int: number of subsystems the operation acts on, or None if any number > 0 is ok
    ns = 1

    def __init__(self, par):
        #: set[RegRef]: extra dependencies due to deferred measurements, used during optimization
        self._measurement_deps = set()
        #: list[Parameter]: operation parameters
        self.p = []

        # convert each parameter into a Parameter instance, keep track of dependenciens
        for q in par:
            if isinstance(q, RegRef):
                raise TypeError("Use RegRef.par for measured parameters.")
            self.p.append(q)
            self._measurement_deps |= par_regref_deps(q)

    def __str__(self):
        """String representation for the Operation using Blackbird syntax.

        Returns:
            str: string representation
        """
        # defaults to the class name
        if not self.p:
            return self.__class__.__name__

        # class name and parameter values
        temp = [par_str(i) for i in self.p]
        return self.__class__.__name__ + "(" + ", ".join(temp) + ")"

    @property
    def measurement_deps(self):
        """Extra dependencies due to parameters that depend on measurements.

        Returns:
            set[RegRef]: dependencies
        """
        return self._measurement_deps

    def __or__(self, reg):
        """Apply the operation to a part of a quantum register.

        Appends the Operation to a :class:`.Program` instance.

        Args:
            reg (RegRef, Sequence[RegRef]): subsystem(s) the operation is acting on

        Returns:
            list[RegRef]: subsystem list as RegRefs
        """
        # into a list of subsystems
        reg = _seq_to_list(reg)
        if (not reg) or (self.ns is not None and self.ns != len(reg)):
            raise ValueError("Wrong number of subsystems.")
        # append it to the Program
        reg = pu.Program_current_context.append(self, reg)
        return reg

    def merge(self, other):
        """Merge the operation with another (acting on the exact same set of subsystems).

        .. note:: For subclass overrides: merge may return a newly created object,
           or self, or other, but it must never modify self or other
           because the same Operation objects may be also used elsewhere.

        Args:
            other (Operation): operation to merge this one with

        Returns:
            Operation, None: other * self. The return value None represents
            the identity gate (doing nothing).

        Raises:
            .MergeFailure: if the two operations cannot be merged
        """
        # todo: Using the return value None to denote the identity is a
        # bit dangerous, since a function with no explicit return statement
        # also returns None, which can lead to puzzling bugs. Maybe return
        # a special singleton Identity object instead?
        raise NotImplementedError

    def decompose(self, reg, **kwargs):
        """Decompose the operation into elementary operations supported by the backend API.

        See :mod:`strawberryfields.backends.base`.

        Args:
            reg (Sequence[RegRef]): subsystems the operation is acting on

        Returns:
            list[Command]: decomposition as a list of operations acting on specific subsystems
        """
        return self._decompose(reg, **kwargs)

    def _decompose(self, reg, **kwargs):
        """Internal decomposition method defined by subclasses.

        NOTE: Does not evaluate Operation parameters, symbolic parameters remain symbolic.

        Args:
            reg (Sequence[RegRef]): subsystems the operation is acting on

        Returns:
            list[Command]: decomposition as a list of operations acting on specific subsystems
        """
        raise NotImplementedError("No decomposition available: {}".format(self))

    def _apply(self, reg, backend, **kwargs):
        """Internal apply method. Uses numeric subsystem referencing.

        Args:
            reg (Sequence[int]): subsystem indices the operation is
                acting on (this is how the backend API wants them)
            backend (BaseBackend): backend to execute the operation

        Returns:
            array[Number] or None: Measurement results, if any; shape == (len(reg), shots).
        """
        raise NotImplementedError("Missing direct implementation: {}".format(self))

    def apply(self, reg, backend, **kwargs):
        """Ask a local backend to execute the operation on the current register state right away.

        Takes care of parameter evaluations and any pending formal
        transformations (like dagger) and then calls :meth:`Operation._apply`.

        Args:
            reg (Sequence[RegRef]): subsystem(s) the operation is acting on
            backend (BaseBackend): backend to execute the operation

        Returns:
            Any: the result of self._apply
        """
        # NOTE: We cannot just replace all parameters with their evaluated
        # numerical values here. If we re-initialize a measured mode and
        # re-measure it, the corresponding MeasuredParameter value should change accordingly
        # when it is used again after the new measurement.

        # convert RegRefs back to indices for the backend API
        temp = [rr.ind for rr in reg]
        # call the child class specialized _apply method
        return self._apply(temp, backend, **kwargs)


# ====================================================================
# Derived operation classes
# ====================================================================


class Preparation(Operation):
    """Abstract base class for operations that demolish
    the previous state of the subsystem entirely.
    """

    def merge(self, other):
        # sequential preparation, only the last one matters
        if isinstance(other, Preparation):
            # give a warning, since this is pointless and probably a user error
            warnings.warn("Two subsequent state preparations, first one has no effect.")
            return other

        raise MergeFailure("For now, Preparations cannot be merged with anything else.")


class Measurement(Operation):
    """Abstract base class for subsystem measurements.

    The measurement is deferred: its result is available only
    after the backend has executed it. The value of the measurement can
    be accessed in the program through the symbolic subsystem reference
    to the measured subsystem.

    When the measurement happens, the state of the system is updated
    to the conditional state corresponding to the measurement result.
    Measurements also support postselection, see below.

    Args:
        select (None, Sequence[Number]): Desired values of the measurement
            results, one for each subsystem the measurement acts on.
            Allows the post-selection of specific measurement results
            instead of randomly sampling. None means no postselection.
    """

    # todo: self.select could support :class:`~strawberryfields.parameters.Parameter` instances.
    ns = None

    def __init__(self, par, select=None):
        super().__init__(par)
        #: None, Sequence[Number]: postselection values, one for each measured subsystem
        self.select = select

    def __str__(self):
        # class name, parameter values, and possibly the select parameter
        temp = super().__str__()

        if self.select is not None:
            if not self.p:
                temp += f"(select={self.select})"
            else:
                temp = f"{temp[:-1]}, select={self.select})"

        return temp

    def merge(self, other):
        raise MergeFailure("For now, measurements cannot be merged with anything else.")

    def apply(self, reg, backend, **kwargs):
        """Ask a backend to execute the operation on the current register state right away.

        Like :func:`Operation.apply`, but also stores the measurement result in the RegRefs.

        Keyword Args:
            shots (int): Number of independent evaluations to perform.
                Only applies to Measurements.
        """
        values = super().apply(reg, backend, **kwargs)

        # store the results in the register reference objects
        for v, r in zip(np.transpose(values), reg):
            r.val = v

        return values


class Decomposition(Operation):
    """Abstract base class for multimode matrix transformations.

    This class provides the base behaviour for decomposing various multimode operations
    into a sequence of gates and state preparations.

    .. note:: The first parameter ``p[0]`` of a Decomposition is always a square matrix, and it cannot be symbolic.
    """

    ns = None  # overridden by child classes in __init__

    @staticmethod
    def _check_p0(p0):
        """Checks that p0 is not symbolic."""
        if par_is_symbolic(p0):
            raise TypeError(
                "The first parameter of a Decomposition is a square matrix, and cannot be symbolic."
            )

    def __init__(self, par, decomp=True):
        self._check_p0(par[0])
        super().__init__(par)
        self.decomp = decomp
        """bool: If False, try to apply the Decomposition as a single primitive operation
        instead of decomposing it."""

    def merge(self, other):
        # can be merged if they are the same class
        if isinstance(other, self.__class__):
            # at the moment, we will assume all state decompositions only
            # take one argument. The only exception currently are state
            # decompositions, which cannot be merged.
            U1 = self.p[0]
            U2 = other.p[0]
            U = U2 @ U1
            # Note: above we strip the Parameter wrapper to make the following check
            # easier to perform. The constructor restores it.
            # Another option would be to add the required methods to Parameter class.
            # check if the matrices cancel
            if np.allclose(U, np.identity(len(U)), atol=_decomposition_merge_tol, rtol=0):
                return None

            return self.__class__(U)

        raise MergeFailure("Not the same decomposition type.")


class Transformation(Operation):
    """Abstract base class for transformations.

    This class provides the base behaviour for operations which
    act on existing states.
    """

    # NOTE: At the moment this is an empty class, and only
    # exists for a nicer inheritence diagram. One option is
    # to remove, and make Channel and Gate top-level derived classes.
    #
    # Are there any useful operations/properties shared by Gate/Channel?


# ====================================================================
# Derived transformation classes
# ====================================================================


class Channel(Transformation):
    """Abstract base class for quantum channels.

    This class provides the base behaviour for non-unitary
    maps and transformations.
    """

    # TODO decide how all Channels should treat the first parameter p[0]
    # (see e.g. https://en.wikipedia.org/wiki/C0-semigroup), cf. p[0] in ops.Gate

    def merge(self, other):
        if not self.__class__ == other.__class__:
            raise MergeFailure("Not the same channel family.")

        # channels can be merged if they are the same class and share all the other parameters
        if self.p[1:] == other.p[1:]:
            # determine the combined first parameter
            T = self.p[0] * other.p[0]
            # if one, replace with the identity
            if T == 1:
                return None

            # return a copy
            # NOTE deepcopy would make copies of the parameters which would mess things up
            temp = copy.copy(self)
            temp.p = [T] + self.p[1:]  # change the parameter list
            return temp

        raise MergeFailure("Don't know how to merge these operations.")


class Gate(Transformation):
    """Abstract base class for unitary quantum gates.

    The first parameter p[0] of the Gate class is special:

    * The value p[0] = 0 corresponds to the identity gate.
    * The inverse gate is obtained by negating p[0].
    * Two gates of this class can be merged by adding the
      first parameters together, assuming all the other parameters match.
    """

    def __init__(self, par):
        super().__init__(par)
        # default: non-dagger form
        self.dagger = False  #: bool: formal inversion of the gate

    def __str__(self):
        """String representation for the gate."""
        # add a dagger symbol to the class name if needed
        temp = super().__str__()
        if self.dagger:
            temp += ".H"
        return temp

    @property
    def H(self):
        """Returns a copy of the gate with the self.dagger flag flipped.

        H stands for hermitian conjugate.

        Returns:
            Gate: formal inverse of this gate
        """
        # HACK Semantically a bad use of @property since this method is not a getter.
        # NOTE deepcopy would make copies of the parameters which would mess things up
        s = copy.copy(self)
        s.dagger = not s.dagger
        return s

    def decompose(self, reg, **kwargs):
        """Decompose the operation into elementary operations supported by the backend API.

        Like :func:`Operation.decompose`, but applies self.dagger.
        """
        seq = self._decompose(reg, **kwargs)
        if self.dagger:
            # apply daggers, reverse the Command sequence
            for cmd in seq:
                cmd.op.dagger = not cmd.op.dagger
            seq = list(reversed(seq))
        return seq

    def apply(self, reg, backend, **kwargs):
        """Ask a backend to execute the operation on the current register state right away.

        Like :func:`Operation.apply`, but takes into account the special nature of
        p[0] and applies self.dagger.

        Returns:
            None: Gates do not return anything, return value is None
        """
        z = self.p[0]
        # if z represents a batch of parameters, then all of these
        # must be zero to skip calling backend
        if np.all(z == 0):
            # identity, no need to apply
            return
        if self.dagger:
            z = -z
        original_p0 = self.p[0]  # store the original Parameter
        self.p[0] = z

        # convert RegRefs back to indices for the backend API
        temp = [rr.ind for rr in reg]
        # call the child class specialized _apply method
        self._apply(temp, backend, **kwargs)
        self.p[0] = original_p0  # restore the original Parameter instance

    def merge(self, other):
        if not self.__class__ == other.__class__:
            raise MergeFailure("Not the same gate family.")

        # gates can be merged if they are the same class and share all the other parameters
        if self.p[1:] == other.p[1:]:
            # make sure the gates have the same dagger flag, if not, invert the second p[0]
            if self.dagger == other.dagger:
                temp = other.p[0]
            else:
                temp = -other.p[0]
            # now we can add up the parameters and keep self.dagger
            p0 = self.p[0] + temp
            if p0 == 0:
                return None  # identity gate

            # return a copy
            # NOTE deepcopy would make copies the parameters which would mess things up
            temp = copy.copy(self)
            temp.p = [p0] + self.p[1:]  # change the parameter list
            return temp

        raise MergeFailure("Don't know how to merge these gates.")


# ====================================================================
# State preparation operations
# ====================================================================


class Vacuum(Preparation):
    r"""Prepare a mode in the vacuum state.

    Can be accessed via the shortcut variable ``Vac``.

    .. note:: By default, newly created modes in Strawberry Fields default to the vacuum state.

    .. details::

        .. admonition:: Definition
            :class: defn

            The vacuum state :math:`\ket{0}` is a Gaussian state defined by

            .. math::
                \ket{0} = \frac{1}{\sqrt[4]{\pi \hbar}}
                \int dx~e^{-x^2/(2 \hbar)}\ket{x} ~~\text{where}~~ \a\ket{0}=0

        .. tip::

            *Available in Strawberry Fields as a NumPy array by*
            :func:`strawberryfields.utils.vacuum_state`

        In the Fock basis, it is represented by Fock state :math:`\ket{0}`,
        and in the Gaussian formulation, by :math:`\bar{\mathbf{r}}=(0,0)`
        and :math:`\mathbf{V}= \frac{\hbar}{2} I`.
    """

    def __init__(self):
        super().__init__([])

    def _apply(self, reg, backend, **kwargs):
        backend.prepare_vacuum_state(*reg)

    def __str__(self):
        # return the shorthand object when the
        # command is printed by the user
        return "Vac"


class Coherent(Preparation):
    r"""Prepare a mode in a coherent state.

    The gate is parameterized so that a user can specify a single complex number :math:`a=\alpha`
    or use the polar form :math:`a = r, p=\phi` and still get the same result.

    Args:
        r (float): displacement magnitude :math:`|\alpha|`
        phi (float): phase angle :math:`\phi`

    .. details::

        .. admonition:: Definition
            :class: defn

            The coherent state :math:`\ket{\alpha}`, :math:`\alpha\in\mathbb{C}`
            is a displaced vacuum state defined by

            .. math::
                \ket{\alpha} = D(\alpha)\ket{0}

        .. tip::

            *Available in Strawberry Fields as a NumPy array by*
            :func:`strawberryfields.utils.coherent_state`

        A coherent state is a minimum uncertainty state, and the
        eigenstate of the annihilation operator:

        .. math:: \a\ket{\alpha} = \alpha\ket{\alpha}

        In the Fock basis, it has the decomposition

        .. math:: |\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^\infty
                  \frac{\alpha^n}{\sqrt{n!}}|n\rangle

        whilst in the Gaussian formulation, :math:`\bar{\mathbf{r}}=2
        \sqrt{\frac{\hbar}{2}}(\text{Re}(\alpha), \text{Im}(\alpha))` and
        :math:`\mathbf{V}= \frac{\hbar}{2} I`.
    """

    def __init__(self, r=0.0, phi=0.0):
        super().__init__([r, phi])

    def _apply(self, reg, backend, **kwargs):
        r = par_evaluate(self.p[0])
        phi = par_evaluate(self.p[1])

        np_args = [arg.numpy() if hasattr(arg, "numpy") else arg for arg in [r, phi]]
        is_complex = any([np.iscomplexobj(np.real_if_close(arg)) for arg in np_args])

        if is_complex:
            raise ValueError("The arguments of Coherent(r, phi) cannot be complex")

        backend.prepare_coherent_state(r, phi, *reg)


class Squeezed(Preparation):
    r"""Prepare a mode in a squeezed vacuum state.

    Args:
        r (float): squeezing magnitude
        p (float): squeezing angle :math:`\phi`

    .. details::

        .. admonition:: Definition
            :class: defn

            The squeezed state :math:`\ket{z}`, :math:`z=re^{i\phi}`
            is a squeezed vacuum state defined by

            .. math::
                \ket{z} = S(z)\ket{0}

        .. tip::

            *Available in Strawberry Fields as a NumPy array by*
            :func:`strawberryfields.utils.squeezed_state`

        A squeezed state is a minimum uncertainty state with unequal
        quadrature variances, and satisfies the following eigenstate equation:

        .. math:: \left(\a\cosh(r)+\ad e^{i\phi}\sinh(r)\right)\ket{z} = 0

        In the Fock basis, it has the decomposition

        .. math:: |z\rangle = \frac{1}{\sqrt{\cosh(r)}}\sum_{n=0}^\infty
                  \frac{\sqrt{(2n)!}}{2^n n!}(-e^{i\phi}\tanh(r))^n|2n\rangle

        whilst in the Gaussian formulation, :math:`\bar{\mathbf{r}} = (0,0)`,
        :math:`\mathbf{V} = \frac{\hbar}{2}R(\phi/2)\begin{bmatrix}e^{-2r} & 0 \\
        0 & e^{2r} \\\end{bmatrix}R(\phi/2)^T`.

        We can use the squeezed vacuum state to approximate the zero position and
        zero momentum eigenstates;

        .. math:: \ket{0}_x \approx S(r)\ket{0}, ~~~~ \ket{0}_p \approx S(-r)\ket{0}

        where :math:`z=r` is sufficiently large.
    """

    def __init__(self, r=0.0, p=0.0):
        super().__init__([r, p])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        backend.prepare_squeezed_state(p[0], p[1], *reg)


class DisplacedSqueezed(Preparation):
    r"""Prepare a mode in a displaced squeezed state.

    A displaced squeezed state is prepared by squeezing a vacuum state, and
    then applying a displacement operator.

    .. math::
       \ket{\alpha,z} = D(\alpha)\ket{0,z} = D(\alpha)S(z)\ket{0},

    where the squeezing parameter :math:`z=re^{i\phi}`.

    Args:
        r_d (float): displacement magnitude
        phi_d (float): displacement angle
        r_s (float): squeezing magnitude
        phi_s (float): squeezing angle :math:`\phi`

    .. details::

        .. admonition:: Definition
            :class: defn

            The displaced squeezed state :math:`\ket{\alpha, z}`, :math:`\alpha\in\mathbb{C}`,
            :math:`z=re^{i\phi}` is a displaced and squeezed vacuum state defined by

            .. math::
                \ket{\alpha, z} = D(\alpha)S(z)\ket{0}

        .. tip::

            *Available in Strawberry Fields as a NumPy array by*
            :func:`strawberryfields.utils.displaced_squeezed_state`

        In the Fock basis, it has the decomposition

        .. math::
            |\alpha,z\rangle = e^{-\frac{1}{2}|\alpha|^2-\frac{1}{2}{\alpha^*}^2
            e^{i\phi}\tanh{(r)}} \sum_{n=0}^\infty\frac{\left[\frac{1}{2}e^{i\phi}
            \tanh(r)\right]^{n/2}}{\sqrt{n!\cosh(r)}} H_n
            \left[ \frac{\alpha\cosh(r)+\alpha^*e^{i\phi}\sinh(r)}{\sqrt{e^{i\phi}\sinh(2r)}} \right]
            |n\rangle


        where :math:`H_n(x)` are the Hermite polynomials defined by
        :math:`H_n(x)=(-1)^n e^{x^2}\frac{d}{dx}e^{-x^2}`. Alternatively,
        in the Gaussian formulation, :math:`\bar{\mathbf{r}} = 2
        \sqrt{\frac{\hbar}{2}}(\text{Re}(\alpha),\text{Im}(\alpha))` and
        :math:`\mathbf{V} = R(\phi/2)\begin{bmatrix}e^{-2r} & 0 \\0 & e^{2r} \\
        \end{bmatrix}R(\phi/2)^T`


        We can use the displaced squeezed states to approximate the :math:`x`
        position and :math:`p` momentum eigenstates;

        .. math::
            \ket{x}_x \approx D\left(\frac{1}{2}x\right)S(r)\ket{0}, ~~~~
            \ket{p}_p \approx D\left(\frac{i}{2}p\right)S(-r)\ket{0}

        where :math:`z=r` is sufficiently large.
    """

    def __init__(self, r_d=0.0, phi_d=0.0, r_s=0.0, phi_s=0.0):
        super().__init__([r_d, phi_d, r_s, phi_s])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)

        np_args = [arg.numpy() if hasattr(arg, "numpy") else arg for arg in p]
        is_complex = any([np.iscomplexobj(np.real_if_close(arg)) for arg in np_args])

        if is_complex:
            raise ValueError(
                "The arguments of DisplacedSqueezed(r_d, phi_d, r_s, phi_s) cannot be complex"
            )

        # prepare the displaced squeezed state directly
        backend.prepare_displaced_squeezed_state(p[0], p[1], p[2], p[3], *reg)

    def _decompose(self, reg, **kwargs):
        # squeezed state preparation followed by a displacement gate
        return [
            Command(Squeezed(self.p[2], self.p[3]), reg),
            Command(Dgate(self.p[0], self.p[1]), reg),
        ]


class Fock(Preparation):
    r"""Prepare a mode in a Fock basis state.

    The prepared mode is traced out and replaced with the Fock state :math:`\ket{n}`.
    As a result the state of the other subsystems may have to be described using a density matrix.

    .. warning::
        The Fock basis is **non-Gaussian**, and thus can
        only be used in the Fock backends, *not* the Gaussian backend.

    Args:
        n (int): Fock state to prepare

    .. details::

        .. admonition:: Definition
            :class: defn

            A single mode state can be decomposed into the Fock basis as follows:

            .. math::
                \ket{\psi} = \sum_n c_n \ket{n}

            if there exists a unique integer :math:`m` such that
            :math:`\begin{cases}c_n=1& n=m\\c_n=0&n\neq m\end{cases}`,
            then the single mode is simply a Fock state or :math:`n` photon state.

        .. tip::

            *Available in Strawberry Fields as a NumPy array by*
            :func:`strawberryfields.utils.fock_state`

            *Arbitrary states in the Fock basis can be applied in Strawberry Fields
            using the state preparation operator* :class:`strawberryfields.ops.Ket`
    """

    def __init__(self, n=0):
        super().__init__([n])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        backend.prepare_fock_state(p[0], *reg)


class Catstate(Preparation):
    r"""Prepare a mode in a cat state.

    A cat state is the coherent superposition of two coherent states,

    .. math::
       \ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{i\phi} \ket{-\alpha}),

    where :math:`N = \sqrt{2 (1+\cos(\phi)e^{-2|\alpha|^2})}` is the normalization factor.

    .. warning::
        Cat states are **non-Gaussian**, and thus can
        only be used in the Fock backends, *not* the Gaussian backend.

    Args:
        alpha (complex): displacement parameter
        p (float): parity, where :math:`\phi=p\pi`. ``p=0`` corresponds to an even
            cat state, and ``p=1`` an odd cat state.

    .. details::

        .. admonition:: Definition
            :class: defn

            The cat state is a non-Gaussian superposition of coherent states

            .. math::
                |cat\rangle = \frac{e^{-|\alpha|^2/2}}{\sqrt{2(1+e^{-2|\alpha|^2}\cos(\phi))}}
                \left(|\alpha\rangle +e^{i\phi}|-\alpha\rangle\right)

            with the even cat state given for :math:`\phi=0`, and the odd cat state
            given for :math:`\phi=\pi`.

        .. tip::

            *Implemented in Strawberry Fields as a NumPy array by*
            :class:`strawberryfields.utils.cat_state`

        In the case where :math:`\alpha<1.2`, the cat state can be approximated by
        the squeezed single photon state :math:`S\ket{1}`.
    """

    def __init__(self, alpha=0, p=0):
        super().__init__([alpha, p])

    def _apply(self, reg, backend, **kwargs):
        alpha = self.p[0]
        phi = np.pi * self.p[1]
        D = backend.get_cutoff_dim()
        l = np.arange(D)[:, np.newaxis]

        # normalization constant
        temp = pf.exp(-0.5 * pf.Abs(alpha) ** 2)
        N = temp / pf.sqrt(2 * (1 + pf.cos(phi) * temp ** 4))

        # coherent states
        c1 = (alpha ** l) / np.sqrt(ssp.factorial(l))
        c2 = ((-alpha) ** l) / np.sqrt(ssp.factorial(l))
        # add them up with a relative phase
        ket = (c1 + pf.exp(1j * phi) * c2) * N

        # in order to support broadcasting, the batch axis has been located at last axis, but backend expects it up as first axis
        ket = np.transpose(ket)

        # drop dummy batch axis if it is not necessary
        ket = np.squeeze(ket)

        # evaluate the array (elementwise)
        ket = par_evaluate(ket)

        backend.prepare_ket_state(ket, *reg)


class Ket(Preparation):
    r"""Prepare mode(s) using the given ket vector(s) in the Fock basis.

    The prepared modes are traced out and replaced with the given ket state
    (in the Fock basis). As a result the state of the other subsystems may have
    to be described using a density matrix.

    The provided kets must be each be of length ``cutoff_dim``, matching
    the cutoff dimension used in calls to :meth:`eng.run <~.Engine.run>`.

    .. warning::
        The Fock basis is **non-Gaussian**, and thus can
        only be used in the Fock backends, *not* the Gaussian backend.

    Args:
        state (array or BaseFockState): state vector in the Fock basis.
            This can be provided as either:

            * a single ket vector, for single mode state preparation
            * a multimode ket, with one array dimension per mode
            * a :class:`BaseFockState` state object.

    .. details::

        .. admonition:: Definition
            :class: defn

            A single mode state can be decomposed into the Fock basis as follows:

            .. math::
                \ket{\psi} = \sum_n c_n \ket{n}
    """
    ns = None

    def __init__(self, state):
        if isinstance(state, BaseFockState):
            if not state.is_pure:
                raise ValueError("Provided Fock state is not pure.")
            super().__init__([state.ket()])
        elif isinstance(state, BaseGaussianState):
            raise ValueError("Gaussian states are not supported for the Ket operation.")
        else:
            super().__init__([state])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        backend.prepare_ket_state(p[0], reg)


class DensityMatrix(Preparation):
    r"""Prepare mode(s) using the given density matrix in the Fock basis.

    The prepared modes are traced out and replaced with the given state
    (in the Fock basis). As a result, the overall state of system
    will also have to be described using a density matrix.

    The provided density matrices must be of size ``[cutoff_dim, cutoff_dim]``,
    matching the cutoff dimension used in calls to :meth:`eng.run <~.Engine.run>`.

    .. warning::
        The Fock basis is **non-Gaussian**, and thus can
        only be used in the Fock backends, *not* the Gaussian backend.

    Args:
        state (array or BaseFockState): density matrix in the Fock basis.
            This can be provided as either:

            * a single mode two-dimensional matrix :math:`\rho_{ij}`,
            * a multimode tensor :math:`\rho_{ij,kl,\dots,mn}`, with two indices per mode,
            * a :class:`BaseFockState` state object.

    .. details::

        When working with an :math:`N`-mode density matrix in the Fock basis,

        .. math::
            \rho = \sum_{n_1}\cdots\sum_{n_N} c_{n_1,\cdots,n_N}
            \ket{n_1,\cdots,n_N}\bra{n_1,\cdots,n_N}

        we use the convention that every pair of consecutive dimensions
        corresponds to a subsystem; i.e.,

        .. math::
            \rho_{\underbrace{ij}_{\text{mode}~0}~\underbrace{kl}_{\text{mode}~1}
            ~\underbrace{mn}_{\text{mode}~2}}

        Thus, using index notation, we can calculate the reduced density matrix
        for mode 2 by taking the partial trace over modes 0 and 1:

        .. math:: \braketT{n}{\text{Tr}_{01}[\rho]}{m} = \sum_{i}\sum_k \rho_{iikkmn}
    """
    ns = None

    def __init__(self, state):
        if isinstance(state, BaseFockState):
            super().__init__([state.dm()])
        elif isinstance(state, BaseGaussianState):
            raise ValueError("Gaussian states are not supported for the Ket operation.")
        else:
            super().__init__([state])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        backend.prepare_dm_state(p[0], reg)


class Thermal(Preparation):
    r"""Prepare a mode in a thermal state.

    The requested mode is traced out and replaced with the thermal state :math:`\rho(\bar{n})`.
    As a result the state will be described using a density matrix.

    Args:
        n (float): mean thermal population of the mode

    .. details::

        .. admonition:: Definition
            :class: defn

            The thermal state is a mixed Gaussian state defined by

            .. math::
                \rho(\nbar) := \sum_{n=0}^\infty\frac{\nbar^n}{(1+\nbar)^{n+1}}\ketbra{n}{n}

            where :math:`\nbar:=\tr{(\rho(\nbar)\hat{n})}` is the mean photon number.
            In the Gaussian formulation one has :math:`\mathbf{V}=(2 \nbar +1) \frac{\hbar}{2} I`
            and :math:`\bar{\mathbf{r}}=(0,0)`.
    """

    def __init__(self, n=0):
        super().__init__([n])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        backend.prepare_thermal_state(p[0], *reg)


# ====================================================================
# Measurements
# ====================================================================


class MeasureFock(Measurement):
    r"""Photon counting measurement: measures a set of modes in the Fock basis.

    Also accessible via the shortcut variable ``Measure``.

    After measurement, the modes are reset to the vacuum state.

    .. warning::
        Photon counting is available in the Gaussian backend,
        but the state of the circuit is not updated after measurement
        (since it would be non-Gaussian).

    .. details::

        .. admonition:: Definition
           :class: defn

           Photon counting is a non-Gaussian projective measurement given by

           .. math:: \ket{n_i}\bra{n_i}
    """

    ns = None

    def __init__(self, select=None, dark_counts=None):
        if dark_counts and select:
            raise NotImplementedError("Post-selection cannot be used together with dark counts.")

        if dark_counts is not None and not isinstance(dark_counts, Sequence):
            dark_counts = [dark_counts]

        if select is not None and not isinstance(select, Sequence):
            select = [select]

        self.dark_counts = dark_counts
        super().__init__([], select)

    def _apply(self, reg, backend, shots=1, **kwargs):
        samples = backend.measure_fock(reg, shots=shots, select=self.select, **kwargs)

        if isinstance(samples, list):
            samples = np.array(samples)

        if self.dark_counts is not None:
            if len(self.dark_counts) != len(reg):
                raise ValueError(
                    "The number of dark counts must be equal to the number of measured modes: "
                    "{} != {}".format(len(self.dark_counts), len(reg))
                )
            samples += np.random.poisson(self.dark_counts, samples.shape)

        return samples

    def __str__(self):
        # class name, parameter values, possible select and dark_counts parameters
        temp = super().__str__()

        if self.dark_counts is not None:
            if not self.select:
                temp += f"(dark_counts={self.dark_counts})"
            else:
                temp = f"{temp[:-1]}, dark_counts={self.dark_counts})"

        return temp


class MeasureThreshold(Measurement):
    """Measures a set of modes with thresholded Fock-state measurements, i.e.,
    measuring whether a mode contain zero or nonzero photons.

    After measurement, the modes are reset to the vacuum state.
    """

    ns = None

    def __init__(self, select=None):
        if select is not None and not isinstance(select, Sequence):
            select = [select]
        super().__init__([], select)

    def _apply(self, reg, backend, shots=1, **kwargs):
        return backend.measure_threshold(reg, shots=shots, select=self.select, **kwargs)


class MeasureHomodyne(Measurement):
    r"""Performs a homodyne measurement, measures one quadrature of a mode.

    * Position basis measurement: :math:`\phi = 0`
      (also accessible via the shortcut variable ``MeasureX``).

    * Momentum basis measurement: :math:`\phi = \pi/2`.
      (also accessible via the shortcut variable ``MeasureP``)

    The measured mode is reset to the vacuum state.

    Args:
        phi (float): measurement angle :math:`\phi`
        select (None, float): (Optional) desired values of measurement result.
            Allows the post-selection of specific measurement results instead of randomly sampling.

    .. details::

        .. admonition:: Definition
           :class: defn

           Homodyne measurement is a Gaussian projective measurement given by projecting the state
           onto the states

           .. math:: \ket{x_\phi}\bra{x_\phi},

           defined as eigenstates of the Hermitian operator

           .. math:: \hat{x}_\phi = \cos(\phi) \hat{x} + \sin(\phi)\hat{p}.

        In the Gaussian backend, this is done by projecting onto finitely squeezed states
        approximating the :math:`x` and :math:`p` eigenstates. Due to the finite squeezing
        approximation, this results in a measurement variance of :math:`\sigma_H^2`, where
        :math:`\sigma_H=2\times 10^{-4}`.

        In the Fock backends, this is done by using Hermite polynomials to calculate the
        :math:`\x_\phi` probability distribution over a specific range and number of bins,
        before taking a random sample.
    """
    ns = 1

    def __init__(self, phi, select=None):
        super().__init__([phi], select)

    def _apply(self, reg, backend, shots=1, **kwargs):
        p = par_evaluate(self.p)
        s = np.sqrt(sf.hbar / 2)  # scaling factor, since the backend API call is hbar-independent
        select = self.select
        if select is not None:
            select = select / s

        return s * backend.measure_homodyne(p[0], *reg, shots=shots, select=select, **kwargs)

    def __str__(self):
        if self.select is None:
            if self.p[0] == 0:
                return "MeasureX"
            if self.p[0] == np.pi / 2:
                return "MeasureP"
        return super().__str__()


class MeasureHeterodyne(Measurement):
    r"""Performs a heterodyne measurement on a mode.

    Also accessible via the shortcut variable ``MeasureHD``.

    Samples the joint Husimi distribution :math:`Q(\vec{\alpha}) =
    \frac{1}{\pi}\bra{\vec{\alpha}}\rho\ket{\vec{\alpha}}`.
    The measured mode is reset to the vacuum state.

    .. warning:: The heterodyne measurement can only be performed in the Gaussian backend.

    Args:
        select (None, complex): (Optional) desired values of measurement result.
            Allows the post-selection of specific measurement results instead of randomly sampling.

    .. details::

        .. admonition:: Definition
           :class: defn

           Heterodyne measurement is a Gaussian projective measurement given by projecting
           the state onto the coherent states,

           .. math:: \frac{1}{\pi} \ket{\alpha}\bra{\alpha}
    """
    ns = 1

    def __init__(self, select=None):
        super().__init__([], select)

    def _apply(self, reg, backend, shots=1, **kwargs):
        return backend.measure_heterodyne(*reg, shots=shots, select=self.select, **kwargs)

    def __str__(self):
        if self.select is None:
            return "MeasureHD"
        return "MeasureHeterodyne(select={})".format(self.select)


# ====================================================================
# Channels
# ====================================================================


class LossChannel(Channel):
    r"""Perform a loss channel operation on the specified mode.

    This channel couples mode :math:`\a` to another bosonic mode :math:`\hat{b}`
    prepared in the vacuum state using the following transformation:

    .. math::
        \a \mapsto \sqrt{T} a+\sqrt{1-T} \hat{b}

    Args:
        T (float): the loss parameter :math:`0\leq T\leq 1`.

    .. details::

        Loss is implemented by a CPTP map whose Kraus representation is

        .. math::
           \mathcal{N}(T)\left\{\ \cdot \ \right\} = \sum_{n=0}^{\infty} E_n(T) \
           \cdot \ E_n(T)^\dagger , \quad E_n(T) = \left(\frac{1-T}{T} \right)^{n/2}
           \frac{\a^n}{\sqrt{n!}} \left(\sqrt{T}\right)^{\ad \a}

        .. admonition:: Definition
            :class: defn

            Loss is implemented by coupling mode :math:`\a` to another bosonic mode
            :math:`\hat{b}` prepared in the vacuum state, by using the following transformation

            .. math::
               \a \to \sqrt{T} \a+\sqrt{1-T} \hat{b}

            and then tracing it out. Here, :math:`T` is the *energy* transmissivity.
            For :math:`T = 0` the state is mapped to the vacuum state, and for
            :math:`T=1` one has the identity map.

        One useful identity is

        .. math::
           \mathcal{N}(T)\left\{\ket{n}\bra{m} \right\}=\sum_{l=0}^{\min(n,m)}
           \left(\frac{1-T}{T}\right)^l \frac{T^{(n+m)/2}}{l!} \sqrt{\frac{n! m!}{(n-l)!(m-l)!}}
           \ket{n-l}\bra{m-l}

        In particular :math:`\mathcal{N}(T)\left\{\ket{0}\bra{0} \right\} =  \pr{0}`.
    """

    def __init__(self, T):
        super().__init__([T])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        backend.loss(p[0], *reg)


class ThermalLossChannel(Channel):
    r"""Perform a thermal loss channel operation on the specified mode.

    This channel couples mode :math:`\a` to another bosonic mode :math:`\hat{b}`
    prepared in a thermal state with mean photon number :math:`\bar{n}`,
    using the following transformation:

    .. math::
       \a \mapsto \sqrt{T} a+\sqrt{1-T} \hat{b}

    Args:
        T (float): the loss parameter :math:`0\leq T\leq 1`.
        nbar (float): mean photon number of the environment thermal state

    .. details::

        .. admonition:: Definition
            :class: defn

            Thermal loss is implemented by coupling mode :math:`\a` to another
            bosonic mode :math:`\hat{b}` prepared in the thermal state
            :math:`\ket{\bar{n}}`, by using the following transformation

            .. math::
               \a \to \sqrt{T} \a+\sqrt{1-T} \hat{b}

            and then tracing it out. Here, :math:`T` is the *energy* transmissivity.
            For :math:`T = 0` the state is mapped to the thermal state :math:`\ket{\bar{n}}`
            with mean photon number :math:`\bar{n}`, and for :math:`T=1` one has the identity map.

        Note that if :math:`\bar{n}=0`, the thermal loss channel is equivalent to the
        :doc:`loss channel <strawberryfields.ops.LossChannel>`.
    """

    def __init__(self, T, nbar):
        super().__init__([T, nbar])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        backend.thermal_loss(p[0], p[1], *reg)


# ====================================================================
# Unitary gates
# ====================================================================


class Dgate(Gate):
    r"""Phase space displacement gate.

    .. math::
        D(\alpha) = \exp(\alpha a^\dagger -\alpha^* a) = \exp\left(-i\sqrt{2}(\re(\alpha) \hat{p} -\im(\alpha) \hat{x})/\sqrt{\hbar}\right)

    where :math:`\alpha = r e^{i\phi}` has magnitude :math:`r\geq 0` and phase :math:`\phi`.

    The gate is parameterized so that a user can specify a single complex number :math:`a=\alpha`
    or use the polar form :math:`a = r, \phi` and still get the same result.

    Args:
        r (float): displacement magnitude :math:`|\alpha|`
        phi (float): displacement angle :math:`\phi`

    .. details::

        .. admonition:: Definition
            :class: defn

            .. math::
                D(\alpha) = \exp( \alpha \ad -\alpha^* \a) = \exp(r (e^{i\phi}\ad -e^{-i\phi}\a)),
                \quad D^\dagger(\alpha) \a D(\alpha)=\a +\alpha\I

            where :math:`\alpha=r e^{i \phi}` with :math:`r \geq 0` and :math:`\phi \in [0,2 \pi)`.

        We obtain for the position and momentum operators

        .. math::
            D^\dagger(\alpha) \x D(\alpha) = \x +\sqrt{2 \hbar } \re(\alpha) \I,\\
            D^\dagger(\alpha) \p D(\alpha) = \p +\sqrt{2 \hbar } \im(\alpha) \I.

        The matrix elements of the displacement operator in the Fock basis were derived by Cahill and Glauber :cite:`cahill1969`:

        .. math::
            \bra{m}\hat D(\alpha) \ket{n}  = \sqrt{\frac{n!}{m!}} \alpha^{m-n} e^{-|\alpha|^2/2} L_n^{m-n}\left( |\alpha|^2 \right)

        where :math:`L_n^{m}(x)` is a generalized Laguerre polynomial :cite:`dlmf`.
    """

    def __init__(self, r, phi=0.0):
        super().__init__([r, phi])

    def _apply(self, reg, backend, **kwargs):
        r, phi = par_evaluate(self.p)

        np_args = [arg.numpy() if hasattr(arg, "numpy") else arg for arg in [r, phi]]
        is_complex = any([np.iscomplexobj(np.real_if_close(arg)) for arg in np_args])

        if is_complex:
            raise ValueError("The arguments of Dgate(r, phi) cannot be complex")

        backend.displacement(r, phi, *reg)


class Xgate(Gate):
    r"""Position displacement gate.

    .. math::
        X(x) = e^{-i x \hat{p}/\hbar}

    Args:
        x (float): position displacement

    .. details::

        .. admonition:: Definition
            :class: defn

            The pure position displacement operator is defined as

            .. math::
                X(x) = D\left( x/\sqrt{2 \hbar}\right)  = \exp(-i x \p /\hbar),
                \quad X^\dagger(x) \x X(x) = \x +x\I,

            where :math:`D` is the :doc:`displacement gate <strawberryfields.ops.Dgate>`.
    """

    def __init__(self, x):
        super().__init__([x])

    def _decompose(self, reg, **kwargs):
        # into a displacement
        r = self.p[0] / np.sqrt(2 * sf.hbar)
        return [Command(Dgate(r, 0), reg)]


class Zgate(Gate):
    r"""Momentum displacement gate.

    .. math::
        Z(p) = e^{i p \hat{x}/\hbar}

    Args:
        p (float): momentum displacement

    .. details::

        .. admonition:: Definition
            :class: defn

            The pure position displacement operator is defined as

            .. math::
                Z(p) = D\left(i p/\sqrt{2 \hbar}\right) = \exp(i p \x /\hbar ),
                \quad Z^\dagger(p) \p Z(p) = \p +p\I,

            where :math:`D` is the :doc:`displacement gate <strawberryfields.ops.Dgate>`.
    """

    def __init__(self, p):
        super().__init__([p])

    def _decompose(self, reg, **kwargs):
        # into a displacement
        r = self.p[0] / np.sqrt(2 * sf.hbar)
        return [Command(Dgate(r, np.pi / 2), reg)]


class Sgate(Gate):
    r"""Phase space squeezing gate.

    .. math::
        S(z) = \exp\left(\frac{1}{2}(z^* a^2 -z {a^\dagger}^2)\right)

    where :math:`z = r e^{i\phi}`.

    Args:
        r (float): squeezing amount
        phi (float): squeezing phase angle :math:`\phi`

    .. details::

        .. admonition:: Definition
            :class: defn

            .. math::
                & S(z) = \exp\left(\frac{1}{2}\left(z^* \a^2-z {\ad}^{2} \right) \right)
                = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right)\\
                & S^\dagger(z) \a S(z) = \a \cosh(r) -\ad e^{i \phi} \sinh r\\
                & S^\dagger(z) \ad S(z) = \ad \cosh(r) -\a e^{-i \phi} \sinh(r)

            where :math:`z=r e^{i \phi}` with :math:`r \geq 0` and :math:`\phi \in [0,2 \pi)`.

        The squeeze gate affects the position and momentum operators as

        .. math::
            S^\dagger(z) \x_{\phi} S(z) = e^{-r}\x_{\phi}, ~~~ S^\dagger(z) \p_{\phi} S(z) = e^{r}\p_{\phi}

        The Fock basis decomposition of displacement and squeezing operations was analysed
        by Krall :cite:`kral1990`, and the following quantity was calculated,

        .. math::
            f_{n,m}(r,\phi,\beta)&=\bra{n}\exp\left(\frac{r}{2}\left(e^{i \phi} \a^2
                -e^{-i \phi} \ad \right) \right) D(\beta) \ket{m} = \bra{n}S(z^*) D(\beta) \ket{m}\\
            &=\sqrt{\frac{n!}{\mu  m!}} e^{\frac{\beta ^2 \nu ^*}{2\mu }-\frac{\left| \beta \right| ^2}{2}}
            \sum_{i=0}^{\min(m,n)}\frac{\binom{m}{i} \left(\frac{1}{\mu  \nu }\right)^{i/2}2^{\frac{i-m}{2}
                +\frac{i}{2}-\frac{n}{2}} \left(\frac{\nu }{\mu }\right)^{n/2}
                \left(-\frac{\nu ^*}{\mu }\right)^{\frac{m-i}{2}} H_{n-i}\left(\frac{\beta }{\sqrt{2}
                \sqrt{\mu  \nu }}\right) H_{m-i}\left(-\frac{\alpha ^*}{\sqrt{2}\sqrt{-\mu  \nu ^*}}\right)}{(n-i)!}

        where :math:`\nu=e^{- i\phi} \sinh(r), \mu=\cosh(r), \alpha=\beta \mu - \beta^* \nu`.

        Two important special cases of the last formula are obtained when :math:`r \to 0`
        and when :math:`\beta \to 0`:

        * For :math:`r \to 0` we can take :math:`\nu \to 1, \mu \to r, \alpha \to \beta` and use
          the fact that for large :math:`x \gg 1` the leading order term of the Hermite
          polynomials is  :math:`H_n(x) = 2^n x^n +O(x^{n-2})` to obtain

          .. math::
              f_{n,m}(0,\phi,\beta) = \bra{n}D(\beta) \ket{m}=\sqrt{\frac{n!}{  m!}}
              e^{-\frac{\left| \beta \right| ^2}{2}} \sum_{i=0}^{\min(m,n)}
              \frac{(-1)^{m-i}}{(n-i)!} \binom{m}{i} \beta^{n-i} (\beta^*)^{m-i}

        * On the other hand if we let :math:`\beta\to 0` we use the fact that

          .. math::
              H_n(0) =\begin{cases}0,  & \mbox{if }n\mbox{ is odd} \\
              (-1)^{\tfrac{n}{2}} 2^{\tfrac{n}{2}} (n-1)!! , & \mbox{if }n\mbox{ is even} \end{cases}

          to deduce that :math:`f_{n,m}(r,\phi,0)` is zero if :math:`n` is even and
          :math:`m` is odd or vice versa.

        When writing the Bloch-Messiah reduction :cite:`cariolaro2016`:cite:`cariolaro2016b`
        of a Gaussian state in the Fock basis one often needs the following matrix element

        .. math::
            \bra{k} D(\alpha) R(\theta) S(r) \ket{l}  = e^{i \theta l }
            \bra{k} D(\alpha) S(r e^{2i \theta}) \ket{l} = e^{i \theta l}
            f^*_{l,k}(-r,-2\theta,-\alpha)
    """

    def __init__(self, r, phi=0.0):
        super().__init__([r, phi])

    def _apply(self, reg, backend, **kwargs):
        r, phi = par_evaluate(self.p)
        backend.squeeze(r, phi, *reg)


class Pgate(Gate):
    r"""Quadratic phase gate.

    .. math::
        P(s) = e^{i \frac{s}{2} \hat{x}^2/\hbar}

    Args:
        s (float): parameter

    .. details::

        .. admonition:: Definition
            :class: defn

            .. math::
                P(s) = \exp\left(i  \frac{s}{2 \hbar} \x^2\right),
                \quad P^\dagger(s) \a P(s) = \a +i\frac{s}{2}(\a +\ad)

        It shears the phase space, preserving position:

        .. math::
            P^\dagger(s) \x P(s) &= \x,\\
            P^\dagger(s) \p P(s) &= \p +s\x.

        This gate can be decomposed as

        .. math::
            P(s) = R(\theta) S(r e^{i \phi})

        where :math:`\cosh(r) = \sqrt{1+(\frac{s}{2})^2}, \quad
        \tan(\theta) = \frac{s}{2}, \quad \phi = -\sign(s)\frac{\pi}{2} -\theta`.
    """

    def __init__(self, s):
        super().__init__([s])

    def _decompose(self, reg, **kwargs):
        # into a squeeze and a rotation
        temp = self.p[0] / 2
        r = pf.acosh(pf.sqrt(1 + temp ** 2))
        theta = pf.atan(temp)
        phi = -np.pi / 2 * pf.sign(temp) - theta
        return [Command(Sgate(r, phi), reg), Command(Rgate(theta), reg)]


class Vgate(Gate):
    r"""Cubic phase gate.

    .. math::
        V(\gamma) = e^{i \frac{\gamma}{3 \hbar} \hat{x}^3}

    .. warning::

        * The cubic phase gate has lower accuracy than the Kerr gate at the same cutoff dimension.

        * The cubic phase gate is **non-Gaussian**, and thus can only be used
          in the Fock backends, *not* the Gaussian backend.

    Args:
        gamma (float): parameter

    .. details::

        .. warning::
            The cubic phase gate can suffer heavily from numerical inaccuracies due to
            finite-dimensional cutoffs in the Fock basis. The gate implementation in
            Strawberry Fields is unitary, but it does not implement an exact cubic phase
            gate. The Kerr gate provides an alternative non-Gaussian gate.

        .. admonition:: Definition
            :class: defn

            .. math::
                V(\gamma) = \exp\left(i \frac{\gamma}{3 \hbar} \x^3\right),
                \quad V^\dagger(\gamma) \a V(\gamma) = \a +i\frac{\gamma}{2\sqrt{2/\hbar}} (\a +\ad)^2

        It transforms the phase space as follows:

        .. math::
            V^\dagger(\gamma) \x V(\gamma) &= \x,\\
            V^\dagger(\gamma) \p V(\gamma) &= \p +\gamma \x^2.
    """

    def __init__(self, gamma):
        super().__init__([gamma])

    def _apply(self, reg, backend, **kwargs):
        gamma_prime = self.p[0] * np.sqrt(sf.hbar / 2)
        # the backend API call cubic_phase is hbar-independent
        backend.cubic_phase(par_evaluate(gamma_prime), *reg)


class Kgate(Gate):
    r"""Kerr gate.

    .. math::
        K(\kappa) = e^{i \kappa \hat{n}^2}

    .. warning::
        The Kerr gate is **non-Gaussian**, and thus can only be used
        in the Fock backends, *not* the Gaussian backend.

    Args:
        kappa (float): parameter

    .. details::

        .. admonition:: Definition
            :class: defn

            The Kerr interaction is given by the Hamiltonian

            .. math::
                H = (\hat{a}^\dagger\hat{a})^2=\hat{n}^2

            which is non-Gaussian and diagonal in the Fock basis.

        We can therefore define the Kerr gate, with parameter :math:`\kappa` as

        .. math::
            K(\kappa) = \exp{(i\kappa\hat{n}^2)}.
    """

    def __init__(self, kappa):
        super().__init__([kappa])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        backend.kerr_interaction(p[0], *reg)


class Rgate(Gate):
    r"""Rotation gate.

    .. math::
        R(\theta) = e^{i \theta a^\dagger a}

    Args:
        theta (float): rotation angle :math:`\theta`.

    .. details::

        .. note::
            We use the convention that a positive value of :math:`\phi`
            corresponds to an **anticlockwise** rotation in the phase space.

        .. admonition:: Definition
            :class: defn

            We write the phase space rotation operator as

            .. math::
                R(\phi) = \exp\left(i \phi \ad \a\right)=
                \exp\left(i \frac{\phi}{2} \left(\frac{\x^2+  \p^2}{\hbar}-\I\right)\right),
                \quad R^\dagger(\phi) \a R(\phi) = \a e^{i \phi}

        It rotates the position and momentum quadratures to each other:

        .. math::
            R^\dagger(\phi)\x R(\phi) = \x \cos \phi -\p \sin \phi,\\
            R^\dagger(\phi)\p R(\phi) = \p \cos \phi +\x \sin \phi.
    """

    def __init__(self, theta):
        super().__init__([theta])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        backend.rotation(p[0], *reg)


class BSgate(Gate):
    r"""BSgate(theta=pi/4, phi=0.)
    Beamsplitter gate.

    .. math::
        B(\theta,\phi) = \exp\left(\theta (e^{i \phi} a_1 a_2^\dagger -e^{-i \phi} a_1^\dagger a_2) \right)

    Args:
        theta (float): Transmittivity angle :math:`\theta`. The transmission amplitude of
            the beamsplitter is :math:`t = \cos(\theta)`.
            The value :math:`\theta=\pi/4` gives the 50-50 beamsplitter (default).
        phi (float): Phase angle :math:`\phi`. The reflection amplitude of the beamsplitter
            is :math:`r = e^{i\phi}\sin(\theta)`.
            The value :math:`\phi = \pi/2` gives the symmetric beamsplitter.

    .. details::

        .. admonition:: Definition
            :class: defn

            For the annihilation and creation operators of two modes, denoted :math:`\a_1`
            and :math:`\a_2`, the beamsplitter is defined by

            .. math::
                B(\theta,\phi) = \exp\left(\theta (e^{i \phi}\a_1 \ad_2 - e^{-i \phi} \ad_1 \a_2) \right)

        **Action on the creation and annihilation operators**

        They will transform the operators according to

        .. math::
            B^\dagger(\theta,\phi) \a_1  B(\theta,\phi) &= \a_1\cos \theta -\a_2 e^{-i \phi} \sin \theta  = t \a_1 -r^* \a_2,\\
            B^\dagger(\theta,\phi) \a_2  B(\theta,\phi) &= \a_2\cos \theta + \a_1  e^{i \phi} \sin \theta= t \a_2 +r \a_1.

        where :math:`t = \cos \theta` and :math:`r = e^{i\phi} \sin \theta` are the
        transmittivity and reflectivity amplitudes of the beamsplitter respectively.

        Therefore, the beamsplitter transforms two input coherent states to two output
        coherent states :math:`B(\theta, \phi) \ket{\alpha,\beta} = \ket{\alpha',\beta'}`, where

        .. math::
            \alpha' &= \alpha\cos \theta-\beta e^{-i\phi}\sin\theta = t\alpha - r^*\beta\\
            \beta' &= \beta\cos \theta+\alpha e^{i\phi}\sin\theta = t\beta + r\alpha\\

        **Action on the quadrature operators**

        By substituting in the definition of the creation and annihilation operators in terms
        of the position and momentum operators, it is possible to derive an expression for
        how the beamsplitter transforms the quadrature operators:

        .. math::
            &\begin{cases}
                B^\dagger(\theta,\phi) \x_1 B(\theta,\phi) = \x_1 \cos(\theta)-\sin(\theta) [\x_2\cos(\phi)+\p_2\sin(\phi)]\\
                B^\dagger(\theta,\phi) \p_1 B(\theta,\phi) = \p_1 \cos(\theta)-\sin(\theta) [\p_2\cos(\phi)-\x_2\sin(\phi)]\\
            \end{cases}\\[12pt]
            &\begin{cases}
                B^\dagger(\theta,\phi) \x_2 B(\theta,\phi) = \x_2 \cos(\theta)+\sin(\theta) [\x_1\cos(\phi)-\p_1\sin(\phi)]\\
                B^\dagger(\theta,\phi) \p_2 B(\theta,\phi) = \p_2 \cos(\theta)+\sin(\theta) [\p_1\cos(\phi)+\x_1\sin(\phi)]
            \end{cases}

        **Action on the position and momentum eigenstates**

        A 50% or **50-50 beamsplitter** has :math:`\theta=\pi/4` and :math:`\phi=0` or
        :math:`\phi=\pi`; consequently :math:`|t|^2 = |r|^2 = \frac{1}{2}`, and it acts as follows:

        .. math::
            & B(\pi/4,0)\xket{x_1}\xket{x_2} = \xket{\frac{1}{\sqrt{2}}(x_1-x_2)}\xket{\frac{1}{\sqrt{2}}(x_1+x_2)}\\
            & B(\pi/4,0)\ket{p_1}_p\ket{p_2}_p = \xket{\frac{1}{\sqrt{2}}(p_1-p_2)}\xket{\frac{1}{\sqrt{2}}(p_1+p_2)}

        and

        .. math::
            & B(\pi/4,\pi)\xket{x_1}\xket{x_2} = \xket{\frac{1}{\sqrt{2}}(x_1+x_2)}\xket{\frac{1}{\sqrt{2}}(x_2-x_1)}\\
            & B(\pi/4,\pi)\ket{p_1}_p\ket{p_2}_p = \xket{\frac{1}{\sqrt{2}}(p_1+p_2)}\xket{\frac{1}{\sqrt{2}}(p_2-p_1)}

        Alternatively, **symmetric beamsplitter** (one that does not distinguish between
        :math:`\a_1` and :math:`\a_2`) is obtained by setting :math:`\phi=\pi/2`.
    """
    ns = 2

    def __init__(self, theta=np.pi / 4, phi=0.0):
        # default: 50% beamsplitter
        super().__init__([theta, phi])

    def _apply(self, reg, backend, **kwargs):
        theta, phi = par_evaluate(self.p)
        backend.beamsplitter(theta, phi, *reg)


class MZgate(Gate):
    r"""Mach-Zehnder interferometer.

    .. math::

        \mathrm{MZ}(\phi_{in}, \phi_{ex}) = BS\left(\frac{\pi}{4}, \frac{\pi}{2}\right)
            (R(\phi_{in})\otimes I) BS\left(\frac{\pi}{4}, \frac{\pi}{2}\right)
            (R(\phi_{ex})\otimes I)

    Args:
        phi_in (float): internal phase
        phi_ex (float): external phase


    This gate becomes the identity for ``phi_in=np.pi`` and ``phi_ex=0``, and permutes the modes
    for ``phi_in=0`` and ``phi_ex=0``.

    """
    ns = 2

    def __init__(self, phi_in, phi_ex):
        super().__init__([phi_in, phi_ex])

    def _decompose(self, reg, **kwargs):
        # into local phase shifts and two 50-50 beamsplitters
        return [
            Command(Rgate(self.p[1]), reg[0]),
            Command(BSgate(np.pi / 4, np.pi / 2), reg),
            Command(Rgate(self.p[0]), reg[0]),
            Command(BSgate(np.pi / 4, np.pi / 2), reg),
        ]


class S2gate(Gate):
    r"""Two-mode squeezing gate.

    .. math::
       S_2(z) = \exp\left(z a_1^\dagger a_2^\dagger - z^* a_1 a_2 \right) = \exp\left(r (e^{i\phi} a_1^\dagger a_2^\dagger e^{-i\phi} a_1 a_2 ) \right)

    where :math:`z = r e^{i\phi}`.

    Args:
        r (float): squeezing amount
        phi (float): squeezing phase angle :math:`\phi`

    .. details::

        .. admonition:: Definition
            :class: defn

            .. math::
                S_2(z) = \exp\left(z \a^\dagger_1\a^\dagger_2 -z^* \a_1 \a_2 \right) =
                \exp\left(r (e^{i\phi} \a^\dagger_1\a^\dagger_2 -e^{-i\phi} \a_1 \a_2 \right)

            where :math:`z=r e^{i \phi}` with :math:`r \geq 0` and :math:`\phi \in [0,2 \pi)`.

        It can be decomposed into two opposite local squeezers sandwiched
        between two 50\% :doc:`beamsplitters <strawberryfields.ops.BSgate>` :cite:`ebs2002`:

        .. math::
            S_2(z) = B^\dagger(\pi/4,0) \: \left[ S(z) \otimes S(-z)\right] \: B(\pi/4,0)

        Two-mode squeezing will transform the operators according to

        .. math::
            S_2(z)^\dagger \a_1  S_2(z) &= \a_1 \cosh(r)+\ad_2 e^{i \phi} \sinh(r),\\
            S_2(z)^\dagger \a_2  S_2(z) &= \a_2 \cosh(r)+\ad_1 e^{i \phi} \sinh(r),\\

        where :math:`z=r e^{i \phi}` with :math:`r \geq 0` and :math:`\phi \in [0,2 \pi)`.
    """
    ns = 2

    def __init__(self, r, phi=0.0):
        super().__init__([r, phi])

    def _apply(self, reg, backend, **kwargs):
        r, phi = par_evaluate(self.p)
        backend.two_mode_squeeze(r, phi, *reg)

    def _decompose(self, reg, **kwargs):
        # two opposite squeezers sandwiched between 50% beamsplitters
        S = Sgate(self.p[0], self.p[1])
        BS = BSgate(np.pi / 4, 0)
        return [Command(BS, reg), Command(S, reg[0]), Command(S.H, reg[1]), Command(BS.H, reg)]


class CXgate(Gate):
    r"""Controlled addition or sum gate in the position basis.

    .. math::
        \text{CX}(s) = \int dx \ket{x}\bra{x} \otimes D\left({\frac{1}{\sqrt{2\hbar}}}s x\right) = e^{-i s \: \hat{x} \otimes \hat{p}/\hbar}

    In the position basis it maps
    :math:`\ket{x_1, x_2} \mapsto \ket{x_1, s x_1 +x_2}`.

    Args:
        s (float): addition multiplier

    .. details::

        .. admonition:: Definition
            :class: defn

            The controlled-X gate, also known as the addition gate or
            the sum gate, is a controlled displacement in position. It is given by

            .. math::
                \text{CX}(s) = \int dx \xket{x}\xbra{x} \otimes
                D\left(\frac{s x}{\sqrt{2\hbar}}\right) =
                \exp\left({-i \frac{s}{\hbar} \: \x_1 \otimes \p_2}\right).

        It is called addition because in the position basis
        :math:`\text{CX}(s) \xket{x_1, x_2} = \xket{x_1, x_2+s x_1}`.

        We can also write the action of the addition gate on the canonical operators:

        .. math::
            \text{CX}(s)^\dagger \x_1 \text{CX}(s) &= \x_1\\
            \text{CX}(s)^\dagger \p_1 \text{CX}(s) &= \p_1- s \ \p_2\\
            \text{CX}(s)^\dagger \x_2 \text{CX}(s) &= \x_2+ s \ \x_1\\
            \text{CX}(s)^\dagger \p_2 \text{CX}(s) &= \p_2 \\
            \text{CX}(s)^\dagger \hat{a}_1 \text{CX}(s) &= \a_1+  \frac{s}{2} (\ad_2 -  \a_2)\\
            \text{CX}(s)^\dagger \hat{a}_2 \text{CX}(s) &= \a_2+  \frac{s}{2} (\ad_1 +  \a_1)\\

        The addition gate can be decomposed in terms of :doc:`single mode squeezers <strawberryfields.ops.Sgate>`
        and :doc:`beamsplitters <strawberryfields.ops.BSgate>` as follows:

        .. math::
            \text{CX}(s) = B(\frac{\pi}{2}+\theta,0)
            \left(S(r,0) \otimes S(-r,0) \right) B(\theta,0),

        where

        .. math::
            \sin(2 \theta) = \frac{-1}{\cosh r}, \ \cos(2 \theta)=-\tanh(r),
            \ \sinh(r) = -\frac{ s}{2}.
    """
    ns = 2

    def __init__(self, s=1):
        super().__init__([s])

    def _decompose(self, reg, **kwargs):
        s = self.p[0]
        r = pf.asinh(-s / 2)
        theta = 0.5 * pf.atan2(-1.0 / pf.cosh(r), -pf.tanh(r))
        return [
            Command(BSgate(theta, 0), reg),
            Command(Sgate(r, 0), reg[0]),
            Command(Sgate(-r, 0), reg[1]),
            Command(BSgate(theta + np.pi / 2, 0), reg),
        ]


class CZgate(Gate):
    r"""Controlled phase gate in the position basis.

    .. math::
        \text{CZ}(s) =  \iint dx dy \: e^{i sxy/\hbar} \ket{x,y}\bra{x,y} = e^{i s \: \hat{x} \otimes \hat{x}/\hbar}

    In the position basis it maps
    :math:`\ket{x_1, x_2} \mapsto e^{i s x_1 x_2/\hbar} \ket{x_1, x_2}`.

    Args:
        s (float): phase shift multiplier

    .. details::

        .. admonition:: Definition
            :class: defn

            .. math::
                \text{CZ}(s) =  \iint dx dy \: e^{i s x_1 x_2/\hbar }
                \xket{x_1,x_2}\xbra{x_1,x_2} = \exp\left({i s \: \hat{x_1}
                \otimes \hat{x_2} /\hbar}\right).

        It is related to the addition gate by a :doc:`phase space rotation <strawberryfields.ops.Rgate>`
        in the second mode:

        .. math::
            \text{CZ}(s) = R_{(2)}(\pi/2) \: \text{CX}(s) \: R_{(2)}^\dagger(\pi/2).

        In the position basis
        :math:`\text{CZ}(s) \xket{x_1, x_2} = e^{i  s x_1 x_2/\hbar} \xket{x_1, x_2}`.

        We can also write the action of the controlled-phase gate on the
        canonical operators:

        .. math::
            \text{CZ}(s)^\dagger \x_1 \text{CZ}(s) &= \x_1\\
            \text{CZ}(s)^\dagger \p_1 \text{CZ}(s) &= \p_1+ s \ \x_2\\
            \text{CZ}(s)^\dagger \x_2 \text{CZ}(s) &= \x_2\\
            \text{CZ}(s)^\dagger \p_2 \text{CZ}(s) &= \p_2+ s \ \x_1 \\
            \text{CZ}(s)^\dagger \hat{a}_1 \text{CZ}(s) &= \a_1+  i\frac{s}{2} (\ad_2 +  \a_2)\\
            \text{CZ}(s)^\dagger \hat{a}_2 \text{CZ}(s) &= \a_2+  i\frac{s}{2} (\ad_1 +  \a_1)\\
    """
    ns = 2

    def __init__(self, s=1):
        super().__init__([s])

    def _decompose(self, reg, **kwargs):
        # phase-rotated CZ
        CX = CXgate(self.p[0])
        return [
            Command(Rgate(-np.pi / 2), reg[1]),
            Command(CX, reg),
            Command(Rgate(np.pi / 2), reg[1]),
        ]


class CKgate(Gate):
    r"""Cross-Kerr gate.

    .. math::
        CK(\kappa) = e^{i \kappa \hat{n}_1\hat{n}_2}

    .. warning::
        The cross-Kerr gate is **non-Gaussian**, and thus can only
        be used in the Fock backends, *not* the Gaussian backend.

    Args:
        kappa (float): parameter

    .. details::

        .. admonition:: Definition
            :class: defn

            The cross-Kerr interaction is given by the Hamiltonian

            .. math::
                H = \hat{n}_1\hat{n_2}

            which is non-Gaussian and diagonal in the Fock basis.

        We can therefore define the cross-Kerr gate, with parameter :math:`\kappa` as

        .. math::
            CK(\kappa) = \exp{(i\kappa\hat{n}_1\hat{n_2})}.
    """
    ns = 2

    def __init__(self, kappa):
        super().__init__([kappa])

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        backend.cross_kerr_interaction(p[0], *reg)


class Fouriergate(Gate):
    r"""Fourier gate.

    Also accessible via the shortcut variable ``Fourier``.

    A special case of the :class:`phase space rotation gate <Rgate>`,
    where :math:`\theta=\pi/2`.

    .. math::
        F = R(\pi/2) = e^{i (\pi/2) a^\dagger a}

    .. details::

        .. admonition:: Definition
            :class: defn

            A special case of the :doc:`rotation operator <strawberryfields.ops.Rgate>`
            is the case :math:`\phi=\pi/2`; this corresponds to the Fourier gate,

            .. math::
                F = R(\pi/2) = e^{i (\pi/2) \ad \a}.

        The Fourier gate transforms the quadratures as follows:

        .. math::
            & F^\dagger\x F = -\p,\\
            & F^\dagger\p F = \x.
    """

    def __init__(self):
        super().__init__([np.pi / 2])

    def _decompose(self, reg, **kwargs):
        # into a rotation
        theta = np.pi / 2
        return [Command(Rgate(theta), reg)]

    def __str__(self):
        """String representation for the gate."""
        temp = "Fourier"
        if self.dagger:
            temp += ".H"
        return temp


# ====================================================================
# Metaoperations
# ====================================================================


# ====================================================================
# Subsystem creation and deletion
# ====================================================================


class MetaOperation(Operation):
    """Abstract base class for metaoperations.

    This includes subsystem creation and deletion.
    """

    def __init__(self):
        super().__init__(par=[])


class _Delete(MetaOperation):
    """Deletes one or more existing modes.
    Also accessible via the shortcut variable ``Del``.

    The deleted modes are traced out.
    After the deletion the state of the remaining subsystems may have to be described using a density operator.
    """

    ns = None

    def __or__(self, reg):
        reg = super().__or__(reg)
        pu.Program_current_context._delete_subsystems(reg)

    def _apply(self, reg, backend, **kwargs):
        backend.del_mode(reg)

    def __str__(self):
        # use the shorthand object
        return "Del"


def New(n=1):
    """Adds new subsystems to the quantum register.

    The new modes are prepared in the vacuum state.

    Must only be called in a :class:`Program` context.

    Args:
        n (int): number of subsystems to add
    Returns:
        tuple[RegRef]: tuple of the newly added subsystem references
    """
    if pu.Program_current_context is None:
        raise RuntimeError("New() can only be called inside a Program context.")
    # create RegRefs for the new modes
    refs = pu.Program_current_context._add_subsystems(n)
    # append the actual Operation to the Program
    pu.Program_current_context.append(_New_modes(n), refs)
    return refs


class _New_modes(MetaOperation):
    """Used internally for adding new modes to the system in a deferred way.

    This class cannot be used with the :meth:`__or__` syntax since it would be misleading.
    Indeed, users should *not* use this class directly, but rather the function :func:`New`.
    """

    ns = 0

    def __init__(self, n=1):
        """
        Args:
            n (int): number of modes to add
        """
        super().__init__()
        self.n = n  # int: store the number of new modes for the __str__ method

    def _apply(self, reg, backend, **kwargs):
        # pylint: disable=unused-variable
        inds = backend.add_mode(len(reg))

    def __str__(self):
        # use the shorthand object
        return "New({})".format(self.n)


class All(MetaOperation):
    """Metaoperation for applying a single-mode operation to every mode in the register.

    Args:
        op (Operation): single-mode operation to apply
    """

    def __init__(self, op):
        if op.ns != 1:
            raise ValueError("Not a one-subsystem operation.")
        super().__init__()
        self.op = op  #: Operation: one-subsystem operation to apply

    def __str__(self):
        return super().__str__() + "({})".format(str(self.op))

    def __or__(self, reg):
        # into a list of subsystems
        reg = _seq_to_list(reg)
        # convert into commands
        # make sure reg does not contain duplicates (we feed them to Program.append() one by one)
        pu.Program_current_context._test_regrefs(reg)
        for r in reg:
            pu.Program_current_context.append(self.op, [r])


# ====================================================================
# Decompositions
# ====================================================================


class Interferometer(Decomposition):
    r"""Apply a linear interferometer to the specified qumodes.

    This operation uses either the rectangular decomposition
    or triangular decomposition to decompose
    a linear interferometer into a sequence of beamsplitters and
    rotation gates.

    By specifying the keyword argument ``mesh``, the scheme used to implement the interferometer
    may be adjusted:

    * ``mesh='rectangular'`` (default): uses the scheme described in
      :cite:`clements2016`, resulting in a *rectangular* array of
      :math:`M(M-1)/2` beamsplitters:

      .. figure:: ../../_static/clements.png
          :align: center
          :width: 30%
          :target: javascript:void(0);

      Local phase shifts appear in the middle of the beamsplitter array.
      Use ``mesh='rectangular_phase_end`` to instead commute all local phase shifts
      to the end of the beamsplitter array.

      By default, the interferometers are decomposed into :class:`~.BSgate` operations.
      To instead decompose the interferometer using the :class:`~.ops.MZgate`,
      use ``mesh='rectangular_symmetric'``.

    * ``mesh='triangular'``: uses the scheme described in :cite:`reck1994`,
      resulting in a *triangular* array of :math:`M(M-1)/2` beamsplitters:

      .. figure:: ../../_static/reck.png
          :align: center
          :width: 30%
          :target: javascript:void(0);

      Local phase shifts appear at the end of the beamsplitter array.

    Args:
        U (array[complex]): an :math:`N\times N` unitary matrix
        mesh (str): the scheme used to implement the interferometer.
            Options include:

            - ``'rectangular'`` - rectangular mesh, with local phase shifts
              applied between interferometers

            - ``'rectangular_phase_end'`` - rectangular mesh, with local phase shifts
              placed after all interferometers

            - ``'rectangular_symmetric'`` - rectangular mesh, with local phase shifts
              placed after all interferometers, and all beamsplitters decomposed into
              pairs of symmetric beamsplitters and phase shifters

            - ``'triangular'`` - triangular mesh

        drop_identity (bool): If ``True``, decomposed gates with trivial parameters,
            such that they correspond to an identity operation, are removed.
        tol (float): the tolerance used when checking if the input matrix is unitary:
            :math:`|U-U^\dagger| <` tol

    .. details::

        The rectangular decomposition allows any passive Gaussian transformation
        to be decomposed into a series of beamsplitters and rotation gates.

        .. admonition:: Definition
            :class: defn

            For every real orthogonal symplectic matrix

            .. math:: O=\begin{bmatrix}X&-Y\\ Y&X\end{bmatrix}\in\mathbb{R}^{2N\times 2N},

            the corresponding unitary matrix :math:`U=X+iY\in\mathbb{C}^{N\times N}`
            representing a multiport interferometer can be decomposed into a set
            of :math:`N(N-1)/2` beamsplitters and single mode rotations with circuit
            depth of :math:`N`.

            For more details, see :cite:`clements2016`.

        .. note::

            The rectangular decomposition as formulated by Clements :cite:`clements2016`
            uses a different beamsplitter convention to Strawberry Fields:

            .. math:: BS_{clements}(\theta, \phi) = BS(\theta, 0) R(\phi)
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, U, mesh="rectangular", drop_identity=True, tol=1e-6):
        super().__init__([U])
        self.ns = U.shape[0]
        self.mesh = mesh
        self.tol = tol
        self.drop_identity = drop_identity

        allowed_meshes = {
            "rectangular",
            "rectangular_phase_end",
            "rectangular_symmetric",
            "triangular",
        }

        if mesh not in allowed_meshes:
            raise ValueError(
                "Unknown mesh '{}'. Mesh must be one of {}".format(mesh, allowed_meshes)
            )

        self.identity = np.allclose(U, np.identity(len(U)), atol=_decomposition_merge_tol, rtol=0)

    def _decompose(self, reg, **kwargs):
        mesh = kwargs.get("mesh", self.mesh)
        tol = kwargs.get("tol", self.tol)
        drop_identity = kwargs.get("drop_identity", self.drop_identity)

        cmds = []

        if not self.identity or not drop_identity:
            decomp_fn = getattr(dec, mesh)
            BS1, R, BS2 = decomp_fn(self.p[0], tol=tol)

            for n, m, theta, phi, _ in BS1:
                theta = theta if np.abs(theta) >= _decomposition_tol else 0
                phi = phi if np.abs(phi) >= _decomposition_tol else 0

                if "symmetric" in mesh:
                    # Mach-Zehnder interferometers
                    cmds.append(
                        Command(
                            MZgate(np.mod(theta, 2 * np.pi), np.mod(phi, 2 * np.pi)),
                            (reg[n], reg[m]),
                        )
                    )

                else:
                    # Clements style beamsplitters
                    if not (drop_identity and phi == 0):
                        cmds.append(Command(Rgate(phi), reg[n]))

                    if not (drop_identity and theta == 0):
                        cmds.append(Command(BSgate(theta, 0), (reg[n], reg[m])))

            for n, expphi in enumerate(R):
                # local phase shifts
                q = np.log(expphi).imag if np.abs(expphi - 1) >= _decomposition_tol else 0
                if not (drop_identity and q == 0):
                    cmds.append(Command(Rgate(np.mod(q, 2 * np.pi)), reg[n]))

            if BS2 is not None:
                # Clements style beamsplitters

                for n, m, theta, phi, _ in reversed(BS2):
                    theta = theta if np.abs(theta) >= _decomposition_tol else 0
                    phi = phi if np.abs(phi) >= _decomposition_tol else 0

                    if not (drop_identity and theta == 0):
                        cmds.append(Command(BSgate(-theta, 0), (reg[n], reg[m])))
                    if not (drop_identity and phi == 0):
                        cmds.append(Command(Rgate(-phi), reg[n]))

        return cmds


class GraphEmbed(Decomposition):
    r"""Embed a graph into an interferometer setup.

    This operation uses the Takagi decomposition to decompose
    an adjacency matrix into a sequence of squeezers and beamsplitters and
    rotation gates.

    Args:
        A (array): an :math:`N\times N` complex or real symmetric matrix
        mean_photon_per_mode (float): guarantees that the mean photon number in the pure Gaussian state
            representing the graph satisfies  :math:`\frac{1}{N}\sum_{i=1}^N sinh(r_{i})^2 ==` :code:``mean_photon``
        make_traceless (boolean): Removes the trace of the input matrix, by performing the transformation
            :math:`\tilde{A} = A-\mathrm{tr}(A) \I/n`. This may reduce the amount of squeezing needed to encode
            the graph but will lead to different photon number statistics for events with more than
            one photon in any mode.
        tol (float): the tolerance used when checking if the input matrix is symmetric:
            :math:`|A-A^T| <` tol
    """

    def __init__(self, A, mean_photon_per_mode=1.0, make_traceless=False, tol=1e-6):
        super().__init__([A])
        self.ns = A.shape[0]

        if np.allclose(A, np.identity(len(A)), atol=_decomposition_merge_tol, rtol=0):
            self.identity = True
        else:
            self.identity = False
            self.sq, self.U = dec.graph_embed(
                A,
                mean_photon_per_mode=mean_photon_per_mode,
                make_traceless=make_traceless,
                atol=tol,
                rtol=0,
            )

    def _decompose(self, reg, **kwargs):
        cmds = []

        if not self.identity:
            for n, s in enumerate(self.sq):
                if np.abs(s) >= _decomposition_tol:
                    cmds.append(Command(Sgate(s), reg[n]))

            if not np.allclose(self.U, np.identity(len(self.U)), atol=_decomposition_tol, rtol=0):
                mesh = kwargs.get("mesh", "rectangular")
                cmds.append(Command(Interferometer(self.U, mesh=mesh), reg))

        return cmds


class BipartiteGraphEmbed(Decomposition):
    r"""Embed a bipartite graph into an interferometer setup.

    A bipartite graph is a graph that consists of two vertex sets :math:`U` and :math:`V`,
    such that every edge in the graph connects a vertex between :math:`U` and :math:`V`.
    That is, there are no edges between vertices in the same vertex set.

    The adjacency matrix of an :math:`N` vertex undirected bipartite graph
    is a :math:`N\times N` symmetric matrix of the form

    .. math:: A = \begin{bmatrix}0 & B \\ B^T & 0\end{bmatrix}

    where :math:`B` is a :math:`N/2\times N/2` matrix representing the (weighted)
    edges between the vertex set.

    This operation decomposes an adjacency matrix into a sequence of two
    mode squeezers, beamsplitters, and rotation gates.

    Args:
        A (array): Either an :math:`N\times N` complex or real symmetric adjacency matrix
            :math:`A`, or an :math:`N/2\times N/2` complex or real matrix :math:`B`
            representing the edges between the vertex sets if ``edges=True``.
        mean_photon_per_mode (float): guarantees that the mean photon number in the pure Gaussian state
            representing the graph satisfies  :math:`\frac{1}{N}\sum_{i=1}^N sinh(r_{i})^2 ==` :code:``mean_photon``
        edges (bool): set to ``True`` if argument ``A`` represents the edges :math:`B`
            between the vertex sets rather than the full adjacency matrix
        drop_identity (bool): If ``True``, decomposed gates with trivial parameters,
            such that they correspond to an identity operation, are removed.
        tol (float): the tolerance used when checking if the input matrix is symmetric:
            :math:`|A-A^T| <` tol
    """

    def __init__(self, A, mean_photon_per_mode=1.0, edges=False, drop_identity=True, tol=1e-6):
        self._check_p0(A)
        self.mean_photon_per_mode = mean_photon_per_mode
        self.tol = tol
        self.identity = np.all(np.abs(A - np.identity(len(A))) < _decomposition_merge_tol)
        self.drop_identity = drop_identity

        if edges:
            self.ns = 2 * A.shape[0]
            B = A
        else:
            self.ns = A.shape[0]

            # check if A is a bipartite graph
            N = A.shape[0] // 2
            A00 = A[:N, :N]
            A11 = A[N:, N:]

            diag_zeros = np.allclose(A00, np.zeros_like(A00), atol=tol, rtol=0) and np.allclose(
                A11, np.zeros_like(A11), atol=tol, rtol=0
            )

            if (not diag_zeros) or (not np.allclose(A, A.T, atol=tol, rtol=0)):
                raise ValueError(
                    "Adjacency matrix {} does not represent a bipartite graph".format(A)
                )

            B = A[:N, N:]

        super().__init__([B])

    def _decompose(self, reg, **kwargs):
        mean_photon_per_mode = kwargs.get("mean_photon_per_mode", self.mean_photon_per_mode)
        tol = kwargs.get("tol", self.tol)
        mesh = kwargs.get("mesh", "rectangular")
        drop_identity = kwargs.get("drop_identity", self.drop_identity)

        cmds = []

        B = self.p[0]
        N = len(B)

        sq, U, V = dec.bipartite_graph_embed(
            B, mean_photon_per_mode=mean_photon_per_mode, atol=tol, rtol=0
        )

        if not self.identity or not drop_identity:
            for m, s in enumerate(sq):
                s = s if np.abs(s) >= _decomposition_tol else 0

                if not (drop_identity and s == 0):
                    cmds.append(Command(S2gate(-s), (reg[m], reg[m + N])))

            for X, _reg in ((U, reg[:N]), (V, reg[N:])):

                if np.allclose(X, np.identity(len(X)), atol=_decomposition_tol, rtol=0):
                    X = np.identity(len(X))

                if not (drop_identity and np.all(X == np.identity(len(X)))):
                    cmds.append(
                        Command(
                            Interferometer(X, mesh=mesh, drop_identity=drop_identity, tol=tol), _reg
                        )
                    )

        return cmds


class GaussianTransform(Decomposition):
    r"""Apply a Gaussian symplectic transformation to the specified qumodes.

    This operation uses the Bloch-Messiah decomposition
    to decompose a symplectic matrix :math:`S`:

    .. math:: S = O_1 R O_2

    where :math:`O_1` and :math:`O_2` are two orthogonal symplectic matrices (and thus passive
    Gaussian transformations), and :math:`R`
    is a squeezing transformation in the phase space (:math:`R=\text{diag}(e^{-z},e^z)`).

    The symplectic matrix describing the Gaussian transformation on :math:`N` modes must satisfy

    .. math:: S\Omega S^T = \Omega, ~~\Omega = \begin{bmatrix}0&I\\-I&0\end{bmatrix}

    where :math:`I` is the :math:`N\times N` identity matrix, and :math:`0` is the zero matrix.

    The two orthogonal symplectic unitaries describing the interferometers are then further
    decomposed via the :class:`~.Interferometer` operator and the
    :ref:`Rectangular decomposition <rectangular>`:

    .. math:: U_i = X_i + iY_i

    where

    .. math:: O_i = \begin{bmatrix}X&-Y\\Y&X\end{bmatrix}

    Args:
        S (array[float]): a :math:`2N\times 2N` symplectic matrix describing the Gaussian transformation.
        vacuum (bool): set to True if acting on a vacuum state. In this case, :math:`O_2 V O_2^T = I`,
            and the unitary associated with orthogonal symplectic :math:`O_2` will be ignored.
        tol (float): the tolerance used when checking if the matrix is symplectic:
            :math:`|S^T\Omega S-\Omega| \leq` tol

    .. details::

        .. admonition:: Definition
            :class: defn

            For every symplectic matrix :math:`S\in\mathbb{R}^{2N\times 2N}`, there
            exists orthogonal symplectic matrices :math:`O_1` and :math:`O_2`, and
            diagonal matrix :math:`Z`, such that

            .. math:: S = O_1 Z O_2

            where :math:`Z=\text{diag}(e^{-r_1},\dots,e^{-r_N},e^{r_1},\dots,e^{r_N})`
            represents a set of one mode squeezing operations with parameters
            :math:`(r_1,\dots,r_N)`.

        Gaussian symplectic transforms can be grouped into two main types; passive
        transformations (those which preserve photon number) and active transformations
        (those which do not). Compared to active transformation, passive transformations
        have an additional constraint - they must preserve the trace of the covariance
        matrix, :math:`\text{Tr}(SVS^T)=\text{Tr}(V)`; this only occurs when the
        symplectic matrix :math:`S` is also orthogonal (:math:`SS^T=\I`).

        The Bloch-Messiah decomposition therefore allows any active symplectic
        transformation to be decomposed into two passive Gaussian transformations
        :math:`O_1` and :math:`O_2`, sandwiching a set of one-mode squeezers, an
        active transformation.

        **Acting on the vacuum**

        In the case where the symplectic matrix :math:`S` is applied to a vacuum state
        :math:`V=\frac{\hbar}{2}\I`, the action of :math:`O_2` cancels out due to its orthogonality:

        .. math::
            SVS^T = (O_1 Z O_2)\left(\frac{\hbar}{2}\I\right)(O_1 Z O_2)^T
            = \frac{\hbar}{2} O_1 Z O_2 O_2^T Z O_1^T = \frac{\hbar}{2}O_1 Z^2 O_1^T

        As such, a symplectic transformation acting on the vacuum is sufficiently
        characterised by single mode squeezers followed by a passive Gaussian
        transformation (:math:`S = O_1 Z`).
    """

    def __init__(self, S, vacuum=False, tol=1e-10):
        super().__init__([S])
        self.ns = S.shape[0] // 2
        self.vacuum = (
            vacuum  #: bool: if True, ignore the first unitary matrix when applying the gate
        )
        N = self.ns  # shorthand

        # check if input symplectic is passive (orthogonal)
        diffn = np.linalg.norm(S @ S.T - np.identity(2 * N))
        self.active = (
            np.abs(diffn) > _decomposition_tol
        )  #: bool: S is an active symplectic transformation

        if not self.active:
            # The transformation is passive, do Clements
            X1 = S[:N, :N]
            P1 = S[N:, :N]
            self.U1 = X1 + 1j * P1
        else:
            # transformation is active, do Bloch-Messiah
            O1, smat, O2 = dec.bloch_messiah(S, tol=tol)
            X1 = O1[:N, :N]
            P1 = O1[N:, :N]
            X2 = O2[:N, :N]
            P2 = O2[N:, :N]

            self.U1 = X1 + 1j * P1  #: array[complex]: unitary matrix corresponding to O_1
            self.U2 = X2 + 1j * P2  #: array[complex]: unitary matrix corresponding to O_2
            self.Sq = np.diagonal(smat)[
                :N
            ]  #: array[complex]: diagonal vector of the squeezing matrix R

    def _decompose(self, reg, **kwargs):
        cmds = []
        mesh = kwargs.get("mesh", "rectangular")

        if self.active:
            if not self.vacuum:
                cmds = [Command(Interferometer(self.U2), reg)]

            for n, expr in enumerate(self.Sq):
                if np.abs(expr - 1) >= _decomposition_tol:
                    r = np.abs(np.log(expr))
                    phi = np.angle(np.log(expr))
                    cmds.append(Command(Sgate(-r, phi), reg[n]))

            cmds.append(Command(Interferometer(self.U1, mesh=mesh), reg))
        else:
            if not self.vacuum:
                cmds = [Command(Interferometer(self.U1, mesh=mesh), reg)]

        return cmds


class Gaussian(Preparation, Decomposition):
    r"""Prepare the specified modes in a Gaussian state.

    This operation uses the Williamson decomposition to prepare
    quantum modes into a given Gaussian state, specified by a
    vector of means and a covariance matrix.

    The Williamson decomposition decomposes the Gaussian state into a Gaussian
    transformation (represented by a symplectic matrix) acting on :class:`~.Thermal`
    states. The Gaussian transformation is then further decomposed into an array
    of beamsplitters and local squeezing and rotation gates, by way of the
    :class:`~.GaussianTransform` and :class:`~.Interferometer` decompositions.

    Alternatively, the decomposition can be explicitly turned off, and the
    backend can be explicitly prepared in the Gaussian state provided. This is
    **only** supported by backends using the Gaussian representation.

    .. note::

        :math:`V` must be a valid quantum state satisfying the uncertainty principle:
        :math:`V+\frac{1}{2}i\hbar\Omega\geq 0`. If this is not the case, the Williamson
        decomposition will return non-physical thermal states with :math:`\bar{n}_i<0`.

    Args:
        V (array[float]): an :math:`2N\times 2N` (real and positive definite) covariance matrix
        r (array[float] or None): Length :math:`2N` vector of means, of the
            form :math:`(\x_0,\dots,\x_{N-1},\p_0,\dots,\p_{N-1})`.
            If None, it is assumed that :math:`r=0`.
        decomp (bool): Should the operation be decomposed into a sequence of elementary gates?
            If False, the state preparation is performed directly via the backend API.
        tol (float): the tolerance used when checking if the matrix is symmetric: :math:`|V-V^T| \leq` tol

    .. details::

        .. admonition:: Definition
            :class: defn

            For every positive definite real matrix :math:`V\in\mathbb{R}^{2N\times 2N}`,
            there exists a symplectic matrix :math:`S` and diagonal matrix :math:`D` such that

            .. math:: V = S D S^T

            where :math:`D=\text{diag}(\nu_1,\dots,\nu_N,\nu_1,\dots,\nu_N)`, and
            :math:`\{\nu_i\}` are the eigenvalues of :math:`|i\Omega V|`, where :math:`||`
            represents the element-wise absolute value.

        The Williamson decomposition allows an arbitrary Gaussian covariance matrix to be
        decomposed into a symplectic transformation acting on the state described
        by the diagonal matrix :math:`D`.

        The matrix :math:`D` can always be decomposed further into a set of
        thermal states with mean photon number given by

        .. math:: \bar{n}_i = \frac{1}{\hbar}\nu_i - \frac{1}{2}, ~~i=1,\dots,N

        **Pure states**

        In the case where :math:`V` represents a pure state (:math:`|V|-(\hbar/2)^{2N}=0`),
        the Williamson decomposition outputs :math:`D=\frac{1}{2}\hbar I_{2N}`; that is,
        a symplectic transformation :math:`S` acting on the vacuum. It follows that the
        original covariance matrix can therefore be recovered simply via :math:`V=\frac{\hbar}{2}SS^T`.
    """
    # pylint: disable=too-many-instance-attributes
    ns = None

    def __init__(self, V, r=None, decomp=True, tol=1e-6):
        self._check_p0(V)
        # internally we eliminate hbar from the covariance matrix V (or equivalently set hbar=2), but not from the means vector r
        V = V / (sf.hbar / 2)
        self.ns = V.shape[0] // 2

        if r is None:
            r = np.zeros(2 * self.ns)
        r = np.asarray(r)

        if len(r) != V.shape[0]:
            raise ValueError("Vector of means must have the same length as the covariance matrix.")

        super().__init__([V, r], decomp=decomp)  # V is hbar-independent, r is not

        self.x_disp = r[: self.ns]
        self.p_disp = r[self.ns :]

        # needed only if decomposed
        th, self.S = dec.williamson(V, tol=tol)
        self.pure = np.abs(np.linalg.det(V) - 1.0) < tol
        self.nbar = 0.5 * (np.diag(th)[: self.ns] - 1.0)

    def _apply(self, reg, backend, **kwargs):
        p = par_evaluate(self.p)
        s = np.sqrt(sf.hbar / 2)  # scaling factor, since the backend API call is hbar-independent
        backend.prepare_gaussian_state(p[1] / s, p[0], reg)

    def _decompose(self, reg, **kwargs):
        # pylint: disable=too-many-branches
        cmds = []

        V = self.p[0]
        D = np.diag(V)
        is_diag = np.all(V == np.diag(D))

        BD = changebasis(self.ns) @ V @ changebasis(self.ns).T
        BD_modes = [BD[i * 2 : (i + 1) * 2, i * 2 : (i + 1) * 2] for i in range(BD.shape[0] // 2)]
        is_block_diag = (not is_diag) and np.all(BD == block_diag(*BD_modes))

        if self.pure and is_diag:
            # covariance matrix consists of x/p quadrature squeezed state
            for n, expr in enumerate(D[: self.ns]):
                if np.abs(expr - 1) >= _decomposition_tol:
                    r = np.abs(np.log(expr) / 2)
                    cmds.append(Command(Squeezed(r, 0), reg[n]))
                else:
                    cmds.append(Command(Vac, reg[n]))

        elif self.pure and is_block_diag:
            # covariance matrix consists of rotated squeezed states
            for n, v in enumerate(BD_modes):
                if not np.all(v - np.identity(2) < _decomposition_tol):
                    r = np.abs(np.arccosh(np.sum(np.diag(v)) / 2)) / 2
                    phi = np.arctan(2 * v[0, 1] / np.sum(np.diag(v) * [1, -1]))
                    cmds.append(Command(Squeezed(r, phi), reg[n]))
                else:
                    cmds.append(Command(Vac, reg[n]))

        elif not self.pure and is_diag and np.all(D[: self.ns] == D[self.ns :]):
            # covariance matrix consists of thermal states
            for n, nbar in enumerate(0.5 * (D[: self.ns] - 1.0)):
                if nbar >= _decomposition_tol:
                    cmds.append(Command(Thermal(nbar), reg[n]))
                else:
                    cmds.append(Command(Vac, reg[n]))

        else:
            if not self.pure:
                # mixed state, must initialise thermal states
                for n, nbar in enumerate(self.nbar):
                    if np.abs(nbar) >= _decomposition_tol:
                        cmds.append(Command(Thermal(nbar), reg[n]))
                    else:
                        cmds.append(Command(Vac, reg[n]))

            else:
                for r in reg:
                    cmds.append(Command(Vac, r))

            cmds.append(Command(GaussianTransform(self.S, vacuum=self.pure), reg))

        cmds += [Command(Xgate(u), reg[n]) for n, u in enumerate(self.x_disp) if u != 0]
        cmds += [Command(Zgate(u), reg[n]) for n, u in enumerate(self.p_disp) if u != 0]

        return cmds


# =======================================================================
# Shorthands, e.g. pre-constructed singleton-like objects

Del = _Delete()
Vac = Vacuum()
MeasureX = MeasureHomodyne(0)
MeasureP = MeasureHomodyne(np.pi / 2)
MeasureHD = MeasureHeterodyne()

Fourier = Fouriergate()

shorthands = ["New", "Del", "Vac", "MeasureX", "MeasureP", "MeasureHD", "Fourier", "All"]

# =======================================================================
# here we list different classes of operations for unit testing purposes

zero_args_gates = (Fouriergate,)
one_args_gates = (Xgate, Zgate, Rgate, Pgate, Vgate, Kgate, CXgate, CZgate, CKgate)
two_args_gates = (Dgate, Sgate, BSgate, MZgate, S2gate)
gates = zero_args_gates + one_args_gates + two_args_gates

channels = (LossChannel, ThermalLossChannel)

simple_state_preparations = (
    Vacuum,
    Coherent,
    Squeezed,
    DisplacedSqueezed,
    Fock,
    Catstate,
    Thermal,
)  # have __init__ methods with default arguments
state_preparations = simple_state_preparations + (Ket, DensityMatrix)

measurements = (MeasureFock, MeasureHomodyne, MeasureHeterodyne, MeasureThreshold)

decompositions = (Interferometer, BipartiteGraphEmbed, GraphEmbed, GaussianTransform, Gaussian)

# =======================================================================
# exported symbols

__all__ = [
    cls.__name__ for cls in gates + channels + state_preparations + measurements + decompositions
] + shorthands

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
.. _gates:

Quantum operations
===================

**Module name:** :mod:`strawberryfields.ops`

.. currentmodule:: strawberryfields.ops

.. note::

    In the :mod:`strawberryfields.ops` API we use the convention :math:`\hbar=2` by default, however
    this can be changed using the global variable :py:data:`strawberryfields.hbar`.

    See :ref:`conventions` for more details.

This module defines and implements the Python-embedded quantum programming language
for continuous-variable (CV) quantum systems.
The syntax is modeled after ProjectQ :cite:`projectq2016`.

Quantum operations (state preparation, unitary gates, measurements, channels) act on
register objects using the following syntax:

.. code-block:: python

    prog = sf.Program(3)
    with prog.context as q:
        G(params) | q
        F(params) | (q[1], q[6], q[2])

Here :samp:`prog` is an instance of :class:`strawberryfields.program.Program`
which defines the context where the commands are stored.
Within each command, the part on the left is an :class:`Operation` instance,
quite typically a constructor call for the requested operation class with the relevant parameters.
The vertical bar calls the :func:`__or__` method of the :class:`Operation` object,
with the part on the right as the parameter. The part on the right is a single
:class:`strawberryfields.engine.RegRef` object or, for multi-mode gates, a sequence of them.
It is of course also possible to construct gates separately and reuse them several times::

    R = Rgate(s)
    with prog.context as q:
        R   | q
        Xgate(t) | q
        R.H | q


There are six kinds of :class:`Operation` objects:

* :class:`Preparation` operations only manipulate the register state::

    with prog.context as q:
        Vac | q[0]
        All(Coherent(0.4, 0.2)) | (q[1], q[2])

* Transformations such as :class:`Gate` and :class:`Channel` operations only manipulate the register state::

    with prog.context as q:
        Dgate(0.3)   | q[0]
        BSgate(-0.5) | q[0:2]

* :class:`Measurement` operations manipulate the register state and produce classical information.
  The information is directly available only after the program has been run up to the point of measurement::

    with prog.context as (alice, bob):
        MeasureFock() | alice

    eng = sf.LocalEngine(backend='fock')
    eng.run(prog)
    print(alice.val)

  Alternatively one may use a symbolic reference to the register containing the measurement result
  by supplying registers as the argument to an :class:`Operation`, in which case the measurement may be deferred,
  i.e., we may symbolically use the measurement result before it exists::

    with prog.context as (alice, bob):
        MeasureFock()| alice
        Dgate(alice) | bob

  One may also include an arbitrary post-processing function for the measurement result, to be applied
  before using it as the argument to another :class:`Operation`. The :func:`~.convert` decorator can be used in Python
  to convert a user-defined function into a post-processing function recognized by the engine::

    @convert
    def square(q):
        return q ** 2

    with prog.context as q:
        MeasureFock()       | q[0]
        Dgate(square(q[0])) | q[1]

  Finally, the lower-level :class:`strawberryfields.engine.RegRefTransform` (RR) and
  an optional lambda function can be used to achieve the same functionality::

    with prog.context as q:
        MeasureFock()   | q[0]
        Dgate(RR(q[0])) | q[1]
        Dgate(RR(q[0], lambda q: q ** 2)) | q[2]

* Modes can be created and deleted during program execution using the
  function :func:`New` and the pre-constructed object :py:data:`Del`.
  Behind the scenes they utilize the meta-operations :class:`_New_modes` and :class:`_Delete`::

    with prog.context as (alice,):
        Sgate(1)    | alice
        bob, charlie = New(2)
        BSgate(0.5) | (alice, bob)
        CXgate(1)   | (alice, charlie)
        Del         | alice
        S2gate(0.4) | (charlie, bob)

* Finally, :class:`Decomposition` operations are a special case, and can act as
  either transformations *or* state preparation, depending on the decomposition used.
  Decompositions calculate the required elementary :class:`Gate` and/or :class:`Preparation`
  objects and parameters in order to decompose specific transformations or states.
  Examples of objects that are supported by decompositions include covariance matrices,
  interferometers, and symplectic transformations.

Hierarchy for operations
------------------------

.. inheritance-diagram:: strawberryfields.ops
   :parts: 1


Base classes
------------

The abstract base class hierarchy exists to provide the correct semantics for the actual operations that inherit them.

.. autosummary::
    Operation
    Preparation
    Transformation
    Gate
    Channel
    Measurement
    Decomposition
    MetaOperation


Operation class
---------------

All Operations have the following methods.

.. currentmodule:: strawberryfields.ops.Operation

.. autosummary::
    __str__
    __or__
    merge
    decompose
    apply
    _apply

.. currentmodule:: strawberryfields.ops


State preparation
-----------------

.. autosummary::
    Vacuum
    Coherent
    Squeezed
    DisplacedSqueezed
    Thermal
    Fock
    Catstate
    Ket
    DensityMatrix
    Gaussian

Measurements
------------

.. autosummary::
    MeasureFock
    MeasureThreshold
    MeasureHomodyne
    MeasureHeterodyne


Channels
-----------

.. autosummary::
    LossChannel
    ThermalLossChannel


Decompositions
--------------

.. autosummary::
    Interferometer
    GraphEmbed
    BipartiteGraphEmbed
    GaussianTransform
    Gaussian


Single-mode gates
-----------------

.. autosummary::
    Dgate
    Xgate
    Zgate
    Sgate
    Rgate
    Pgate
    Vgate
    Fouriergate

Two-mode gates
--------------

.. autosummary::
    BSgate
    MZgate
    S2gate
    CXgate
    CZgate
    CKgate

Meta-operations
---------------

.. autosummary::
    All
    _New_modes
    _Delete


Operations shortcuts
---------------------

Several of the operation classes described below come with variables that point to pre-constructed instances;
this is to provide shorthands for operations that accept no arguments, as well as for common variants of operations that do.

.. raw:: html

    <style>
      .widetable {
         width:100%;
      }
    </style>

.. rst-class:: longtable widetable

======================   =================================================================================
**Shorthand variable**   **Operation**
``New``                  :class:`~._New_modes`
``Del``                  :class:`~._Delete`
``Vac``                  :class:`~.Vacuum`
``Fourier``              :class:`~.Fouriergate`
``MeasureX``             :class:`~.MeasureHomodyne` (:math:`\phi=0`), :math:`x` quadrature measurement
``MeasureP``             :class:`~.MeasureHomodyne` (:math:`\phi=\pi/2`), :math:`p` quadrature measurement
``MeasureHD``            :class:`~.MeasureHeterodyne`
``RR``                   Alias for :class:`~.RegRefTransform`
======================   =================================================================================


Code details
~~~~~~~~~~~~

"""
from collections.abc import Sequence
import copy
import types
import sys
import warnings

import numpy as np
from numpy import pi

from scipy.linalg import block_diag
from scipy.special import factorial as fac

import strawberryfields as sf
import strawberryfields.program_utils as pu
from .backends.states import BaseFockState, BaseGaussianState
from .backends.shared_ops import changebasis
from .program_utils import (Command, RegRefTransform, MergeFailure)
from .parameters import (Parameter, _unwrap, matmul, sign, abs, exp, log, sqrt,
                         sin, cos, cosh, tanh, arcsinh, arccosh, arctan, arctan2,
                         transpose, squeeze)

from . import decompositions as dec

# pylint: disable=abstract-method
# pylint: disable=protected-access

# numerical tolerances
_decomposition_merge_tol = 1e-13
_decomposition_tol = 1e-13  # TODO this tolerance is used for various purposes and is not well-defined


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """User warning formatter"""
    # pylint: disable=unused-argument
    return '{}:{}: {}: {}\n'.format(filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


def _seq_to_list(s):
    "Converts a Sequence or a single object into a list."
    if not isinstance(s, Sequence):
        s = [s]
    return list(s)


class Operation:
    """Abstract base class for quantum operations acting on one or more subsystems.

    The extra_deps instance variable is a set containing the :class:`.RegRef`
    the :class:`Operation` depends on. In the quantum circuit diagram notation it
    corresponds to the vertical double lines of classical information
    entering the :class:`Operation` that originate in a measurement of a subsystem.

    This abstract base class may be initialised with parameters; see the
    :class:`~strawberryfields.parameters.Parameter` class for more details.

    Args:
        par (Sequence[Any]): Operation parameters. An empty sequence if no parameters
            are required.
    """
    # default: one-subsystem operation
    #: int: number of subsystems the operation acts on,
    # or None if any number of subsystems > 0 is okay
    ns = 1

    def __init__(self, par):
        #: set[RegRef]: extra dependencies due to deferred measurements, used during optimization
        self._extra_deps = set()
        #: list[Parameter]: operation parameters
        self.p = []

        # convert each parameter into a Parameter instance, keep track of dependenciens
        for q in par:
            if not isinstance(q, Parameter):
                q = Parameter(q)
            self.p.append(q)
            self._extra_deps.update(q.deps)

    def __str__(self):
        """String representation for the Operation using Blackbird syntax.

        Returns:
            str: string representation
        """
        # defaults to the class name
        if not self.p:
            return self.__class__.__name__

        # class name and parameter values
        temp = [str(i) for i in self.p]
        return self.__class__.__name__+'('+', '.join(temp)+')'

    @property
    def extra_deps(self):
        """Extra dependencies due to parameters that depend on measurements.

        Returns:
            set[RegRef]: dependencies
        """
        return self._extra_deps

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
        # todo: For now decompose() works on unevaluated Parameters.
        # This causes an error if a :class:`.RegRefTransform`-based Parameter is used, and
        # decompose() tries to do arithmetic on it.
        return self._decompose(reg, **kwargs)

    def _decompose(self, reg, **kwargs):
        """Internal decomposition method defined by subclasses.

        Args:
            reg (Sequence[RegRef]): subsystems the operation is acting on

        Returns:
            list[Command]: decomposition as a list of operations acting on specific subsystems
        """
        raise NotImplementedError('No decomposition available: {}'.format(self))

    def _apply(self, reg, backend, **kwargs):
        """Internal apply method. Uses numeric subsystem referencing.

        Args:
            reg (Sequence[int]): subsystem indices the operation is
                acting on (this is how the backend API wants them)
            backend (BaseBackend): backend to execute the operation

        Returns:
            array[Number] or None: Measurement results, if any; shape == (len(reg), shots).
        """
        raise NotImplementedError('Missing direct implementation: {}'.format(self))

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
        # NOTE: We cannot just replace all RegRefTransform parameters with their
        # numerical values here. If we re-initialize a measured mode and
        # re-measure it, the RegRefTransform value should change accordingly
        # when it is used again after the new measurement.

        original_p = self.p  # store the original parameters
        # Evaluate the Parameters, restore the originals later:
        self.p = [x.evaluate() for x in self.p]

        # convert RegRefs back to indices for the backend API
        temp = [rr.ind for rr in reg]
        # call the child class specialized _apply method
        result = self._apply(temp, backend, **kwargs)

        # restore original unevaluated Parameter instances
        self.p = original_p
        return result


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
            warnings.warn('Two subsequent state preparations, first one has no effect.')
            return other

        raise MergeFailure('For now, Preparations cannot be merged with anything else.')


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
            temp = temp[:-1] + ', select={})'.format(self.select)
        return temp

    def merge(self, other):
        raise MergeFailure('For now, measurements cannot be merged with anything else.')

    def apply(self, reg, backend, **kwargs):
        """Ask a backend to execute the operation on the current register state right away.

        Like :func:`Operation.apply`, but also stores the measurement result in the RegRefs.

        Keyword Args:
            shots (int): Number of independent evaluations to perform.
                Only applies to Measurements.
        """
        values = super().apply(reg, backend, **kwargs)
        # convert the returned values into an iterable with the measured modes indexed along
        # the first axis and shots along second axis (if larger than 1), so that we can assign
        # register values
        shots = kwargs.get("shots", 1)
        if self.ns == 1:
            values = [values]  # values is either a scalar, or has shape (shots,)
        else:
            if shots > 1:
                values = values.T  # shape of values would be (shots, num_meas,)

        # store the results in the register reference objects
        for v, r in zip(values, reg):
            r.val = v


class Decomposition(Operation):
    """Abstract base class for multimode matrix transformations.

    This class provides the base behaviour for decomposing various multimode operations
    into a sequence of gates and state preparations.

    The first parameter p[0] of the Decomposition is always a square matrix.
    """
    ns = None  # overridden by child classes in __init__

    def __init__(self, par, decomp=True):
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
            U = matmul(U2, U1).x
            # Note: above we strip the Parameter wrapper to make the following check
            # easier to perform. The constructor restores it.
            # Another option would be to add the required methods to Parameter class.
            # check if the matrices cancel
            if np.allclose(U, np.identity(len(U)), atol=_decomposition_merge_tol, rtol=0):
                return None

            return self.__class__(U)

        raise MergeFailure('Not the same decomposition type.')


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

    def merge(self, other):
        # check that other is an identical channel, and that there are
        # no extra dependencies <=> no RegRefTransform parameters
        if isinstance(other, self.__class__) \
                and len(self._extra_deps)+len(other._extra_deps) == 0:
            # check that all other parameters are identical
            if np.all(self.p[1:] != other.p[1:]):
                raise MergeFailure('Other parameters differ.')

            # determine the new loss parameter
            T = self.p[0] * other.p[0]

            # if one, replace with the identity
            if T == 1:
                return None

            if len(self.p) == 1:
                return self.__class__(T)

            return self.__class__(T, *self.p[1:])

        raise MergeFailure('Not the same operation family.')


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
        # NOTE deepcopy would make copies of RegRefs inside a possible
        # RegRefTransformation parameter, RegRefs must not be copied.
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
        z = self.p[0].evaluate()
        # if z represents a batch of parameters, then all of these
        # must be zero to skip calling backend
        if np.all(z == 0):
            # identity, no need to apply
            return
        if self.dagger:
            z = -z
        original_p = self.p  # store the original Parameters
        # evaluate the rest of the Parameters, restore the originals later
        self.p = [z] + [x.evaluate() for x in self.p[1:]]

        # convert RegRefs back to indices for the backend API
        temp = [rr.ind for rr in reg]
        # call the child class specialized _apply method
        self._apply(temp, backend, **kwargs)
        self.p = original_p  # restore original unevaluated Parameter instances

    def merge(self, other):
        if not self.__class__ == other.__class__:
            raise MergeFailure('Not the same gate family.')

        if len(self._extra_deps)+len(other._extra_deps) == 0:
            # no extra dependencies <=> no RegRefTransform parameters,
            # with which we cannot do arithmetic at the moment

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
                # HACK: some of the subclass constructors only take a single parameter,
                # some take two, none take three
                if len(self.p) == 1:
                    temp = self.__class__(p0)
                else:
                    temp = self.__class__(p0, *self.p[1:])
                # NOTE deepcopy would make copies of RegRefs inside a possible
                # RegRefTransformation parameter, RegRefs must not be copied.
                # OTOH copy results in temp having the same p list as self,
                # which we would modify below.
                #temp = copy.copy(self)
                #temp.p[0] = p0
                temp.dagger = self.dagger
                return temp

        else:
            # gates have RegRefTransform parameters:
            # without knowing anything more specific about the gates, we
            # can only merge them if they are each others' inverses
            if self.p == other.p and self.dagger != other.dagger:
                return None

        raise MergeFailure("Don't know how to merge these gates.")



# ====================================================================
# State preparation operations
# ====================================================================


class Vacuum(Preparation):
    """Prepare a mode in the :ref:`vacuum state <vacuum_state>`.

    Can be accessed via the shortcut variable ``Vac``.
    """

    def __init__(self):
        super().__init__([])

    def _apply(self, reg, backend, **kwargs):
        backend.prepare_vacuum_state(*reg)

    def __str__(self):
        # return the shorthand object when the
        # command is printed by the user
        return 'Vac'


class Coherent(Preparation):
    r"""Prepare a mode in a :ref:`coherent state <coherent_state>`.

    The gate is parameterized so that a user can specify a single complex number :math:`a=\alpha`
    or use the polar form :math:`a = r, p=\phi` and still get the same result.

    Args:
        a (complex): displacement parameter :math:`\alpha`
        p (float): phase angle :math:`\phi`
    """

    def __init__(self, a=0., p=0.):
        super().__init__([a, p])

    def _apply(self, reg, backend, **kwargs):
        z = self.p[0] * exp(1j * self.p[1])
        backend.prepare_coherent_state(z.x, *reg)


class Squeezed(Preparation):
    r"""Prepare a mode in a :ref:`squeezed vacuum state <squeezed_state>`.

    Args:
        r (float): squeezing magnitude
        p (float): squeezing angle :math:`\phi`
    """

    def __init__(self, r=0., p=0.):
        super().__init__([r, p])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.prepare_squeezed_state(p[0], p[1], *reg)


class DisplacedSqueezed(Preparation):
    r"""Prepare a mode in a :ref:`displaced squeezed state <displaced_squeezed_state>`.

    A displaced squeezed state is prepared by squeezing a vacuum state, and
    then applying a displacement operator.

    .. math::
       \ket{\alpha,z} = D(\alpha)\ket{0,z} = D(\alpha)S(z)\ket{0},

    where the squeezing parameter :math:`z=re^{i\phi}`.

    Args:
        alpha (complex): displacement parameter
        r (float): squeezing magnitude
        p (float): squeezing angle :math:`\phi`
    """

    def __init__(self, alpha=0., r=0., p=0.):
        super().__init__([alpha, r, p])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        # prepare the displaced squeezed state directly
        backend.prepare_displaced_squeezed_state(p[0], p[1], p[2], *reg)

    def _decompose(self, reg, **kwargs):
        # squeezed state preparation followed by a displacement gate
        return [
            Command(Squeezed(self.p[1], self.p[2]), reg),
            Command(Dgate(self.p[0]), reg)
        ]


class Fock(Preparation):
    r"""Prepare a mode in a :ref:`fock_basis` state.

    The prepared mode is traced out and replaced with the Fock state :math:`\ket{n}`.
    As a result the state of the other subsystems may have to be described using a density matrix.

    Args:
        n (int): Fock state to prepare
    """

    def __init__(self, n=0):
        super().__init__([n])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.prepare_fock_state(p[0], *reg)


class Catstate(Preparation):
    r"""Prepare a mode in a :ref:`cat state <cat_state>`.

    A cat state is the coherent superposition of two coherent states,

    .. math::
       \ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{i\phi} \ket{-\alpha}),

    where :math:`N = \sqrt{2 (1+\cos(\phi)e^{-2|\alpha|^2})}` is the normalization factor.

    Args:
        alpha (complex): displacement parameter
        p (float): parity, where :math:`\phi=p\pi`. ``p=0`` corresponds to an even
            cat state, and ``p=1`` an odd cat state.
    """

    def __init__(self, alpha=0, p=0):
        super().__init__([alpha, p])

    def _apply(self, reg, backend, **kwargs):
        alpha = self.p[0]
        phi = pi*self.p[1]
        D = backend.get_cutoff_dim()
        l = np.arange(D)[:, np.newaxis]

        # normalization constant
        temp = exp(-0.5 * abs(alpha)**2)
        N = temp / sqrt(2*(1 + cos(phi) * temp**4))

        # coherent states
        c1 = (alpha ** l) / sqrt(fac(l))
        c2 = ((-alpha) ** l) / sqrt(fac(l))
        # add them up with a relative phase
        ket = (c1 + exp(1j*phi) * c2) * N

        # in order to support broadcasting, the batch axis has been located at last axis, but backend expects it up as first axis
        ket = transpose(ket)

        # drop dummy batch axis if it is not necessary
        ket = squeeze(ket)

        backend.prepare_ket_state(ket.x, *reg)


class Ket(Preparation):
    r"""Prepare mode(s) using the given ket vector(s) in the Fock basis.

    The prepared modes are traced out and replaced with the given ket state
    (in the Fock basis). As a result the state of the other subsystems may have
    to be described using a density matrix.

    The provided kets must be each be of length ``cutoff_dim``, matching
    the cutoff dimension used in calls to :meth:`eng.run <~.Engine.run>`.

    Args:
        state (array or BaseFockState): state vector in the Fock basis.
            This can be provided as either:

            * a single ket vector, for single mode state preparation
            * a multimode ket, with one array dimension per mode
            * a :class:`BaseFockState` state object.
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
        p = _unwrap(self.p)
        backend.prepare_ket_state(p[0], reg)


class DensityMatrix(Preparation):
    r"""Prepare mode(s) using the given density matrix in the Fock basis.

    The prepared modes are traced out and replaced with the given state
    (in the Fock basis). As a result, the overall state of system
    will also have to be described using a density matrix.

    The provided density matrices must be of size ``[cutoff_dim, cutoff_dim]``,
    matching the cutoff dimension used in calls to :meth:`eng.run <~.Engine.run>`.

    Args:
        state (array or BaseFockState): density matrix in the Fock basis.
            This can be provided as either:

            * a single mode two-dimensional matrix :math:`\rho_{ij}`,
            * a multimode tensor :math:`\rho_{ij,kl,\dots,mn}`, with two indices per mode,
            * a :class:`BaseFockState` state object.
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
        p = _unwrap(self.p)
        backend.prepare_dm_state(p[0], reg)


class Thermal(Preparation):
    r"""Prepare a mode in a :ref:`thermal state <thermal_state>`.

    The requested mode is traced out and replaced with the thermal state :math:`\rho(\bar{n})`.
    As a result the state will be described using a density matrix.

    Args:
        n (float): mean thermal population of the mode
    """

    def __init__(self, n=0):
        super().__init__([n])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.prepare_thermal_state(p[0], *reg)


# ====================================================================
# Measurements
# ====================================================================


class MeasureFock(Measurement):
    """:ref:`photon_counting`: measures a set of modes in the Fock basis.

    Also accessible via the shortcut variable ``Measure``.

    After measurement, the modes are reset to the vacuum state.
    """
    ns = None

    def __init__(self, select=None):
        if select is not None and not isinstance(select, Sequence):
            select = [select]
        super().__init__([], select)

    def _apply(self, reg, backend, shots=1, **kwargs):
        return backend.measure_fock(reg, shots=shots, select=self.select, **kwargs)


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
    r"""Performs a :ref:`homodyne measurement <homodyne>`, measures one quadrature of a mode.

    * Position basis measurement: :math:`\phi = 0`
      (also accessible via the shortcut variable ``MeasureX``).

    * Momentum basis measurement: :math:`\phi = \pi/2`.
      (also accessible via the shortcut variable ``MeasureP``)

    The measured mode is reset to the vacuum state.

    Args:
        phi (float): measurement angle :math:`\phi`
        select (None, float): (Optional) desired values of measurement result.
            Allows the post-selection of specific measurement results instead of randomly sampling.
    """
    ns = 1

    def __init__(self, phi, select=None):
        super().__init__([phi], select)

    def _apply(self, reg, backend, shots=1, **kwargs):
        p = _unwrap(self.p)
        s = sqrt(sf.hbar / 2)  # scaling factor, since the backend API call is hbar-independent
        select = self.select
        if select is not None:
            select = select / s

        return s * backend.measure_homodyne(p[0], *reg, shots=shots, select=select, **kwargs)

    def __str__(self):
        if self.select is None:
            if self.p[0] == 0:
                return 'MeasureX'
            if self.p[0] == pi/2:
                return 'MeasureP'
        return super().__str__()


class MeasureHeterodyne(Measurement):
    r"""Performs a :ref:`heterodyne measurement <heterodyne>` on a mode.

    Also accessible via the shortcut variable ``MeasureHD``.

    Samples the joint Husimi distribution :math:`Q(\vec{\alpha}) = \frac{1}{\pi}\bra{\vec{\alpha}}\rho\ket{\vec{\alpha}}`.
    The measured mode is reset to the vacuum state.

    Args:
        select (None, complex): (Optional) desired values of measurement result.
            Allows the post-selection of specific measurement results instead of randomly sampling.
    """
    ns = 1

    def __init__(self, select=None):
        super().__init__([], select)

    def _apply(self, reg, backend, shots=1, **kwargs):
        return backend.measure_heterodyne(*reg, shots=shots, select=self.select, **kwargs)

    def __str__(self):
        if self.select is None:
            return 'MeasureHD'
        return 'MeasureHeterodyne(select={})'.format(self.select)


# ====================================================================
# Channels
# ====================================================================


class LossChannel(Channel):
    r"""Perform a :ref:`loss channel <loss>` operation on the specified mode.

    This channel couples mode :math:`\a` to another bosonic mode :math:`\hat{b}`
    prepared in the vacuum state using the following transformation:

    .. math::
       \a \mapsto \sqrt{T} a+\sqrt{1-T} \hat{b}

    Args:
        T (float): the loss parameter :math:`0\leq T\leq 1`.
    """

    def __init__(self, T):
        super().__init__([T])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.loss(p[0], *reg)


class ThermalLossChannel(Channel):
    r"""Perform a :ref:`thermal loss channel <thermal_loss>` operation on the specified mode.

    This channel couples mode :math:`\a` to another bosonic mode :math:`\hat{b}`
    prepared in a thermal state with mean photon number :math:`\bar{n}`,
    using the following transformation:

    .. math::
       \a \mapsto \sqrt{T} a+\sqrt{1-T} \hat{b}

    Args:
        T (float): the loss parameter :math:`0\leq T\leq 1`.
        nbar (float): mean photon number of the environment thermal state
    """

    def __init__(self, T, nbar):
        super().__init__([T, nbar])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.thermal_loss(p[0], p[1], *reg)


# ====================================================================
# Unitary gates
# ====================================================================


class Dgate(Gate):
    r"""Phase space :ref:`displacement <displacement>` gate.

    .. math::
       D(\alpha) = \exp(\alpha a^\dagger -\alpha^* a) = \exp\left(-i\sqrt{2}(\re(\alpha) \hat{p} -\im(\alpha) \hat{x})/\sqrt{\hbar}\right)

    where :math:`\alpha = r e^{i\phi}` has magnitude :math:`r\geq 0` and phase :math:`\phi`.

    The gate is parameterized so that a user can specify a single complex number :math:`a=\alpha`
    or use the polar form :math:`a = r, \phi` and still get the same result.

    Args:
        a (complex): displacement parameter :math:`\alpha`
        phi (float): extra (optional) phase angle :math:`\phi`
    """

    def __init__(self, a, phi=0.):
        super().__init__([a, phi])

    def _apply(self, reg, backend, **kwargs):
        z = self.p[0] * exp(1j * self.p[1])
        backend.displacement(z.x, *reg)


class Xgate(Gate):
    r"""Position :ref:`displacement <displacement>` gate.

    .. math::
       X(x) = e^{-i x \hat{p}/\hbar}

    Args:
        x (float): position displacement
    """

    def __init__(self, x):
        super().__init__([x])

    def _apply(self, reg, backend, **kwargs):
        z = self.p[0] / sqrt(2 * sf.hbar)
        backend.displacement(z.x, *reg)


class Zgate(Gate):
    r"""Momentum :ref:`displacement <displacement>` gate.

    .. math::
       Z(p) = e^{i p \hat{x}/\hbar}

    Args:
        p (float): momentum displacement
    """

    def __init__(self, p):
        super().__init__([p])

    def _apply(self, reg, backend, **kwargs):
        z = self.p[0] * 1j/sqrt(2 * sf.hbar)
        backend.displacement(z.x, *reg)


class Sgate(Gate):
    r"""Phase space :ref:`squeezing <squeezing>` gate.

    .. math::
       S(z) = \exp\left(\frac{1}{2}(z^* a^2 -z {a^\dagger}^2)\right)

    where :math:`z = r e^{i\phi}`.

    Args:
        r (float): squeezing amount
        phi (float): squeezing phase angle :math:`\phi`
    """

    def __init__(self, r, phi=0.):
        super().__init__([r, phi])

    def _apply(self, reg, backend, **kwargs):
        z = self.p[0] * exp(1j * self.p[1])
        backend.squeeze(z.x, *reg)


class Pgate(Gate):
    r""":ref:`Quadratic phase <quadratic>` gate.

    .. math::
       P(s) = e^{i \frac{s}{2} \hat{x}^2/\hbar}

    Args:
        s (float): parameter
    """

    def __init__(self, s):
        super().__init__([s])

    def _decompose(self, reg, **kwargs):
        # into a squeeze and a rotation
        temp = self.p[0] / 2
        r = arccosh(sqrt(1+temp**2))
        theta = arctan(temp)
        phi = -pi/2 * sign(temp) - theta
        return [
            Command(Sgate(r, phi), reg),
            Command(Rgate(theta), reg)
        ]


class Vgate(Gate):
    r""":ref:`Cubic phase <cubic>` gate.

    .. math::
       V(\gamma) = e^{i \frac{\gamma}{3 \hbar} \hat{x}^3}

    .. warning:: The cubic phase gate has lower accuracy than the Kerr gate at the same cutoff dimension.

    Args:
        gamma (float): parameter
    """

    def __init__(self, gamma):
        super().__init__([gamma])

    def _apply(self, reg, backend, **kwargs):
        gamma_prime = self.p[0] * sqrt(sf.hbar / 2)
        # the backend API call cubic_phase is hbar-independent
        backend.cubic_phase(gamma_prime.x, *reg)


class Kgate(Gate):
    r""":ref:`Kerr <kerr>` gate.

    .. math::
       K(\kappa) = e^{i \kappa \hat{n}^2}

    Args:
        kappa (float): parameter
    """

    def __init__(self, kappa):
        super().__init__([kappa])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.kerr_interaction(p[0], *reg)


class Rgate(Gate):
    r""":ref:`Rotation <rotation>` gate.

    .. math::
       R(\theta) = e^{i \theta a^\dagger a}

    Args:
        theta (float): rotation angle :math:`\theta`.
    """

    def __init__(self, theta):
        super().__init__([theta])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.rotation(p[0], *reg)


class BSgate(Gate):
    r"""BSgate(theta=pi/4, phi=0.)
    :ref:`Beamsplitter <beamsplitter>` gate.

    .. math::
       B(\theta,\phi) = \exp\left(\theta (e^{i \phi} a_1 a_2^\dagger -e^{-i \phi} a_1^\dagger a_2) \right)

    Args:
        theta (float): Transmittivity angle :math:`\theta`. The transmission amplitude of the beamsplitter is :math:`t = \cos(\theta)`.
            The value :math:`\theta=\pi/4` gives the 50-50 beamsplitter (default).
        phi (float): Phase angle :math:`\phi`. The reflection amplitude of the beamsplitter is :math:`r = e^{i\phi}\sin(\theta)`.
            The value :math:`\phi = \pi/2` gives the symmetric beamsplitter.
    """
    ns = 2

    def __init__(self, theta=pi/4, phi=0.):
        # default: 50% beamsplitter
        super().__init__([theta, phi])

    def _apply(self, reg, backend, **kwargs):
        t = cos(self.p[0])
        r = sin(self.p[0]) * exp(1j * self.p[1])
        backend.beamsplitter(t.x, r.x, *reg)


class MZgate(Gate):
    r"""Mach-Zehnder interferometer.

    .. math::

        \mathrm{MZ}(\phi_{ex}, \phi_{in}) = BS\left(\frac{\pi}{4}, \frac{\pi}{2}\right)
            (R(\phi_{in})\otimes I) BS\left(\frac{\pi}{4}, \frac{\pi}{2}\right)
            (R(\phi_{ex})\otimes I)

    Args:
        phi_ex (float): external phase
        phi_in (float): internal phase
    """
    ns = 2

    def __init__(self, phi_ex, phi_in):
        super().__init__([phi_ex, phi_in])

    def _decompose(self, reg, **kwargs):
        # into local phase shifts and two 50-50 beamsplitters
        return [
            Command(Rgate(self.p[0].x), reg[0]),
            Command(BSgate(np.pi/4, np.pi/2), reg),
            Command(Rgate(self.p[1].x), reg[0]),
            Command(BSgate(np.pi/4, np.pi/2), reg)
        ]


class S2gate(Gate):
    r""":ref:`Two-mode squeezing <two_mode_squeezing>` gate.

    .. math::
       S_2(z) = \exp\left(z^* ab -z a^\dagger b^\dagger \right) = \exp\left(r (e^{-i\phi} ab -e^{i\phi} a^\dagger b^\dagger \right)

    where :math:`z = r e^{i\phi}`.

    Args:
        r (float): squeezing amount
        phi (float): squeezing phase angle :math:`\phi`
    """
    ns = 2

    def __init__(self, r, phi=0.):
        super().__init__([r, phi])

    def _decompose(self, reg, **kwargs):
        # two opposite squeezers sandwiched between 50% beamsplitters
        S = Sgate(self.p[0], self.p[1])
        BS = BSgate(pi/4, 0)
        return [
            Command(BS, reg),
            Command(S, reg[0]),
            Command(S.H, reg[1]),
            Command(BS.H, reg)
        ]


class CXgate(Gate):
    r""":ref:`Controlled addition or sum <CX>` gate in the position basis.

    .. math::
       \text{CX}(s) = \int dx \ket{x}\bra{x} \otimes D\left({\frac{1}{\sqrt{2\hbar}}}s x\right) = e^{-i s \: \hat{x} \otimes \hat{p}/\hbar}

    In the position basis it maps
    :math:`\ket{x_1, x_2} \mapsto \ket{x_1, s x_1 +x_2}`.

    Args:
        s (float): addition multiplier
    """
    ns = 2

    def __init__(self, s=1):
        super().__init__([s])

    def _decompose(self, reg, **kwargs):
        s = self.p[0]
        r = arcsinh(-s/2)
        theta = 0.5*arctan2(-1.0/cosh(r), -tanh(r))

        BS1 = BSgate(theta, 0)
        BS2 = BSgate(theta+pi/2, 0)
        return [
            Command(BS1, reg),
            Command(Sgate(r, 0), reg[0]),
            Command(Sgate(-r, 0), reg[1]),
            Command(BS2, reg)
        ]


class CZgate(Gate):
    r""":ref:`Controlled phase <CZ>` gate in the position basis.

    .. math::
       \text{CZ}(s) =  \iint dx dy \: e^{i sxy/\hbar} \ket{x,y}\bra{x,y} = e^{i s \: \hat{x} \otimes \hat{x}/\hbar}

    In the position basis it maps
    :math:`\ket{x_1, x_2} \mapsto e^{i s x_1 x_2/\hbar} \ket{x_1, x_2}`.

    Args:
        s (float): phase shift multiplier
    """
    ns = 2

    def __init__(self, s=1):
        super().__init__([s])

    def _decompose(self, reg, **kwargs):
        # phase-rotated CZ
        CX = CXgate(self.p[0])
        return [
            Command(Rgate(-pi/2), reg[1]),
            Command(CX, reg),
            Command(Rgate(pi/2), reg[1])
        ]


class CKgate(Gate):
    r""":ref:`Cross-Kerr <cross_kerr>` gate.

    .. math::
       CK(\kappa) = e^{i \kappa \hat{n}_1\hat{n}_2}

    Args:
        kappa (float): parameter
    """
    ns = 2

    def __init__(self, kappa):
        super().__init__([kappa])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.cross_kerr_interaction(p[0], *reg)


class Fouriergate(Gate):
    r""":ref:`Fourier <fourier>` gate.

    Also accessible via the shortcut variable ``Fourier``.

    A special case of the :class:`phase space rotation gate <Rgate>`, where :math:`\theta=\pi/2`.

    .. math::
       F = R(\pi/2) = e^{i (\pi/2) a^\dagger a}
    """

    def __init__(self):
        super().__init__([pi/2])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.rotation(p[0], *reg)

    def __str__(self):
        """String representation for the gate."""
        temp = 'Fourier'
        if self.dagger:
            temp += '.H'
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
        return 'Del'


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
        raise RuntimeError('New() can only be called inside a Program context.')
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
        return 'New({})'.format(self.n)


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
        return super().__str__() + '({})'.format(str(self.op))

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

    This operation uses either the :ref:`rectangular decomposition <rectangular>`
    or triangular decomposition to decompose
    a linear interferometer into a sequence of beamsplitters and
    rotation gates.

    By specifying the keyword argument ``mesh``, the scheme used to implement the interferometer
    may be adjusted:

    * ``mesh='rectangular'`` (default): uses the scheme described in
      :cite:`clements2016`, resulting in a *rectangular* array of
      :math:`M(M-1)/2` beamsplitters:

      .. figure:: ../_static/clements.png
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

      .. figure:: ../_static/reck.png
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
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, U, mesh="rectangular", drop_identity=True, tol=1e-6):
        super().__init__([U])
        self.ns = U.shape[0]
        self.mesh = mesh
        self.tol = tol
        self.drop_identity = drop_identity

        allowed_meshes = {"rectangular", "rectangular_phase_end", "rectangular_symmetric", "triangular"}

        if mesh not in allowed_meshes:
            raise ValueError("Unknown mesh '{}'. Mesh must be one of {}".format(mesh, allowed_meshes))

        self.identity = np.allclose(U, np.identity(len(U)), atol=_decomposition_merge_tol, rtol=0)

    def _decompose(self, reg, **kwargs):
        mesh = kwargs.get("mesh", self.mesh)
        tol = kwargs.get("tol", self.tol)
        drop_identity = kwargs.get("drop_identity", self.drop_identity)

        cmds = []

        if not self.identity or not drop_identity:
            decomp_fn = getattr(dec, mesh)
            BS1, R, BS2 = decomp_fn(self.p[0].x, tol=tol)

            for n, m, theta, phi, _ in BS1:
                theta = theta if np.abs(theta) >= _decomposition_tol else 0
                phi = phi if np.abs(phi) >= _decomposition_tol else 0

                if "symmetric" in mesh:
                    # Mach-Zehnder interferometers
                    cmds.append(Command(MZgate(np.mod(phi, 2*np.pi), np.mod(theta, 2*np.pi)), (reg[n], reg[m])))

                else:
                    # Clements style beamsplitters
                    if not (drop_identity and phi == 0):
                        cmds.append(Command(Rgate(phi), reg[n]))

                    if not (drop_identity and theta == 0):
                        cmds.append(Command(BSgate(theta, 0), (reg[n], reg[m])))

            for n, expphi in enumerate(R):
                # local phase shifts
                q = log(expphi).imag if np.abs(expphi - 1) >= _decomposition_tol else 0
                if not (drop_identity and q == 0):
                    cmds.append(Command(Rgate(q), reg[n]))

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
                A, mean_photon_per_mode=mean_photon_per_mode, make_traceless=make_traceless, atol=tol, rtol=0
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

    where :math:`B` is a :math:N/2\times N/2` matrix representing the (weighted)
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
        self.mean_photon_per_mode = mean_photon_per_mode
        self.tol = tol
        self.identity = np.all(np.abs(A - np.identity(len(A))) < _decomposition_merge_tol)
        self.drop_identity = drop_identity

        if edges:
            self.ns = 2*A.shape[0]
            B = A
        else:
            self.ns = A.shape[0]

            # check if A is a bipartite graph
            N = A.shape[0]//2
            A00 = A[:N, :N]
            A11 = A[N:, N:]

            diag_zeros = np.allclose(A00, np.zeros_like(A00), atol=tol, rtol=0) \
                and np.allclose(A11, np.zeros_like(A11), atol=tol, rtol=0)

            if (not diag_zeros) or (not np.allclose(A, A.T, atol=tol, rtol=0)):
                raise ValueError("Adjacency matrix {} does not represent a bipartite graph".format(A))

            B = A[:N, N:]

        super().__init__([B])

    def _decompose(self, reg, **kwargs):
        mean_photon_per_mode = kwargs.get("mean_photon_per_mode", self.mean_photon_per_mode)
        tol = kwargs.get("tol", self.tol)
        mesh = kwargs.get("mesh", "rectangular")
        drop_identity = kwargs.get("drop_identity", self.drop_identity)

        cmds = []

        B = self.p[0].x
        N = len(B)

        sq, U, V = dec.bipartite_graph_embed(B, mean_photon_per_mode=mean_photon_per_mode, atol=tol, rtol=0)

        if not self.identity or not drop_identity:
            for m, s in enumerate(sq):
                s = s if np.abs(s) >= _decomposition_tol else 0

                if not (drop_identity and s == 0):
                    cmds.append(Command(S2gate(-s), (reg[m], reg[m+N])))

            for X, _reg in ((U, reg[:N]), (V, reg[N:])):

                if np.allclose(X, np.identity(len(X)), atol=_decomposition_tol, rtol=0):
                    X = np.identity(len(X))

                if not (drop_identity and np.all(X == np.identity(len(X)))):
                    cmds.append(Command(Interferometer(X, mesh=mesh, drop_identity=drop_identity, tol=tol), _reg))

        return cmds


class GaussianTransform(Decomposition):
    r"""Apply a Gaussian symplectic transformation to the specified qumodes.

    This operation uses the :ref:`Bloch-Messiah decomposition <bloch_messiah>`
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
    """
    def __init__(self, S, vacuum=False, tol=1e-10):
        super().__init__([S])
        self.ns = S.shape[0] // 2
        self.vacuum = vacuum  #: bool: if True, ignore the first unitary matrix when applying the gate
        N = self.ns  # shorthand

        # check if input symplectic is passive (orthogonal)
        diffn = np.linalg.norm(S @ S.T - np.identity(2*N))
        self.active = (np.abs(diffn) > _decomposition_tol)  #: bool: S is an active symplectic transformation

        if not self.active:
            # The transformation is passive, do Clements
            X1 = S[:N, :N]
            P1 = S[N:, :N]
            self.U1 = X1+1j*P1
        else:
            # transformation is active, do Bloch-Messiah
            O1, smat, O2 = dec.bloch_messiah(S, tol=tol)
            X1 = O1[:N, :N]
            P1 = O1[N:, :N]
            X2 = O2[:N, :N]
            P2 = O2[N:, :N]

            self.U1 = X1+1j*P1  #: array[complex]: unitary matrix corresponding to O_1
            self.U2 = X2+1j*P2  #: array[complex]: unitary matrix corresponding to O_2
            self.Sq = np.diagonal(smat)[:N]  #: array[complex]: diagonal vector of the squeezing matrix R

    def _decompose(self, reg, **kwargs):
        cmds = []
        mesh = kwargs.get("mesh", "rectangular")

        if self.active:
            if not self.vacuum:
                cmds = [Command(Interferometer(self.U2), reg)]

            for n, expr in enumerate(self.Sq):
                if np.abs(expr - 1) >= _decomposition_tol:
                    r = abs(log(expr))
                    phi = np.angle(log(expr))
                    cmds.append(Command(Sgate(-r, phi), reg[n]))

            cmds.append(Command(Interferometer(self.U1, mesh=mesh), reg))
        else:
            if not self.vacuum:
                cmds = [Command(Interferometer(self.U1, mesh=mesh), reg)]

        return cmds


class Gaussian(Preparation, Decomposition):
    r"""Prepare the specified modes in a Gaussian state.

    This operation uses the :ref:`Williamson decomposition <williamson>` to prepare
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

    Args:
        V (array[float]): an :math:`2N\times 2N` (real and positive definite) covariance matrix
        r (array[float] or None): Length :math:`2N` vector of means, of the
            form :math:`(\x_0,\dots,\x_{N-1},\p_0,\dots,\p_{N-1})`.
            If None, it is assumed that :math:`r=0`.
        decomp (bool): Should the operation be decomposed into a sequence of elementary gates?
            If False, the state preparation is performed directly via the backend API.
        tol (float): the tolerance used when checking if the matrix is symmetric: :math:`|V-V^T| \leq` tol
    """
    # pylint: disable=too-many-instance-attributes
    ns = None

    def __init__(self, V, r=None, decomp=True, tol=1e-6):
        # internally we eliminate hbar from the covariance matrix V (or equivalently set hbar=2), but not from the means vector r
        V = V / (sf.hbar / 2)
        self.ns = V.shape[0] // 2

        if r is None:
            r = np.zeros(2*self.ns)
        r = np.asarray(r)

        if len(r) != V.shape[0]:
            raise ValueError('Vector of means must have the same length as the covariance matrix.')

        super().__init__([V, r], decomp=decomp)  # V is hbar-independent, r is not

        self.x_disp = r[:self.ns]
        self.p_disp = r[self.ns:]

        # needed only if decomposed
        th, self.S = dec.williamson(V, tol=tol)
        self.pure = np.abs(np.linalg.det(V) - 1.0) < tol
        self.nbar = 0.5 * (np.diag(th)[:self.ns] - 1.0)

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        s = sqrt(sf.hbar / 2)  # scaling factor, since the backend API call is hbar-independent
        backend.prepare_gaussian_state(p[1]/s, p[0], reg)

    def _decompose(self, reg, **kwargs):
        # pylint: disable=too-many-branches
        cmds = []

        V = self.p[0].x
        D = np.diag(V)
        is_diag = np.all(V == np.diag(D))

        BD = changebasis(self.ns) @ V @ changebasis(self.ns).T
        BD_modes = [BD[i*2:(i+1)*2, i*2:(i+1)*2]
                    for i in range(BD.shape[0]//2)]
        is_block_diag = (not is_diag) and np.all(BD == block_diag(*BD_modes))

        if self.pure and is_diag:
            # covariance matrix consists of x/p quadrature squeezed state
            for n, expr in enumerate(D[:self.ns]):
                if np.abs(expr - 1) >= _decomposition_tol:
                    r = abs(log(expr)/2)
                    cmds.append(Command(Squeezed(r, 0), reg[n]))
                else:
                    cmds.append(Command(Vac, reg[n]))

        elif self.pure and is_block_diag:
            # covariance matrix consists of rotated squeezed states
            for n, v in enumerate(BD_modes):
                if not np.all(v - np.identity(2) < _decomposition_tol):
                    r = np.abs(arccosh(np.sum(np.diag(v)) / 2)) / 2
                    phi = arctan(2 * v[0, 1] / np.sum(np.diag(v) * [1, -1]))
                    cmds.append(Command(Squeezed(r, phi), reg[n]))
                else:
                    cmds.append(Command(Vac, reg[n]))

        elif not self.pure and is_diag and np.all(D[:self.ns] == D[self.ns:]):
            # covariance matrix consists of thermal states
            for n, nbar in enumerate(0.5 * (D[:self.ns] - 1.0)):
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

        cmds += [Command(Xgate(u), reg[n])
                 for n, u in enumerate(self.x_disp) if u != 0]
        cmds += [Command(Zgate(u), reg[n])
                 for n, u in enumerate(self.p_disp) if u != 0]

        return cmds


#=======================================================================
# Shorthands, e.g. pre-constructed singleton-like objects

Del = _Delete()
Vac = Vacuum()
MeasureX = MeasureHomodyne(0)
MeasureP = MeasureHomodyne(pi/2)
MeasureHD = MeasureHeterodyne()

Fourier = Fouriergate()

RR = RegRefTransform

shorthands = ['New', 'Del', 'Vac', 'Measure', 'MeasureX', 'MeasureP', 'MeasureHD', 'Fourier', 'RR',
              'All']

#=======================================================================
# here we list different classes of operations for unit testing purposes

zero_args_gates = (Fouriergate,)
one_args_gates = (Xgate, Zgate, Rgate, Pgate, Vgate,
                  Kgate, CXgate, CZgate, CKgate)
two_args_gates = (Dgate, Sgate, BSgate, MZgate, S2gate)
gates = zero_args_gates + one_args_gates + two_args_gates

channels = (LossChannel, ThermalLossChannel)

simple_state_preparations = (Vacuum, Coherent, Squeezed, DisplacedSqueezed, Fock, Catstate, Thermal)  # have __init__ methods with default arguments
state_preparations = simple_state_preparations + (Ket, DensityMatrix)

measurements = (MeasureFock, MeasureHomodyne, MeasureHeterodyne, MeasureThreshold)

decompositions = (Interferometer, BipartiteGraphEmbed, GraphEmbed, GaussianTransform, Gaussian)

#=======================================================================
# exported symbols

__all__ = [cls.__name__ for cls in gates + channels + state_preparations + measurements + decompositions] + shorthands


#=======================================================================
# Module wrapper for deprecating shorthands


class Wrapper(types.ModuleType):
    """Wrapper class to modify the module level
    attribute lookup.

    This allows module attributes to be deprecated.

    Current list of deprecated attributes:

    * ``Measure``: instead use ``MeasureFock``

    .. note::

        With Python 3.7+, there is new support for a module-level
        ``__getattr__`` function, which should enable this functionality
        without needing to modify ``sys.modules``.
    """
    deprecation_map = {"Measure": "MeasureFock"}

    def __init__(self, mod):
        self.mod = mod
        self.__dict__.update(mod.__dict__)
        super().__init__("strawberryfields.ops", doc=sys.modules[__name__].__doc__)

    def __getattr__(self, name):
        if name in self.deprecation_map:
            new_name = self.deprecation_map[name]

            warnings.warn("The shorthand '{}' has been deprecated, "
                          "please use '{}()' instead.".format(name, new_name))

            return getattr(self.mod, new_name)()

        return getattr(self.mod, name)

    def __dir__(self):
        return __all__


sys.modules[__name__] = Wrapper(sys.modules[__name__])

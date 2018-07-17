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
.. _gates:

Quantum operations
===================

**Module name:** :mod:`strawberryfields.ops`

.. currentmodule:: strawberryfields.ops

.. note::

  In the :mod:`strawberryfields.ops` API we use the convention :math:`\hbar=2` by default, however
  this can be changed on engine initialisation.

  See :ref:`conventions` for more details.

This module defines and implements the Python-embedded quantum programming language
for continuous-variable (CV) quantum systems.
The syntax is modeled after ProjectQ :cite:`projectq2016`.

Quantum operations (state preparation, unitary gates, measurements, channels) act on
register objects using the following syntax::

  with eng:
    G(params) | q
    F(params) | (q[1], q[6], q[2])

Here :samp:`eng` is an instance of :class:`strawberryfields.engine.Engine` and defines the context in which
the commands are executed.
Within each command, the part on the left is an :class:`Operation` instance,
quite typically a constructor call for the requested operation class with the relevant parameters.
The vertical bar calls the :func:`__or__` method of the :class:`Operation` object, with the part on the right as the parameter.
The part on the right is a single :class:`strawberryfields.engine.RegRef` object or, for multi-mode gates, a sequence of them.
It is of course also possible to construct gates separately and reuse them several times::

  R = Rgate(s)
  with eng:
    R   | q
    Xgate(t) | q
    R.H | q


There are six kinds of :class:`Operation` objects:

* :class:`Preparation` operations only manipulate the register state::

    eng, q = sf.Engine(3)
    with eng:
      Vac | q[0]
      All(Coherent(0.4, 0.2)) | (q[1], q[2])

* Transformations such as :class:`Gate` and :class:`Channel` operations only manipulate the register state::

    eng, q = sf.Engine(2)
    with eng:
      Dgate(0.3)   | q[0]
      BSgate(-0.5) | q[0:2]

* :class:`Measurement` operations manipulate the register state and produce classical information.
  The information is directly available only after the simulation has been run up to the point of measurement::

    eng, q = sf.Engine(2)
    with eng:
      Measure       | q[0]
      eng.run()
      Dgate(q[0].val) | q[1]

  Alternatively one may use a symbolic reference to the register containing the measurement result
  by supplying registers as the argument to an :class:`Operation`, in which case the measurement may be deferred,
  i.e., we may symbolically use the measurement result before it exists::

    eng, (alice, bob) = sf.Engine(2)
    with eng:
      Measure   | alice
      Dgate(alice) | bob
    eng.run()

  One may also include an arbitrary post-processing function for the measurement result, to be applied
  before using it as the argument to another :class:`Operation`. The :func:`~.convert` decorator can be used in Python
  to convert a user-defined function into a post-processing function recognized by the engine::

    @convert
    def square(q):
      return q ** 2

    eng, q = sf.Engine(2)
    with eng:
      Measure           | q[0]
      Dgate(square(q[0])) | q[1]
    eng.run()

  Finally, the lower-level :class:`strawberryfields.engine.RegRefTransform` (RR) and
  an optional lambda function can be used to achieve the same functionality::

    eng, q = sf.Engine(3)
    with eng:
      Measure       | q[0]
      Dgate(RR(q[0])) | q[1]
      Dgate(RR(q[0], lambda q: q ** 2)) | q[2]
    eng.run()


* Meta-operations such as :class:`Delete` and :class:`New_modes` Operations delete
  and create modes during program execution.
  In practice the user only deals with the pre-constructed
  instances :py:data:`Del` and :py:data:`New`::

    eng, (alice,) = sf.Engine(1)
    with eng:
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
   MeasureHomodyne
   MeasureHeterodyne


Channels
-----------

.. autosummary::
    LossChannel


Decompositions
--------------

.. autosummary::
    Interferometer
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
   S2gate
   CXgate
   CZgate
   CKgate

Meta-operations
---------------

.. autosummary::
   All
   New_modes
   Delete


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
``New``                  :class:`~.New_modes`
``Del``                  :class:`~.Delete`
``Vac``                  :class:`~.Vacuum`
``Fourier``              :class:`~.Fouriergate`
``Measure``              :class:`~.MeasureFock`
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
import warnings

import numpy as np
from numpy import pi

from scipy.linalg import block_diag
from scipy.special import factorial as fac

from .backends.states import BaseFockState, BaseGaussianState
from .backends.shared_ops import changebasis
from .engine import Engine as _Engine, Command, RegRefTransform, MergeFailure
from .parameters import (Parameter, _unwrap, matmul, sign, abs, exp, log, sqrt,
                         sin, cos, cosh, tanh, arcsinh, arccosh, arctan, arctan2,
                         transpose, squeeze)
from .decompositions import clements, bloch_messiah, williamson

# pylint: disable=abstract-method
# pylint: disable=protected-access
# pylint: disable=too-many-arguments


# numerical tolerances
_decomposition_merge_tol = 1e-13


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
    the :class:`Operation` depends on. In the quantum diagram notation it
    corresponds to the vertical double lines of classical information
    entering the :class:`Operation` that originate in a measurement of a subsystem.

    This abstract base class may be initialised with parameters; see the
    :class:`~strawberryfields.parameters.Parameter` class for more details.

    Args:
        par (Sequence[...]): parameters. len(par) >= 1. Alternatively,
            set to [] or None to allow initialisation with no parameters.
    """
    # default: one-subsystem operation
    #: int: number of subsystems the operation acts on,
    # or None if any number of subsystems > 0 is okay
    ns = 1

    def __init__(self, par):
        #: set[RegRef]: extra dependencies due to deferred measurements, used during optimization
        self._extra_deps = set()
        #: list[Parameter]
        self.p = []

        if par:
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
        if self.p is None:
            return self.__class__.__name__

        # class name and parameter values
        temp = [str(i) for i in self.p]
        return self.__class__.__name__+'('+', '.join(temp)+')'

    @property
    def extra_deps(self):
        """Extra dependencies due to parameters that depend on deferred measurements.

        Returns:
            set[RegRef]: dependencies
        """
        return self._extra_deps

    def __or__(self, reg):
        """Apply the operation to a part of a quantum register.

        Dispatches the Operation to the :class:`~strawberryfields.engine.Engine`
        command queue for later execution.

        Args:
            reg (RegRef, Sequence[RegRef]): subsystem(s) the operation is acting on

        Returns:
            list[RegRef]: subsystem list as RegRefs
        """
        # into a list of subsystems
        reg = _seq_to_list(reg)
        if (not reg) or (self.ns != None and self.ns != len(reg)):
            raise ValueError("Wrong number of subsystems.")
        # send it to the engine
        reg = _Engine._current_context.append(self, reg)
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
            ~strawberryfields.engine.MergeFailure: if the two
                operations cannot be merged
        """
        # todo: Using the return value None to denote the identity is a
        # bit dangerous, since a function with no explicit return statement
        # also returns None, which can lead to puzzling bugs. Maybe return
        # a special singleton Identity object instead?
        raise NotImplementedError

    def decompose(self, reg):
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
        raise NotImplementedError('No decomposition available: {}'.format(self))

    def _apply(self, reg, backend, **kwargs):
        """Internal apply method. Uses numeric subsystem referencing.

        Args:
            reg (Sequence[int]): subsystem indices the operation is
                acting on (this is how the backend API wants them)
            backend (BaseBackend): backend to execute the operation
        """
        raise NotImplementedError('Missing direct implementation: {}'.format(self))

    def apply(self, reg, backend, hbar, **kwargs):
        """Ask a backend to execute the operation on the current register state right away.

        Takes care of parameter evaluations and any pending formal
        transformations (like dagger) and then calls _apply.

        Args:
            reg (Sequence[RegRef]): subsystem(s) the operation is acting on
            backend (BaseBackend): backend to execute the operation

        Keyword Args:
            eval_params (bool): set this to False to explicitly turn off the
                evaluation of parameters in the Operation.apply method. This is
                useful if the parameters are pre-evaluated prior to calling this method.

        Returns:
            : The result of self._apply
        """
        eval_params = kwargs.get('eval_params', True)
        original_p = self.p  # store the original parameters

        if eval_params and original_p:
            # NOTE: We cannot just replace all RegRefTransform parameters with their
            # numerical values here. If we re-initialize a measured mode and
            # re-measure it, the RegRefTransform value should change accordingly
            # when it is used again after the new measurement.

            # Evaluate the Parameters, restore the originals later:
            self.p = [x.evaluate() for x in self.p]

        # convert RegRefs back to indices for the backend API
        temp = [rr.ind for rr in reg]
        self.hbar = hbar  # pylint: disable=attribute-defined-outside-init
        # call the child class specialized _apply method
        result = self._apply(temp, backend, **kwargs)

        if eval_params and original_p:
            # restore original unevaluated Parameter instances
            self.p = original_p

        return result


#====================================================================
# Derived operation classes
#====================================================================


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
        else:
            raise MergeFailure('For now, Preparations cannot be merged with anything else.')


class Measurement(Operation):
    """Abstract base class for projective subsystem measurements.

    The measurement is deferred: its result is available only
    after the backend has executed it. The value of the measurement can
    be accessed in the program through the symbolic subsystem reference
    to the measured subsystem.

    When the measurement happens, the state of the circuit is updated
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
        self.select = select  #: None, Sequence[Number]: postselection values, one for each measured subsystem

    def __str__(self):
        # class name, parameter values, and possibly the select parameter
        temp = super().__str__()
        if self.select is not None:
            temp = temp[:-1] +', select={})'.format(self.select)
        return temp

    def merge(self, other):
        raise MergeFailure('For now, measurements cannot be merged with anything else.')

    def apply(self, reg, backend, hbar, **kwargs):
        """Ask a backend to execute the operation on the current register state right away.

        Like :func:`Operation.apply`, but also stores the measurement result in the RegRefs.
        """
        values = super().apply(reg, backend, hbar, **kwargs)
        # measurement can act on multiple modes
        if self.ns == 1:
            values = [values]
        # store the results in the register reference objects
        for v, r in zip(values, reg):
            r.val = v


class Decomposition(Operation):
    """Abstract base class for decompositions.

    This class provides the base behaviour for decomposing various objects
    into a sequence of gates and state preparations.
    """
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
            if np.all(np.abs(U -np.identity(len(U))) < _decomposition_merge_tol):
                return None
            new_decomp = self.__class__(U)
            return new_decomp
        else:
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
    pass


#====================================================================
# Derived transformation classes
#====================================================================


class Channel(Transformation):
    """Abstract base class for channels.

    This class provides the base behaviour for non-unitary
    maps and transformations.
    """
    # NOTE: At the moment this is an empty class, and only
    # exists for a nicer inheritence diagram.
    # This class will likely get filled in when we add more channels.
    pass


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

    def apply(self, reg, backend, hbar, **kwargs):
        """Ask a backend to execute the operation on the current register state right away.

        Like :func:`Operation.apply`, but takes into account the special nature of
        p[0] and applies self.dagger.

        Returns:
            None: Gates do not return anything, return value is None
        """
        z = self.p[0].evaluate()
        # if z represents a batch of parameters, then all of these
        #must be zero to skip calling backend
        if np.all(z == 0):
            # identity, no need to apply
            return
        if self.dagger:
            z = -z
        temp = self.p  # store the original Parameters
        # evaluate the rest of the Parameters, restore the originals later
        self.p = [z] +[x.evaluate() for x in self.p[1:]]
        # calling the parent apply, skipping re-evaluation of self.p
        #(which wouldn't hurt but is unnecessary)
        super().apply(reg, backend, hbar, eval_params=False, **kwargs)
        self.p = temp  # restore original unevaluated Parameter instances

    def merge(self, other):
        # can be merged if they are the same class and share all the other parameters
        if isinstance(other, self.__class__) and self.p[1:] == other.p[1:] \
           and len(self._extra_deps)+len(other._extra_deps) == 0:
            # no extra dependencies <=> no RegRefTransform parameters,
            # with which we cannot do arithmetic at the moment
            # make sure the gates have the same dagger flag, if not, invert the second p[0]
            if self.dagger == other.dagger:
                temp = other.p[0]
            else:
                temp = -other.p[0]
            # now we can add up the parameters and keep self.dagger
            p0 = self.p[0] +temp
            if p0 == 0:
                return None  # identity gate
            else:
                # return a copy
                # HACK: some of the subclass constructors only take a single parameter,
                #some take two, none take three
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
            raise MergeFailure('Not the same gate family.')

        if isinstance(other, self.__class__):
            # without knowing anything more specific about the gates, we
            # can only merge them if they are each others' inverses
            if self.dagger != other.dagger:
                return None
            else:
                raise MergeFailure("Don't know how to merge these gates.")
        else:
            raise MergeFailure('Not the same gate family.')


#====================================================================
# State preparation operations
#====================================================================


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
        # command queue is printed by the user
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
        # prepare the squeezed state
        backend.prepare_squeezed_state(p[1], p[2], *reg)
        # displace the state by alpha
        backend.displacement(p[0], *reg)


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

    This is a Strawberry Fields quantum gate operator, and thus is used within an engine
    context as follows:

    .. code-block:: python

        with eng:
            Catstate(1, 0.2) | q[0]

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


#====================================================================
# Measurements
#====================================================================


class MeasureFock(Measurement):
    """:ref:`photon_counting`: measures a set of modes in the Fock basis.

    Also accessible via the shortcut variable ``Measure``.

    The modes are projected to the Fock state corresponding to the result of the measurement.
    """
    ns = None
    def __init__(self, select=None):
        if select is not None and not isinstance(select, Sequence):
            select = [select]
        super().__init__([], select)

    def _apply(self, reg, backend, **kwargs):
        return backend.measure_fock(reg, select=self.select, **kwargs)

    def __str__(self):
        if self.select is None:
            return 'Measure'
        return 'MeasureFock(select={})'.format(self.select)


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

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        return backend.measure_homodyne(p[0], *reg, select=self.select, **kwargs)

    def __str__(self):
        if self.select is None:
            if self.p[0] == 0:
                return 'MeasureX'
            elif self.p[0] == pi/2:
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

    def _apply(self, reg, backend, **kwargs):
        return backend.measure_heterodyne(*reg, select=self.select, **kwargs)

    def __str__(self):
        if self.select is None:
            return 'MeasureHD'
        return 'MeasureHeterodyne(select={})'.format(self.select)


#====================================================================
# Channels
#====================================================================


class LossChannel(Channel):
    r"""Perform a :ref:`loss channel <loss>` operation on the specified mode.

    This channel couples mode :math:`a` to another bosonic mode :math:`b`
    prepared in the vacuum state using the following transformation:

    .. math::
       a \mapsto \sqrt{T} a+\sqrt{1-T} b

    Args:
        T (float): the loss parameter :math:`0\leq T\leq 1`.
    """
    def __init__(self, T):
        super().__init__([T])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.loss(p[0], *reg)

    def merge(self, other):
        # check that other is also a LossChannel, and that
        # no extra dependencies <=> no RegRefTransform parameters
        if isinstance(other, self.__class__) \
        and len(self._extra_deps)+len(other._extra_deps) == 0:
            # determine the new loss parameter
            T = self.p[0] * other.p[0]
            # if one, replace with the identity
            if T == 1:
                return None
            return self.__class__(T)
        else:
            raise MergeFailure('Not the same operation family.')


#====================================================================
# Unitary gates
#====================================================================


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
        z = self.p[0] / sqrt(2 * self.hbar)
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
        z = self.p[0] * 1j/sqrt(2 * self.hbar)
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

    def decompose(self, reg):
        # into a squeeze and a rotation
        temp = self.p[0] / 2
        r = arccosh(sqrt(1+temp**2))
        theta = arctan(temp)
        phi = -pi/2 * sign(temp) -theta
        return [
            Command(Sgate(r, phi), reg, decomp=True),
            Command(Rgate(theta), reg, decomp=True)
        ]


class Vgate(Gate):
    r""":ref:`Cubic phase <cubic>` gate.

    .. math::
       V(\gamma) = e^{i \frac{\gamma}{3} \hat{x}^3/\hbar}

    .. warning:: The cubic phase gate has lower accuracy than the Kerr gate at the same cutoff dimension.

    Args:
        gamma (float): parameter
    """
    def __init__(self, gamma):
        super().__init__([gamma])

    def _apply(self, reg, backend, **kwargs):
        p = _unwrap(self.p)
        backend.cubic_phase(p[0], *reg)


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
       B(\theta,\phi) = \exp\left(\theta (e^{i \phi} a^\dagger b -e^{-i \phi}a b^\dagger) \right)

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

    def decompose(self, reg):
        # two opposite squeezers sandwiched between 50% beamsplitters
        S = Sgate(self.p[0], self.p[1])
        BS = BSgate(pi/4, 0)
        return [
            Command(BS, reg, decomp=True),
            Command(S, reg[0], decomp=True),
            Command(S.H, reg[1], decomp=True),
            Command(BS.H, reg, decomp=True)
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

    def decompose(self, reg):
        s = self.p[0]
        r = arcsinh(-s/2)
        theta = 0.5*arctan2(-1.0/cosh(r), -tanh(r))

        BS1 = BSgate(theta, 0)
        BS2 = BSgate(theta+pi/2, 0)
        return [
            Command(BS1, reg, decomp=True),
            Command(Sgate(r, 0), reg[0], decomp=True),
            Command(Sgate(-r, 0), reg[1], decomp=True),
            Command(BS2, reg, decomp=True)
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

    def decompose(self, reg):
        # phase-rotated CZ
        CX = CXgate(self.p[0])
        return [
            Command(Rgate(-pi/2), reg[1], decomp=True),
            Command(CX, reg, decomp=True),
            Command(Rgate(pi/2), reg[1], decomp=True)
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


#====================================================================
# Metaoperations
#====================================================================


#====================================================================
# Subsystem creation and deletion
#====================================================================


class MetaOperation(Operation):
    """Abstract base class for metaoperations.

    This includes subsystem creation and deletion.
    """
    def __init__(self):
        super().__init__(par=[])


class Delete(MetaOperation):
    """Deletes one or more existing modes.
    Also accessible via the shortcut variable ``Del``.

    The deleted modes are traced out.
    After the deletion the state of the remaining subsystems may have to be described using a density operator.
    """
    ns = None

    def __or__(self, reg):
        reg = super().__or__(reg)
        _Engine._current_context._delete_subsystems(reg)

    def _apply(self, reg, backend, **kwargs):
        backend.del_mode(reg)

    def __str__(self):
        # the shorthand object
        return 'Del'


class New_modes(MetaOperation):
    """Used for adding new modes to the system.
    Also accessible via the shortcut variable ``New``.

    The new modes are prepapred in the vacuum state.

    This class cannot be used with the :meth:`__or__` syntax since it would be misleading,
    instead we use :meth:`__call__` on a single instance to dispatch the command to the engine.
    """
    ns = 0

    def __call__(self, n=1):
        """Adds one or more new modes to the system in a deferred way.

        Dispatches the command to the command queue.
        """
        # pylint: disable=attribute-defined-outside-init
        self.n = n  # int: store the number of new modes for the __str__ method
        # create RegRef placeholders for the new modes
        refs = _Engine._current_context._add_subsystems(n)
        # send the actual creation command to the engine
        _Engine._current_context.append(self, refs)
        return refs

    def _apply(self, reg, backend, **kwargs):
        # pylint: disable=unused-variable
        inds = backend.add_mode(len(reg))

    def __str__(self):
        # HACK, trailing space signals "do not print anything more" to Command.__str__.
        return 'New({}) '.format(self.n)


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
        return super().__str__() +'({})'.format(str(self.op))

    def __or__(self, reg):
        # into a list of subsystems
        reg = _seq_to_list(reg)
        # convert into commands
        _Engine._current_context._test_regrefs(reg)  # make sure reg does not contain duplicates (we feed them to Engine.append() one by one)
        for r in reg:
            _Engine._current_context.append(self.op, [r])


#====================================================================
# Decompositions
#====================================================================


class Interferometer(Decomposition):
    r"""Apply a linear interferometer to the specified qumodes.

    This operation uses the Clements decomposition to decompose
    a linear interferometer into a sequence of beamsplitters and
    rotation gates.

    Args:
        U (array): an :math:`N\times N` complex unitary matrix.
    """
    ns = None
    def __init__(self, U):
        super().__init__([U])

        if np.all(np.abs(U - np.identity(len(U))) < _decomposition_merge_tol):
            self.identity = True
        else:
            self.identity = False
            self.BS1, self.BS2, self.R = clements(U, tol=11)
            self.ns = U.shape[0]

    def decompose(self, reg):
        cmds = []

        if not self.identity:
            for n, m, theta, phi, N in self.BS1: # pylint: disable=unused-variable
                if np.round(phi, 13) != 0:
                    cmds.append(Command(Rgate(phi), reg[n], decomp=True))
                if np.round(theta, 13) != 0:
                    cmds.append(Command(BSgate(theta, 0), (reg[n], reg[m]), decomp=True))

            for n, expphi in enumerate(self.R):
                if np.round(expphi, 13) != 1.0:
                    q = log(expphi).imag
                    cmds.append(Command(Rgate(q), reg[n], decomp=True))

            for n, m, theta, phi, N in reversed(self.BS2):  # pylint: disable=unused-variable
                if np.round(theta, 13) != 0:
                    cmds.append(Command(BSgate(-theta, 0), (reg[n], reg[m]), decomp=True))
                if np.round(phi, 13) != 0:
                    cmds.append(Command(Rgate(-phi), reg[n], decomp=True))

        return cmds


class GaussianTransform(Decomposition):
    r"""Apply a Gaussian symplectic transformation to the specified qumodes.

    This operation uses the Bloch-Messiah decomposition to decompose a symplectic
    matrix :math:`S`:

    .. math:: S = O_1 R O_2

    where :math:`O_1` and :math:`O_2` are two orthogonal symplectic matrices (and thus passive
    Gaussian transformations), and :math:`R`
    is a squeezing transformation in the phase space (:math:`R=\text{diag}(e^{-z},e^z)`).

    The symplectic matrix describing the Gaussian transformation on :math:`N` modes must satisfy

    .. math:: S\Omega S^T = \Omega, ~~\Omega = \begin{bmatrix}0&I\\-I&0\end{bmatrix}

    where :math:`I` is the :math:`N\times N` identity matrix, and :math:`0` is the zero matrix.

    The two orthogonal symplectic unitaries describing the interferometers are then further
    decomposed via the :class:`~.Interferometer` operator and the Clements decomposition:

    .. math:: U_i = X_i + iY_i

    where

    .. math:: O_i = \begin{bmatrix}X&-Y\\Y&X\end{bmatrix}

    Args:
        S (array): a :math:`2N\times 2N` symplectic matrix describing the Gaussian transformation.
        hbar (float): the value of :math:`\hbar` used in the definition of the :math:`\x`
            and :math:`\p` quadrature operators. Note that if used inside of an engine
            context, the hbar value of the engine will override this keyword argument.
        vacuum (bool): set to True if acting on a vacuum state. In this case, :math:`O_2 V O_2^T = I`,
            and the unitary associated with orthogonal symplectic :math:`O_2` will be ignored.
    """
    ns = None
    def __init__(self, S, hbar=None, vacuum=False):
        super().__init__([S])

        try:
            self.hbar = _Engine._current_context.hbar
        except AttributeError:
            if hbar is None:
                raise ValueError("Either specify the hbar keyword argument, "
                                 "or use this operator inside an engine context.")
            else:
                self.hbar = hbar

        N = S.shape[0]//2

        # check if input symplectic is passive (orthogonal)
        diffn = np.linalg.norm(S @ S.T - np.identity(2*N))

        if np.round(diffn, 11) == 0.0:
            # The transformation is passive, do Clements
            self.active = False
            X1 = S[:N, :N]
            P1 = S[N:, :N]
            self.U1 = X1+1j*P1
        else:
            # transformation is active, do Bloch-Messiah
            self.active = True
            O1, smat, O2 = bloch_messiah(S, tol=10)
            N = S.shape[0]//2

            X1 = O1[:N, :N]
            P1 = O1[N:, :N]
            X2 = O2[:N, :N]
            P2 = O2[N:, :N]

            self.U1 = X1+1j*P1
            self.U2 = X2+1j*P2
            self.Sq = np.diagonal(smat)[:N]

        self.ns = N
        self.vacuum = vacuum

    def decompose(self, reg):
        cmds = []

        if self.active:
            if not self.vacuum:
                cmds = [Command(Interferometer(self.U2), reg, decomp=True)]

            for n, expr in enumerate(self.Sq):
                if np.abs(np.round(expr, 13)) != 1.0:
                    r = abs(log(expr))
                    phi = np.angle(log(expr))
                    cmds.append(Command(Sgate(-r, phi), reg[n], decomp=True))

            cmds.append(Command(Interferometer(self.U1), reg, decomp=True))
        else:
            if not self.vacuum:
                cmds = [Command(Interferometer(self.U1), reg, decomp=True)]

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

    Args:
        V (array): the :math:`2N\times 2N` (real and positive definite) covariance matrix.
        r (array): a length :math:`2N` vector of means, of the
            form :math:`(\x_0,\dots,\x_{N-1},\p_0,\dots,\p_{N-1})`.
            If None, it is assumed that :math:`r=0`.
        decomp (bool): if False, no decomposition is applied, and the specified modes
            are explicity prepared in the provided Gaussian state.
        hbar (float): the value of :math:`\hbar` used in the definition of the :math:`\x`
            and :math:`\p` quadrature operators. Note that if used inside of an engine
            context, the hbar value of the engine will override this keyword argument.
    """
    # pylint: disable=too-many-instance-attributes
    ns = None
    def __init__(self, V, r=None, decomp=True, hbar=None):
        try:
            self.hbar = _Engine._current_context.hbar
        except AttributeError:
            if hbar is None:
                raise ValueError("Either specify the hbar keyword argument, "
                                 "or use this operator inside an engine context.")
            else:
                self.hbar = hbar

        self.ns = V.shape[0]//2

        if r is None:
            # all zeroes
            r = [0] * (2*self.ns)
        r = np.asarray(r)

        if len(r) != V.shape[0]:
            raise ValueError('Vector of means must have the same length as the covariance matrix.')

        self.x_disp = r[:self.ns]
        self.p_disp = r[self.ns:]

        self.decomp = False
        if decomp:
            self.decomp = True
            th, self.S = williamson(V, tol=11)
            self.pure = np.abs(np.linalg.det(V) - (self.hbar/2)**(2*self.ns)) < 1e-6
            self.nbar = np.diag(th)[:self.ns]/self.hbar - 0.5

        super().__init__([V, r])

    def _apply(self, reg, backend, **kwargs):
        if self.decomp:
            # explicity raise a NotImplementedError,
            # so that the engine knows to perform a decomposition.
            raise NotImplementedError

        p = _unwrap(self.p)
        backend.prepare_gaussian_state(p[1], p[0], reg)

    def decompose(self, reg):
        # pylint: disable=too-many-branches
        cmds = []

        V = self.p[0].x
        D = np.diag(V)
        is_diag = np.all(V == np.diag(D))

        BD = changebasis(self.ns) @ V @ changebasis(self.ns).T
        BD_modes = [BD[i*2:(i+1)*2, i*2:(i+1)*2] for i in range(BD.shape[0]//2)]
        is_block_diag = (not is_diag) and np.all(BD == block_diag(*BD_modes))

        if self.pure and is_diag:
            # covariance matrix consists of x/p quadrature squeezed state
            for n, expr in enumerate(D[:self.ns]*2/self.hbar):
                if np.abs(np.round(expr, 13)) != 1.0:
                    r = abs(log(expr)/2)
                    cmds.append(Command(Sgate(r, 0), reg[n], decomp=True))

        elif self.pure and is_block_diag:
            # covariance matrix consists of rotated squeezed states
            for n, v in enumerate(BD_modes):
                if not np.all(v - np.identity(2)*self.hbar/2 < 1e-10):
                    r = np.abs(arccosh(np.sum(np.diag(v/self.hbar)))/2)
                    phi = arctan(2*v[0, 1] / np.sum(np.diag(v)*[1, -1]))
                    cmds.append(Command(Sgate(r, phi), reg[n], decomp=True))

        elif not self.pure and is_diag and np.all(D[:self.ns] == D[self.ns:]):
            # covariance matrix consists of thermal states
            for n, nbar in enumerate(D[:self.ns]/self.hbar - 0.5):
                if np.round(nbar, 13) != 0:
                    cmds.append(Command(Thermal(nbar), reg[n], decomp=True))

        else:
            if not self.pure:
                # mixed state, must initialise thermal states
                for n, nbar in enumerate(self.nbar):
                    if np.round(nbar, 13) != 0:
                        cmds.append(Command(Thermal(nbar), reg[n], decomp=True))

            cmds.append(
                Command(GaussianTransform(self.S, hbar=self.hbar, vacuum=self.pure), reg, decomp=True)
            )

        cmds += [Command(Xgate(u), reg[n], decomp=True) for n, u in enumerate(self.x_disp) if u != 0]
        cmds += [Command(Zgate(u), reg[n], decomp=True) for n, u in enumerate(self.p_disp) if u != 0]

        return cmds



#====================================================================
# Shorthand pre-constructed objects

New = New_modes()
Del = Delete()
Vac = Vacuum()
Measure = MeasureFock()
MeasureX = MeasureHomodyne(0)
MeasureP = MeasureHomodyne(pi/2)
MeasureHD = MeasureHeterodyne()

Fourier = Fouriergate()

RR = RegRefTransform


#====================================================================
# here we list different classes of operations for unit testing purposes

zero_args_gates = (Fourier,)  # all these are pre-constructed objects, not classes
one_args_gates = (Xgate, Zgate, Rgate, Pgate, Vgate, Kgate, CXgate, CZgate, CKgate)
two_args_gates = (Dgate, Sgate, BSgate, S2gate)
gates = zero_args_gates + one_args_gates + two_args_gates

channels = (LossChannel,)

state_preparations = (Vacuum, Coherent, Squeezed, DisplacedSqueezed, Fock, Thermal, Catstate)

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
The classes in this module represent parameters passed to the
quantum operations represented by :class:`~.Operation` subclasses.

Parameter types
---------------

There are three basic types of parameters:

1. **Numerical parameters** (bound and fixed): An immediate, immutable numerical object
   (float, complex, int, numerical array).
   Implemented as-is, not encapsulated in a class.

2. **Measured parameters** (bound but not fixed): Certain quantum circuits/protocols require that
   Operations can be conditioned on measurement results obtained during the execution of the
   circuit. In this case the parameter value is not known/fixed until the measurement is made
   (or simulated). Represented by :class:`MeasuredParameter` instances.
   Constructed from the :class:`.RegRef` instance storing the measurement
   result using the :meth:`.RegRef.par` method.

3. **Free parameters** (not bound nor fixed): A *parametrized circuit template* is a circuit that
   depends on a number of unbound (free) parameters. These parameters need to be bound to fixed
   numerical values before the circuit can be executed on a hardware quantum device or a numeric
   simulator. Represented by :class:`FreeParameter` instances.
   Simulators with symbolic capability can accept a parametrized circuit as input (and should
   return symbolic expressions representing the measurement results, with the same free parameters,
   as output).
   Free parameters belong to a single :class:`.Program` instance, are constructed using the
   :meth:`.Program.params` method, and are bound using :meth:`.Program.bind_params`.

:class:`.Operation` subclass constructors accept parameters that are functions or algebraic
combinations of any number of these basic parameter types. This is made possible by
:class:`MeasuredParameter` and :class:`FreeParameter` inheriting from :class:`sympy.Symbol`.

.. note:: Binary arithmetic operations between sympy symbols and numpy arrays produces numpy object arrays containing sympy symbols.


Operation lifecycle
-------------------

The normal lifecycle of an Operation object and its associated parameters is as follows:

* An Operation instance is constructed, and given some input arguments.
  In :meth:`.Operation.__init__`,
  the RegRef dependencies of measured parameters are added to :attr:`.Operation._measurement_deps`.

* The Operation instance is applied using its :meth:`~ops.Operation.__or__`
  method inside a :class:`.Program` context.
  This creates a :class:`.Command` instance that wraps
  the Operation and the RegRefs it acts on, which is appended to :attr:`.Program.circuit`.

* Before the Program is run, it is compiled and optimized for a specific backend. This involves
  checking that the Program only contains valid Operations, decomposing non-elementary Operations
  using :meth:`~ops.Operation.decompose`, and finally merging and commuting Commands inside
  the graph representing the quantum circuit.
  The circuit graph is built using the knowledge of which subsystems the Commands act and depend on.

* Decompositions, merges, and commutations often involve the creation of new Operations with algebraically
  transformed parameters.
  For example, merging two :class:`.Gate` instances of the same subclass involves
  adding their first parameters after equality-comparing the others. This is easily done if
  all the parameters have an immediate numerical value.
  Measured and free parameters are handled symbolically by Sympy.

* The compiled Program is run by a :class:`.BaseEngine` instance, which calls the
  :meth:`~ops.Operation.apply` method of each Operation in turn.

* :meth:`~ops.Operation.apply` then calls :meth:`~ops.Operation._apply` which is redefined by each Operation subclass.
  It evaluates the value of the parameters using :func:`par_evaluate`, and
  may perform additional numeric transformations on them.
  The parameter values are finally passed to the appropriate backend API method.
  It is up to the backend to either accept NumPy arrays and TensorFlow objects as parameters, or not.


What we cannot do at the moment:

* Use anything except integers and RegRefs (or Sequences thereof) as the subsystem argument
  for the :meth:`~ops.Operation.__or__` method.
  Technically we could allow any parameters that evaluate into an integer.
"""
# pylint: disable=too-many-ancestors,unused-argument,protected-access

import collections.abc
import functools
import types

import numpy as np
import sympy
import sympy.functions as sf


def wrap_mathfunc(func):
    """Applies the wrapped sympy function elementwise to NumPy arrays.

    Required because the sympy math functions cannot deal with NumPy arrays.
    We implement no broadcasting; if the first argument is a NumPy array, we assume
    all the arguments are arrays of the same shape.
    """

    @functools.wraps(func)
    def wrapper(*args):
        temp = [isinstance(k, np.ndarray) for k in args]
        if any(temp):
            if not all(temp):
                raise ValueError(
                    "Parameter functions with array arguments: all the arguments must be arrays of the same shape."
                )
            for k in args[1:]:
                if len(k) != len(args[0]):
                    raise ValueError(
                        "Parameter functions with array arguments: all the arguments must be arrays of the same shape."
                    )
            # apply func elementwise, recursively, on the args
            return np.array([wrapper(*k) for k in zip(*args)])
        return func(*args)

    return wrapper


par_funcs = types.SimpleNamespace(
    **{name: wrap_mathfunc(getattr(sf, name)) for name in dir(sf) if name[0] != "_"}
)
"""SimpleNamespace: Namespace of mathematical functions for manipulating Parameters.
Consists of all :mod:`sympy.functions` public members, which we wrap with :func:`wrap_mathfunc`.
"""


class ParameterError(RuntimeError):
    """Exception raised when the Parameter classes encounter an illegal operation.

    E.g., trying to use a measurement result before it is available.
    """


def is_object_array(p):
    """Returns True iff p is an object array.

    Args:
        p (Any): object to be checked

    Returns:
        bool: True iff p is a NumPy object array
    """
    return isinstance(p, np.ndarray) and p.dtype == object


def par_evaluate(params, dtype=None):
    """Evaluate an Operation parameter sequence.

    Any parameters descending from :class:`sympy.Basic` are evaluated, others are returned as-is.
    Evaluation means that free and measured parameters are replaced by their numeric values.
    NumPy object arrays are evaluated elementwise.

    Alternatively, evaluates a single parameter and returns its value.

    Args:
        params (Sequence[Any]): parameters to evaluate
        dtype (None, np.dtype, tf.dtype): NumPy or TensorFlow datatype to optionally cast atomic symbols
            to *before* evaluating the parameter expression. Note that if the atom
            is a TensorFlow tensor, a NumPy datatype can still be passed; ``tensorflow.dtype.as_dtype()``
            is used to determine the corresponding TensorFlow dtype internally.

    Returns:
        list[Any]: evaluated parameters
    """
    scalar = False
    if not isinstance(params, collections.abc.Sequence):
        scalar = True
        params = [params]

    def do_evaluate(p):
        """Evaluates a single parameter."""
        if is_object_array(p):
            return np.array([do_evaluate(k) for k in p])

        if not par_is_symbolic(p):
            return p

        # using lambdify we can also substitute np.ndarrays and tf.Tensors for the atoms
        atoms = list(p.atoms(MeasuredParameter, FreeParameter))
        # evaluate the atoms of the expression
        vals = [k._eval_evalf(None) for k in atoms]
        # use the tensorflow printer if any of the symbolic parameter values are TF objects
        # (we do it like this to avoid importing tensorflow if it's not needed)
        is_tf = (type(v).__module__.startswith("tensorflow") for v in vals)
        printer = "tensorflow" if any(is_tf) else "numpy"
        func = sympy.lambdify(atoms, p, printer)

        if dtype is not None:
            # cast the input values
            if printer == "tensorflow":
                import tensorflow as tf

                tfdtype = tf.as_dtype(dtype)
                vals = [tf.cast(v, dtype=tfdtype) for v in vals]
            else:
                vals = [dtype(v) for v in vals]

        return func(*vals)

    ret = list(map(do_evaluate, params))
    if scalar:
        return ret[0]
    return ret


def par_is_symbolic(p):
    """Returns True iff p is a symbolic Operation parameter instance.

    If a parameter inherits :class:`sympy.Basic` it is symbolic.
    A NumPy object array is symbolic if any of its elements are.
    All other objects are considered not symbolic parameters.

    Note that :data:`strawberryfields.math` functions applied to numerical (non-symbolic) parameters return
    symbolic parameters.
    """
    if is_object_array(p):
        return any(par_is_symbolic(k) for k in p)
    return isinstance(p, sympy.Basic)


def par_convert(args, prog):
    """Convert Blackbird symbolic Operation arguments into their SF counterparts.

    Args:
        args (Iterable[Any]): Operation arguments
        prog (Program): program containing the Operations.

    Returns:
        list[Any]: converted arguments
    """

    def do_convert(a):
        if isinstance(a, sympy.Basic):
            # substitute SF symbolic parameter objects for Blackbird ones
            s = {}
            for k in a.atoms(sympy.Symbol):
                if k.name[0] == "q":
                    s[k] = MeasuredParameter(prog.register[int(k.name[1:])])
                else:
                    s[k] = prog.params(k.name)  # free parameter
            return a.subs(s)
        return a  # return non-symbols as-is

    return [do_convert(a) for a in args]


def par_regref_deps(p):
    """RegRef dependencies of an Operation parameter.

    Returns the RegRefs that the parameter depends on through the :class:`MeasuredParameter`
    atoms it contains.

    Args:
        p (Any): Operation parameter

    Returns:
        set[RegRef]: RegRefs the parameter depends on
    """
    ret = set()
    if is_object_array(p):
        # p is an object array, possibly containing symbols
        for k in p:
            ret.update(par_regref_deps(k))
    elif isinstance(p, sympy.Basic):
        # p is a Sympy expression, possibly containing measured parameters
        for k in p.atoms(MeasuredParameter):
            ret.add(k.regref)
    return ret


def par_str(p):
    """String representation of the Operation parameter.

    Args:
        p (Any): Operation parameter

    Returns:
        str: string representation
    """
    if isinstance(p, np.ndarray):
        np.set_printoptions(precision=4)
        return str(p)
    if par_is_symbolic(p):
        return str(p)
    return "{:.4g}".format(p)  # scalar parameters


class MeasuredParameter(sympy.Symbol):
    """Single measurement result used as an Operation parameter.

    A MeasuredParameter instance, given as a parameter to a
    :class:`~strawberryfields.ops.Operation` constructor, represents
    a dependence of the Operation on classical information obtained by
    measuring a subsystem of the register.

    Used for deferred measurements, i.e., using a measurement's value
    symbolically in defining a gate before the numeric value of that
    measurement is available.

    Former RegRefTransform (SF <= 0.11) functionality is provided by the sympy.Symbol base class.

    Args:
        regref (RegRef): register reference responsible for storing the measurement result
    """

    def __new__(cls, regref):
        # sympy.Basic.__new__ wants a name, other arguments must not end up in self._args
        return super().__new__(cls, "q" + str(regref.ind))

    def __init__(self, regref):
        if not regref.active:
            raise ValueError("Trying to use an inactive RegRef.")
        #: RegRef: the value of the parameter depends on this RegRef, and can only be evaluated after the corresponding subsystem has been measured
        self.regref = regref

    def _sympystr(self, printer):
        """Blackbird notation.

        The Sympy printing system uses this method instead of __str__.
        """
        return "q{}".format(self.regref.ind)

    def _eval_evalf(self, prec):
        """Returns the numeric result of the measurement if it is available.

        Returns:
            Any: measurement result

        Raises:
            ParameterError: iff the parameter has not been measured yet
        """
        res = self.regref.val
        if res is None:
            raise ParameterError(
                "{}: trying to use a nonexistent measurement result (e.g., before it has been measured).".format(
                    self
                )
            )

        # remove unnecessary dims when returning measurement
        try:
            res = np.squeeze(res).item()
        except ValueError:
            res = np.squeeze(res)

        return res


class FreeParameter(sympy.Symbol):
    """Named symbolic Operation parameter.

    Args:
        name (str): name of the free parameter
    """

    def __init__(self, name):
        #: str: name of the free parameter
        self.name = name
        #: Any: value of the parameter, None means unbound
        self.val = None
        #: Any: default value of the parameter, used if unbound
        self.default = None

    def _sympystr(self, printer):
        """Blackbird notation.

        The Sympy printing system uses this method instead of __str__.
        """
        return "{{{}}}".format(self.name)

    def _eval_evalf(self, prec):
        """Returns the value of the parameter if it has been bound, or the default value if not.

        Returns:
            Any: bound value, or the default value if not bound

        Raises:
            ParameterError: iff the parameter has not been bound, and has no default value
        """
        if self.val is None:
            if self.default is None:
                raise ParameterError("{}: unbound parameter with no default value.".format(self))
            return self.default
        return self.val

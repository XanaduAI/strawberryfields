"""
Gate parameters
===============

**Module name:** :mod:`strawberryfields.parameters`

.. currentmodule:: strawberryfields.parameters

The :class:`Parameter` class is an abstraction of a parameter passed to the
quantum circuit operations represented by :class:`~strawberryfields.ops.ParOperation`.
The parameter objects can represent a number, a NumPy array, a value measured from the quantum register
(:class:`~strawberryfields.engine.RegRefTransform`),
or a TensorFlow object.

.. currentmodule:: strawberryfields

The normal lifecycle of a ParOperation object and its associated Parameter instances is as follows:

* A ParOperation instance is created, and given some parameters as input.

* :meth:`ParOperation.__init__` converts the inputs into Parameter instances.
  Plain :class:`~engine.RegRef` instances are wrapped in a trivial
  :class:`~engine.RegRefTransform`.
  RegRefTransforms add their RegRef dependencies to the Parameter and consequently to the Operation.

* The Operation instance is applied using its :func:`~ops.Operation.__or__`
  method inside an :class:`~engine.Engine` context.
  This creates a :class:`~engine.Command` instance that wraps
  the Operation and the RegRefs it acts on, which is appended to the Engine command queue.

* Once the entire program is inputted, Engine optimizes it. This involves merging and commuting Commands
  inside the circuit graph, the building of which requires knowledge of their dependencies, both direct and Parameter-based.

* Merging two :class:`~ops.Gate` instances of the same subclass involves
  adding their first parameters after equality-comparing the others. This is easily done if
  all the parameters have an immediate value. RegRefTransforms and TensorFlow objects are more complicated,
  but could in principle be handled.

For now, we simply don't do the merge if RegRefTransforms or TensorFlow objects are involved.

* The optimized command queue is run by Engine, which calls the :func:`~ops.Operation.apply` method
  of each Operation in turn (and tries :func:`~ops.Operation.decompose`
  if a :py:exc:`NotImplementedError` exception is raised).

* :func:`~ops.ParOperation.apply` evaluates the numeric value of any
  RegRefTransform-based Parameters using :func:`Parameter.evaluate` (other types of Parameters are simply passed through).
  The parameter values and the subsystem indices are passed to :func:`~ops.Operation._apply`.

* :func:`~ops.Operation._apply` "unwraps" the Parameter instances. There are three different cases:

  1. We still need to do some arithmetic, unwrap after it is done using p.x.
  2. No arithmetic required, use :func:`~parameters._unwrap`.
  3. No parameters are used, do nothing.

  Finally, _apply calls the appropriate backend API method using the unwrapped parameters.
  It is up to the backend to either accept NumPy arrays and Tensorflow objects as parameters, or not.

What we cannot do at the moment:

* Use anything except integers and RegRefs (or Sequences thereof) as the subsystem parameter
  for the :func:`~ops.Operation.__or__` method.
  Technically we could allow any Parameters or valid Parameter initializers that evaluate into an integer.
* Do arithmetic with RegRefTransforms.

Parameter methods
-----------------

.. currentmodule:: strawberryfields.parameters.Parameter

.. autosummary::
   evaluate


Code details
~~~~~~~~~~~~

"""

import numbers

import numpy as np
import tensorflow as tf

from .engine import (RegRef, RegRefTransform)



# supported TF classes
_tf_classes = (tf.Tensor, tf.Variable)


def _unwrap(params):
    """Unwrap a parameter sequence.

    Args:
      params (Sequence[Parameter]): parameters to unwrap

    Returns:
      tuple[Number, array, Tensor, Variable]: unwrapped Parameters
    """
    return tuple(p.x for p in params)


class Parameter():
    """Represents a parameter passed to a :class:`strawberryfields.ops.Operation` subclass constructor.

    The supported parameter types are Python and NumPy numeric types, NumPy arrays, :class:`RegRef` instances,
    :class:`RegRefTransform` instances, and certain TensorFlow objects. RegRef instances are internally represented as
    trivial RegRefTransforms.

    All but the RR and TensorFlow parameters represent an immediate numeric value that
    will not change. RR parameters can only be evaluated after the corresponding register
    subsystems have been measured. TF parameters can be evaluated whenever, but they depend on TF objects that
    are evaluated using :meth:`tf.Session.run`.

    The class supports various arithmetic operations which may change the internal representation of the result.
    If a TensorFlow object is involved, the result will always be a TensorFlow object.

    Args:
      x (Number, array, Tensor, Variable, RegRef, RegRefTransform): parameter value
    """
    # turn off the NumPy ufunc dispatching mechanism which is incompatible with our approach (see https://docs.scipy.org/doc/numpy-1.14.0/neps/ufunc-overrides.html)
    # NOTE: Another possible approach would be to use https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
    __array_ufunc__ = None

    def __init__(self, x):
        if isinstance(x, Parameter):
            raise TypeError('Tried initializing a Parameter using a Parameter.')

        self.deps = set()  #: set[RegRef]: parameter value depends on these RegRefs (if any), it can only be evaluated after the corresponding subsystems have been measured

        # wrap RegRefs in the identity RegRefTransform
        if isinstance(x, RegRef):
            x = RegRefTransform(x)
        elif isinstance(x, (numbers.Number, np.ndarray, _tf_classes, RegRefTransform)):
            pass
        else:
            raise TypeError('Unsupported base object type: ' +x.__class__.__name__)

        # add extra dependencies due to RegRefs
        if isinstance(x, RegRefTransform):
            self.deps.update(x.regrefs)
        self.x = x  #: parameter value, or reference

    def __str__(self):
        if isinstance(self.x, numbers.Number):
            return '{:.4g}'.format(self.x)
        return self.x.__str__()

    def __format__(self, format_spec):
        return self.x.__format__(format_spec)

    def evaluate(self):
        """Evaluate the numerical value of a RegRefTransform-based parameter.

        Returns:
          Parameter: self, unless self.x is a RegRefTransform in which case it is evaluated and a new Parameter instance is constructed on the result and returned
        """
        if isinstance(self.x, RegRefTransform):
            return Parameter(self.x.evaluate())
        return self

    def _unwrap_and_cast(self, other):
        """Unwrap Parameters and cast TensorFlow-type parameters to other dtypes during arithmetic.

        The main reason we need this is that TensorFlow does not automatically promote int to float or float to complex but requires an explicit cast.

        Args:
          other: the other input of a binary arithmetic operation
        Returns:
          (self, other) unwrapped, cast into compatible dtypes
        """
        # todo: Decide whether to cast to single or double precision by default (both float and complex).
        t = self.x
        if isinstance(other, Parameter):
            other = other.x
        swap = False
        if not isinstance(t, _tf_classes):
            if isinstance(other, _tf_classes):
                t, other = other, t  # make sure other is the non-tf type for simplicity
                swap = True
            else:
                # only TensorFlow types are cast
                return t, other

        if t.dtype.is_complex:
            if isinstance(other, _tf_classes) and not other.dtype.is_complex:
                other = tf.cast(other, tf.complex128)
        else:
            if (np.iscomplexobj(other) or
                    (isinstance(other, _tf_classes) and other.dtype.is_complex)):
                t = tf.cast(t, tf.complex128)
            elif t.dtype.is_integer:
                if (isinstance(other, float) or
                        (isinstance(other, np.ndarray) and np.issubdtype(other.dtype, np.floating)) or
                        (isinstance(other, _tf_classes) and other.dtype.is_floating)):
                    t = tf.cast(t, tf.float32)
            elif t.dtype.is_floating:
                if isinstance(other, _tf_classes) and other.dtype.is_integer:
                    other = tf.cast(other, tf.float32)

        if swap:
            return other, t
        return t, other

    @staticmethod
    def _wrap(x):
        """Wraps x inside a Parameter instance, unless x is a Parameter instance itself.

        Needed because of the way the reverse binary arithmetic methods work.

        Returns:
          Parameter: x as a Parameter instance
        """
        if isinstance(x, Parameter):
            return x
        return Parameter(x)

    # the arithmetic methods below basically are just responsible for calling _unwrap_and_cast which exposes self.x, then the arithmetic ops of the supported parameter types take over
    def __add__(self, other):
        temp, other = self._unwrap_and_cast(other)
        return self._wrap(temp +other)

    def __radd__(self, other):
        temp, other = self._unwrap_and_cast(other)
        return self._wrap(other +temp)

    def __sub__(self, other):
        temp, other = self._unwrap_and_cast(other)
        return self._wrap(temp -other)

    def __rsub__(self, other):
        temp, other = self._unwrap_and_cast(other)
        return self._wrap(other -temp)

    def __mul__(self, other):
        temp, other = self._unwrap_and_cast(other)
        return self._wrap(temp * other)

    def __rmul__(self, other):
        temp, other = self._unwrap_and_cast(other)
        return self._wrap(other * temp)

    def __truediv__(self, other):
        temp, other = self._unwrap_and_cast(other)
        return self._wrap(temp / other)

    def __rtruediv__(self, other):
        temp, other = self._unwrap_and_cast(other)
        return self._wrap(other / temp)

    def __pow__(self, other):
        temp, other = self._unwrap_and_cast(other)
        return self._wrap(temp ** other)

    def __rpow__(self, other):
        temp, other = self._unwrap_and_cast(other)
        return self._wrap(other ** temp)

    def __neg__(self):
        # pylint: disable=invalid-unary-operand-type
        return Parameter(-self.x)

    # other properties
    @property
    def shape(self):
        """Returns the shape of array parameters."""
        try:
            return self.x.shape
        except AttributeError:
            return None

    # comparisons
    def __eq__(self, other):
        """Equality comparison.

        .. note:: This method may be too permissive, maybe it should return False if either parameter is not a numbers.Number or a np.ndarray?

        Returns:
          bool: True iff both self and other have immediate, equal values, or identical dependence on measurements, otherwise False.
        """
        if isinstance(other, Parameter):
            other = other.x
        # see RegRefTransform.__eq__
        return self.x == other

# corresponding numpy and tensorflow functions
np_math_fns = {"abs": (np.abs, tf.abs),
               "sign": (np.sign, tf.sign),
               "sin": (np.sin, tf.sin),
               "cos": (np.cos, tf.cos),
               "cosh": (np.cosh, tf.cosh),
               "tanh": (np.tanh, tf.tanh),
               "exp": (np.exp, tf.exp),
               "log": (np.log, tf.log),
               "sqrt": (np.sqrt, tf.sqrt),
               "arctan": (np.arctan, tf.atan),
               "arctan2": (np.arctan2, tf.atan2),
               "arcsinh": (np.arcsinh, tf.asinh),
               "arccosh": (np.arccosh, tf.acosh),
               "matmul": (np.matmul, tf.matmul),
               "expand_dims": (np.expand_dims, tf.expand_dims),
               "squeeze": (np.squeeze, tf.squeeze),
               "transpose": (np.transpose, tf.transpose),
               "reshape": (np.reshape, tf.reshape)
              }

def math_fn_wrap(np_fn, tf_fn):
    """Wrapper function for the standard math functions.

    It checks the type of the incoming object and calls the appropriate NumPy or TensorFlow function.
    """
    def wrapper(*args, **kwargs):
        """wrapper function"""
        if any([isinstance(a, _tf_classes) for a in args]):
            # if anything is a tf object, use the tensorflow version of the function
            return tf_fn(*args, **kwargs)
        elif any([isinstance(a, Parameter) for a in args]):
            # for Parameters, call the function on the data and construct a new instance
            temp = (a.x if isinstance(a, Parameter) else a for a in args)
            return Parameter(wrapper(*temp))
        # otherwise, default to numpy version
        return np_fn(*args, **kwargs)

    wrapper.__name__ = np_fn.__name__
    wrapper.__doc__ = math_fn_wrap.__doc__
    return wrapper

# HACK, edit the global namespace to have sort-of single dispatch overloading for the standard math functions
for name, fn in np_math_fns.items():
    globals()[name] = math_fn_wrap(*fn)

"""
Gate parameters
===============

**Module name:** :mod:`strawberryfields.parameters`

.. currentmodule:: strawberryfields.parameters

The :class:`Parameter` class is an abstraction of a parameter passed to the
quantum circuit operations represented by :class:`~strawberryfields.ops.Operation`.
The parameter objects can represent a number, a NumPy array, a value measured from the quantum register,
or a TensorFlow object.


The normal lifecycle of a :class:`~strawberryfields.ops.ParametrizedOperation` object is as follows:

* A ParametrizedOperation instance is created, and given some parameters as input.
* The initializer converts the inputs into :class:`Parameter` instances.
  Plain :class:`~strawberryfields.engine.RegRef` instances are wrapped in a trivial
  :class:`~strawberryfields.engine.RegRefTransform`.
  RegRefTransforms add their RegRef dependencies to the Parameter and consequently to the Operation.
* The Operation instance is applied using its :func:`~strawberryfields.ops.Operation.__or__`
  method inside an :class:`~strawberryfields.engine.Engine` context.
  This creates a :class:`~strawberryfields.engine.Command` instance that wraps
  the Operation and the RegRefs it acts on, which is appended to the Engine command queue.
* Once the entire program is inputted, the Engine optimizes it. This involves merging and commuting Commands
  inside the circuit graph, the building of which requires knowledge of their dependencies, both direct and Parameter-based.
* Merging two :class:`~strawberryfields.ops.Gate` instances of the same subclass involves
  adding their first parameters after equality-comparing the others. This is easily done if
  all the parameters have an immediate value. RegRefTransforms and TensorFlow objects are more complicated,
  but could in principle be handled. TODO for now we simply don't do the merge if they're involved.
* The optimized command queue is run by the Engine, which calls the :func:`~strawberryfields.ops.Operation.apply` method
  of each Operation in turn (and tries :func:`~strawberryfields.ops.Operation.decompose`
  if a :py:exc:`NotImplementedError` exception is raised).
* :func:`~strawberryfields.ops.ParametrizedOperation.apply` evaluates the numeric value of any
  RegRefTransform-based Parameters using :func:`Parameter.evaluate` (other types of Parameters are simply passed through).
  The parameter values and the subsystem indices are finally passed to :func:`~strawberryfields.ops.Operation._apply`
  which calls the appropriate backend API method. It is up to the backend to either accept
  NumPy arrays and Tensorflow objects as parameters, or not.


What we cannot do at the moment:

* Use anything except integers and RegRefs (or Sequences thereof) as the subsystem parameter
  for the :func:`~strawberryfields.ops.Operation.__or__` method.
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
#from numpy import ndarray
from numpy import cos, sin, exp, sqrt, arctan, arccosh, sign, arctan2, arcsinh, cosh, tanh, log, matmul

from tensorflow import (Tensor, Variable)
from tensorflow import cos as tfcos, sin as tfsin, exp as tfexp, sqrt as tfsqrt, atan as tfatan, acosh as tfacosh, sign as tfsign, \
    atan2 as tfatan2, asinh as tfasinh, cosh as tfcosh, tanh as tftanh, log as tflog, matmul as tfmatmul


from .engine import (RegRef, RegRefTransform)


tf_objs = (Tensor, Variable)


# wrap math functions so they call the correct underlying function for the input type
tf_math_fns = {"sin": tfsin,
               "cos": tfcos,
               "exp": tfexp,
               "sqrt": tfsqrt,
               "arctan": tfatan,
               "arccosh": tfacosh,
               "sign": tfsign,
               "arctan2": tfatan2,
               "arcsinh": tfasinh,
               "cosh": tfcosh,
               "tanh": tftanh,
               "log": tflog,
               "matmul": tfmatmul}
np_math_fns = {"sin": sin,
               "cos": cos,
               "exp": exp,
               "sqrt": sqrt,
               "arctan": arctan,
               "arccosh": arccosh,
               "sign": sign,
               "arctan2": arctan2,
               "arcsinh": arcsinh,
               "cosh": cosh,
               "tanh": tanh,
               "log": log,
               "matmul": matmul}

def check_type(math_fn):
    "Wrapper function which checks the type of the incoming object and calls the appropriate tf/np function"
    fn_name = math_fn.__name__
    def wrapper(*args, **kwargs):
        """wrapper function"""
        if any([isinstance(x, (Variable, Tensor)) for x in args]):
            # if anything is a tf object, use the tensorflow version of the function
            math_fn = tf_math_fns[fn_name]
        else:
            # otherwise, default to numpy version
            math_fn = np_math_fns[fn_name]
        return math_fn(*args, **kwargs)
    return wrapper


# HACK, edit the global namespace to have single dispatch overloading for the standard math functions
for k, mfn in np_math_fns.items():
    globals()[k] = check_type(mfn)



class Parameter():
    """Represents a parameter passed to a :class:`strawberryfields.ops.Operation` subclass constructor.

    The supported parameter types are Python and NumPy numeric types, NumPy arrays, :class:`RegRef` instances,
    :class:`RegRefTransform` instances, and certain TensorFlow objects.

    All but the RegRef and TensorFlow objects are guaranteed to have an immediate numeric value that can be evaluated
    and will not change.

    The class supports various arithmetic operations which may change the internal representation of the result.
    If a TensorFlow object is involved, the result will always be a TensorFlow object.

    Args:
      x (): parameter value
    """
    def __init__(self, x):
        if isinstance(x, Parameter):
            raise ValueError('sdfsdfsf')

        self.deps = set() #: set[RegRef]: parameter value depends on these RegRefs, it can only be evaluated after the corresponding subsystems have been measured

        # wrap RegRefs in the identity RegRefTransform
        if isinstance(x, RegRef):
            x = RegRefTransform(x)
        elif isinstance(x, tf_objs):
            pass
        elif isinstance(x, np.ndarray):
            pass

        # add extra dependencies due to RegRefs
        if isinstance(x, RegRefTransform):
            self.deps.update(x.regrefs)
        self.x = x     #: parameter value, or reference

    def __str__(self):
        if isinstance(self.x, numbers.Number):
            return '{:.4g}'.format(self.x)
        else:
            return self.x.__str__()

    def __format__(self, format_spec):
        return self.x.__format__(format_spec)

    def evaluate(self):
        """Evaluate the numerical value of the parameter.

        Returns:
          Number, array, Tensor:
        """
        if isinstance(self.x, (numbers.Number, np.ndarray)):
            return self.x
        elif isinstance(self.x, RegRefTransform):
            return self.x.evaluate()
        elif isinstance(self.x, tf_objs):
            return self.x

    def _maybe_cast(self, other):
        if isinstance(other, complex):
            t = tf.cast(self.tensor, def_type)
        elif isinstance(other, float):
            if self.tensor.dtype.is_integer:
                t = tf.cast(self.tensor, tf.float64) # cast ints to float
            else:
                t = self.tensor # but dont cast other dtypes (i.e., complex) to float
        elif isinstance(other, (tf.Tensor, tf.Variable)) and other.dtype.is_complex:
            t = tf.cast(self.tensor, def_type)
        else:
            t = self.tensor
        return t

    @staticmethod
    def _wrap(x):
        """Wraps x inside a Parameter instance, unless x is a Parameter instance itself.

        Needed because of the way the reverse arithmetic methods work.
        """
        if isinstance(x, Parameter):
            return x
        return Parameter(x)

    # the arithmetic methods below basically are just responsible for exposing self.x to the arithmetic ops of the supported parameter types
    def __add__(self, other):
        return self._wrap(self.x +other)

    def __radd__(self, other):
        return self.__add__(other)  # addition commutes

    def __sub__(self, other):
        return self._wrap(self.x -other)

    def __rsub__(self, other):
        return self._wrap(other -self.x)

    def __mul__(self, other):
        return self._wrap(self.x * other)

    def __rmul__(self, other):
        return self.__mul__(other)  # multiplication commutes

    def __truediv__(self, other):
        return self._wrap(self.x / other)

    def __rtruediv__(self, other):
        return self._wrap(other / self.x)

    def __pow__(self, other):
        return self._wrap(self.x ** other)

    def __rpow__(self, other):
        return self._wrap(other ** self.x)

    def __neg__(self):
        return Parameter(-self.x)

    # comparisons
    def __eq__(self, other):
        """Equality comparison.

        .. note:: This method may be too permissive, maybe it should return False if either parameter is not a numbers.Number or a np.ndarray?

        Returns:
          bool: True iff both self and other have an immediate value and the values are equal, otherwise False.
        """
        if isinstance(other, Parameter):
            other = other.x
        return self.x == other

    # Do all NumPy one-parameter ufuncs magically call parameter.functionname() if no native implementation exists??
    def abs(self):
        return Parameter(abs(self.x))  # TODO tf.abs for Tensorflow objs?!?

    def sqrt(self):
        return Parameter(sqrt(self.x))

#    def cos(self):
#        return tf.cos(self.x)

#    def sin(self):
#        return tf.sin(self.x)

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
This module defines and implements decorators that can be used for example with
context managers. The :class:`~.strawberryfields.utils.operation` decorator
allows functions containing quantum operations acting on a qumode to be used as
an operation itself within a :class:`~.Program` context.
"""
import collections
from inspect import signature


__all__ = [
    "operation",
]


class operation:
    """Groups a sequence of gates into a single operation to be used
    within a Program context.

    For example:

    .. code-block:: python

        @sf.operation(3)
        def custom_operation(v1, v2, q):
            CZgate(v1) | (q[0], q[1])
            Vgate(v2) | q[2]

    Here, the ``operation`` decorator must recieve an argument
    detailing the number of subsystems the resulting custom
    operation acts on.

    The function it acts on can contain arbitrary
    Python and Blackbird code that may normally be placed within a
    Program context. Note that it must always accept the register
    ``q`` it acts on as the *last* argument of the function.

    Once defined, it can be used like any other quantum operation:

    .. code-block:: python

        prog = sf.Program(3)
        with prog.context as q:
            custom_operation(0.5719, 2.0603) | (q[0], q[1], q[3])

    Note that here, we do not pass the register ``q`` directly
    to the function - instead, it is defined on the right hand side
    of the ``|`` operation, like all other Blackbird code.

    Args:
        ns (int): number of subsystems required by the operation
    """

    def __init__(self, ns):
        self.ns = ns
        self.func = None
        self.args = None

    def __or__(self, reg):
        """Apply the operation to a part of a quantum register.

        Redirects the execution flow to the wrapped function.

        Args:
            reg (RegRef, Sequence[RegRef]): subsystem(s) the operation is
                acting on

        Returns:
            list[RegRef]: subsystem list as RegRefs
        """
        if (not reg) or (not self.ns):
            raise ValueError("Wrong number of subsystems")

        reg_len = 1
        if isinstance(reg, collections.abc.Sized):
            reg_len = len(reg)

        if reg_len != self.ns:
            raise ValueError("Wrong number of subsystems")

        return self._call_function(reg)

    def _call_function(self, reg):
        """Executes the wrapped function and passes the quantum registers.

        Args:
            reg (RegRef, Sequence[RegRef]): subsystem(s) the operation is
                acting on

        Returns:
            list[RegRef]: subsystem list as RegRefs
        """
        func_sig = signature(self.func)
        num_params = len(func_sig.parameters)

        if num_params == 0:
            raise ValueError("Operation must receive the qumode register as an argument.")

        if num_params != len(self.args) + 1:
            raise ValueError("Mismatch in the number of arguments")

        # pass parameters and subsystems to the function
        if num_params == 1:
            self.func(reg)
        else:
            self.func(*self.args, reg)

        return reg

    def __call__(self, func):
        self.func = func

        def f_proxy(*args):
            """
            Proxy for function execution. Function will actually execute in __or__
            """
            self.args = args
            return self

        return f_proxy

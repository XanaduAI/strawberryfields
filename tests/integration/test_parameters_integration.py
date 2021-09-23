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
"""Integration tests for the parameters.py module."""

import inspect
import itertools
import warnings

import pytest
import numpy as np

import strawberryfields.program_utils as pu
from strawberryfields import ops
from strawberryfields.parameters import ParameterError, par_funcs as pf


# ops.Fock requires an integer parameter
scalar_arg_preparations = (
    ops.Coherent,
    ops.Squeezed,
    ops.DisplacedSqueezed,
    ops.Thermal,
    ops.Catstate,
)

# Operation subclasses to be tested
testset = ops.one_args_gates + ops.two_args_gates + ops.channels + scalar_arg_preparations


def test_free_parameters(setup_eng, tol):
    """Programs with free parameters."""

    eng, prog = setup_eng(1)
    x = prog.params("x")  # free parameter
    with prog.context as q:
        ops.Dgate(x) | q
        ops.Sgate(-1.2 * x * pf.sin(x ** 2 - 0.1)) | q

    with pytest.raises(ParameterError, match="Unknown free parameter"):
        eng.run(prog, args={"foo": 1.0})
    with pytest.raises(ParameterError, match="unbound parameter with no default value"):
        eng.run(prog)

    # successful run
    eng.run(prog, args={x: 0.0})
    assert np.all(eng.backend.is_vacuum(tol))
    eng.reset()

    # now set a default value for the free parameter
    x.default = 0.0
    eng.run(prog)
    assert np.all(eng.backend.is_vacuum(tol))
    eng.reset()

    # override the default
    x.default = 1.0
    eng.run(prog, args={x: 0.0})
    assert np.all(eng.backend.is_vacuum(tol))


@pytest.fixture(scope="function")
def eng_prog_params(batch_size, setup_eng):
    """Engine and Program instances, and an attached set of Operation parameters."""

    def func1(x):
        return abs(2 * x ** 2 - 3 * x + 1)

    def func2(x, y):
        return abs(2 * x * y - y ** 2 + 0.5)

    eng, prog = setup_eng(2)
    r = prog.register

    # fixed and different types of measured parameters
    # (note that some Operations expect nonnegative parameter values)
    params = [0.14, r[0].par, func1(r[0].par), func2(r[0].par, r[1].par)]
    if batch_size is not None:
        # test batched input
        params.append(np.random.uniform(size=(batch_size,)))

    return eng, prog, params


@pytest.mark.parametrize("G", testset)
def test_parameters_with_operations(eng_prog_params, G):
    """Test all combinations of different types of Parameters with different Operation subclasses.

    This test is successful if no exceptions are raised.

    Some operation/backend combinations are forbidden by the Compiler instance of the backend.
    We catch these exceptions and convert them into warnings.
    """
    kwargs = {}
    eng, prog, params = eng_prog_params

    def check(G, par):
        """Check an Operation/Parameters combination."""

        # construct the op using the given tuple of Parameters as args
        G = G(*par)
        assert isinstance(G, ops.Operation)

        # clear the program for reuse
        prog.locked = False
        prog.circuit = []
        with prog.context as r:
            # fake a measurement for speed
            r[0].val = 0.1
            r[1].val = 0.2
            if G.ns == 1:
                G | r[0]
            else:
                G | (r[0], r[1])

        try:
            eng.run(prog, **kwargs)
        except pu.CircuitError as err:
            # record Compiler-forbidden op/backend combinations here
            warnings.warn(str(err))
        eng.reset()

    sig = inspect.signature(G.__init__)
    n_args = len(sig.parameters) - 1  # number of parameters __init__ takes, minus self
    assert n_args >= 1
    # shortcut, only test cartesian products up to two parameter types
    n_args = min(n_args, 2)
    # check all combinations of Parameter types
    for p in itertools.product(params, repeat=n_args):
        check(G, p)

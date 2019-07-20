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
r"""Integration tests to make sure parameters are correctly evaluated
before passing them to the backends"""

import inspect
import itertools
import logging

import pytest
import numpy as np

import strawberryfields as sf
from strawberryfields import ops


# Fock requires an integer parameter
scalar_arg_preparations = (
    ops.Coherent,
    ops.Squeezed,
    ops.DisplacedSqueezed,
    ops.Thermal,
    ops.Catstate,
)

testset = ops.one_args_gates + ops.two_args_gates + ops.channels + scalar_arg_preparations


def test_parameters_with_operations(batch_size, setup_eng):
    """Test using different types of Parameters with different classes of Operations."""

    # TODO: this test is very hard to understand. Should be split up,
    # and separated into many individual tests that do one thing only.

    # TODO: In the original test suite, this test appears to 'work' on the
    # Gaussian backend, even though it never explicitly ignores non-Gaussian
    # operations. It even worked on the Fock backends, even though the Fock
    # backend doesn't support the ThermalLoss channel?!

    # TODO: This test doesn't assert anything!!!!!
    eng, prog = setup_eng(2)
    r = prog.register

    def func1(x):
        return abs(2 * x ** 2 - 3 * x + 1)

    def func2(x, y):
        return abs(2 * x * y - y ** 2 + 0.5)

    kwargs = {}

    # test fixed and measured parameters
    # (note that some Operations expect nonnegative parameter values)
    params = [0.14, r[0].par, func1(r[0].par), func2(r[0].par, r[1].par)]
    if batch_size is not None:
        # test batched input
        params.append(np.random.uniform(size=(batch_size,)))

    def check(G, par, measure=False):
        """Check an Operation/Parameters combination"""
        # construct the op using the given tuple of Parameters as args
        G = G(*par)
        with prog.context as r:
            if measure:
                # fake a measurement for speed
                r[0].val = 0.1
                r[1].val = 0.2
            if G.ns == 1:
                G | r[0]
            else:
                G | (r[0], r[1])

        #prog.optimize()
        try:
            eng.run(prog, **kwargs)
        except NotApplicableError as err:
            # catch unapplicable op/backend combinations here
            logging.debug(err)
        except NotImplementedError as err:
            # catch not yet implemented op/backend combinations here
            logging.debug(err)

        prog.locked = False
        prog.circuit = []


    print(testset)
    for G in testset:
        print(G)
        sig = inspect.signature(G.__init__)
        n_args = len(sig.parameters) - 1  # number of parameters __init__ takes, minus self
        if n_args < 1:
            logging.debug(
                "Unexpected number of args ({}) for {}, check the testset.".format(n_args, G)
            )

        # shortcut, only test cartesian products up to two parameter types
        n_args = min(n_args, 2)

        # check all combinations of Parameter types
        for p in itertools.product(params, repeat=n_args):
            check(G, p, measure=True)
            eng.reset()

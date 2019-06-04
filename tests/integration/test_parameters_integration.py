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
# import tensorflow as tf

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.parameters import Parameter
# from strawberryfields.backends.tfbackend import TFBackend

TFBackend = int # hack to avoid removing TFBackend from test


# TODO: broken for TensorFlow. Skipping tf test here for now.
@pytest.mark.backends("fock")
def test_parameters_with_operations(batch_size, setup_eng, backend, hbar):
    """Test using different types of Parameters with different classes
    of ParOperations."""

    # TODO: this test is very hard to understand. Should be split up,
    # and separated into many individual tests that do one thing only.

    # TODO: In the original test suite, this test appears to 'work' on the
    # Gaussian backend, even though it never explicitly ignores non-Gaussian
    # operations. It even worked on the Fock backends, even though the Fock
    # backend doesn't support the ThermalLoss channel?!

    # TODO: This test doesn't assert anything!!!!!
    eng, prog = setup_eng(2)
    r = prog.register

    @sf.convert
    def func1(x):
        return abs(2 * x ** 2 - 3 * x + 1)

    @sf.convert
    def func2(x, y):
        return abs(2 * x * y - y ** 2 + 0.5)

    kwargs = {}

    # RegRefTransforms for deferred measurements (note that some operations
    # expect nonnegative parameter values)
    rr_inputs = [ops.RR(r[0], lambda x: x ** 2), func1(r[0]), func2(*r)]
    rr_pars = tuple(Parameter(k) for k in rr_inputs)

    # other types of parameters
    other_inputs = [0.14]
    if isinstance(eng.backend, TFBackend):
        # HACK: add placeholders for tf.Variables
        other_inputs.append(np.inf)
        if batch_size is not None:
            # test batched input
            other_inputs.append(np.random.uniform(size=(batch_size,)))

    other_pars = tuple(Parameter(k) for k in other_inputs)

    def check(G, par, measure=False):
        """Check a ParOperation/Parameters combination"""
        # construct the op using the given tuple of Parameters as args
        G = G(*par)
        with prog:
            if measure:
                r[0].val = 0.1
                r[1].val = 0.2
            if G.ns == 1:
                G | r[0]
            else:
                G | (r[0], r[1])

        prog.optimize()

        try:
            eng.run(prog, **kwargs)
        except NotApplicableError as err:
            # catch unapplicable op/backend combinations here
            logging.debug(err)
            # unsuccessful run means the queue was not emptied.
            eng.reset_queue()
        except NotImplementedError as err:
            # catch not yet implemented op/backend combinations here
            logging.debug(err)
            # unsuccessful run means the queue was not emptied.
            eng.reset_queue()

        prog.locked = False

    scalar_arg_preparations = (
        ops.Coherent,
        ops.Squeezed,
        ops.DisplacedSqueezed,
        ops.Thermal,
        ops.Catstate,
    )
    # Fock requires an integer parameter
    testset = set(
        ops.one_args_gates + ops.two_args_gates + ops.channels + scalar_arg_preparations
    ) - {ops.ThermalLossChannel}
    print(testset)

    for G in testset:
        sig = inspect.signature(G.__init__)
        n_args = (
            len(sig.parameters) - 1
        )  # number of parameters __init__ takes, minus self
        if n_args < 1:
            logging.debug(
                "Unexpected number of args ({}) for {}, check the testset.".format(
                    n_args, G
                )
            )

        # shortcut, only test cartesian products up to two parameter types
        # n_args = min(n_args, 2)

        # check all combinations of Parameter types
        for p in itertools.product(rr_pars + other_pars, repeat=n_args):
            # TODO: For now decompose() works on unevaluated Parameters.
            # This causes an error if a :class:`.RegRefTransform`-based Parameter is used, and
            # decompose() tries to do arithmetic on it.
            if isinstance(p[0].x, ops.RR):
                continue

            if isinstance(eng.backend, TFBackend):
                # declare tensorflow Variables here (otherwise they will be on a different graph when we do backend.reset below)
                # replace placeholders with tf.Variable instances
                def f(q):
                    if np.any(q.x == np.inf):  # detect the HACK
                        if f.v is None:  # only construct one instance, only when needed
                            f.v = Parameter(tf.Variable(0.8, name="sf_parameter"))
                        return f.v
                    return q

                f.v = None  # kind of like a static variable in a function in C
                p = tuple(map(f, p))

                # create new tf.Session (which will be associated with the default graph) and initialize variables
                session = tf.Session()
                kwargs["session"] = session
                session.run(tf.global_variables_initializer())
                check(G, p, measure=True)
                session.close()

            else:
                check(G, p, measure=True)

            eng.backend.reset()

# Copyright 2019-2021 Xanadu Quantum Technologies Inc.

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
This module contains functions for loading and saving Strawberry Fields
:class:`~.Program` objects from/to Blackbird scripts and Strawberry Fields
code.
"""
# pylint: disable=protected-access,too-many-nested-blocks
import numpy as np

import blackbird

import strawberryfields.parameters as sfpar
from strawberryfields.program import Program
from strawberryfields.tdm import TDMProgram, is_ptype
from strawberryfields import ops


def from_blackbird(bb: blackbird.BlackbirdProgram) -> Program:
    """Convert a Blackbird program to a Strawberry Fields program.

    Args:
        bb (blackbird.BlackbirdProgram): the input Blackbird program object

    Returns:
        Program: corresponding Strawberry Fields program

    Raises:
        NameError: if an applied quantum operation is not defined in Strawberry Fields
    """
    # create a SF program
    prog = Program(max(bb.modes) + 1, name=bb.name)

    # append the quantum operations
    with prog.context as q:
        for op in bb.operations:
            # check if operation name is in the list of
            # defined StrawberryFields operations.
            # This is used by checking against the ops.py __all__
            # module attribute, which contains the names
            # of all defined quantum operations
            if op["op"] in ops.__all__:
                # get the quantum operation from the sf.ops module
                gate = getattr(ops, op["op"])
            else:
                raise NameError("Quantum operation {} not defined!".format(op["op"]))

            # create the list of regrefs
            regrefs = [q[i] for i in op["modes"]]

            if "args" in op and "kwargs" in op:
                # the gate has arguments
                args = op["args"]
                kwargs = op["kwargs"]

                # Convert symbolic expressions in args/kwargs containing measured and free parameters to
                # symbolic expressions containing the corresponding MeasuredParameter and FreeParameter instances.
                args = sfpar.par_convert(args, prog)
                vals = sfpar.par_convert(kwargs.values(), prog)
                kwargs = dict(zip(kwargs.keys(), vals))
                gate(*args, **kwargs) | regrefs  # pylint:disable=expression-not-assigned
            else:
                # the gate has no arguments
                gate | regrefs  # pylint:disable=expression-not-assigned,pointless-statement

    prog._target = bb.target["name"]

    if "shots" in bb.target["options"]:
        prog.run_options["shots"] = bb.target["options"]["shots"]

    if "cutoff_dim" in bb.target["options"]:
        prog.backend_options["cutoff_dim"] = bb.target["options"]["cutoff_dim"]

    return prog


# pylint:disable=too-many-branches
def from_blackbird_to_tdm(bb: blackbird.BlackbirdProgram) -> TDMProgram:
    """Convert a ``BlackbirdProgram`` to a ``TDMProgram``.

    Args:
        bb (blackbird.BlackbirdProgram): the input Blackbird program object

    Returns:
        TDMProgram: corresponding ``TDMProgram``

    Raises:
        NameError: if an applied quantum operation is not defined in Strawberry Fields
    """
    prog = TDMProgram(max(bb.modes) + 1, name=bb.name)

    def is_free_param(param):
        return isinstance(param, str) and is_ptype(param)

    # retrieve all the free parameters in the Blackbird program (e.g. "p0", "p1"
    # etc.) and add their corresponding values to args
    args = []
    for k in bb._var.keys():
        if is_free_param(k):
            v = bb._var[k].flatten()
            args.append(v)

    # append the quantum operations
    with prog.context(*args) as (p, q):
        for op in bb.operations:
            # check if operation name is in the list of
            # defined StrawberryFields operations.
            # This is used by checking against the ops.py __all__
            # module attribute, which contains the names
            # of all defined quantum operations
            if op["op"] in ops.__all__:
                # get the quantum operation from the sf.ops module
                gate = getattr(ops, op["op"])
            else:
                raise NameError("Quantum operation {} not defined!".format(op["op"]))

            # create the list of regrefs
            regrefs = [q[i] for i in op["modes"]]

            if "args" in op:
                # the gate has arguments
                args = op["args"]
                kwargs = op["kwargs"]

                for i, p in enumerate(args):
                    if is_free_param(p):
                        args[i] = sfpar.FreeParameter(p)
                for k, v in kwargs.items():
                    if is_free_param(v):
                        kwargs[k] = sfpar.FreeParameter(v)

                # Convert symbolic expressions in args/kwargs containing measured and free parameters to
                # symbolic expressions containing the corresponding MeasuredParameter and FreeParameter instances.
                args = sfpar.par_convert(args, prog)
                vals = sfpar.par_convert(kwargs.values(), prog)
                kwargs = dict(zip(kwargs.keys(), vals))
                gate(*args, **kwargs) | regrefs  # pylint:disable=expression-not-assigned
            else:
                # the gate has no arguments
                gate | regrefs  # pylint:disable=expression-not-assigned,pointless-statement

    prog._target = bb.target["name"]

    if "shots" in bb.target["options"]:
        prog.run_options["shots"] = bb.target["options"]["shots"]

    if "cutoff_dim" in bb.target["options"]:
        prog.backend_options["cutoff_dim"] = bb.target["options"]["cutoff_dim"]

    return prog


def to_blackbird(prog: Program, version: str = "1.0") -> blackbird.BlackbirdProgram:
    """Convert a Strawberry Fields Program to a Blackbird Program.

    Args:
        prog (Program): the Strawberry Fields program
        version (str): Blackbird script version number

    Returns:
        blackbird.BlackbirdProgram:
    """
    bb = blackbird.BlackbirdProgram(name=prog.name, version=version)
    bb._modes = set(prog.reg_refs.keys())

    isMeasuredParameter = lambda x: isinstance(x, sfpar.MeasuredParameter)

    # not sure if this makes sense: the program has *already been* compiled using this target
    if prog.target is not None:
        # set the target
        bb._target["name"] = prog.target

        # set the run options
        if prog.run_options:
            bb._target["options"].update(prog.run_options)

        if prog.backend_options:
            bb._target["options"].update(prog.backend_options)

    # fill in the quantum circuit
    for cmd in prog.circuit:
        op = {"kwargs": {}, "args": []}

        op["op"] = cmd.op.__class__.__name__
        op["modes"] = [i.ind for i in cmd.reg]

        if "Measure" in op["op"]:
            # special case to take into account 'select' keyword argument
            if cmd.op.select is not None:
                op["kwargs"]["select"] = cmd.op.select

            if cmd.op.p:
                # argument is quadrature phase
                op["args"] = cmd.op.p

            if op["op"] == "MeasureFock":
                # special case to take into account 'dark_counts' keyword argument
                if cmd.op.dark_counts is not None:
                    op["kwargs"]["dark_counts"] = cmd.op.dark_counts

        else:
            for a in cmd.op.p:
                if sfpar.par_is_symbolic(a):
                    # SymPy object, convert to string
                    if any(map(isMeasuredParameter, a.free_symbols)):
                        # check if there are any measured parameters in `a`
                        a = blackbird.RegRefTransform(a)
                    else:
                        a = str(a)
                op["args"].append(a)

        # If program is a TDMProgram then add the looped-over arrays to the
        # blackbird program. `prog.loop_vars` are symbolic parameters (e.g.
        # `{p0}`), which should be replaced with `p.name` (e.g. `p0`) inside the
        # Blackbird operation (keyword) arguments.
        if isinstance(prog, TDMProgram):
            for p in prog.loop_vars:
                for i, ar in enumerate(op["args"]):
                    if str(p) == str(ar):
                        op["args"][i] = p.name
                for k, v in op["kwargs"].items():
                    if str(p) == str(v):
                        op["kwargs"][k] = p.name

        bb._operations.append(op)
    # add the specific "tdm" metadata to the Blackbird program
    if isinstance(prog, TDMProgram):
        bb._type["name"] = "tdm"
        bb._type["options"].update(
            {
                "temporal_modes": prog.timebins,
            }
        )
        bb._var.update(
            {f"{p.name}": np.array([prog.tdm_params[i]]) for i, p in enumerate(prog.loop_vars)}
        )

    return bb

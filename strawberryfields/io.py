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
Blackbird I/O
=============

This module contains functions for loading and saving Strawberry
Fields :class:`~.Program` objects from/to Blackbird scripts.

Summary
-------

.. autosummary::
    to_blackbird
    to_program
    save
    load

Code details
~~~~~~~~~~~~
"""
# pylint: disable=protected-access,too-many-nested-blocks
import os
import blackbird

from . import ops
from .program import (Program,)
from .program_utils import (RegRefTransform,)


def to_blackbird(prog, version="1.0"):
    """Convert a Strawberry Fields Program to a Blackbird Program.

    Args:
        prog (Program): the Strawberry Fields program
        version (str): Blackbird script version number

    Returns:
        blackbird.BlackbirdProgram:
    """
    bb = blackbird.BlackbirdProgram(name=prog.name, version=version)

    # TODO not sure if this makes sense: the program has *already been* compiled using this target
    if prog.target is not None:
        # set the target
        bb._target["name"] = prog.target

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
                op["kwargs"]["phi"] = cmd.op.p[0].x

        else:
            if cmd.op.p is not None:
                for a in cmd.op.p:
                    # check if reg ref transform
                    if isinstance(a.x, RegRefTransform):
                        # if a.x.func_str is not None:
                            # TODO: will not satisfy all use cases
                            # as the RegRefTransform string cannot be checked
                            # to determine if it is a valid function for serialization!
                            #
                            # Possible solutions:
                            #
                            #   * Use SymPy to represent functions, as
                            #     SymPy provides methods for converting to
                            #     Python functions as well as serialization
                            #
                            #   * Don't allow classical processing of measurements
                            #     on remote backends
                            #
                            # op["args"].append(a.x.func_str)
                        # else:
                        raise ValueError(
                            "The RegRefTransform in operation {} "
                            "is not supported by Blackbird.".format(cmd.op)
                        )
                    # else:
                    op["args"].append(a.x)
        bb._operations.append(op)

    return bb


def to_program(bb):
    """Convert a Blackbird Program to a Strawberry Fields Program.

    Args:
        bb (blackbird.BlackbirdProgram): the input Blackbird program object

    Returns:
        Program: corresponding SF program
    """
    # create a SF program
    if not bb.modes:
        # we can't return an empty program, since we don't know how many modes
        # to initialize the Program object with.
        raise ValueError("Blackbird program contains no quantum operations!")

    prog = Program(max(bb.modes)+1, name=bb.name)

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

            if 'args' in op:
                # the gate has arguments
                gate(*op["args"], **op["kwargs"]) | regrefs #pylint:disable=expression-not-assigned
            else:
                # the gate has no arguments
                gate | regrefs #pylint:disable=expression-not-assigned,pointless-statement

    # compile the program if a compile target is given
    targ = bb.target
    if targ["name"] is not None:
        prog = prog.compile(targ["name"], **targ["options"])

    return prog


def save(f, prog):
    """Saves a quantum program to a Blackbird .xbb file.

    Args:
        f (Union[file, str, pathlib.Path]): File or filename to which
            the data is saved. If file is a file-object, then the filename
            is unchanged. If file is a string or Path, a .xbb extension will
            be appended to the file name if it does not already have one.
        prog (Program): Strawberry Fields program
    """
    own_file = False
    bb = to_blackbird(prog).serialize()

    if hasattr(f, "read"):
        # argument file is a file-object
        fid = f
    else:
        # argument file is a string or Path
        filename = os.fspath(f)

        if not filename.endswith(".xbb"):
            filename = filename + ".xbb"

        fid = open(filename, "w")

        # this function owns the open file,
        # must remember to safely close it.
        own_file = True

    try:
        fid.write(bb)
    finally:
        if own_file:
            # safely close the file
            fid.close()


def load(f):
    """Load a quantum program from a Blackbird .xbb file.

    Args:
        f (Union[file, str, pathlib.Path]): File or filename to which
            the data is saved. If file is a file-object, then the filename
            is unchanged. If file is a string or Path, a .xbb extension will
            be appended to the file name if it does not already have one.

    Returns:
        prog (Program): Strawberry Fields program
    """
    own_file = False

    try:
        if hasattr(f, "read"):
            # argument file is a file-object
            fid = f
        else:
            # argument file is a Path or string
            filename = os.fspath(f)
            fid = open(filename, "r")
            own_file = True

    except TypeError:
        raise ValueError("file must be a string, pathlib.Path, or file-like object")

    try:
        bb_str = fid.read()
    finally:
        if own_file:
            # safely close the file
            fid.close()

    # load blackbird program
    bb = blackbird.loads(bb_str)
    return to_program(bb)

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

from numpy.compat import os_PathLike

import blackbird

from . import ops
from .program import Program


def to_blackbird(prog, version="1.0"):
    """Convert a Strawberry Fields Program to a Blackbird Program.

    Args:
        prog (Program): the Strawberry Fields program
        version (str): Blackbird script version number

    Returns:
        blackbird.BlackbirdProgram:
    """
    bb = blackbird.BlackbirdProgram(name=prog.name, version=version)

    if prog.backend is not None:
        # set the target
        bb._target["name"] = prog.backend

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
                    if isinstance(a.x, ops.RegRefTransform):
                        if a.x.func_str is not None:
                            # Note: not currently guaranteed to work,
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
                            op["args"].append(a.x.func_str)
                        else:
                            raise ValueError(
                                "The RegRefTransform in operation {} "
                                "is not supported by Blackbird.".format(cmd.op)
                            )
                    else:
                        op["args"].append(a.x)

        bb._operations.append(op)

    return bb


def to_program(bb):
    """Convert a Blackbird Program to a Strawberry Fields Program.

    Args:
        bb (blackbird.BlackbirdProgram): the input Blackbird program object

    Returns:
        program:
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
            try:
                if 'args' in op:
                    getattr(ops, op["op"])(*op["args"], **op["kwargs"]) | [q[i] for i in op["modes"]]
                else:
                    getattr(ops, op["op"]) | [q[i] for i in op["modes"]]
            except AttributeError:
                raise NameError("Quantum operation {} not defined!".format(op["op"]))

    # compile the program if a target exists
    if bb.target["name"] is not None:
        return prog.compile(bb.target["name"])

    return prog


def save(file, prog):
    """Saves a quantum program to a Blackbird .xbb file.

    Args:
        file (Union[file, str, pathlib.Path]): File or filename to which
            the data is saved. If file is a file-object, then the filename
            is unchanged. If file is a string or Path, a .xbb extension will
            be appended to the file name if it does not already have one.
        prog (Program): Strawberry Fields program
    """
    own_file = False
    bb = to_blackbird(prog).serialize()

    if hasattr(file, "read"):
        # argument file is a file-object
        fid = file
    else:
        # argument file is a string or Path
        f = os.fspath(file)

        if not f.endswith(".xbb"):
            f = f + ".xbb"

        fid = open(f, "w")

        # this function owns the open file,
        # must remember to safely close it.
        own_file = True

    try:
        fid.write(bb)
    finally:
        if own_file:
            # safely close the file
            fid.close()


def load(file):
    """Load a quantum program from a Blackbird .xbb file.

    Args:
        file (Union[file, str, pathlib.Path]): File or filename to which
            the data is saved. If file is a file-object, then the filename
            is unchanged. If file is a string or Path, a .xbb extension will
            be appended to the file name if it does not already have one.

    Returns:
        prog (Program): Strawberry Fields program
    """
    own_file = False

    try:
        if hasattr(file, "read"):
            # argument file is a file-object
            fid = file
        else:
            # argument file is a Path or string
            f = os.fspath(file)
            fid = open(f, "r")
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

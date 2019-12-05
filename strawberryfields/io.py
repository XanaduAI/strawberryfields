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
from .program import Program
from .parameters import par_is_symbolic, par_convert


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
                op["kwargs"]["phi"] = cmd.op.p[0]

        else:
            for a in cmd.op.p:
                if par_is_symbolic(a):
                    # SymPy object, convert to string
                    a = str(a)
                op["args"].append(a)

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
                args = op['args']
                kwargs = op['kwargs']

                # Convert symbolic expressions in args/kwargs containing measured and free parameters to
                # symbolic expressions containing the corresponding MeasuredParameter and FreeParameter instances.
                args = par_convert(args, prog)
                vals = par_convert(kwargs.values(), prog)
                kwargs = dict(zip(kwargs.keys(), vals))
                gate(*args, **kwargs) | regrefs  #pylint:disable=expression-not-assigned
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


def loads(s):
    """Load a quantum program from a string.

    Args:
        s (str): string containing the Blackbird circuit
    Returns:
        prog (Program): Strawberry Fields program

    """
    bb = blackbird.loads(s)
    return to_program(bb)

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
    return loads(bb_str)

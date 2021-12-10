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
This module contains functions for loading and saving Strawberry Fields :class:`~.Program` objects
and converting from/to Blackbird and XIR scripts to Strawberry Fields code.
"""
import os

from typing import Union, TextIO
from pathlib import Path

import xir
import blackbird

from strawberryfields.program import Program

from .blackbird_io import to_blackbird, from_blackbird, from_blackbird_to_tdm
from .xir_io import to_xir, from_xir, from_xir_to_tdm
from .utils import generate_code

# for automodapi, do not include the classes that should appear under the top-level strawberryfields namespace
__all__ = ["to_blackbird", "to_xir", "to_program", "loads", "generate_code"]


def to_program(prog: Union[blackbird.BlackbirdProgram, xir.Program]) -> Program:
    """Convert a Blackbird or an XIR program to a Strawberry Fields program.

    Args:
        prog (blackbird.BlackbirdProgram, xir.Program): the input program object

    Returns:
        Program: corresponding Strawberry Fields program

    Raises:
        ValueError: if the Blackbird program contains no quantum operations
        TypeError: if the program has an invalid type
    """
    if isinstance(prog, blackbird.BlackbirdProgram):
        if not prog.modes:
            # we can't return an empty program, since we don't know how many modes
            # to initialize the Program object with.
            raise ValueError("Blackbird program contains no quantum operations!")

        if prog.programtype["name"] == "tdm":
            return from_blackbird_to_tdm(prog)
        return from_blackbird(prog)

    if isinstance(prog, xir.Program):
        if prog.options.get("_type_") == "tdm":
            return from_xir_to_tdm(prog)
        return from_xir(prog)

    raise TypeError(f"Cannot convert {type(prog)}' to Strawberry Fields Program")


def save(f: Union[TextIO, str, Path], prog: Program, ir: str = "blackbird", **kwargs) -> None:
    """Saves a quantum program to a Blackbird ``.xbb`` or an XIR ``.xir`` file.

    **Example:**

    .. code-block:: python3

        prog = sf.Program(3)

        with prog.context as q:
            ops.Sgate(0.543) | q[1]
            ops.BSgate(0.6, 0.1) | (q[2], q[0])
            ops.MeasureFock() | q

        sf.save("program1.xbb", prog, ir="blackbird")

    This will create the following Blackbird file:

    .. code-block:: pycon

        >>> f = open("program1.xbb").read()
        >>> print(f)
        name None
        version 1.0

        Sgate(0.543, 0.0) | 1
        BSgate(0.6, 0.1) | [2, 0]
        MeasureFock() | [0, 1, 2]

    Args:
        f (Union[file, str, pathlib.Path]): File or filename to which
            the data is saved. If file is a file-object, then the filename
            is unchanged. If file is a string or Path, a .xbb extension will
            be appended to the file name if it does not already have one.
        prog (Program): Strawberry Fields program
        ir (str): Intermediate representation language to use. Can be either "blackbird" or "xir".

    Keyword Args:
        add_decl (bool): Whether gate and output declarations should be added to
            the XIR program. Default is ``False``.
    """
    own_file = False

    if ir == "xir":
        prog_str = to_xir(prog, **kwargs).serialize()
    elif ir == "blackbird":
        prog_str = to_blackbird(prog).serialize()
    else:
        raise ValueError(
            f"'{ir}' not recognized as a valid IR option. Valid options are 'xir' and 'blackbird'."
        )

    if hasattr(f, "read"):
        # argument file is a file-object
        fid = f
    else:
        # argument file is a string or Path
        filename = os.fspath(f)

        if ir == "blackbird" and not filename.endswith(".xbb"):
            filename = filename + ".xbb"
        elif ir == "xir" and not filename.endswith(".xir"):
            filename = filename + ".xir"

        fid = open(filename, "w")

        # this function owns the open file,
        # must remember to safely close it.
        own_file = True

    try:
        fid.write(prog_str)
    finally:
        if own_file:
            # safely close the file
            fid.close()


def loads(s: str, ir: str = "blackbird") -> Program:
    """Load a quantum program from a string.

    Args:
        s (str): string containing the Blackbird or XIR circuit
        ir (str): Intermediate representation language to use. Can be either "blackbird" or "xir".

    Returns:
        prog (Program): Strawberry Fields program

    Raises:
        ValueError: if an invalid IR name is passed
    """
    if ir == "xir":
        prog = xir.parse_script(s)
    elif ir == "blackbird":
        prog = blackbird.loads(s)
    else:
        raise ValueError(
            f"'{ir}' not recognized as a valid IR option. Valid options are 'xir' and 'blackbird'."
        )
    return to_program(prog)


def load(f: Union[TextIO, str, Path], ir: str = "blackbird") -> Program:
    """Load a quantum program from a Blackbird .xbb or an XIR .xir file.

    **Example:**

    The following Blackbird file, ``program1.xbb``,

    .. code-block:: python3

        name test_program
        version 1.0

        Sgate(0.543, 0.0) | 1
        BSgate(0.6, 0.1) | [2, 0]
        MeasureFock() | [0, 1, 2]

    can be imported into Strawberry Fields using the ``loads``
    function:

    >>> sf.loads("program1.xbb", ir="blackbird")
    >>> prog.name
    'test_program'
    >>> prog.num_subsystems
    3
    >>> prog.print()
    Sgate(0.543, 0) | (q[1])
    BSgate(0.6, 0.1) | (q[2], q[0])
    MeasureFock | (q[0], q[1], q[2])

    Args:
        f (Union[file, str, pathlib.Path]): File or filename from which
            the data is loaded. If file is a string or Path, a value with the
            .xbb extension is expected.
        ir (str): Intermediate representation language to use. Can be either "blackbird" or "xir".

    Returns:
        prog (Program): Strawberry Fields program

    Raises:
        ValueError: if file is not a string, pathlib.Path, or file-like object
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

    except TypeError as e:
        raise ValueError("file must be a string, pathlib.Path, or file-like object") from e

    try:
        prog_str = fid.read()
    finally:
        if own_file:
            # safely close the file
            fid.close()

    # load Blackbird or XIR program
    return loads(prog_str, ir=ir)

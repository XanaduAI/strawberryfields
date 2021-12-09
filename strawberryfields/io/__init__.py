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

from typing import Optional, Union, List, TextIO
from numbers import Number
from pathlib import Path

import xir
import blackbird
import numpy as np

from strawberryfields.tdm.tdmprogram import TDMProgram
from strawberryfields.program import Program
from strawberryfields.engine import Engine, RemoteEngine

from .blackbird_io import to_blackbird, from_blackbird, from_blackbird_to_tdm
from .xir_io import to_xir, from_xir, from_xir_to_tdm

# for automodapi, do not include the classes that should appear under the top-level strawberryfields namespace
__all__ = ["to_blackbird", "to_xir", "to_program", "loads"]


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


def generate_code(prog: Program, eng: Optional[Engine] = None) -> str:
    """Converts a Strawberry Fields program into valid Strawberry Fields code.

    **Example:**

    .. code-block:: python3

        prog = sf.Program(3)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})

        with prog.context as q:
            ops.Sgate(2*np.pi/3) | q[1]
            ops.BSgate(0.6, 0.1) | (q[2], q[0])
            ops.MeasureFock() | q

        results = eng.run(prog)

        code_str = sf.io.generate_code(prog, eng=eng)

    This will create the following string:

    .. code-block:: pycon

        >>> print(code_str)
        import strawberryfields as sf
        from strawberryfields import ops

        prog = sf.Program(3)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})

        with prog.context as q:
            ops.Sgate(2*np.pi/3, 0.0) | q[1]
            ops.BSgate(0.6, 0.1) | (q[2], q[0])
            ops.MeasureFock() | (q[0], q[1], q[2])

        results = eng.run(prog)

    Args:
        prog (Program): the Strawberry Fields program
        eng (Engine): The Strawberryfields engine. If ``None``, only the Program
            parts will be in the resulting code-string.

    Returns:
        str: the Strawberry Fields code, for constructing the program, as a string
    """
    code_seq = ["import strawberryfields as sf", "from strawberryfields import ops\n"]

    if isinstance(prog, TDMProgram):
        code_seq.append(f"prog = sf.TDMProgram(N={prog.N})")
    else:
        code_seq.append(f"prog = sf.Program({prog.num_subsystems})")

    # check if an engine is supplied; if so, format and add backend/target
    # along with backend options
    if eng:
        if isinstance(eng, RemoteEngine):
            code_seq.append(f'eng = sf.RemoteEngine("{eng.target}")')
        else:
            if "cutoff_dim" in eng.backend_options:
                formatting_str = (
                    f'"{eng.backend_name}", backend_options='
                    + f'{{"cutoff_dim": {eng.backend_options["cutoff_dim"]}}}'
                )
                code_seq.append(f"eng = sf.Engine({formatting_str})")
            else:
                code_seq.append(f'eng = sf.Engine("{eng.backend_name}")')

    # check if program is a TDMProgram and format the context as appropriate
    if isinstance(prog, TDMProgram):
        # if the context arrays contain pi-values, factor out multiples of np.pi
        tdm_params = [f"[{_factor_out_pi(par)}]" for par in prog.tdm_params]
        code_seq.append("\nwith prog.context(" + ", ".join(tdm_params) + ") as (p, q):")
    else:
        code_seq.append("\nwith prog.context as q:")

    # add the operations, and replace any free parameters with e.g. `p[0]`, `p[1]`
    for cmd in prog.circuit or []:
        name = cmd.op.__class__.__name__
        if isinstance(prog, TDMProgram):
            format_dict = {k: f"p[{k[1:]}]" for k in prog.parameters.keys()}
            params_str = _factor_out_pi(cmd.op.p).format(**format_dict)
        else:
            params_str = _factor_out_pi(cmd.op.p)

        modes = [f"q[{r.ind}]" for r in cmd.reg]
        if len(modes) == 1:
            modes_str = ", ".join(modes)
        else:
            modes_str = "(" + ", ".join(modes) + ")"
        op = f"    ops.{name}({params_str}) | {modes_str}"

        code_seq.append(op)

    if eng:
        code_seq.append("\nresults = eng.run(prog)")

    return "\n".join(code_seq)


def _factor_out_pi(num_list: List[Union[Number, str]], denominator: int = 12) -> str:
    """Factors out pi, divided by the denominator value, from all number in a list
    and returns a string representation.

    Args:
        num_list (list[Number, str]): a list of numbers and/or strings
        denominator (int): factor out pi divided by denominator;
            e.g. default would be to factor out np.pi/12

    Return:
        string: containing strings of values and/or input string objects
    """
    factor = np.pi / denominator

    a = []
    for p in num_list:
        # if list element is not a number then append its string representation
        if not isinstance(p, (int, float)):
            a.append(str(p))
            continue

        if np.isclose(p % factor, [0, factor]).any() and p != 0:
            gcd = np.gcd(int(p / factor), denominator)
            if gcd == denominator:
                if int(p / np.pi) == 1:
                    a.append("np.pi")
                else:
                    a.append(f"{int(p / np.pi)}*np.pi")
            else:
                coeff = int(p / factor / gcd)
                if coeff == 1:
                    a.append(f"np.pi/{int(denominator / gcd)}")
                else:
                    a.append(f"{coeff}*np.pi/{int(denominator / gcd)}")
        else:
            a.append(str(p))
    return ", ".join(a)


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

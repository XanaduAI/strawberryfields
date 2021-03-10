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
This module contains functions for loading and saving Strawberry Fields
:class:`~.Program` objects from/to Blackbird scripts and Strawberry Fields
code.
"""
# pylint: disable=protected-access,too-many-nested-blocks
import os
from numbers import Number

import numpy as np

import blackbird
import strawberryfields.program as sfp
import strawberryfields.parameters as sfpar
from . import ops


# for automodapi, do not include the classes that should appear under the top-level strawberryfields namespace
__all__ = ["to_blackbird", "to_program", "loads"]


# pylint:disable=too-many-branches
def to_blackbird(prog, version="1.0"):
    """Convert a Strawberry Fields Program to a Blackbird Program.

    Args:
        prog (Program): the Strawberry Fields program
        version (str): Blackbird script version number

    Returns:
        blackbird.BlackbirdProgram:
    """
    bb = blackbird.BlackbirdProgram(name=prog.name, version=version)
    bb._modes = list(prog.reg_refs.keys())

    # TODO not sure if this makes sense: the program has *already been* compiled using this target
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
                op["kwargs"]["phi"] = cmd.op.p[0]

            if op["op"] == "MeasureFock":
                # special case to take into account 'dark_counts' keyword argument
                if cmd.op.dark_counts is not None:
                    op["kwargs"]["dark_counts"] = cmd.op.dark_counts

        else:
            for a in cmd.op.p:
                if sfpar.par_is_symbolic(a):
                    # SymPy object, convert to string
                    a = str(a)
                op["args"].append(a)

        # If program type is "tdm" then add the looped-over arrays to the
        # blackbird program. `prog.loop_vars` are symbolic parameters (e.g.
        # `{p0}`), which should be replaced with `p.name` (e.g. `p0`) inside the
        # Blackbird operation (keyword) arguments.
        if prog.type == "tdm":
            for p in prog.loop_vars:
                for i, ar in enumerate(op["args"]):
                    if str(p) == str(ar):
                        op["args"][i] = p.name
                for k, v in op["kwargs"].items():
                    if str(p) == str(v):
                        op["kwargs"][k] = p.name

        bb._operations.append(op)
    # add the specific "tdm" metadata to the Blackbird program
    if prog.type == "tdm":
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

    if bb.programtype["name"] == "tdm":
        return _to_tdm_program(bb)

    prog = sfp.Program(max(bb.modes) + 1, name=bb.name)

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
def _to_tdm_program(bb):
    # pylint: disable=import-outside-toplevel
    from strawberryfields.tdm.tdmprogram import TDMProgram

    prog = TDMProgram(max(bb.modes) + 1, name=bb.name)

    def is_free_param(param):
        return isinstance(param, str) and param[0] == "p" and param[1:].isdigit()

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


def generate_code(prog, eng=None):
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

    if prog.type == "tdm":
        code_seq.append(f"prog = sf.TDMProgram(N={prog.N})")
    else:
        code_seq.append(f"prog = sf.Program({prog.num_subsystems})")

    # check if an engine is supplied; if so, format and add backend/target
    # along with backend options
    if eng:
        eng_type = eng.__class__.__name__
        if eng_type == "RemoteEngine":
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

    # check if program is of TDM type and format the context as appropriate
    if prog.type == "tdm":
        # if the context arrays contain pi-values, factor out multiples of np.pi
        tdm_params = [f"[{_factor_out_pi(par)}]" for par in prog.tdm_params]
        code_seq.append("\nwith prog.context(" + ", ".join(tdm_params) + ") as (p, q):")
    else:
        code_seq.append("\nwith prog.context as q:")

    # add the operations, and replace any free parameters with e.g. `p[0]`, `p[1]`
    for cmd in prog.circuit:
        name = cmd.op.__class__.__name__
        if prog.type == "tdm":
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


def _factor_out_pi(num_list, denominator=12):
    """Factors out pi, divided by the denominator value, from all number in a list
    and returns a string representation.

    Args:
        num_list (list[Number, string]): a list of numbers and/or strings
        denominator (int): factor out pi divided by denominator;
            e.g. default would be to factor out np.pi/12

    Return:
        string: containing strings of values and/or input string objects
    """
    factor = np.pi / denominator

    a = []
    for p in num_list:
        # if list element is not a number then append its string representation
        if not isinstance(p, Number):
            a.append(str(p))
            continue

        if np.isclose(p % factor, [0, factor]).any() and p != 0:
            gcd = np.gcd(int(p / factor), denominator)
            if gcd == denominator:
                if int(p / np.pi) == 1:
                    a.append("np.pi")
                else:
                    a.append(f"{int(p/np.pi)}*np.pi")
            else:
                coeff = int(p / factor / gcd)
                if coeff == 1:
                    a.append(f"np.pi/{int(denominator/gcd)}")
                else:
                    a.append(f"{coeff}*np.pi/{int(denominator / gcd)}")
        else:
            a.append(str(p))
    return ", ".join(a)


def save(f, prog):
    """Saves a quantum program to a Blackbird .xbb file.

    **Example:**

    .. code-block:: python3

        prog = sf.Program(3)

        with prog.context as q:
            ops.Sgate(0.543) | q[1]
            ops.BSgate(0.6, 0.1) | (q[2], q[0])
            ops.MeasureFock() | q

        sf.save("program1.xbb", prog)

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

    >>> sf.loads("program1.xbb")
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

    except TypeError as e:
        raise ValueError("file must be a string, pathlib.Path, or file-like object") from e

    try:
        bb_str = fid.read()
    finally:
        if own_file:
            # safely close the file
            fid.close()

    # load blackbird program
    return loads(bb_str)

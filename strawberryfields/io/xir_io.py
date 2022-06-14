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
:class:`~.Program` objects from/to XIR scripts and Strawberry Fields
code.
"""
# pylint: disable=protected-access,too-many-nested-blocks
from decimal import Decimal
from typing import Iterable, List, Sequence

import numpy as np

import xir

import strawberryfields.parameters as sfpar
from strawberryfields.program import Program
from strawberryfields.tdm import TDMProgram, is_ptype
from strawberryfields import ops


def get_expanded_statements(prog: xir.Program) -> Sequence[xir.Statement]:
    """Get a list of statements with all gate definitions expanded.

    Args:
        prog (xir.Program): XIR program with statements and definitions

    Returns:
        list[xir.Statement]: list of expanded XIR statements
    """

    def expand_statements(statements: Sequence[xir.Statement]) -> Sequence[xir.Statement]:
        flattened_statements = []
        for op in statements:
            sub_statements = expand_statements(prog.gates.get(op.name, []))
            if sub_statements:
                wire_mapping = dict(zip(prog.search("gate", "wires", op.name), op.wires))
                param_mapping = dict(zip(prog.search("gate", "params", op.name), op.params))

                # create a new statement object with substituted parameters and wires
                for stmt in sub_statements:
                    wires = tuple(wire_mapping[w] for w in stmt.wires)
                    params = [param_mapping[w] for w in stmt.params]
                    flattened_statements.append(xir.Statement(stmt.name, params, wires))
            else:
                flattened_statements.append(op)
        return flattened_statements

    return expand_statements(prog.statements)


# pylint: disable=too-many-branches
def from_xir(xir_prog: xir.Program) -> Program:
    """Convert an XIR Program to a Strawberry Fields program.

    Args:
        xir_prog (xir.Program): the input XIR program object

    Returns:
        Program: corresponding Strawberry Fields program

    Raises:
        ValueError: if the XIR program is empty
    """
    # only script-level statements are part of `xir_prog.statements`, which can only have integer
    # wires, leading to `xir_prog.wires` only containing integer wire labels
    if not xir_prog.wires:
        raise ValueError(
            "The XIR program is empty and cannot be transformed "
            "into a Strawberry Fields program."
        )

    num_of_modes = int(max(xir_prog.wires)) + 1
    name = xir_prog.options.get("_name_", "sf_from_xir")
    prog = Program(num_of_modes, name=name)

    # append the quantum operations
    with prog.context as q:
        for op in get_expanded_statements(xir_prog):
            # check if operation name is in the list of
            # defined StrawberryFields operations.
            # This is used by checking against the ops.py __all__
            # module attribute, which contains the names
            # of all defined quantum operations
            if op.name in ops.__all__:
                # get the quantum operation from the sf.ops module
                gate = getattr(ops, op.name)
            else:
                raise NameError(f"Quantum operation {op.name!r} not defined!")

            # create the list of regrefs
            regrefs = [q[i] for i in op.wires]

            if op.params:
                # convert symbolic expressions to symbolic expressions containing the corresponding
                # MeasuredParameter and FreeParameter instances.
                if isinstance(op.params, dict):
                    vals = sfpar.par_convert(op.params.values(), prog)
                    params = dict(zip(op.params.keys(), vals))
                    gate(**params) | regrefs  # pylint:disable=expression-not-assigned
                else:
                    params = []
                    for p in op.params:
                        if isinstance(p, Decimal):
                            params.append(float(p))
                        elif isinstance(p, Iterable):
                            params.append(np.array(_listr(p)))
                        else:
                            params.append(p)
                    params = sfpar.par_convert(params, prog)
                    gate(*params) | regrefs  # pylint:disable=expression-not-assigned
            else:
                gate() | regrefs  # pylint:disable=expression-not-assigned,pointless-statement

    prog._target = xir_prog.options.get("_target_", None)  # pylint: disable=protected-access

    if "shots" in xir_prog.options:
        prog.run_options["shots"] = xir_prog.options["shots"]
    if "cutoff_dim" in xir_prog.options:
        prog.backend_options["cutoff_dim"] = xir_prog.options["cutoff_dim"]

    return prog


# pylint: disable=too-many-branches
def from_xir_to_tdm(xir_prog: xir.Program) -> TDMProgram:
    """Convert an XIR Program to a ``TDMProgram``.

    Args:
        xir_prog (xir.Program): the input XIR program object

    Returns:
        TDMProgram: corresponding ``TDMProgram``

    Raises:
        ValueError: if the number of modes 'N' is missing from the XIR program options
        NameError: if an applied quantum operation is not defined in Strawberry Fields
    """
    N = xir_prog.options.get("N")
    if not N:
        raise ValueError("Number of modes 'N' is missing from the XIR program options.")

    prog = TDMProgram(N, name=xir_prog.options.get("_name_", "xir"))

    # extract the tdm gate arguments from the xir program constants
    args = [val for key, val in xir_prog.constants.items() if is_ptype(key)]

    # convert arguments to float/complex if stored as Decimal/DecimalComplex objects
    for i, params in enumerate(args):
        for j, p in enumerate(params):
            if isinstance(p, Decimal):
                args[i][j] = float(p)
            elif isinstance(p, xir.DecimalComplex):
                args[i][j] = complex(p)

    # append the quantum operations
    with prog.context(*args) as (p, q):
        for op in get_expanded_statements(xir_prog):
            # check if operation name is in the list of
            # defined StrawberryFields operations.
            # This is used by checking against the ops.py __all__
            # module attribute, which contains the names
            # of all defined quantum operations
            if op.name in ops.__all__:
                # get the quantum operation from the sf.ops module
                gate = getattr(ops, op.name)
            else:
                raise NameError(f"Quantum operation {op.name!r} not defined!")

            # create the list of regrefs
            regrefs = [q[int(i)] for i in op.wires]

            if op.params:
                # convert symbolic expressions to symbolic expressions containing the corresponding
                # MeasuredParameter and FreeParameter instances.
                if isinstance(op.params, dict):
                    vals = sfpar.par_convert(op.params.values(), prog)
                    params = dict(zip(op.params.keys(), vals))
                    for key, val in params.items():
                        if is_ptype(val):
                            params[key] = p[int(val[1:])]
                    gate(**params) | regrefs  # pylint:disable=expression-not-assigned
                else:
                    params = []
                    for param in op.params:
                        if isinstance(param, Decimal):
                            params.append(float(param))
                        elif isinstance(param, (list, np.ndarray)):
                            params.append(np.array(_listr(param)))
                        elif isinstance(param, str) and is_ptype(param):
                            params.append(p[int(param[1:])])
                        else:
                            params.append(param)
                    params = sfpar.par_convert(params, prog)
                    gate(*params) | regrefs  # pylint:disable=expression-not-assigned
            else:
                gate() | regrefs  # pylint:disable=expression-not-assigned,pointless-statement

    prog._target = xir_prog.options.get("target", None)  # pylint: disable=protected-access

    if "shots" in xir_prog.options:
        prog.run_options["shots"] = xir_prog.options["shots"]

    return prog


def to_xir(prog: Program, **kwargs) -> xir.Program:
    """Convert a Strawberry Fields Program to an XIR Program.

    Args:
        prog (Program): the Strawberry Fields program

    Keyword Args:
        add_decl (bool): Whether gate and output declarations should be added to
            the XIR program. Default is ``False``.

    Returns:
        xir.Program
    """
    xir_prog = xir.Program()
    add_decl = kwargs.get("add_decl", False)

    if isinstance(prog, TDMProgram):
        xir_prog.add_option("_type_", "tdm")
        xir_prog.add_option("N", prog.N)
        for i, p in enumerate(prog.tdm_params):
            xir_prog.add_constant(f"p{i}", _listr(p))

    if prog.name:
        xir_prog.add_option("_name_", prog.name)
    if prog.target:
        xir_prog.add_option("target", prog.target)  # pylint: disable=protected-access
    if "cutoff_dim" in prog.backend_options:
        xir_prog.add_option("cutoff_dim", prog.backend_options["cutoff_dim"])
    if "shots" in prog.run_options:
        xir_prog.add_option("shots", prog.run_options["shots"])

    # fill in the quantum circuit
    for cmd in prog.circuit or []:

        name = cmd.op.__class__.__name__
        wires = tuple(i.ind for i in cmd.reg)

        if "Measure" in name:
            if add_decl:
                output_decl = xir.Declaration(name, type_="out", wires=wires)
                xir_prog.add_declaration(output_decl)

            params = {}
            if cmd.op.p:
                # argument is quadrature phase
                a = cmd.op.p[0]
                if a in getattr(prog, "loop_vars", ()):
                    params["phi"] = a.name
                else:
                    params["phi"] = a

            # special case to take into account 'select' keyword argument
            if cmd.op.select is not None:
                params["select"] = cmd.op.select

            if name == "MeasureFock":
                # special case to take into account 'dark_counts' keyword argument
                if cmd.op.dark_counts is not None:
                    params["dark_counts"] = cmd.op.dark_counts
        else:
            if add_decl:
                if name not in [gdecl.name for gdecl in xir_prog.declarations["gate"]]:
                    params = [f"p{i}" for i, _ in enumerate(cmd.op.p)]
                    gate_decl = xir.Declaration(
                        name, type_="gate", params=params, wires=tuple(range(len(wires)))
                    )
                    xir_prog.add_declaration(gate_decl)

            params = []
            for i, a in enumerate(cmd.op.p):
                if sfpar.par_is_symbolic(a):
                    # try to evaluate symbolic parameter
                    try:
                        a = sfpar.par_evaluate(a)
                    except sfpar.ParameterError:
                        # if a tdm param
                        if a in getattr(prog, "loop_vars", ()):
                            a = a.name
                        # if a pure symbol (free parameter), convert to string
                        elif a.is_symbol:
                            a = a.name
                        # else, assume it's a symbolic function and replace all free parameters
                        # with string representations
                        else:
                            symbolic_func = a.copy()
                            for s in symbolic_func.free_symbols:
                                symbolic_func = symbolic_func.subs(s, s.name)
                            a = str(symbolic_func)

                elif isinstance(a, str):
                    pass
                elif isinstance(a, Iterable):
                    # if an iterable, make sure it only consists of lists and Python types
                    a = _listr(a)
                params.append(a)

        op = xir.Statement(name, params, wires)
        xir_prog.add_statement(op)

    return xir_prog


def _listr(mixed_iterable: Iterable) -> List:
    """Casts a nested iterable to a list recursively, maintaining the same shape.

    Any iterable will be cast to a list, including casting all internal types to native Python
    types (e.g., ``Decimal`` and ``np.floating`` to ``float``); Python strings will be cast to
    lists of strings containing a single character each.

    .. warning:

        Currently, strings cannot be passed to the function; an error will be raised if
        ``mixed_iterable`` is of type ``str``.
    """
    if isinstance(mixed_iterable, str):
        raise TypeError("Strings cannot be passed to _listr().")

    list_ = []

    for l in mixed_iterable:
        # if string, then create a list of chars
        if isinstance(l, str):
            list_.append(l)
        elif isinstance(l, Iterable):
            list_.append(_listr(l))
        else:
            if isinstance(l, (Decimal, np.floating)):
                list_.append(float(l))
            elif isinstance(l, (xir.DecimalComplex, np.complexfloating)):
                list_.append(complex(l))
            else:
                try:
                    # if a NumPy-like object, extract the internal object
                    # with native Python type (e.g., `np.int` to Python `int`)
                    list_.append(l.item())
                except AttributeError:
                    list_.append(l)

    return list_

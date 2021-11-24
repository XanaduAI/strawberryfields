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
from decimal import Decimal
from typing import Iterable

import numpy as np

import xir

import strawberryfields.program as sfp
from strawberryfields.tdm.tdmprogram import TDMProgram
import strawberryfields.parameters as sfpar
from strawberryfields import ops


# pylint: disable=too-many-branches
# TODO: fix TDM conversion: need to use kwargs
def from_xir(xir_prog):
    """Convert an XIR Program to a Strawberry Fields program.

    Args:
        xir_prog (xir.Program): the input XIR program object

    Returns:
        Program: corresponding Strawberry Fields program
    """
    # only script-level statements are part of `xir_prog.statements`, which can only have integer
    # wires, leading to `xir_prog.wires` only containing integer wire labels
    num_of_modes = int(max(xir_prog.wires or [-1])) + 1
    name = xir_prog.options.get("name", "xir")
    if num_of_modes == 0:
        raise ValueError(
            "The XIR program is empty and cannot be transformed into a Strawberry Fields program"
        )
    prog = sfp.Program(num_of_modes, name=name)

    # append the quantum operations
    with prog.context as q:
        for op in xir_prog.statements:
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
                if callable(gate):
                    gate() | regrefs  # pylint:disable=expression-not-assigned,pointless-statement
                else:
                    gate | regrefs  # pylint:disable=expression-not-assigned,pointless-statement

    prog._target = xir_prog.options.get("target", None)  # pylint: disable=protected-access

    if "shots" in xir_prog.options:
        prog.run_options["shots"] = xir_prog.options["shots"]
    if "cutoff_dim" in xir_prog.options:
        prog.backend_options["cutoff_dim"] = xir_prog.options["cutoff_dim"]

    return prog

# pylint: disable=too-many-branches
# TODO: fix this function; need to use kwargs from to_xir function above
def from_xir_to_tdm(xir_prog):
    prog = TDMProgram(max(xir_prog.wires) + 1, name=xir_prog.options.get("name", "xir"))

    args = []
    for stmt in xir_prog.statements:
        for p in stmt.params:
            if isinstance(p, list):
                args.append(p)
            elif p == "phi":
                args.append(stmt.params[p])

    # append the quantum operations
    with prog.context(*args) as (p, q):
        for op in xir_prog.statements:
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
                if callable(gate):
                    gate() | regrefs  # pylint:disable=expression-not-assigned,pointless-statement
                else:
                    gate | regrefs  # pylint:disable=expression-not-assigned,pointless-statement

    prog._target = xir_prog.options.get("target", None)  # pylint: disable=protected-access

    if "shots" in xir_prog.options:
        prog.run_options["shots"] = xir_prog.options["shots"]
    if "cutoff_dim" in xir_prog.options:
        prog.backend_options["cutoff_dim"] = xir_prog.options["cutoff_dim"]

    return prog


def to_xir(prog, **kwargs):
    """Convert a Strawberry Fields Program to an XIR Program.

    Args:
        prog (Program): the Strawberry Fields program

    Keyword Args:
        add_decl (bool): Whether gate and output declarations should be added to
            the XIR program. Default is ``False``.
        version (str): Version number for the program. Default is ``0.1.0``.

    Returns:
        xir.Program
    """
    version = kwargs.get("version", "0.1.0")
    xir_prog = xir.Program(version=version)

    if isinstance(prog, TDMProgram):
        xir_prog.add_option("type", "tdm")

    if prog._target:
        xir_prog.add_option("target", prog._target)  # pylint: disable=protected-access
    if "cutoff_dim" in prog.run_options:
        xir_prog.add_option("cutoff_dim", prog.run_options["cutoff_dim"])
    if "name" in prog.run_options:
        xir_prog.add_option("name", prog.run_options["name"])
    if "shots" in prog.run_options:
        xir_prog.add_option("shots", prog.run_options["shots"])

    # fill in the quantum circuit
    for cmd in prog.circuit or []:

        name = cmd.op.__class__.__name__
        wires = tuple(i.ind for i in cmd.reg)

        if "Measure" in name:
            if kwargs.get("add_decl", False):
                output_decl = xir.Declaration(name, type_="out")
                xir_prog.add_declaration(output_decl)

            params = {}
            if cmd.op.p:
                # argument is quadrature phase
                a = cmd.op.p[0]
                if a in getattr(prog, "loop_vars", ()):
                    a = _listr(prog.tdm_params[prog.loop_vars.index(a)])
                params["phi"] = a

            # special case to take into account 'select' keyword argument
            if cmd.op.select is not None:
                params["select"] = cmd.op.select

            if name == "MeasureFock":
                # special case to take into account 'dark_counts' keyword argument
                if cmd.op.dark_counts is not None:
                    params["dark_counts"] = cmd.op.dark_counts
        else:
            if kwargs.get("add_decl", False):
                if name not in [gdecl.name for gdecl in xir_prog.declarations["gate"]]:
                    params = [f"p{i}" for i, _ in enumerate(cmd.op.p)]
                    gate_decl = xir.Declaration(
                        name, type_="gate", params=params, wires=tuple(range(len(wires)))
                    )
                    xir_prog.add_declaration(gate_decl)

            params = []
            for a in cmd.op.p:
                if sfpar.par_is_symbolic(a):
                    # try to evaluate symbolic parameter
                    try:
                        a = sfpar.par_evaluate(a)
                    except sfpar.ParameterError:
                        # if a tdm param
                        if a in getattr(prog, "loop_vars", ()):
                            a = _listr(prog.tdm_params[prog.loop_vars.index(a)])
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
                elif isinstance(a, Iterable):
                    # if an iterable, make sure it only consists of lists and Python types
                    a = _listr(a)
                params.append(a)

        op = xir.Statement(name, params, wires)
        xir_prog.add_statement(op)

    return xir_prog


def _listr(mixed_list):
    """Casts a nested iterable to a list recursively"""
    list_ = []

    for l in mixed_list:
        if isinstance(l, Iterable):
            list_.append(_listr(l))
        else:
            if isinstance(l, Decimal):
                list_.append(float(l))
            elif isinstance(l, xir.DecimalComplex):
                list_.append(complex(l))
            else:
                try:
                    list_.append(l.item())
                except AttributeError:
                    list_.append(l)

    return list_

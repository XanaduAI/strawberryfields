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
"""Utility functions for the io module."""
from __future__ import annotations

from typing import List, Union
from numbers import Number

import numpy as np

from strawberryfields.program import Program
from strawberryfields.tdm import TDMProgram


def generate_code(prog: Program, eng=None) -> str:
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

        code_str = sf.io.utils.generate_code(prog, eng=eng)

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
        if hasattr(eng, "connection"):
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
    """Factors out pi, divided by the denominator value, from all numbers in a list
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

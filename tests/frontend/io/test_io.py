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
r"""Unit tests for the IO module"""
import pytest
import textwrap

import numpy as np
import blackbird

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields import io

pytestmark = pytest.mark.frontend


@pytest.fixture
def eng(backend):
    """Engine fixture."""
    return sf.LocalEngine(backend)


prog_txt = textwrap.dedent(
    """\
    import strawberryfields as sf
    from strawberryfields import ops

    prog = sf.Program(3)
    eng = sf.Engine({engine_args})

    with prog.context as q:
        ops.Sgate(0.54, 0) | q[0]
        ops.BSgate(0.45, np.pi/2) | (q[0], q[2])
        ops.Sgate(3*np.pi/2, 0) | q[1]
        ops.BSgate(2*np.pi, 0.62) | (q[0], q[1])
        ops.MeasureFock() | q[0]

    results = eng.run(prog)
"""
)

prog_txt_no_engine = textwrap.dedent(
    """\
    import strawberryfields as sf
    from strawberryfields import ops

    prog = sf.Program(3)

    with prog.context as q:
        ops.Sgate(0.54, 0) | q[0]
        ops.BSgate(0.45, np.pi/2) | (q[0], q[2])
        ops.Sgate(3*np.pi/2, 0) | q[1]
        ops.BSgate(2*np.pi, 0.62) | (q[0], q[1])
        ops.MeasureFock() | q[0]
"""
)

prog_txt_tdm = textwrap.dedent(
    """\
    import strawberryfields as sf
    from strawberryfields import ops

    prog = sf.TDMProgram(N=[2, 3])
    eng = sf.Engine("gaussian")

    with prog.context([np.pi, 3*np.pi/2, 0], [1, 0.5, np.pi], [0, 0, 0]) as (p, q):
        ops.Sgate(0.123, np.pi/4) | q[2]
        ops.BSgate(p[0], 0.0) | (q[1], q[2])
        ops.Rgate(p[1]) | q[2]
        ops.MeasureHomodyne(p[0]) | q[0]
        ops.MeasureHomodyne(p[2]) | q[2]

    results = eng.run(prog)
"""
)


class TestGenerateCode:
    """Tests for the generate_code function"""

    def test_generate_code_no_engine(self):
        """Test generating code for a regular program with no engine"""
        prog = sf.Program(3)

        with prog.context as q:
            ops.Sgate(0.54, 0) | q[0]
            ops.BSgate(0.45, np.pi / 2) | (q[0], q[2])
            ops.Sgate(3 * np.pi / 2, 0) | q[1]
            ops.BSgate(2 * np.pi, 0.62) | (q[0], q[1])
            ops.MeasureFock() | q[0]

        code = io.generate_code(prog)

        code_list = code.split("\n")
        expected = prog_txt_no_engine.split("\n")

        for i, row in enumerate(code_list):
            assert row == expected[i]

    @pytest.mark.parametrize(
        "engine_kwargs",
        [
            {"backend": "fock", "backend_options": {"cutoff_dim": 5}},
            {"backend": "gaussian"},
        ],
    )
    def test_generate_code_with_engine(self, engine_kwargs):
        """Test generating code for a regular program with an engine"""
        prog = sf.Program(3)
        eng = sf.Engine(**engine_kwargs)

        with prog.context as q:
            ops.Sgate(0.54, 0) | q[0]
            ops.BSgate(0.45, np.pi / 2) | (q[0], q[2])
            ops.Sgate(3 * np.pi / 2, 0) | q[1]
            ops.BSgate(2 * np.pi, 0.62) | (q[0], q[1])
            ops.MeasureFock() | q[0]

        results = eng.run(prog)

        code = io.generate_code(prog, eng)

        code_list = code.split("\n")
        formatting_str = f"\"{engine_kwargs['backend']}\""
        if "backend_options" in engine_kwargs:
            formatting_str += (
                ", backend_options="
                f'{{"cutoff_dim": {engine_kwargs["backend_options"]["cutoff_dim"]}}}'
            )
        expected = prog_txt.format(engine_args=formatting_str).split("\n")

        for i, row in enumerate(code_list):
            assert row == expected[i]

    def test_generate_code_tdm(self):
        """Test generating code for a TDM program with an engine"""
        prog = sf.TDMProgram(N=[2, 3])
        eng = sf.Engine("gaussian")

        with prog.context([np.pi, 3 * np.pi / 2, 0], [1, 0.5, np.pi], [0, 0, 0]) as (p, q):
            ops.Sgate(0.123, np.pi / 4) | q[2]
            ops.BSgate(p[0]) | (q[1], q[2])
            ops.Rgate(p[1]) | q[2]
            ops.MeasureHomodyne(p[0]) | q[0]
            ops.MeasureHomodyne(p[2]) | q[2]

        results = eng.run(prog)

        code = io.generate_code(prog, eng)

        code_list = code.split("\n")
        expected = prog_txt_tdm.split("\n")

        for i, row in enumerate(code_list):
            assert row == expected[i]

    @pytest.mark.parametrize(
        "value",
        [
            "np.pi",
            "np.pi/2",
            "np.pi/12",
            "2*np.pi",
            "2*np.pi/3",
            "0",
            "42",
            "9.87",
            "-0.2",
            "no_number",
            "{p3}",
        ],
    )
    def test_factor_out_pi(self, value):
        """Test that the factor_out_pi function is able to convert floats
        that are equal to a pi expression, to strings containing a pi
        expression.

        For example, the float 6.28318530718 should be converted to the string "2*np.pi"
        """
        try:
            val = eval(value)
        except NameError:
            val = value
        res = sf.io.utils._factor_out_pi([val])
        expected = value

        assert res == expected


class DummyResults:
    """Dummy results object"""


def dummy_run(self, program, args, compile_options=None, **eng_run_options):
    """A dummy run function, that when called returns a dummy
    results object, with run options available as an attribute.
    This allows run_options to be returned and inspected after calling eng.run
    """
    results = DummyResults()
    results.run_options = eng_run_options
    return results


class TestEngineIntegration:
    """Test that target options interface correctly with eng.run"""

    def test_shots(self, eng, monkeypatch):
        """Test that passing shots correctly propagates to an engine run"""
        bb_script = """\
        name test_program
        version 1.0
        target gaussian (shots=100)
        Pgate(0.54) | 0
        MeasureX | 0
        """
        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert prog.run_options == {"shots": 100}

        with monkeypatch.context() as m:
            m.setattr("strawberryfields.engine.BaseEngine._run", dummy_run)
            results = eng.run(prog)

        assert results.run_options == {"shots": 100}

    def test_shots_overwritten(self, eng, monkeypatch):
        """Test if run_options are passed to eng.run, they
        overwrite those stored in the compiled program"""
        bb_script = """\
        name test_program
        version 1.0
        target gaussian (shots=100)
        Pgate(0.54) | 0
        MeasureX | 0
        """
        bb = blackbird.loads(bb_script)
        prog = io.to_program(bb)

        assert prog.run_options == {"shots": 100}

        with monkeypatch.context() as m:
            m.setattr("strawberryfields.engine.BaseEngine._run", dummy_run)
            results = eng.run(prog, shots=1000)

        assert results.run_options == {"shots": 1000}

    def test_program_sequence(self, eng, monkeypatch):
        """Test that program run_options are successively
        updated if a sequence of programs is executed."""
        bb_script = """\
        name test_program
        version 1.0
        target gaussian (shots=100)
        Pgate(0.54) | 0
        MeasureX | 0
        """
        bb = blackbird.loads(bb_script)
        prog1 = io.to_program(bb)

        bb_script = """\
        name test_program
        version 1.0
        target gaussian (shots=1024)
        Pgate(0.54) | 0
        MeasureX | 0
        """
        bb = blackbird.loads(bb_script)
        prog2 = io.to_program(bb)

        assert prog1.run_options == {"shots": 100}
        assert prog2.run_options == {"shots": 1024}

        with monkeypatch.context() as m:
            m.setattr("strawberryfields.engine.BaseEngine._run", dummy_run)
            results = eng.run([prog1, prog2])

        assert results.run_options == prog2.run_options

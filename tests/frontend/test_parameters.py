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
"""Unit tests for the parameters.py module."""

import pytest
import numpy as np

import strawberryfields as sf
from strawberryfields.parameters import (
    par_is_symbolic,
    par_regref_deps,
    par_str,
    par_evaluate,
    MeasuredParameter,
    FreeParameter,
    par_funcs as pf,
    ParameterError,
)
from strawberryfields.program_utils import RegRef


pytestmark = pytest.mark.frontend


# make test deterministic
np.random.seed(32)

SCALAR_TEST_VALUES = [3, 0.14, 4.2 + 0.5j]
TEST_VALUES = SCALAR_TEST_VALUES + [np.array([0.1, 0.987654])]


def binary_arithmetic(pp, qq, p, q):
    """Test the correctness of basic binary arithmetic expressions."""
    assert par_evaluate(pp + qq) == pytest.approx(p + q)
    assert par_evaluate(pp - qq) == pytest.approx(p - q)
    assert par_evaluate(pp * qq) == pytest.approx(p * q)
    assert par_evaluate(pp / qq) == pytest.approx(p / q)
    assert par_evaluate(pp ** qq) == pytest.approx(p ** q)


class TestParameter:
    """Basic parameter functionality."""

    def test_measured_par(self):
        """Compare two ways of creating measured parameter instances from RegRefs."""
        r = RegRef(0)
        s = RegRef(1)
        p = MeasuredParameter(r)
        assert p == r.par
        assert p != s.par

    @pytest.mark.parametrize("r", TEST_VALUES)
    def test_par_is_symbolic(self, r):
        """Recognizing symbolic parameters."""
        p = FreeParameter("x")
        q = MeasuredParameter(RegRef(0))

        assert not par_is_symbolic(r)
        assert par_is_symbolic(pf.sin(r))
        assert par_is_symbolic(q)
        assert par_is_symbolic(p)
        assert par_is_symbolic(pf.sin(p))
        assert par_is_symbolic(p + r)
        assert par_is_symbolic(p - r)
        assert par_is_symbolic(p * r)
        assert par_is_symbolic(p / r)
        assert par_is_symbolic(p ** r)
        assert par_is_symbolic(p - p)  # no simplification

        # object array with symbols
        a = np.array([[0.1, 3, 0], [0.3, 2, p], [1, 2, 4]])
        assert a.dtype == object
        assert par_is_symbolic(a)

        # object array, no symbols
        a = np.array([[0.1, 3, 0], [0.3, 2, 0], [1, 2, 4]], dtype=object)
        assert a.dtype == object
        assert not par_is_symbolic(a)

        # float array, no symbols
        a = np.array([[0.1, 3, 0], [0.3, 2, 0], [1, 2, 4]])
        assert a.dtype != object
        assert not par_is_symbolic(a)
        assert par_is_symbolic(pf.sin(a))

    def test_par_regref_deps(self):
        """RegRef dependencies of parameters."""
        r = [RegRef(k) for k in range(2)]
        R0 = set([r[0]])
        R = set(r)

        s = 0.4
        x = FreeParameter("x")
        p = MeasuredParameter(r[0])
        q = MeasuredParameter(r[1])

        assert par_regref_deps(s) == set()
        assert par_regref_deps(x) == set()
        assert par_regref_deps(p) == R0
        assert par_regref_deps(s * p) == R0
        assert par_regref_deps(x * p) == R0
        assert par_regref_deps(p + q) == R
        assert par_regref_deps(s * p + q ** x) == R

    def test_par_str(self):
        """String representations of parameters."""

        a = 0.1234567
        b = np.array([0, 0.987654])
        c = MeasuredParameter(RegRef(1))
        d = FreeParameter("x")

        assert par_str(a) == "0.1235"  # rounded to 4 decimals
        assert par_str(b) == "[0.     0.9877]"
        assert par_str(c) == "q1"
        assert par_str(d) == "{x}"
        assert (
            par_str(b * d) == "[0 0.987654*{x}]"
        )  # not rounded to 4 decimals due to Sympy's internal settings (object array!)

    def test_raw_parameter_printing(self):
        """Raw printing of parameter expressions."""

        c = MeasuredParameter(RegRef(1))
        d = FreeParameter("x")

        assert str(c) == "q1"
        assert str(d) == "{x}"
        assert str(0.1234567 * d) == "0.1234567*{x}"
        assert str(np.array([0, 1, -3, 0.987654]) * d) == "[0 1.0*{x} -3.0*{x} 0.987654*{x}]"
        assert str(pf.exp(1 + c) / d ** 2) == "exp(q1 + 1)/{x}**2"

    def test_par_functions_with_arrays(self):
        """Parameter functions with array arguments."""
        a = np.ones((2, 2))
        b = np.ones((2, 3))
        c = np.ones((2,))
        d = np.ones((2, 2))

        with pytest.raises(ValueError, match="all the arguments must be arrays of the same shape"):
            pf.beta(a, b)
        with pytest.raises(ValueError, match="all the arguments must be arrays of the same shape"):
            pf.beta(a, 1)
        with pytest.raises(ValueError, match="all the arguments must be arrays of the same shape"):
            pf.beta(1, a)
        with pytest.raises(ValueError, match="all the arguments must be arrays of the same shape"):
            pf.beta(a, c)

        res = pf.beta(a, d)
        assert res.shape == a.shape
        assert res.dtype == object

    @pytest.mark.parametrize("p", TEST_VALUES)
    def test_par_evaluate(self, p):
        x = FreeParameter("x")
        with pytest.raises(ParameterError, match="unbound parameter with no default value"):
            par_evaluate(x)

        # val only
        x.val = p
        assert np.all(par_evaluate(x) == p)

        # default only
        x.val = None
        x.default = p
        assert np.all(par_evaluate(x) == p)

        # both val and default
        x.val = p
        x.default = 0.0
        assert np.all(par_evaluate(x) == p)

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    @pytest.mark.parametrize("p", TEST_VALUES)
    def test_par_evaluate_dtype_numpy(self, p, dtype):
        """Test the numpy parameter evaluation works when a dtype is provided"""
        x = FreeParameter("x")
        x.val = p
        res = par_evaluate(x, dtype=dtype)
        assert res.dtype.type is dtype

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    @pytest.mark.parametrize("p", TEST_VALUES)
    def test_par_evaluate_dtype_TF(self, p, dtype):
        """Test the TF parameter evaluation works when a dtype is provided"""
        pytest.importorskip("tensorflow", minversion="2.0")
        import tensorflow as tf

        x = FreeParameter("x")
        x.val = tf.Variable(p)
        res = par_evaluate(x, dtype=dtype)
        assert res.dtype is tf.as_dtype(dtype)

    @pytest.mark.parametrize("p", TEST_VALUES)
    @pytest.mark.parametrize("q", TEST_VALUES)
    def test_parameter_arithmetic(self, p, q):
        """Test parameter arithmetic works as expected,"""
        pp = FreeParameter("x")
        qq = FreeParameter("y")
        pp.val = p
        qq.val = q
        binary_arithmetic(pp, qq, p, q)

    @pytest.mark.parametrize("p", TEST_VALUES)
    @pytest.mark.parametrize("q", SCALAR_TEST_VALUES)
    # adding an array to an array works elementwise in numpy, but symbolically one array is added to each element of the other array...
    def test_parameter_left_literal_arithmetic(self, p, q):
        """Test parameter arithmetic works as expected."""
        qq = FreeParameter("x")
        qq.val = q
        binary_arithmetic(p, qq, p, q)

    @pytest.mark.parametrize("p", TEST_VALUES)
    @pytest.mark.parametrize("q", SCALAR_TEST_VALUES)
    def test_parameter_right_literal_arithmetic(self, p, q):
        """Test parameter arithmetic works as expected."""
        pp = FreeParameter("x")
        pp.val = p
        binary_arithmetic(pp, q, p, q)

    @pytest.mark.parametrize("p", TEST_VALUES)
    def test_parameter_unary_negation(self, p):
        """Test unary negation works as expected."""
        pp = FreeParameter("x")
        pp.val = p
        assert par_evaluate(-p) == pytest.approx(-p)
        assert par_evaluate(-pp) == pytest.approx(-p)


@pytest.fixture
def applied_cmds(monkeypatch):
    """This fixture returns a list that dynamically keeps track
    of any applied commands within a test.

    Simply include this fixture in a test; any local calls to
    ``eng.run`` will subsequently be reflected in the fixture.

    Returns:
        list (Command): commands that were applied to the engine
    """
    applied = []

    def mock_run_prog(self, prog, **kwargs):
        """Mock function that is used for extracting
        the engine queue when running programs."""
        values = []
        values_dict = {}
        for cmd in prog.circuit:
            try:
                val = cmd.op.apply(cmd.reg, self.backend, **kwargs)
                if val is not None:
                    for i, r in enumerate(cmd.reg):
                        values.append(val[:, i])
                        if r.ind not in values_dict:
                            values_dict[r.ind] = []
                        values_dict[r.ind].append(val[:, i])
                applied.append(cmd)
            except NotImplementedError:
                for c in cmd.op._decompose(cmd.reg, **kwargs):
                    c.op.apply(c.reg, self.backend, **kwargs)
                    applied.append(c)
        return applied, values, values_dict

    with monkeypatch.context() as m:
        m.setattr(sf.LocalEngine, "_run_program", mock_run_prog)
        yield applied

    # tear down
    applied = []


class TestParameterTFIntegration:
    """Test integration of the parameter handling system with
    various gates and TensorFlow"""

    pytest.importorskip("tensorflow", minversion="2.0")

    @staticmethod
    def create_program(gate, mapping):
        """Utility method for constructing a program
        consisting of a single parametrized gate.

        Args:
            gate (strawberryfields.ops.Gate): gate to apply to the program
            mapping (dict[str, Any]): mapping from parameter name to value
        """
        prog = sf.Program(gate.ns)

        # create symbolic parameters
        params = []
        for param_name in mapping:
            params.append(prog.params(param_name))

        # construct program
        with prog as q:
            gate(*params) | q

        # bind numeric values
        prog.bind_params(mapping)
        return prog

    @pytest.mark.parametrize("gate", [sf.ops.Dgate, sf.ops.Sgate, sf.ops.Coherent])
    def test_single_mode_gate_complex_phase(self, backend, gate, applied_cmds):
        """Test non-decomposed single mode gates with complex phase arguments."""
        import tensorflow as tf

        mapping = {"r": tf.Variable(0.1), "phi": tf.Variable(0.2)}
        prog = self.create_program(gate, mapping)

        # verify bound parameters are correct
        assert prog.free_params["r"].val is mapping["r"]
        assert prog.free_params["phi"].val is mapping["phi"]

        # assert executed program is constructed correctly
        eng = sf.LocalEngine(backend)
        result = eng.run(prog, args=mapping)

        assert len(applied_cmds) == 1
        assert isinstance(applied_cmds[0].op, gate)
        assert applied_cmds[0].op.p[0].val == mapping["r"]
        assert applied_cmds[0].op.p[1].val == mapping["phi"]

    @pytest.mark.parametrize("gate", [sf.ops.BSgate, sf.ops.S2gate])
    def test_two_mode_gate_complex_phase(self, backend, gate, applied_cmds):
        """Test non-decomposed two-mode gates with complex phase arguments."""
        import tensorflow as tf

        mapping = {"r": tf.Variable(0.1), "phi": tf.Variable(0.2)}
        prog = self.create_program(gate, mapping)

        # verify bound parameters are correct
        assert prog.free_params["r"].val is mapping["r"]
        assert prog.free_params["phi"].val is mapping["phi"]

        # assert executed program is constructed correctly
        eng = sf.LocalEngine(backend)
        result = eng.run(prog, args=mapping)

        assert len(applied_cmds) == 1
        assert isinstance(applied_cmds[0].op, gate)
        assert applied_cmds[0].op.p[0].val == mapping["r"]
        assert applied_cmds[0].op.p[1].val == mapping["phi"]

    def test_zgate_decompose(self, backend, hbar, applied_cmds):
        """Test parameter processing occuring within the Zgate._decompose method."""
        import tensorflow as tf

        mapping = {"p": tf.Variable(0.1)}
        prog = self.create_program(sf.ops.Zgate, mapping)

        # verify bound parameters are correct
        assert prog.free_params["p"].val is mapping["p"]

        # assert executed program is constructed correctly
        eng = sf.LocalEngine(backend)
        result = eng.run(prog, args=mapping)

        assert len(applied_cmds) == 1
        assert isinstance(applied_cmds[0].op, sf.ops.Dgate)
        assert par_evaluate(applied_cmds[0].op.p[0]) == mapping["p"] / np.sqrt(2 * hbar)
        assert applied_cmds[0].op.p[1] == np.pi / 2

    @pytest.mark.parametrize(
        "gate", [sf.ops.Dgate, sf.ops.Coherent, sf.ops.DisplacedSqueezed, sf.ops.Catstate]
    )
    def test_complex_symbolic_tf(self, gate):
        """Test that passing a TF Variable to gates that previously accepted
        complex parameters raises an error when using the TF backend."""
        import tensorflow as tf

        with pytest.raises(ValueError, match="cannot be complex"):

            prog = sf.Program(1)
            alpha = prog.params("alpha")

            with prog.context as q:
                gate(alpha) | q[0]

            eng = sf.Engine("tf", backend_options={"cutoff_dim": 5})

            with tf.GradientTape() as tape:
                res = eng.run(prog, args={"alpha": tf.Variable(0.5 + 1j)})

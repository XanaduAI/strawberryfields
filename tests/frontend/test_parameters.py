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

#import strawberryfields as sf
from strawberryfields.parameters import par_is_symbolic, par_regref_deps, par_str, par_evaluate, MeasuredParameter
from strawberryfields.program_utils import RegRef


pytestmark = pytest.mark.frontend


# make test deterministic
np.random.seed(32)

TEST_VALUES = [3, 0.14, 4.2 + 0.5j]  #, np.random.random(3)]


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
        p = MeasuredParameter(RegRef(0))

        assert not par_is_symbolic(r)
        assert par_is_symbolic(p)
        assert par_is_symbolic(p + r)
        assert par_is_symbolic(p - r)
        assert par_is_symbolic(p * r)
        assert par_is_symbolic(p / r)
        assert par_is_symbolic(p ** r)
        assert par_is_symbolic(p / p)  # no simplification

    def test_par_regref_deps(self):
        """RegRef dependencies of parameters."""
        r = [RegRef(k) for k in range(2)]
        R0 = set([r[0]])
        R = set(r)

        s = 0.4
        p = MeasuredParameter(r[0])
        q = MeasuredParameter(r[1])

        assert par_regref_deps(s) == set()
        assert par_regref_deps(p) == R0
        assert par_regref_deps(s * p) == R0
        assert par_regref_deps(p + q) == R
        assert par_regref_deps(s * p + q ** s) == R

    def test_par_str(self):
        """String representations of parameters."""
        p = RegRef(0).par
        assert par_str(2.1) == "2.1"
        assert par_str(p) == "q[0].par"
        assert par_str(3*p) == "3*q[0].par"

    @pytest.mark.parametrize("p", TEST_VALUES)
    @pytest.mark.parametrize("q", TEST_VALUES)
    def test_parameter_arithmetic(self, p, q):
        """Test parameter arithmetic works as expected,"""
        pp = MeasuredParameter(RegRef(0))
        qq = MeasuredParameter(RegRef(1))
        pp.regref.val = p
        qq.regref.val = q
        #binary_arithmetic(pp, qq, p, q)

    @pytest.mark.parametrize("p", TEST_VALUES)
    @pytest.mark.parametrize("q", TEST_VALUES)
    def test_parameter_left_literal_arithmetic(self, p, q):
        """Test parameter arithmetic works as expected."""
        qq = MeasuredParameter(RegRef(0))
        qq.regref.val = q
        binary_arithmetic(p, qq, p, q)

    @pytest.mark.parametrize("p", TEST_VALUES)
    @pytest.mark.parametrize("q", TEST_VALUES)
    def test_parameter_right_literal_arithmetic(self, p, q):
        """Test parameter arithmetic works as expected."""
        pp = MeasuredParameter(RegRef(0))
        pp.regref.val = p
        binary_arithmetic(pp, q, p, q)

    @pytest.mark.parametrize("p", TEST_VALUES)
    def test_parameter_unary_negation(self, p):
        """Test unary negation works as expected."""
        pp = MeasuredParameter(RegRef(0))
        pp.regref.val = p
        assert par_evaluate(-p) == pytest.approx(-p)
        assert par_evaluate(-pp) == pytest.approx(-p)

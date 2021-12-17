# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Unit tests for the Xstrict compiler"""
import pytest

import strawberryfields as sf
from strawberryfields import ops

from conftest import program_equivalence, X8_spec, generate_X8_params

pytestmark = pytest.mark.frontend


class TestCompilation:
    """Tests for compilation using the X8_01 circuit specification"""

    def test_exact_template(self, tol):
        """Test compilation works for the exact circuit"""
        params = generate_X8_params(1, 0.3)
        prog = X8_spec.create_program(**params)
        prog2 = prog.compile(device=X8_spec, compiler="Xstrict")
        assert program_equivalence(prog, prog2, atol=tol, rtol=0)

    def test_invalid_parameter(self, tol):
        """Test exception is raised with invalid parameter values"""
        params = generate_X8_params(1, 0.3)
        prog = X8_spec.create_program(**params)
        prog.circuit[0].op.p = [0.5, 0]

        with pytest.raises(ValueError, match="has invalid value"):
            prog.compile(device=X8_spec, compiler="Xstrict")

    def test_invalid_gate(self, tol):
        """Test exception is raised with invalid primitive"""
        params = generate_X8_params(1, 0.3)
        prog = X8_spec.create_program(**params)
        prog.circuit[0].op.__class__ = sf.ops.CXgate

        with pytest.raises(
            sf.program_utils.CircuitError, match="CXgate cannot be used with the compiler"
        ):
            prog.compile(device=X8_spec, compiler="Xstrict")

    def test_invalid_topology(self, tol):
        """Test exception is raised with invalid topology"""
        params = generate_X8_params(1, 0.3)
        prog = X8_spec.create_program(**params)
        del prog.circuit[0]

        with pytest.raises(sf.program_utils.CircuitError, match="incompatible topology"):
            prog.compile(device=X8_spec, compiler="Xstrict")

# Copyright 2019-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Unit tests for the CircuitSpec class"""
import pytest
import sympy
import inspect

from strawberryfields.compilers.compiler import Compiler
from strawberryfields.program_utils import CircuitError


pytestmark = pytest.mark.frontend


class TestAbstractCircuitSpec:
    """Tests for the abstract CircuitSpec class.
    This class primarily consists of abstract methods,
    so only the methods containing logic are tested.
    """

    def test_program_topology_construction(self):
        """If a circuit spec includes a Blackbird program,
        the topology property should return the equivalent
        directed acyclic graph.
        """

        class DummyCompiler(Compiler):
            """Dummy circuit used to instantiate
            the abstract base class"""

            modes = 0
            remote = False
            local = True
            interactive: bool = True
            primitives: set = set()
            decompositions: set = set()

            circuit: str = inspect.cleandoc(
                """
                name test
                version 0.0

                Sgate(0.543, 0) | 0
                Dgate(-7.123) | 1
                BSgate(0.54) | 0, 1
                MeasureFock() | 0
                MeasureFock() | 2
                """
            )

        dummy = DummyCompiler()
        top = dummy.graph

        circuit = top.nodes().data()

        # check circuit is correct length
        assert len(circuit) == 5

        # check gates are correct
        assert circuit[0]["name"] == "Sgate"
        assert circuit[1]["name"] == "Dgate"
        assert circuit[2]["name"] == "BSgate"
        assert circuit[3]["name"] == "MeasureFock"
        assert circuit[4]["name"] == "MeasureFock"

        # check topology/edges between nodes
        edges = {(i, j) for i, j, d in top.edges().data()}
        assert edges == {(0, 2), (1, 2), (2, 3)}

    def test_template_topology_construction(self):
        """If a circuit spec includes a Blackbird template,
        the topology property should return the equivalent
        directed acyclic graph.
        """

        class DummyCompiler(Compiler):
            """Dummy circuit used to instantiate
            the abstract base class"""

            modes = 0
            remote = False
            local = True
            interactive: bool = True
            primitives: set = set()
            decompositions: set = set()

            circuit: str = inspect.cleandoc(
                """
                name test
                version 0.0

                Sgate({sq}, 0) | 0
                Dgate(-7.123) | 1
                BSgate({theta}) | 0, 1
                MeasureFock() | 0
                MeasureFock() | 2
                """
            )

        dummy = DummyCompiler()
        top = dummy.graph

        circuit = top.nodes().data()

        # check circuit is correct length
        assert len(circuit) == 5

        # check gates are correct
        assert circuit[0]["name"] == "Sgate"
        assert circuit[1]["name"] == "Dgate"
        assert circuit[2]["name"] == "BSgate"
        assert circuit[3]["name"] == "MeasureFock"
        assert circuit[4]["name"] == "MeasureFock"

        # check arguments
        assert isinstance(circuit[0]["args"][0], sympy.Symbol)
        assert circuit[0]["args"][1] == 0
        assert circuit[1]["args"] == [-7.123]
        assert isinstance(circuit[2]["args"][0], sympy.Symbol)

        # check topology/edges between nodes
        edges = {(i, j) for i, j, d in top.edges().data()}
        assert edges == {(0, 2), (1, 2), (2, 3)}

    def test_init_circuit(self):
        """Test initializing the circuit layout in a compiler."""

        class DummyCompiler(Compiler):
            """Dummy circuit used to instantiate
            the abstract base class"""

            modes = 0
            remote = False
            local = True
            interactive: bool = True
            primitives: set = set()
            decompositions: set = set()

        dummy = DummyCompiler()

        circuit = inspect.cleandoc(
            """
            name test
            version 0.0

            Sgate({sq}, 0) | 0
            Dgate(-7.123) | 1
            BSgate({theta}) | 0, 1
            MeasureFock() | 0
            MeasureFock() | 2
            """
        )

        # circuit layout is None if not set
        assert dummy.circuit is None

        # initialize circuit layout and assert that it is set
        dummy.init_circuit(circuit)
        assert dummy.circuit == circuit

        # initialize again to make sure it works
        dummy.init_circuit(circuit)
        assert dummy.circuit == circuit

        # creating a different circuit layout and attempt to initialize it should raise an error
        circuit = inspect.cleandoc(
            """
            name test
            version 0.0

            Sgate({sq}, 0) | 1
            Dgate(-7.123) | 0
            BSgate({theta}) | 0, 1
            MeasureFock() | 1
            """
        )
        with pytest.raises(CircuitError, match="Circuit already set in compiler"):
            dummy.init_circuit(circuit)

        # resetting the circuit layout and then initializing it should work
        dummy.reset_circuit()
        dummy.init_circuit(circuit)
        assert dummy.circuit == circuit

    def test_init_circuit_wrong_type(self):
        """Test initializing a circuit layout that is not a string in a compiler."""

        class DummyCompiler(Compiler):
            """Dummy circuit used to instantiate
            the abstract base class"""

            modes = 0
            remote = False
            local = True
            interactive: bool = True
            primitives: set = set()
            decompositions: set = set()

        dummy = DummyCompiler()
        assert dummy.circuit is None

        with pytest.raises(
            TypeError, match="Layout must be a string representing the Blackbird circuit."
        ):
            dummy.init_circuit(layout=42)

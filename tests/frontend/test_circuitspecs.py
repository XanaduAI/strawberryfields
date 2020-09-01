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
r"""Unit tests for the CircuitSpec class"""
import textwrap

import pytest

from strawberryfields.circuitspecs.circuit_specs import CircuitSpecs


pytestmark = pytest.mark.frontend


class TestAbstractCircuitSpec:
	"""Tests for the abstract CircuitSpec class.
	This class primarily consists of abstract methods,
	so only the methods containing logic are tested.
	"""

	def test_no_parameter_ranges(self):
		"""If not defined, the parameter_ranges property
		should return an empty dictionary."""

		class DummyCircuit(CircuitSpecs):
			"""Dummy circuit used to instantiate
			the abstract base class"""
			modes = 0
			remote = False
			local = True
			interactive = True
			primitives = set()
			decompositions = set()

		dummy = DummyCircuit()

		assert isinstance(dummy.parameter_ranges, dict)
		assert not dummy.parameter_ranges

	def test_program_topology_construction(self):
		"""If a circuit spec includes a Blackbird program,
		the topology property should return the equivalent
		directed acyclic graph.
		"""

		class DummyCircuit(CircuitSpecs):
			"""Dummy circuit used to instantiate
			the abstract base class"""
			modes = 0
			remote = False
			local = True
			interactive = True
			primitives = set()
			decompositions = set()

			circuit = textwrap.dedent(
				"""\
				name test
				version 0.0

				Sgate(0.543, 0) | 0
				Dgate(-7.123) | 1
				BSgate(0.54) | 0, 1
				MeasureFock() | 0
				MeasureFock() | 2
				"""
			)

		dummy = DummyCircuit()
		top = dummy.graph

		circuit = top.nodes().data()

		# check circuit is correct length
		assert len(circuit) == 5

		# check gates are correct
		assert circuit[0]['name'] == 'Sgate'
		assert circuit[1]['name'] == 'Dgate'
		assert circuit[2]['name'] == 'BSgate'
		assert circuit[3]['name'] == 'MeasureFock'
		assert circuit[4]['name'] == 'MeasureFock'

		# check topology/edges between nodes
		edges = {(i, j) for i, j, d in top.edges().data()}
		assert edges == {(0, 2), (1, 2), (2, 3)}

	def test_template_topology_construction(self):
		"""If a circuit spec includes a Blackbird template,
		the topology property should return the equivalent
		directed acyclic graph, with all parameters set to zero.
		"""

		class DummyCircuit(CircuitSpecs):
			"""Dummy circuit used to instantiate
			the abstract base class"""
			modes = 0
			remote = False
			local = True
			interactive = True
			primitives = set()
			decompositions = set()

			circuit = textwrap.dedent(
				"""\
				name test
				version 0.0

				Sgate({sq}, 0) | 0
				Dgate(-7.123) | 1
				BSgate({theta}) | 0, 1
				MeasureFock() | 0
				MeasureFock() | 2
				"""
			)

		dummy = DummyCircuit()
		top = dummy.graph

		circuit = top.nodes().data()

		# check circuit is correct length
		assert len(circuit) == 5

		# check gates are correct
		assert circuit[0]['name'] == 'Sgate'
		assert circuit[1]['name'] == 'Dgate'
		assert circuit[2]['name'] == 'BSgate'
		assert circuit[3]['name'] == 'MeasureFock'
		assert circuit[4]['name'] == 'MeasureFock'

		# check arguments
		assert circuit[0]['args'] == [0, 0]
		assert circuit[1]['args'] == [-7.123]
		assert circuit[2]['args'] == [0]

		# check topology/edges between nodes
		edges = {(i, j) for i, j, d in top.edges().data()}
		assert edges == {(0, 2), (1, 2), (2, 3)}

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
r"""
Unit tests for strawberryfields.apps.clique
"""
# pylint: disable=no-self-use,unused-argument
import os
import functools

import networkx as nx
import numpy as np
import pytest

from strawberryfields.apps import clique
from strawberryfields.cli import command_line_interface

import sys


pytestmark = pytest.mark.cli

class TestStarshipCli:
    """Tests for the Strawberry Fields command line interface."""
    
    def test_ping(self, monkeypatch):

        class MockConnection:

            def __init__(self):
                self.ping = None

            def ping(self):
                self.ping = "SuccessfulPing" 

        with monkeypatch.context() as m:
            mock_connection = MockConnection()
            m.setattr("argparse.ArgumentParser", "parse_args", mock_connection)
            os.system("starship --ping")
            assert mock_connection.ping == "SuccessfulPing"

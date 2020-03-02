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
import strawberryfields as sf

import sys
from os.path import expanduser, join

# use expanduser to locate its home dir and join bin and candy module paths
starship_module_path =  join(expanduser("~"), "xanadu", "strawberryfields", "starship")

# load the module without .py extension
starship = imp.load_source("starship", starship_module_path)
sys.path.extend(['~/xanadu/strawberryfields'])
import starship

pytestmark = pytest.mark.cli

@pytest.fixture
def run(testdir):
    def do_run(*args):
        args = ["pyconv"] + list(args)
        return testdir._run(*args)
    return do_run

class TestStarshipCli:
    """Tests for the Strawberry Fields command line interface."""
    
    def test_ping(self, monkeypatch):

        class MockConnection:

            def __init__(self):
                self.ping = None

            def ping(self):
                self.ping = "SuccessfulPing" 

        print(dir(starship))
        with monkeypatch.context() as m:
            mock_connection = MockConnection()
            m.setattr("starship", "connection", mock_connection)
            os.system("starship --ping")
            assert mock_connection.ping == "SuccessfulPing"

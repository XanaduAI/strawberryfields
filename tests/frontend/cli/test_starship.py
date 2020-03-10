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
Unit tests for the Strawberry Fields command line interface.
"""
# pylint: disable=no-self-use,unused-argument
import os
import functools
import argparse

import networkx as nx
import numpy as np
import pytest

from strawberryfields.apps import clique
from strawberryfields import cli as cli

import sys


pytestmark = pytest.mark.cli

class TestCreateParser:
    # TODO
    def test_ping(self):
        # TODO

        parser = cli.create_parser()
        args = parser.parse_args(['--ping'])
        assert args.ping

    def test_run(self):
        # TODO

        parser = cli.create_parser()
        args = parser.parse_args(['run', 'SomePath'])
        assert args.input == 'SomePath'

    def test_output(self):
        # TODO

        parser = cli.create_parser()
        args = parser.parse_args(['run', 'SomeInputPath', '--output', 'SomeOutputPath'])
        assert args.input == 'SomeInputPath'
        assert args.output == 'SomeOutputPath'

    grouped_argument_options = [
                                ('--ping',),
                                ('--token', 'SomeAuth'),
                                ('--configure',),
                                ('run', 'SomePath')
                                ]

    # Output of a programatic matching of every possible pair (without
    # repetition):
    # list(itertools.combinations(grouped_argument_options, r=2))
    combination_of_grouped = [(('--ping',), ('--token', 'SomeAuth')),
                             (('--ping',), ('--configure',)),
                             (('--ping',), ('run', 'SomePath')),
                             (('--token', 'SomeAuth'), ('--configure',)),
                             (('--token', 'SomeAuth'), ('run', 'SomePath')),
                             (('--configure',), ('run', 'SomePath'))]

    @pytest.mark.parametrize('option1, option2', combination_of_grouped)
    def test_error_mutually_exclusive_group(self, option1, option2):
        parser = cli.create_parser()
        with pytest.raises(SystemExit, match='2'):
            with pytest.raises(argparse.ArgumentError, match='not allowed with argument'):
                args = parser.parse_args([*option1, *option2])

class MockConnection:
    # TODO
    def __init__(self):
        self.pinging = None

    def ping(self):
        self.pinging = "SuccessfulPing"

class MockSysStdout:
    # TODO

    def __init__(self):
        self.write_output = []

    def write(self, message):
        self.write_output = message


class TestPing:
    # TODO

    def test_correct(self, monkeypatch):
        # TODO

        with monkeypatch.context() as m:
            mock_sys_stdout = MockSysStdout()

            m.setattr(cli, "Connection", MockConnection)
            m.setattr(sys, "stdout", mock_sys_stdout)

            assert mock_connection.pinging is None

            with pytest.raises(SystemExit):
                cli.ping()
                assert mock_sys_stdout.write_output == "You have successfully authenticated to the platform!\n"
                assert mock_connection.pinging == "SuccessfulPing"



#class TestConfigureEverything:



class TestRunProgram:

    def test_correct_arguments(self, monkeypatch):

        mocked_stdout = MockSysStdout()
        with monkeypatch.context() as m:
            m.setattr(cli.StarshipEngine, "__init__", lambda: None)
            m.setattr(sys, "stdout", mocked_stdout)



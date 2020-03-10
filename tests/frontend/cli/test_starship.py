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
from strawberryfields.configuration import store_account
from strawberryfields import cli as cli

import sys


pytestmark = pytest.mark.cli

class TestMain:
    def test_ping(self):
        # TODO

        parser = cli.create_parser()
        args = parser.parse_args(['--ping'])
        assert args.ping
 
class TestCreateParser:
    """Tests for creating a parser object."""

    def test_general_details(self):
        """Test the general details of the parser created."""
        parser = cli.create_parser()
        parser._optionals.title = "General Options"
        parser.usage="starship <command> [<args>]",
        parser.description="These are common options when working on the Xanadu cloud platform."
        assert parser.add_help

    @pytest.mark.parametrize("option", ['--ping', '-p'])
    def test_ping(self, option):
        """Test that specifying --ping to the CLI sets the correct attribute."""
        parser = cli.create_parser()

        args = parser.parse_args([option])
        assert args.ping

    @pytest.mark.parametrize("token_option", ['--token', '-t'])
    def test_configure_token(self, token_option):
        """Test that specifying configure, --token and passing an argument to
        the CLI sets the correct attribute."""
        parser = cli.create_parser()

        args = parser.parse_args(['configure', token_option, 'SomeToken'])
        assert args.func is cli.configure
        assert args.token == 'SomeToken'
        assert not args.local

    def test_configure_everything(self):
        """Test that specifying configure, --local to the CLI sets the correct
        attribute."""
        parser = cli.create_parser()
        args = parser.parse_args(['configure'])

        assert args.func is cli.configure
        assert not args.local

    @pytest.mark.parametrize("token_option", ['--token', '-t'])
    @pytest.mark.parametrize("local_option", ['--local', '-l'])
    def test_configure_token_locally(self, token_option, local_option):
        """Test that specifying configure, --token, --local and passing an argument to
        the CLI sets the correct attribute."""
        parser = cli.create_parser()

        args = parser.parse_args(['configure', token_option, 'SomeToken', local_option])
        assert args.func is cli.configure
        assert args.token == 'SomeToken'

    @pytest.mark.parametrize("option", ['--local', '-l'])
    def test_configure_everything_locally(self, option):
        """Test that specifying configure, --local to the CLI sets the correct
        attribute."""
        parser = cli.create_parser()
        args = parser.parse_args(['configure', option])

        assert args.func is cli.configure
        assert args.local

    def test_run(self):
        """Test that specifying input and passing an argument to the CLI sets
        the correct attribute."""
        parser = cli.create_parser()
        args = parser.parse_args(['run', 'SomePath'])

        assert args.func is cli.run_blackbird_script
        assert args.input == 'SomePath'

    def test_output(self):
        """Test that specifying input, --output and passing the arguments to
        the CLI sets the correct attributes."""
        parser = cli.create_parser()
        args = parser.parse_args(['run', 'SomeInputPath', '--output', 'SomeOutputPath'])

        assert args.func is cli.run_blackbird_script
        assert args.input == 'SomeInputPath'
        assert args.output == 'SomeOutputPath'

class MockArgs:
    # TODO
    def __init__(self):
        self.token = None
        self.local = None

class MockStoreAccount:
    def __init__(self):
        self.kwargs = None

    def store_account(self, kwargs):
        self.kwargs = kwargs

class TestConfigure:

    def test_token(self):
        with monkeypatch.context() as m:
            mock_store_account = MockStoreAccount()
            m.setattr(cli, "store_account", mock_store_account.store_account)
            args = MockArgs()
            args.token = "SomeToken"
            cli.configure(args)
            assert mock_store_account.kwargs == {"authentication_token": "SomeToken"}

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
    """Tests for the pinging mechanism of the CLI."""

    def test_success(self, monkeypatch):
        """Test that pinging was successful."""
        with monkeypatch.context() as m:
            mock_sys_stdout = MockSysStdout()

            m.setattr(cli, "Connection", MockConnection)
            m.setattr(sys, "stdout", mock_sys_stdout)

            with pytest.raises(SystemExit):
                cli.ping()
                assert mock_sys_stdout.write_output == "You have successfully authenticated to the platform!\n"
                assert mock_connection.pinging == "SuccessfulPing"

    def test_fail(self, monkeypatch):
        """Test that pinging failed."""
        with monkeypatch.context() as m:
            mock_sys_stdout = MockSysStdout()

            m.setattr(cli, "Connection", MockConnection)
            m.setattr(sys, "stdout", mock_sys_stdout)

            with pytest.raises(SystemExit):
                cli.ping()
                assert mock_sys_stdout.write_output == "There was a problem when authenticating to the platform!\n"
                assert mock_connection.pinging == "SuccessfulPing"

#class TestConfigureEverything:



class TestRunProgram:

    def test_correct_arguments(self, monkeypatch):

        mocked_stdout = MockSysStdout()
        with monkeypatch.context() as m:
            m.setattr(cli.StarshipEngine, "__init__", lambda: None)
            m.setattr(sys, "stdout", mocked_stdout)



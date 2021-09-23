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

from strawberryfields.api import Result
from strawberryfields.apps import clique
from strawberryfields.configuration import store_account, ConfigurationError
from strawberryfields import cli as cli

import sys
import builtins

pytestmark = pytest.mark.frontend


class TestCreateParser:
    """Tests for creating a parser object."""

    def test_general_details(self):
        """Test the general details of the parser created."""
        parser = cli.create_parser()
        assert parser._optionals.title == "General Options"
        assert (
            parser.description
            == "See below for available options and commands for working with the Xanadu cloud platform."
        )
        assert parser.add_help

    @pytest.mark.parametrize("option", ["--ping", "-p"])
    def test_ping(self, option):
        """Test that specifying --ping to the CLI sets the correct attribute."""
        parser = cli.create_parser()

        args = parser.parse_args([option])
        assert args.ping

    @pytest.mark.parametrize("token_option", ["--token", "-t"])
    def test_configure_token(self, token_option):
        """Test that specifying configure, --token and passing an argument to
        the CLI sets the correct attribute."""
        parser = cli.create_parser()

        args = parser.parse_args(["configure", token_option, "SomeToken"])
        assert args.func is cli.configure
        assert args.token == "SomeToken"
        assert not args.local

    def test_configuration_wizard(self):
        """Test that specifying configure, --local to the CLI sets the correct
        attribute."""
        parser = cli.create_parser()
        args = parser.parse_args(["configure"])

        assert args.func is cli.configure
        assert not args.local

    @pytest.mark.parametrize("token_option", ["--token", "-t"])
    @pytest.mark.parametrize("local_option", ["--local", "-l"])
    def test_configure_token_locally(self, token_option, local_option):
        """Test that specifying configure, --token, --local and passing an argument to
        the CLI sets the correct attribute."""
        parser = cli.create_parser()

        args = parser.parse_args(["configure", token_option, "SomeToken", local_option])
        assert args.func is cli.configure
        assert args.token == "SomeToken"

    @pytest.mark.parametrize("option", ["--local", "-l"])
    def test_configuration_wizard_locally(self, option):
        """Test that specifying configure, --local to the CLI sets the correct
        attribute."""
        parser = cli.create_parser()
        args = parser.parse_args(["configure", option])

        assert args.func is cli.configure
        assert args.local

    def test_run(self):
        """Test that specifying input and passing an argument to the CLI sets
        the correct attribute."""
        parser = cli.create_parser()
        args = parser.parse_args(["run", "SomePath"])

        assert args.func is cli.run_blackbird_script
        assert args.input == "SomePath"

    def test_output(self):
        """Test that specifying input, --output and passing the arguments to
        the CLI sets the correct attributes."""
        parser = cli.create_parser()
        args = parser.parse_args(["run", "SomeInputPath", "--output", "SomeOutputPath"])

        assert args.func is cli.run_blackbird_script
        assert args.input == "SomeInputPath"
        assert args.output == "SomeOutputPath"


class MockArgs:
    """A mock class used for mocking the args that are parsed from the command
    line."""

    def __init__(self):
        self.token = None
        self.local = None
        self.input = None
        self.output = None


class MockStoreAccount:
    """A mock class used for capturing the arguments with which the store_account
    function is being called."""

    def __init__(self):
        self.kwargs = None

    def store_account(self, **kwargs):
        self.kwargs = kwargs


EXPECTED_KWARGS = {
    "authentication_token": "",
    "hostname": "platform.strawberryfields.ai",
    "use_ssl": True,
    "port": 443,
}


class TestConfigure:
    """Unit tests for the configure function checking that the lines of
    execution is correct."""

    def test_token(self, monkeypatch):
        """Tests that if a token was given as a command line argument then
        configuration takes place accordingly."""
        with monkeypatch.context() as m:
            mock_store_account = MockStoreAccount()
            m.setattr(cli, "store_account", mock_store_account.store_account)

            args = MockArgs()
            args.token = "SomeToken"

            cli.configure(args)
            assert mock_store_account.kwargs == {"authentication_token": "SomeToken"}

    def test_configuration_wizard(self, monkeypatch):
        """Tests that if no token was given as a command line argument then
        configuration takes place using the configuration_wizard function."""
        with monkeypatch.context() as m:
            mock_store_account = MockStoreAccount()
            m.setattr(cli, "configuration_wizard", lambda: cli.create_config()["api"])
            m.setattr(cli, "store_account", mock_store_account.store_account)

            args = MockArgs()
            args.token = False

            cli.configure(args)
            assert mock_store_account.kwargs == EXPECTED_KWARGS

    def test_token_local(self, monkeypatch):
        """Tests that if a token was given as a command line argument and
        local configuration was specified then configuration takes place
        accordingly."""
        with monkeypatch.context() as m:
            mock_store_account = MockStoreAccount()
            m.setattr(cli, "store_account", mock_store_account.store_account)

            args = MockArgs()
            args.token = "SomeToken"
            args.local = True

            cli.configure(args)
            assert mock_store_account.kwargs == {
                "authentication_token": "SomeToken",
                "location": "local",
            }

    def test_configuration_wizard_local(self, monkeypatch):
        """Tests that if no token was given as a command line argument and
        local configuration was specified then configuration takes place using
        the configuration_wizard function."""
        with monkeypatch.context() as m:
            mock_store_account = MockStoreAccount()
            m.setattr(cli, "configuration_wizard", lambda: cli.create_config()["api"])
            m.setattr(cli, "store_account", mock_store_account.store_account)

            args = MockArgs()
            args.token = False
            args.local = True

            cli.configure(args)
            m.setitem(EXPECTED_KWARGS, "location", "local")

            assert mock_store_account.kwargs == EXPECTED_KWARGS


class MockSuccessfulConnection:
    """A Connection class mocking a successful establishment of connection."""

    def ping(self):
        return True


class MockFailedConnection:
    """A Connection class mocking a failed establishment of connection."""

    def ping(self):
        return False


class TestPing:
    """Tests for the pinging mechanism of the CLI."""

    def test_success(self, monkeypatch, capsys):
        """Test that pinging was successful."""
        with monkeypatch.context() as m:
            m.setattr(cli, "Connection", MockSuccessfulConnection)
            cli.ping()

        out, _ = capsys.readouterr()
        assert out == "You have successfully authenticated to the platform!\n"

    def test_fail(self, monkeypatch, capsys):
        """Test that pinging failed."""
        with monkeypatch.context() as m:
            m.setattr(cli, "Connection", MockFailedConnection)
            out, _ = capsys.readouterr()

            cli.ping()

        out, _ = capsys.readouterr()
        assert out == "There was a problem when authenticating to the platform!\n"


# Keys are adjusted to the prompt message displayed to the user
MOCK_PROMPTS = {
    "token": "MyAuth",
    "hostname": "MyHost",
    "port": 123,
    "SSL": "n",
}

EXPECTED_KWARGS_FOR_PROMPTS = {
    "authentication_token": "MyAuth",
    "hostname": "MyHost",
    "port": 123,
    "use_ssl": False,
}


def mock_input(arg):
    """A mock function that substitutes the built-in input function."""
    option = {k: v for k, v in MOCK_PROMPTS.items() if k in arg}
    if option and len(option) == 1:
        return list(option.values())[0]


class TestConfigureEverything:
    """Unit tests for the configuration_wizard function."""

    def test_no_auth_exit_with_message(self, monkeypatch, capsys):
        """Test that by default the configuration_wizard function exits with a
        relevant message."""
        with monkeypatch.context() as m:
            m.setattr(builtins, "input", lambda *args: False)
            with pytest.raises(SystemExit):
                cli.configuration_wizard()

            out, _ = capsys.readouterr()

        assert out == "No authentication token was provided, please configure again."

    def test_auth_correct(self, monkeypatch):
        """Test that by default the configuration_wizard function works
        correctly, once the authentication token is passed."""
        with monkeypatch.context() as m:
            auth_prompt = "Please enter the authentication token"
            default_config = cli.create_config()["api"]
            default_auth = "SomeAuth"
            default_config["authentication_token"] = default_auth

            m.setattr(builtins, "input", lambda arg: default_auth if (auth_prompt in arg) else "")
            assert cli.configuration_wizard() == default_config

    def test_correct_inputs(self, monkeypatch):
        """Test that the configuration_wizard function returns a dictionary
        based on the inputs, when each configuration detail was inputted."""
        with monkeypatch.context() as m:

            auth_prompt = "Please enter the authentication token"
            default_config = cli.create_config()["api"]
            default_auth = "SomeAuth"
            default_config["authentication_token"] = default_auth

            m.setattr(builtins, "input", mock_input)
            assert cli.configuration_wizard() == EXPECTED_KWARGS_FOR_PROMPTS


class MockProgram:
    """A mock class used for capturing the arguments with which the
    the Program class is instantiated."""

    def __init__(self):

        self.target = "X8_01"
        self.result = None


class MockRemoteEngine:
    """A mock class used for capturing the arguments with which the
    the RemoteEngine class is instantiated and its run method is called."""

    def __init__(self, target):
        self.result = None
        self.target = target

    def run(self, program):
        if program:
            return program.result


class MockWriteScriptResults:
    """A mock class used for capturing the arguments with which the
    write_script_results function is being called."""

    def __init__(self):
        self.called = False
        self.output = None

    def write_script_results(self, output):
        self.called = True
        self.output = output


TEST_SCRIPT = """\
name template_1x2_X8_01     # Name of the program
version 1.0                 # Blackbird version number
target X8_01 (shots = 50)   # This program will run on X8_01 for 50 shots

# Define the interferometer phase values
float phi0 = 0.574
float phi1 = 1.33
MZgate(phi0, phi1) | [0, 1]
MZgate(phi0, phi1) | [4, 5]

# Perform a photon number counting measurement
MeasureFock() | [0, 1, 2, 3, 4, 5, 6, 7]
"""


class TestRunBlackbirdScript:
    """Unit tests for the run_blackbird_script function."""

    def test_exit_if_file_not_found(self, monkeypatch, capsys):
        """Tests that if the input script file was not found then a system exit
        occurs along with a message being outputted."""
        mocked_program = MockProgram()
        mocked_args = MockArgs()

        def mock_load(arg):
            raise FileNotFoundError

        with monkeypatch.context() as m:
            m.setattr(cli, "load", mock_load)

            with pytest.raises(SystemExit):
                cli.run_blackbird_script(mocked_args)

        out, _ = capsys.readouterr()
        assert "blackbird script was not found" in out

    def test_result_is_none(self, monkeypatch, capsys):
        """Tests that the write_script_results function is not called if the
        results from the run method of the engine returned a None."""
        mocked_program = MockProgram()
        mocked_args = MockArgs()
        mocked_write_script_results = MockWriteScriptResults()

        with monkeypatch.context() as m:
            m.setattr(cli, "load", lambda arg: mocked_program)
            m.setattr(cli, "RemoteEngine", MockRemoteEngine)
            m.setattr(cli, "write_script_results", mocked_write_script_results.write_script_results)

            with pytest.raises(SystemExit):
                cli.run_blackbird_script(mocked_args)

        out, _ = capsys.readouterr()
        assert "Executing program on remote hardware..." in out

        # Check that the write_script_results function was not called
        assert not mocked_write_script_results.called


test_samples = [1, 2, 3, 4]


class MockRemoteEngineIntegration:
    """A mock class used for capturing the arguments with which the
    the RemoteEngine class is instantiated and its run method is called when
    multiple components are tested."""

    def __init__(self, target):
        self.result = None
        self.target = target

    def run(self, program):
        if program:
            return Result(test_samples)


class TestRunBlackbirdScriptIntegration:
    """Tests for the run_blackbird_script function that integrate multiple
    components."""

    def test_integration_std_out(self, tmpdir, monkeypatch, capsys):
        """Tests that a blackbird script was loaded and samples were written to
        the standard output using the run_blackbird_script function."""

        filepath = tmpdir.join("test_script.xbb")

        with open(filepath, "w") as f:
            f.write(TEST_SCRIPT)

        mocked_args = MockArgs()
        mocked_args.input = filepath

        with monkeypatch.context() as m:
            m.setattr(cli, "RemoteEngine", MockRemoteEngineIntegration)
            cli.run_blackbird_script(mocked_args)

        out, err = capsys.readouterr()

        execution_message = "Executing program on remote hardware...\n"

        outputs = execution_message + str(Result(test_samples).samples)
        assert outputs == out

    def test_integration_file(self, tmpdir, monkeypatch, capsys):
        """Tests that a blackbird script was loaded and samples were written to
        the specified output file using the run_blackbird_script function."""

        filepath = tmpdir.join("test_script.xbb")

        with open(filepath, "w") as f:
            f.write(TEST_SCRIPT)

        mocked_args = MockArgs()
        mocked_args.input = filepath

        out_filepath = tmpdir.join("test_script.xbb")
        mocked_args.output = out_filepath

        with monkeypatch.context() as m:
            m.setattr(cli, "RemoteEngine", MockRemoteEngineIntegration)
            cli.run_blackbird_script(mocked_args)

        with open(filepath, "r") as f:
            results_from_file = f.read()

        out, _ = capsys.readouterr()
        assert out == "Executing program on remote hardware...\n"
        assert results_from_file == str(Result(test_samples).samples)


class TestWriteScriptResults:
    """Tests for the write_script_results function."""

    def test_write_to_file(self, tmpdir):
        """Tests that the write_script_results function writes to file
        correctly."""
        some_samples = [1, 2, 3, 4, 5]
        filepath = tmpdir.join("test_script.xbb")

        cli.write_script_results(some_samples, output_file=filepath)

        with open(filepath, "r") as f:
            results_from_file = f.read()

        assert results_from_file == str(some_samples)

    def test_write_to_std_out(self, monkeypatch, capsys):
        """Tests that the write_script_results function writes to the standard
        output correctly."""
        some_samples = [1, 2, 3, 4, 5]

        with monkeypatch.context() as m:
            cli.write_script_results(some_samples)

        out, _ = capsys.readouterr()
        assert out == str(some_samples)
